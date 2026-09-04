# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT MiniMax-M2 model overrides.

Corresponds to: vllm/model_executor/models/minimax_m2.py
Target   : arch ``MiniMaxM2ForCausalLM``
Baseline : main_4cd213ead42e
Why      : Iluvatar keeps the router weight in BF16 while returning FP32
           logits, and routes DP/EP MoE execution through ``IluFusedMoEEP``.
How      : Promote community MoE and gate modules in place after construction,
           then select the causal-LM subclass with ``ModelRegistry``.
Scope    : MiniMax-M2 family models on Iluvatar, including DP/EP execution.
Remove   : when upstream supports the Iluvatar router and DP/EP contracts.
"""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.router.gate_linear import (
    GateLinear as _GateLinear,
)
from vllm.model_executor.models.minimax_m2 import (
    MiniMaxM2ForCausalLM as _MiniMaxM2ForCausalLM,
)
from vllm.model_executor.models.minimax_m2 import MiniMaxM2MoE as _MiniMaxM2MoE

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
    IluFusedMoEEP,
    set_max_num_seqs,
)


class IluMiniMaxM2GateLinear(_GateLinear):
    """MiniMax router gate using BF16 weights and FP32 logits."""

    @staticmethod
    def promote(layer: _GateLinear) -> None:
        from vllm.model_executor.utils import replace_parameter

        replace_parameter(layer, "weight", layer.weight.to(torch.bfloat16))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # Bypass GateLinear's CUDA-specific dispatch and use its
        # ReplicatedLinear parent with the promoted BF16 weight.
        output, output_bias = super(_GateLinear, self).forward(x)
        return output.to(torch.float32), output_bias


class IluMiniMaxM2MoE(_MiniMaxM2MoE, IluFusedMoEEP):
    """Community MiniMax MoE block with Iluvatar DP/EP execution."""

    @staticmethod
    def promote(layer: _MiniMaxM2MoE) -> None:
        """Promote a constructed community MoE without rebuilding parameters."""
        layer.__class__ = IluMiniMaxM2MoE
        layer.prefix = layer.gate.prefix.removesuffix(".gate")
        layer.gate.__class__ = IluMiniMaxM2GateLinear
        IluMiniMaxM2GateLinear.promote(layer.gate)
        layer.init_ep()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self._should_use_forward_dp_ep():
            final_hidden_states = self.forward_dp_ep(hidden_states)
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        return final_hidden_states.view(num_tokens, hidden_dim)


def minimax_m2_moe_overrides(model) -> None:
    """Promote MiniMax-M2 MoE blocks for BF16 routing and Iluvatar DP/EP.

    Remove this override when the community router supports BF16 weights with
    FP32 logits and its MoE path supports Iluvatar DP/EP execution.
    """
    for layer in getattr(model, "layers", []):
        moe = getattr(layer, "block_sparse_moe", None)
        if isinstance(moe, _MiniMaxM2MoE):
            IluMiniMaxM2MoE.promote(moe)


class MiniMaxM2ForCausalLM(_MiniMaxM2ForCausalLM):
    """MiniMax-M2 causal LM wired for Iluvatar router and DP/EP behavior."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        set_max_num_seqs(vllm_config.scheduler_config.max_num_seqs)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        minimax_m2_moe_overrides(self.model)


def register_model() -> None:
    """Register the Iluvatar MiniMax-M2 causal-LM implementation."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "MiniMaxM2ForCausalLM",
        "vllm_iluvatar.model_executor.models.minimax_m2:MiniMaxM2ForCausalLM",
    )
