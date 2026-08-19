# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT GLM4-MoE model override.

Corresponds to: vllm/model_executor/models/glm4_moe.py

Target   : vllm.model_executor.models.glm4_moe.Glm4MoeForCausalLM
           (arch "Glm4MoeForCausalLM")
Baseline : main_6e448d0
Why      : the fork runs the GLM4-MoE router gate in bf16 for Iluvatar grouped
           top-k kernels and makes q/k/v contiguous after qkv split before the
           attention backend consumes them.
How      : Use the official ``ModelRegistry.register_model`` hook. The subclass
           keeps community construction unchanged, then swaps the affected
           ``Glm4MoeAttention`` / ``Glm4MoE`` instances in-place and converts the
           router gate parameters before checkpoint loading.
Scope    : GLM-4.5/4.6/4.7 MoE models on Iluvatar.
Remove   : when the community model can run these paths unchanged on Iluvatar.

Registration: ``register_model()`` below is invoked by
``vllm_iluvatar.ilu_register.apply_oot_registrations`` during register_general.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.glm4_moe import Glm4MoE as _Glm4MoE
from vllm.model_executor.models.glm4_moe import (
    Glm4MoeAttention as _Glm4MoeAttention,
)
from vllm.model_executor.models.glm4_moe import (
    Glm4MoeForCausalLM as _Glm4MoeForCausalLM,
)


class Glm4MoeAttention(_Glm4MoeAttention):
    """GLM4-MoE attention that provides contiguous q/k/v tensors."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if self.use_qk_norm:
            q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(
                q.shape
            )
            k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(
                k.shape
            )

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Glm4MoE(_Glm4MoE):
    """GLM4-MoE block that feeds bf16 router logits to FusedMoE."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states.to(dtype=torch.bfloat16))
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return final_hidden_states.view(num_tokens, hidden_dim)


def glm4_moe_attention_contiguous_overrides(model) -> None:
    """Swap GLM4-MoE attention modules to the contiguous-qkv subclass."""
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if not isinstance(attn, _Glm4MoeAttention):
            continue
        attn.__class__ = Glm4MoeAttention


def glm4_moe_bf16_router_overrides(model) -> None:
    """Run the GLM4-MoE router gate and correction bias in bf16."""
    for layer in getattr(model, "layers", []):
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, _Glm4MoE):
            continue
        mlp.__class__ = Glm4MoE
        mlp.gate.weight.data = mlp.gate.weight.data.to(torch.bfloat16)
        if mlp.gate.bias is not None:
            mlp.gate.bias.data = mlp.gate.bias.data.to(torch.bfloat16)
        if hasattr(mlp.gate, "e_score_correction_bias"):
            mlp.gate.e_score_correction_bias.data = (
                mlp.gate.e_score_correction_bias.data.to(torch.bfloat16)
            )


class Glm4MoeForCausalLM(_Glm4MoeForCausalLM):
    """GLM4-MoE causal LM wired for Iluvatar router and attention deltas."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        glm4_moe_attention_contiguous_overrides(self.model)
        glm4_moe_bf16_router_overrides(self.model)


def register_model() -> None:
    """Model registration hook: point GLM4-MoE arch at the Iluvatar class."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Glm4MoeForCausalLM",
        "vllm_iluvatar.model_executor.models.glm4_moe:Glm4MoeForCausalLM",
    )
