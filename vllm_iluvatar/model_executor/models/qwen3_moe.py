# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT Qwen3-MoE model override.

Corresponds to:
    vllm/model_executor/models/qwen3_moe.py

Target:
    vllm.model_executor.models.qwen3_moe.Qwen3MoeForCausalLM
    (arch "Qwen3MoeForCausalLM").
Baseline:
    main_6e448d0.
Why:
    The fork replaces the two separate q_norm/k_norm passes in
    ``Qwen3MoeAttention.forward`` with one fused q/k RMSNorm, and keeps the MoE
    router gate unquantized by constructing it with ``quant_config=None``.
How:
    Use the official ``ModelRegistry.register_model`` hook. The subclass keeps
    community model construction unchanged, then swaps each layer's
    ``self_attn.__class__`` to a fused attention subclass and attaches the
    plugin-owned ``RMSNormQK`` op. Sparse MoE router gates are rebuilt before
    weight loading with the same prefix and dimensions but no quantization.
Scope:
    Qwen3-MoE models on Iluvatar.
Remove:
    When community Qwen3-MoE attention fuses q/k norm natively and router gate
    quantization is safe for Iluvatar.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeAttention as _Qwen3MoeAttention,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeForCausalLM as _Qwen3MoeForCausalLM,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeSparseMoeBlock as _Qwen3MoeSparseMoeBlock,
)

from vllm_iluvatar.model_executor.layers.layernorm import RMSNormQK


class _Qwen3MoeAttentionFusedMixin:
    """Provides fused q/k norm forward for Qwen3-MoE attention."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        out_q, out_k = self.qk_norm(
            q_by_head,
            k_by_head,
            self.q_norm.weight.data,
            self.k_norm.weight.data,
        )
        q = out_q.view(q.shape)
        k = out_k.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3MoeAttention(_Qwen3MoeAttentionFusedMixin, _Qwen3MoeAttention):
    """Qwen3-MoE attention with fused q/k RMSNorm."""


def qwen3_moe_attention_fusion(model) -> None:
    """Swap each decoder layer's self-attention to the fused subclass."""
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn.__class__ = Qwen3MoeAttention
        attn.qk_norm = RMSNormQK(
            attn.head_dim,
            attn.head_dim,
            eps=attn.q_norm.variance_epsilon,
        )


def qwen3_moe_router_gate_overrides(model) -> None:
    """Rebuild sparse MoE router gates without quantization.

    The community module is constructed before checkpoint loading, so replacing
    the gate here preserves the state-dict prefix while avoiding quantized router
    GEMMs on Iluvatar.
    """
    for layer in getattr(model, "layers", []):
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, _Qwen3MoeSparseMoeBlock):
            continue

        old_gate = mlp.gate
        gate = ReplicatedLinear(
            old_gate.input_size,
            old_gate.output_size,
            bias=old_gate.bias is not None,
            skip_bias_add=old_gate.skip_bias_add,
            params_dtype=old_gate.params_dtype,
            # Keep the router gate unquantized to preserve accuracy for
            # Qwen3-30B-A3B-Thinking-2507-W4A8.
            quant_config=None,
            prefix=old_gate.prefix,
            return_bias=old_gate.return_bias,
            disable_tp=getattr(old_gate, "disable_tp", False),
        )
        mlp.gate = gate
        mlp.experts.gate = gate
        if hasattr(mlp.experts, "_combined_gate_weight"):
            mlp.experts._combined_gate_weight = None
        if hasattr(mlp.experts, "_fse_fuse_gate"):
            mlp.experts._fse_fuse_gate = (
                gate is not None
                and getattr(mlp.experts, "shared_expert_gate", None) is not None
            )


class Qwen3MoeForCausalLM(_Qwen3MoeForCausalLM):
    """Qwen3-MoE model wired for Iluvatar fused q/k norm attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        qwen3_moe_attention_fusion(self.model)
        qwen3_moe_router_gate_overrides(self.model)


def register_model() -> None:
    """Model registration hook: point the Qwen3-MoE arch at the Iluvatar class."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM",
        "vllm_iluvatar.model_executor.models.qwen3_moe:Qwen3MoeForCausalLM",
    )
