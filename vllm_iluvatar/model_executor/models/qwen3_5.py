# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/model_executor/models/qwen3_5.py

Iluvatar OOT Qwen3.5 models: optimize
``Qwen3NextAttention._project_qkv_gate`` with fused kernels.

Target   : vllm.model_executor.models.qwen3_5.
           Qwen3_5ForConditionalGeneration and
           Qwen3_5MoeForConditionalGeneration (arches with the same names)
Baseline : main_6e448d0
Why      : the fork replaces the Torch implementation of q/k RMSNorm, rotary
           embedding, and output-gate projection with fused Iluvatar kernels.
How      : ModelRegistry.register_model points both arches at the subclasses
           below; each subclass inherits the community model unchanged and,
           after ``super().__init__()``, swaps every layer's
           ``self_attn.__class__`` to an optimized ``Qwen3NextAttention``
           subclass (weights preserved), overriding only
           ``_project_qkv_gate``.
Scope    : Qwen3.5 dense and MoE models on the Iluvatar platform.
Remove   : when community Qwen3NextAttention uses equivalent fused kernels.

Registration: ``register_model()`` below is invoked by
``vllm_iluvatar.ilu_register.apply_oot_registrations`` during register_general.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration as _Qwen3_5ForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5MoeForConditionalGeneration as _Qwen3_5MoeForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention as _Qwen3NextAttention,
)


class Qwen3NextAttention(_Qwen3NextAttention):
    """Qwen3NextAttention with fused q/k RMSNorm with rope gate."""

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.use_fused_qk_norm_rope_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            pos = positions[0] if positions.ndim == 2 else positions

            q, k, gate = torch.ops._C.fused_qk_rmsnorm_rope_gate(
                q_gate,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                pos,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
            )
            return q, k, v, gate

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )

            q, k, gate = torch.ops._C.fused_qk_rmsnorm_gate(
                q_gate,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
                -1, self.num_heads * self.head_dim
            )
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
                -1, self.num_kv_heads * self.head_dim
            )

        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate


def qwen35_attention_fusion(model) -> None:
    """Swap each decoder layer's ``self_attn`` to the fused subclass.

    ``__class__`` reassignment preserves all weights / state and only changes the
    resolved ``forward``.  See Qwen3NextAttention._project_qkv_gate for details.
    """
    for layer in model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if attn.__class__.__name__ == "Qwen3NextAttention":
            attn.__class__ = Qwen3NextAttention


class Qwen3_5ForConditionalGeneration(_Qwen3_5ForConditionalGeneration):
    """Qwen3_5 dense model wired for Iluvatar fused q/k norm attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        qwen35_attention_fusion(self.language_model.model)


class Qwen3_5MoeForConditionalGeneration(_Qwen3_5MoeForConditionalGeneration):
    """Qwen3_5 moe model wired for Iluvatar fused q/k norm attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        qwen35_attention_fusion(self.language_model.model)


def register_model() -> None:
    """Model registration hook: point the Qwen3 dense arch at the Iluvatar class."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Qwen3_5ForConditionalGeneration",
        "vllm_iluvatar.model_executor.models.qwen3_5:Qwen3_5ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "Qwen3_5MoeForConditionalGeneration",
        "vllm_iluvatar.model_executor.models.qwen3_5:Qwen3_5MoeForConditionalGeneration",
    )
