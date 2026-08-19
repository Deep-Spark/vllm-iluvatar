# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/model_executor/models/qwen3.py

Iluvatar OOT Qwen3 (dense) model: fuses the per-head q/k RMSNorm into a single
``RMSNormQK`` op inside ``Qwen3Attention.forward``.

Target   : vllm.model_executor.models.qwen3.Qwen3ForCausalLM (arch
           "Qwen3ForCausalLM")
Baseline : main_6e448d0
Why      : the fork replaces the two separate q_norm/k_norm passes with one
           fused q/k RMSNorm (ixformer rms_norm_qk).
How      : ModelRegistry.register_model points the arch at the subclass below;
           the subclass inherits the community model unchanged and, after
           ``super().__init__()``, swaps each layer's ``self_attn.__class__`` to
           a fused attention subclass (weights preserved) and attaches a
           ``qk_norm`` op.  vLLM has no finer-grained (per-method) model hook, so
           whole-class re-registration is the only official extension point.
Scope    : Qwen3 dense models on the Iluvatar platform.
Remove   : when community Qwen3 attention fuses q/k norm natively.

Registration: ``register_model()`` below is invoked by
``vllm_iluvatar.ilu_register.apply_oot_registrations`` during register_general.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3 import Qwen3Attention as _Qwen3Attention
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM as _Qwen3ForCausalLM

from vllm_iluvatar.model_executor.layers.layernorm import RMSNormQK


class _Qwen3AttentionFusedMixin:
    """Provides the fused q/k norm forward for Qwen3 attention.

    Requires a ``qk_norm`` (:class:`RMSNormQK`) attribute on the instance,
    attached by :func:`qwen3_attention_fusion` after the community
    ``__init__`` has built ``q_norm`` / ``k_norm``.
    """

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


class Qwen3Attention(_Qwen3AttentionFusedMixin, _Qwen3Attention):
    """Qwen3Attention with fused q/k RMSNorm."""


def qwen3_attention_fusion(model) -> None:
    """Swap each decoder layer's ``self_attn`` to the fused subclass.

    ``__class__`` reassignment preserves all weights / state and only changes the
    resolved ``forward``.  A ``qk_norm`` op is attached using the head dim and
    eps already carried by the community ``q_norm``.
    """
    for layer in model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn.__class__ = Qwen3Attention
        attn.qk_norm = RMSNormQK(
            attn.head_dim,
            attn.head_dim,
            eps=attn.q_norm.variance_epsilon,
        )


class Qwen3ForCausalLM(_Qwen3ForCausalLM):
    """Qwen3 dense model wired for Iluvatar fused q/k norm attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        qwen3_attention_fusion(self.model)


def register_model() -> None:
    """Model registration hook: point the Qwen3 dense arch at the Iluvatar class."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "vllm_iluvatar.model_executor.models.qwen3:Qwen3ForCausalLM",
    )
