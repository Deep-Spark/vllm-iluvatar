# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT Pixtral vision attention override.

Corresponds to: vllm/model_executor/models/pixtral.py

Target   : vllm.model_executor.models.pixtral.PixtralForConditionalGeneration
           (arch "PixtralForConditionalGeneration")
Baseline : main_6e448d0
Why      : Iluvatar uses ixformer unpadded flash attention for the Pixtral vision
           encoder while reusing the community xformers block-diagonal sequence
           metadata.
How      : Use ``ModelRegistry.register_model``. The subclass keeps community
           construction, vision-transformer forward, and weight loading unchanged,
           then swaps only the vision attention class in place. The platform import
           hook forces the community ``USE_XFORMERS_OPS`` flag to ``True`` before
           this model runs.
Scope    : Pixtral vision encoder on Iluvatar.
Remove   : when community Pixtral has an Iluvatar-compatible unpadded attention
           path or the platform no longer needs this override.

Registration: ``register_model()`` below is invoked by
``vllm_iluvatar.ilu_register.apply_oot_registrations`` during register_general.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.pixtral import (
    Attention as _Attention,
)
from vllm.model_executor.models.pixtral import (
    PixtralForConditionalGeneration as _PixtralForConditionalGeneration,
)
from vllm.model_executor.models.pixtral import apply_rotary_emb_vit


class Attention(_Attention):
    """Pixtral vision attention using ixformer unpadded flash attention."""

    def forward(
        self,
        x: torch.Tensor,
        mask,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        import ixformer.inference.functions as ixf

        batch, patches, _ = x.shape

        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, patches, self.n_heads, self.head_dim)
        k = k.reshape(batch, patches, self.n_heads, self.head_dim)
        v = v.reshape(batch * patches, self.n_heads, self.head_dim)

        q, k = apply_rotary_emb_vit(q, k, freqs_cis=freqs_cis)

        q = q.reshape(batch * patches, self.n_heads, self.head_dim)
        k = k.reshape(batch * patches, self.n_heads, self.head_dim)
        out = ixf.ixinfer_flash_attn_unpad(
            q,
            k,
            v,
            mask.q_seqinfo.seqstart.to(q.device),
            mask.k_seqinfo.seqstart.to(q.device),
            mask.q_seqinfo.max_seqlen,
            mask.k_seqinfo.max_seqlen,
        )

        out = out.reshape(batch, patches, self.n_heads * self.head_dim)
        out, _ = self.o_proj(out)
        return out


def pixtral_vision_overrides(model) -> None:
    """Swap Pixtral vision attention class while preserving loaded parameters."""
    vision_encoder = getattr(model, "vision_encoder", None)
    if vision_encoder is None:
        return
    transformer = getattr(vision_encoder, "transformer", None)
    for layer in getattr(transformer, "layers", []):
        attention = getattr(layer, "attention", None)
        if attention is not None:
            attention.__class__ = Attention


class PixtralForConditionalGeneration(_PixtralForConditionalGeneration):
    """Pixtral model wired for Iluvatar vision attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        pixtral_vision_overrides(self)


def register_model() -> None:
    """Point Pixtral architecture resolution at the Iluvatar subclass."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "PixtralForConditionalGeneration",
        "vllm_iluvatar.model_executor.models.pixtral:PixtralForConditionalGeneration",
    )
