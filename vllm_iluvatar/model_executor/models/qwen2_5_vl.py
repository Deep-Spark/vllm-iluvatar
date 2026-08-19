# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT Qwen2.5-VL vision encoder overrides.

Corresponds to:
    vllm/model_executor/models/qwen2_5_vl.py

Target:
    ``Qwen2_5_VLForConditionalGeneration``.
Baseline:
    main_6e448d0.
Why:
    The fork fused QKV split, rotary, and head-dim padding for Qwen2.5-VL ViT
    attention, and accepted CUDA-graph ``max_seqlen`` metadata that may arrive as
    tensors on Iluvatar/CoreX runs.
How:
    Use the official ``ModelRegistry.register_model`` hook. The subclass keeps
    community model construction unchanged, then swaps only the vision attention
    and block classes in-place so loaded weights and module ownership remain
    untouched.
Scope:
    Qwen2.5-VL vision encoder on Iluvatar.
Remove:
    When community Qwen2.5-VL uses the fused ViT QKV preprocessing path and
    tensor ``max_seqlen`` metadata is handled upstream.
"""

from __future__ import annotations

import einops
import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionAttention as _Qwen2_5_VisionAttention,
)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionBlock as _Qwen2_5_VisionBlock,
)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as _Qwen2_5_VLForConditionalGeneration,
)

from vllm_iluvatar.v1.attention.ops.ilu_vit_attn_wrappers import (
    vit_fused_qkv_preproc,
)


def _as_int(value: int | torch.Tensor) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


class Qwen2_5_VisionAttention(_Qwen2_5_VisionAttention):
    """Qwen2.5-VL vision attention with fused QKV preprocessing."""

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)
        seq_len, batch_size, _ = x.shape

        q, k, v = vit_fused_qkv_preproc(
            qkv_packed=x,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=self.num_attention_heads_per_partition,
            head_dim=self.hidden_size_per_attention_head,
            padded_head_dim=self.attn.padded_head_dim,
        )
        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

        context_layer = einops.rearrange(
            context_layer, "b s h d -> s b (h d)", b=batch_size
        ).contiguous()

        output, _ = self.proj(context_layer)
        return output


class Qwen2_5_VisionBlock(_Qwen2_5_VisionBlock):
    """Vision block that accepts tensor max-seqlen metadata."""

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward(
            x=x,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=_as_int(max_seqlen),
            sequence_lengths=sequence_lengths,
        )


def qwen2_5_vl_vision_overrides(model) -> None:
    """Swap the vision tower classes in place while preserving parameters."""
    visual = getattr(model, "visual", None)
    if visual is None:
        return
    for block in getattr(visual, "blocks", []):
        block.__class__ = Qwen2_5_VisionBlock
        attn = getattr(block, "attn", None)
        if attn is not None:
            attn.__class__ = Qwen2_5_VisionAttention


class Qwen2_5_VLForConditionalGeneration(_Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL model wired for Iluvatar vision encoder overrides."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        qwen2_5_vl_vision_overrides(self)


def register_model() -> None:
    """Point Qwen2.5-VL architecture resolution at the Iluvatar subclass."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Qwen2_5_VLForConditionalGeneration",
        "vllm_iluvatar.model_executor.models.qwen2_5_vl:"
        "Qwen2_5_VLForConditionalGeneration",
    )
