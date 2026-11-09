# Source: vllm_iluvatar/v1/attention/ops/ilu_vit_attn_wrappers.py at 47fb804; independent packed ViT layout and rotary.
# SPDX-License-Identifier: Apache-2.0
"""Iluvatar ViT attention preprocessing helpers.

Corresponds to:
    vllm/v1/attention/ops/vit_attn_wrappers.py

Baseline:
    main_6e448d0.
Why:
    Iluvatar's Qwen2.5-VL and Qwen3.5 OOT models fuse packed QKV split,
    rotary, and head-dim padding before calling the community MM encoder
    attention op, and fold FA output with view / permute instead of einops.
How:
    Keep the helpers in the plugin and call them from the OOT vision models
    instead of adding Iluvatar-only APIs to the community ViT attention wrappers.
Scope:
    Qwen2.5-VL and Qwen3.5 vision attention on Iluvatar.
Remove:
    When community vLLM exposes equivalent fused ViT input and output helpers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import vllm.envs as envs

from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_torch_op

_VIT_QKV_LAYOUT = None
_VIT_QKV_ROPE = None


# Bind packed-QKV layout before any model captures the reference.
def configure_vit_qkv_layout_provider():
    global _VIT_QKV_LAYOUT
    implementation, reason = load_ixaite_torch_op("vit_qkv_layout")
    if implementation is None:
        raise RuntimeError(f"ixaite vit_qkv_layout is unavailable: {reason}")
    _VIT_QKV_LAYOUT = implementation


# A missing fused rotary entry cannot become a native rotary call later.
def configure_vit_qkv_rope_provider():
    global _VIT_QKV_ROPE
    implementation, reason = load_ixaite_torch_op("vit_qkv_rope")
    if implementation is None:
        raise RuntimeError(f"ixaite vit_qkv_rope is unavailable: {reason}")
    _VIT_QKV_ROPE = implementation


def vit_fused_qkv_preproc(
    qkv_packed: torch.Tensor,
    rotary_pos_emb_cos: torch.Tensor | None,
    rotary_pos_emb_sin: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed QKV, apply rotary to Q/K, and pad head_dim in one region."""
    del batch_size, seq_len
    has_rotary = rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None
    pad = padded_head_dim - head_dim
    # Quantized and FP16 paths retain their existing preprocessing and rounding.
    fuse_pad = (
        qkv_packed.dtype == torch.bfloat16
        and pad > 0
        and envs.VLLM_ILU_VIT_ATTN_OPT_LEVEL == 0
    )
    padding = {"padded_head_dim": padded_head_dim} if fuse_pad else {}
    if has_rotary:
        q, k, v = _VIT_QKV_ROPE(
            qkv_packed,
            rotary_pos_emb_cos.contiguous(),
            rotary_pos_emb_sin.contiguous(),
            num_heads,
            head_dim,
            **padding,
        )
    else:
        q, k, v = _VIT_QKV_LAYOUT(qkv_packed, num_heads, head_dim, **padding)

    if pad > 0 and not fuse_pad:
        q = F.pad(q, [0, pad], value=0)
        k = F.pad(k, [0, pad], value=0)
        v = F.pad(v, [0, pad], value=0)

    return q, k, v


def vit_attn_output_layout(context_layer: torch.Tensor) -> torch.Tensor:
    """Fold attention heads after restoring sequence-major layout."""
    batch_size, seq_len, num_heads, head_dim = context_layer.shape
    return (
        context_layer.permute(1, 0, 2, 3)
        .reshape(seq_len, batch_size, num_heads * head_dim)
        .contiguous()
    )
