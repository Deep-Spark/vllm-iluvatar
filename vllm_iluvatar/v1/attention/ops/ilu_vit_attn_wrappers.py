# SPDX-License-Identifier: Apache-2.0
"""Iluvatar ViT attention preprocessing helpers.

Corresponds to:
    vllm/v1/attention/ops/vit_attn_wrappers.py

Baseline:
    main_6e448d0.
Why:
    Iluvatar's Qwen2.5-VL OOT model fuses packed QKV split, rotary, and
    head-dim padding before calling the community MM encoder attention op.
How:
    Keep the helper in the plugin and call it from the OOT Qwen2.5-VL model
    instead of adding Iluvatar-only APIs to the community ViT attention wrappers.
Scope:
    Qwen2.5-VL vision attention on Iluvatar.
Remove:
    When community vLLM exposes an equivalent fused ViT QKV preprocessing helper.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb


@torch.compile
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
    qkv = qkv_packed.view(seq_len, batch_size, 3, num_heads, head_dim).permute(
        1, 0, 2, 3, 4
    )

    if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
        qk = qkv[:, :, :2]
        v = qkv[:, :, 2]
        qk_reshaped = (
            qk.permute(2, 0, 1, 3, 4)
            .reshape(2 * batch_size, seq_len, num_heads, head_dim)
            .contiguous()
        )
        qk_rotated = ApplyRotaryEmb.forward_static(
            qk_reshaped,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
            True,  # is_neox_style
            False,  # enable_fp32_compute
        )
        qk_rotated = qk_rotated.view(2, batch_size, seq_len, num_heads, head_dim)
        q, k = qk_rotated.unbind(dim=0)
    else:
        q, k, v = qkv.unbind(dim=2)

    pad = padded_head_dim - head_dim
    if pad > 0:
        q = F.pad(q, [0, pad], value=0)
        k = F.pad(k, [0, pad], value=0)
        v = F.pad(v, [0, pad], value=0)

    return q, k, v
