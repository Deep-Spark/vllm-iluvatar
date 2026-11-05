# Source: layers/rotary_embedding/ilu_mrope.py at 47fb804.
# SPDX-License-Identifier: Apache-2.0
"""Fused MRoPE (gather + rotate) Triton kernel for Iluvatar.

Corresponds to: vllm/model_executor/layers/rotary_embedding/mrope.py

The community ``MRotaryEmbedding.forward_cuda`` runs three CUDA kernels per
full-attention layer on the prefill critical path:

1. ``cos_sin_cache[positions]`` — an ``index_elementwise`` gather that
   materialises ``cos_sin`` ([3, T, rotary_dim]),
2. ``cos.contiguous()`` / ``sin.contiguous()`` — two ``direct_copy`` kernels
   because ``chunk(2, dim=-1)`` leaves strided halves,
3. ``_triton_mrope_forward`` — the rotation itself.

This module fuses all three into a single Triton kernel that gathers the
cos/sin rows straight from ``cos_sin_cache`` inside the rotation kernel.
The gather is an exact copy, so the result is bit-identical to the community
path while saving three kernel launches (and the intermediate tensor traffic)
per layer per step.

Load masks are clamped to ``rotary_dim // 2`` lanes. The community kernel
deliberately loads past the row (those lanes are masked out of the q/k tiles
and never consumed); reading past the last cache row would be out of bounds
for the fused in-kernel gather, so the out-of-range lanes are zeroed here —
numerically identical for every lane that is actually used.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _ilu_fused_mrope_kernel(
    q_ptr,
    k_ptr,
    pos_ptr,
    pos_stride0,
    cache_ptr,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_half_rd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
):
    # Adapted from vllm's _triton_mrope_forward (itself adapted from
    # Liger-Kernel), with the cos/sin plane loads replaced by an in-kernel
    # gather from cos_sin_cache rows indexed by positions[0/1/2, pid].
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    half_rd: tl.constexpr = rd // 2

    # positions: [3, num_tokens] with unit token stride (a column slice of
    # the persistent mrope buffer in vllm, so the plane stride is runtime).
    pos_t = tl.load(pos_ptr + pid)
    pos_h = tl.load(pos_ptr + pos_stride0 + pid)
    pos_w = tl.load(pos_ptr + 2 * pos_stride0 + pid)

    cos_offsets = tl.arange(0, pad_half_rd)
    in_range = cos_offsets < half_rd
    if is_interleaved:
        h_mask = (
            ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h) & in_range
        )
        w_mask = (
            ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w) & in_range
        )
        t_mask = ~(h_mask | w_mask) & in_range
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = (cos_offsets < mrope_section_t) & in_range
        h_mask = ((t_end <= cos_offsets) & (cos_offsets < h_end)) & in_range
        w_mask = ((h_end <= cos_offsets) & (cos_offsets < half_rd)) & in_range

    # cos_sin_cache row layout: [cos(half_rd) | sin(half_rd)], row width rd.
    cache_t = cache_ptr + pos_t * rd
    cache_h = cache_ptr + pos_h * rd
    cache_w = cache_ptr + pos_w * rd

    t_cos_row = tl.load(cache_t + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(cache_h + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(cache_w + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(cache_t + half_rd + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(cache_h + half_rd + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(cache_w + half_rd + cos_offsets, mask=w_mask, other=0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_half_rd)[None, :]
    )
    first_half_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_half_rd)[None, :]
    )
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_half_rd)[None, :] < half_rd
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_half_rd)[None, :] < half_rd
    )

    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(
        sin_row.dtype
    )

    second_half_q_offsets = first_half_q_offsets + half_rd
    second_half_k_offsets = first_half_k_offsets + half_rd
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask

    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(
        sin_row.dtype
    )
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(
        sin_row.dtype
    )

    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)


def ilu_fused_mrope(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    mrope_section: list[int],
    head_size: int,
    rotary_dim: int,
    mrope_interleaved: bool,
) -> None:
    """In-place fused MRoPE: gather cos/sin rows and rotate q/k.

    Args:
        positions: [3, num_tokens] with unit stride on dim 1 (T/H/W rows);
            arbitrary plane stride, e.g. a column slice of vllm's persistent
            mrope buffer.
        query: [num_tokens, num_heads * head_size], rotated in place.
        key: [num_tokens, num_kv_heads * head_size], rotated in place.
        cos_sin_cache: [max_position, rotary_dim] contiguous, row layout
            ``[cos | sin]`` with ``rotary_dim // 2`` lanes each.
        mrope_section: [t, h, w] section sizes.
        head_size: per-head dim.
        rotary_dim: rotary dim (<= head_size); must be even.
        mrope_interleaved: section layout flag (see community kernel).
    """
    n_row, n_q_head_head_dim = query.shape
    n_q_head = n_q_head_head_dim // head_size
    n_kv_head = key.shape[1] // head_size
    half_rd = rotary_dim // 2
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    pad_half_rd = max(triton.next_power_of_2(half_rd), 16)

    _ilu_fused_mrope_kernel[(n_row,)](
        query,
        key,
        positions,
        positions.stride(0),
        cos_sin_cache,
        n_row,
        n_q_head,
        n_kv_head,
        head_size,
        rotary_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_half_rd,
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        mrope_interleaved,
    )


def ilu_fused_mrope_supported(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> bool:
    """Return whether the owned kernel accepts this tensor layout."""
    return (
        positions.ndim == 2
        and positions.shape[0] == 3
        and positions.stride(1) == 1
        and query.ndim == 2
        and key.ndim == 2
        and positions.shape[1] == query.shape[0] == key.shape[0]
        and key.dtype == query.dtype
        and positions.device == query.device == key.device
        and query.is_contiguous()
        and key.is_contiguous()
        and cos_sin_cache.is_contiguous()
        and cos_sin_cache.dtype == query.dtype
        and cos_sin_cache.device == query.device
    )
