# SPDX-License-Identifier: Apache-2.0
"""Fuse elementwise gates while preserving the BF16 activation rounding point."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_product(
    GATE,
    VALUE,
    OUT,
    WIDTH: tl.constexpr,
    TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = col < WIDTH
    offset = row.to(tl.int64) * WIDTH + col
    val_offset = offset
    if HEAD_DIM:
        val_offset = (
            (col // HEAD_DIM).to(tl.int64) * TOKENS + row
        ) * HEAD_DIM + col % HEAD_DIM
    gate = tl.load(GATE + offset, mask, other=0.0).to(tl.float32)
    activated = tl.sigmoid(gate)
    if SILU:
        activated *= gate
    value = tl.load(VALUE + val_offset, mask, other=0.0).to(tl.float32)
    tl.store(OUT + offset, activated.to(tl.bfloat16).to(tl.float32) * value, mask)


def gate_product(gate, value, *, head_dim=0):
    tokens, width = gate.shape
    out = torch.empty_like(gate)
    _gate_product[(triton.cdiv(width, 256), tokens)](
        gate,
        value,
        out,
        width,
        tokens,
        head_dim,
        head_dim == 0,
        256,
    )
    return out
