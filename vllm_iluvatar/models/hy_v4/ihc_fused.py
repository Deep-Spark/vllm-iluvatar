# SPDX-License-Identifier: Apache-2.0
"""Fused iHC staging and epilogue around the original FP32 projection."""

import torch
import triton
import triton.language as tl


@triton.jit
def _cast_and_norm(
    X, FLOAT, INV, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    offset = row * WIDTH + col
    x = tl.load(X + offset, col < WIDTH, other=0).to(tl.float32)
    tl.store(FLOAT + offset, x, col < WIDTH)
    tl.store(INV + row, tl.rsqrt(tl.sum(x * x, 0) / WIDTH + EPS))


@triton.jit
def _merge_channels(
    X,
    INV,
    MIXES,
    SCALE,
    BASE,
    Y,
    POST,
    H: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    HAS_POST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    d = tile * BLOCK + tl.arange(0, BLOCK)
    c = tl.arange(0, C)
    width = 2 * C if HAS_POST else C
    inv = tl.load(INV + row)
    mix = tl.load(MIXES + row * width + c) * inv
    pre = tl.sigmoid(mix * tl.load(SCALE) + tl.load(BASE + c)) + EPS
    residual = tl.load(
        X + (row * C + c[:, None]) * H + d[None, :], d[None, :] < H, other=0
    )
    merged = tl.sum(residual.to(tl.float32) * pre[:, None], 0)
    tl.store(Y + row * H + d, merged, d < H)
    if HAS_POST and tile == 0:
        mix = tl.load(MIXES + row * width + C + c) * inv
        gate = tl.sigmoid(mix * tl.load(SCALE + 1) + tl.load(BASE + C + c))
        tl.store(POST + row * C + c, MAGNITUDE * gate + EPS)


def fp32_ihc(x, projection, scale, base, norm_eps, hc_eps, magnitude=None):
    """Merge contiguous BF16 channels with gates projected in FP32."""
    rows, channels, hidden = x.shape
    width = channels * hidden
    flat = x.new_empty((rows, width), dtype=torch.float32)
    inv = x.new_empty((rows,), dtype=torch.float32)
    _cast_and_norm[(rows,)](
        x,
        flat,
        inv,
        width,
        norm_eps,
        triton.next_power_of_2(width),
        num_warps=8,
    )
    mixes = projection(flat)[0]
    y = x.new_empty((rows, hidden))
    post = (
        x.new_empty((rows, channels), dtype=torch.float32)
        if magnitude is not None
        else None
    )
    _merge_channels[(rows, triton.cdiv(hidden, 256))](
        x,
        inv,
        mixes,
        scale,
        base,
        y,
        post,
        hidden,
        channels,
        hc_eps,
        magnitude if magnitude is not None else 1.0,
        post is not None,
        256,
        enable_fp_fusion=False,
    )
    return y, post
