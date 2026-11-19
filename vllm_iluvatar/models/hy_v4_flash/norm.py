# SPDX-License-Identifier: Apache-2.0
"""Fused normalization of a sum rounded to the activation dtype."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sum_rms_norm(
    X,
    RES,
    WEIGHT,
    OUT,
    SUM,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    KEEP_SUM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    offset = row * WIDTH + col
    x = tl.load(X + offset, valid, other=0).to(tl.float32)
    res = tl.load(RES + offset, valid, other=0).to(tl.float32)
    summed = (x + res).to(X.dtype.element_ty)
    if KEEP_SUM:
        tl.store(SUM + offset, summed, valid)
    v = summed.to(tl.float32)
    inv = tl.rsqrt(tl.sum(v * v, 0) / WIDTH + EPS)
    weight = tl.load(WEIGHT + col, valid, other=0).to(tl.float32)
    tl.store(OUT + offset, (v * inv) * weight, valid)


def sum_rms_norm(x, residual, norm, *, keep_sum=False):
    rows, width = x.shape
    result = torch.empty_like(x)
    summed = torch.empty_like(x) if keep_sum else None
    _sum_rms_norm[(rows,)](
        x,
        residual,
        norm.weight,
        result,
        summed,
        width,
        norm.variance_epsilon,
        triton.next_power_of_2(width),
        keep_sum,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result, summed
