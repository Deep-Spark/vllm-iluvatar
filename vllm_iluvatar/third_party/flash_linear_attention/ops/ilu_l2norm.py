# SPDX-License-Identifier: Apache-2.0
import os

import torch
from vllm.triton_utils import tl, triton

BT_LIST = [8, 16, 32, 64, 128]

USE_DEFAULT_FLA_NORM = int(os.getenv("USE_DEFAULT_FLA_NORM", "0"))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["D"],
)
@triton.jit
def l2norm_fwd_kernel1(
    x,
    x1,
    eps,
    token_stride,
    H,
    D,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * token_stride + tl.program_id(1) * D
    x1 += i_t * token_stride + tl.program_id(1) * D

    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    b_y = b_x * b_rstd
    tl.store(x + cols, b_y, mask=mask)

    b_x = tl.load(x1 + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    b_y = b_x * b_rstd
    tl.store(x1 + cols, b_y, mask=mask)


@triton.jit
def l2norm_fwd_kernel2(X, X1, eps, token_stride, N: tl.constexpr, MBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * token_stride
    X += xoffset
    X1 += xoffset

    h_offset = tl.program_id(1) * MBLOCK
    row_idx = h_offset + tl.arange(0, MBLOCK)[:, None]
    rindex = tl.arange(0, N)[None, :]

    xs = tl.load(
        X + (rindex + N * row_idx),
    ).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, N])
    square_sum = tl.sum(square, 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(X + (rindex + N * row_idx), xs * rsqrt)

    xs = tl.load(
        X1 + (rindex + N * row_idx),
    ).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, N])
    square_sum = tl.sum(square, 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(X1 + (rindex + N * row_idx), xs * rsqrt)


def l2norm_fwd_inplace(
    x: torch.Tensor,
    x1: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
):
    assert output_dtype is None
    assert x.dim() == 3 and x1.dim() == 3
    x_shape_og = x.shape
    B, H, D = x.shape[0], x.shape[1], x.shape[2]

    assert x.stride(-1) == 1 and x1.stride(-1) == 1
    assert x.stride() == x1.stride()
    token_stride = x.stride(0)
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if not USE_DEFAULT_FLA_NORM and H % 4 == 0:
        MBLOCK = 4
        # M, N = x.shape
        l2norm_fwd_kernel2[(B, H // 4)](
            x,
            x1,
            eps,
            token_stride,
            D,
            MBLOCK,
        )
    else:
        l2norm_fwd_kernel1[(B, H)](
            x,
            x1,
            eps,
            token_stride,
            H,
            D,
            BD,
        )

    return x.view(x_shape_og), x1.view(x_shape_og)
