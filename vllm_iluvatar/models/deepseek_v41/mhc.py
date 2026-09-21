# SPDX-License-Identifier: Apache-2.0
"""Multi-stream Hyper-Connections (MHC) for DeepSeek-V4.1."""

from __future__ import annotations

import ixformer.inference.functions as ixfops
import torch
import torch.nn.functional as F


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    post_mult: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``[T, mix_hc]`` into ``(pre, post, comb)``. Layout is ``[pre | post | comb]``.

    Sinkhorn: softmax, then ``sinkhorn_iters`` column and row scalings (first column
    scale is outside the loop). ``eps`` is added after the softmax divide.
    """
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = post_mult * torch.sigmoid(
        mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )

    comb = mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]
    comb = comb.unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_mixes(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
    post_mult: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xf = x.flatten(1).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(xf, hc_fn) * rsqrt
    return hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, post_mult
    )


def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=1).to(x.dtype)


def hc_pre_mixes(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre_mix_in: torch.Tensor,
    norm_eps: float,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
    post_mult: float = 2.0,
    hc_fn_io: torch.Tensor | None = None,
    sqrsum: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project, split and collapse ``x``.

    ``sqrsum`` is the sum of squares of ``x`` per token, from
    :func:`hc_post_sqrsum`. Given it, the epilogue has no token-wide reduction
    left to do and streams instead, which is ~2x on the kernel; without it the
    op reduces the rms itself.
    """
    fn = hc_fn if hc_fn_io is None else hc_fn_io
    if sqrsum is None:
        post, comb, layer_input, pre_out = ixfops.ds4_v41_mhc_pre(
            x.contiguous(),
            fn,
            hc_scale,
            hc_base,
            norm_eps,
            hc_eps,
            hc_eps,
            post_mult,
            sinkhorn_iters,
            pre_mix_in,
        )
    else:
        post, comb, layer_input, pre_out = ixfops.ds4_v41_mhc_pre_1pass(
            x.contiguous(),
            fn,
            hc_scale,
            hc_base,
            sqrsum,
            norm_eps,
            hc_eps,
            hc_eps,
            post_mult,
            sinkhorn_iters,
            pre_mix_in,
        )
    return post.squeeze(-1), comb, layer_input, pre_out


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return ixfops.ds4_mhc_post(
        x.contiguous(),
        residual.contiguous(),
        post.contiguous(),
        comb.contiguous(),
    )


def hc_post_sqrsum(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``hc_post`` that also returns the sum of squares of the residual it writes.

    The next sub-block's ``pre`` needs exactly that to form its rms, and post
    holds the values in registers anyway, so the second output is free. Pass it
    to :func:`hc_pre_mixes` as ``sqrsum`` -- but only if nothing rewrites the
    residual in between.
    """
    return ixfops.ds4_v41_mhc_post_sqrsum(
        x.contiguous(),
        residual.contiguous(),
        post.contiguous(),
        comb.contiguous(),
    )
