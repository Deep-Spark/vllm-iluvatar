# SPDX-License-Identifier: Apache-2.0
"""Multi-stream Hyper-Connections (MHC) for GLM-5.3-Flash (glm5next)."""

from __future__ import annotations

import ixformer.inference.functions as ixfops
import torch
from ixformer.inference.functions.models import deepseek_v41 as ixf_mhc

_PRE_V2_WORKSPACES: dict[torch.device, torch.Tensor] = {}


def _get_pre_v2_workspace(device: torch.device) -> torch.Tensor:
    ws = _PRE_V2_WORKSPACES.get(device)
    if ws is None:
        ws = torch.zeros(1, dtype=torch.int32, device=device)
        _PRE_V2_WORKSPACES[device] = ws
    return ws


def allreduce_mhc_post_supported(config, tp_size: int) -> bool:
    """Whether ``allreduce_mhc_post_with_sqrsum`` can fuse attention TP AllReduce + MHC post."""
    return (
        tp_size in (2, 4, 8)
        and getattr(config, "hidden_size", None) in (4096, 5120, 7168)
        and getattr(config, "mhc_num_residual_streams", None) == 4
    )


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_mult: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    sqrsum: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``(post_mix, comb_mix, layer_input)`` for GLM5Next MHC pre.

    When ``sqrsum`` is provided from :func:`mhc_post_with_sqrsum` or
    :func:`allreduce_mhc_post_with_sqrsum`, uses ``mhc_pre_with_sqrsum`` to
    eliminate the token-wide RMS reduction pass and fold ``norm_weight`` in a
    single streaming epilogue (with ``ixmm`` GEMM for prefill and fused split-K
    for decode). When ``sqrsum`` is ``None`` (Layer 0's first attention pre or
    cross-PP stage entry), uses ``ds4_mhc_pre_v2`` with a preallocated workspace.
    """
    residual = residual.contiguous()
    if sqrsum is not None:
        post, comb, layer_input, _ = ixf_mhc.mhc_pre_with_sqrsum(
            residual,
            fn,
            scale,
            base,
            sqrsum,
            rms_eps,
            hc_eps,
            hc_eps,
            post_mult,
            sinkhorn_repeat,
            None,
            norm_weight,
            norm_eps,
            fuse_reductions=True,
        )
        return post, comb, layer_input

    fuse_norm = norm_weight is not None and ixfops.enable_hc_pre_norm_fusion(
        *residual.shape, dtype=residual.dtype
    )
    post, comb, layer_input = ixfops.ds4_mhc_pre_v2(
        residual=residual,
        fn=fn,
        hc_scale=scale,
        hc_base=base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_sinkhorn_eps=hc_eps,
        hc_post_mult_value=post_mult,
        sinkhorn_repeat=sinkhorn_repeat,
        workspace=_get_pre_v2_workspace(residual.device),
        need_rms=fuse_norm,
        rms_weight=norm_weight if fuse_norm else None,
        layer_rms_eps=norm_eps,
    )
    if norm_weight is not None and not fuse_norm:
        layer_input = ixfops.rms_norm(layer_input, norm_weight, norm_eps)
    return post, comb, layer_input


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Apply MHC post-projection without returning sqrsum (used after the final layer)."""
    return ixfops.mhc_post(
        x.contiguous(),
        residual.contiguous(),
        post.contiguous(),
        comb.contiguous(),
    )


def mhc_post_with_sqrsum(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MHC post-projection and emit the sum of squares of the updated residual."""
    x_c = x.contiguous()
    res_c = residual.contiguous()
    post_c = post.contiguous()
    comb_c = comb.contiguous()
    if x_c.shape[0] <= 16:
        return ixf_mhc.mhc_post_partials(x_c, res_c, post_c, comb_c)
    return ixf_mhc.mhc_post_with_sqrsum(x_c, res_c, post_c, comb_c)


def allreduce_mhc_post_with_sqrsum(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused TP AllReduce of ``x`` + MHC post-projection + sqrsum emission."""
    return ixf_mhc.allreduce_mhc_post_with_sqrsum(
        x.contiguous(),
        residual.contiguous(),
        post.contiguous(),
        comb.contiguous(),
        group=group,
    )


def mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_mult: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    *,
    reduce_x: bool = False,
    tp_group=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused MHC post + next sub-block MHC pre (via explicit sqrsum handoff)."""
    if reduce_x:
        residual, sqrsum = allreduce_mhc_post_with_sqrsum(
            x, residual, post, comb, group=tp_group
        )
    else:
        residual, sqrsum = mhc_post_with_sqrsum(x, residual, post, comb)

    new_post, new_comb, layer_input = mhc_pre(
        residual,
        fn,
        scale,
        base,
        rms_eps,
        hc_eps,
        post_mult,
        sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        sqrsum=sqrsum,
    )
    return residual, new_post, new_comb, layer_input
