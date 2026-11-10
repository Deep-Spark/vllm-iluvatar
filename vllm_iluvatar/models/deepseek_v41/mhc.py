# SPDX-License-Identifier: Apache-2.0
"""Multi-stream Hyper-Connections (MHC) for DeepSeek-V4.1."""

from __future__ import annotations

import os

import torch
from ixformer.inference.functions.models import deepseek_v41 as ixfops

from vllm_iluvatar.ilu_util.arch import is_ivcore30

_fuse_reductions = os.getenv("VLLM_ILU_V41_MHC_FUSE_REDUCTIONS", "1") != "0"
# Fold the fused_wqa_wkv per-token fp8 activation quant into the pre-norm kernel.
_fuse_fp8_norm = os.getenv("VLLM_ILU_V41_FUSE_FP8_NORM", "1") != "0"
# Allow the fused fp8 epilogue for small batches (decode T<=16).
_fuse_fp8_norm_small_batch = os.getenv("VLLM_ILU_V41_FP8_NORM_SMALL_BATCH", "1") != "0"


def _want_fused_fp8(
    want_fp8: bool,
    layer_norm_weight: torch.Tensor | None,
    sqrsum: torch.Tensor,
    num_tokens: int,
) -> bool:
    """Whether to emit the fp8 activation from the pre-norm kernel.

    The fp8 epilogue folds per-token e4m3 dynamic quantisation of
    ``layer_input`` into the pre-norm kernel, eliminating a separate
    ``fp8_per_token_dynamic_scaled`` call.

    Requirements:
    - ivcore30 only (quant consumer is ``apply_block_scaled_mm``).
    - ``layer_norm_weight`` must be provided (folded-norm path).
    - ``sqrsum`` must be dense (1-D) — the fp8 kernel has no partial variant.

    For T<=16 (decode), the non-fused fp8 kernel launches 1 block per token
    which is equivalent to the fused small-batch path for T=1, so enabling it
    saves one kernel launch with no regression.
    """
    if not is_ivcore30():
        return False
    if not (_fuse_fp8_norm and want_fp8 and layer_norm_weight is not None):
        return False
    if _fuse_reductions and num_tokens <= 16 and not _fuse_fp8_norm_small_batch:
        return False
    return not (sqrsum.dim() == 2 and sqrsum.shape[1] > 1)


def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return ixfops.hc_collapse(x.contiguous(), pre_mix.contiguous())


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
    layer_norm_weight: torch.Tensor | None = None,
    layer_norm_eps: float = 1e-6,
    want_fp8: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Project, split and collapse ``x``.

    ``sqrsum`` carries the sum of squares of ``x`` per token (dense or partial), from
    :func:`hc_post_sqrsum`. Given it, the epilogue has no token-wide reduction
    left to do and streams instead, which is ~2x on the kernel; without it the
    op reduces the rms itself.

    ``layer_norm_weight`` folds the layer's own RMSNorm into ``layer_input``, so
    the caller does not run it separately over the row this just wrote. Only the
    streaming path can do that, so callers must apply the norm themselves when
    they pass no ``sqrsum``.
    """
    fn = hc_fn if hc_fn_io is None else hc_fn_io
    fp8_h: torch.Tensor | None = None
    h_scale: torch.Tensor | None = None
    if sqrsum is None:
        assert layer_norm_weight is None
        post, comb, layer_input, pre_out = ixfops.mhc_pre_with_pre_mix(
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
            fuse_reductions=_fuse_reductions,
        )
    elif _want_fused_fp8(want_fp8, layer_norm_weight, sqrsum, x.shape[0]):
        # Fold the per-token fp8 activation quant of ``layer_input`` into the
        # pre-norm epilogue. Only the non-fused folded-norm path (T>16, dense
        # sqrsum) has this kernel; the small-batch fused path does not, so the
        # dispatch guard above keeps decode on the plain op. The fp8 pair is
        # bit-identical to ``fp8_per_token_dynamic_scaled(layer_input)``.
        (
            post,
            comb,
            layer_input,
            pre_out,
            fp8_h,
            h_scale,
        ) = ixfops.mhc_pre_with_sqrsum_fp8(
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
            layer_norm_weight,
            layer_norm_eps,
        )
    else:
        post, comb, layer_input, pre_out = ixfops.mhc_pre_with_sqrsum(
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
            layer_norm_weight,
            layer_norm_eps,
            fuse_reductions=_fuse_reductions,
        )
    return post.squeeze(-1), comb, layer_input, pre_out, fp8_h, h_scale


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return ixfops.mhc_post(
        x.contiguous(),
        residual.contiguous(),
        post.contiguous(),
        comb.contiguous(),
    )


def allreduce_hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused attention-output all-reduce + V4.1 hc_post.

    The all-reduce of the (tensor-parallel) attention output is folded into
    mhc_post, and the kernel still emits the per-token sum of squares of the
    residual it writes -- the same second output :func:`hc_post_sqrsum` returns.
    ``x`` is the local, unreduced o_proj output (``wo_b`` with
    ``reduce_results=False``). Numerically this matches the two-step path
    except for floating-point reduction order (NCCL ring/tree vs the kernel's
    own tree). Set ``VLLM_ILU_V41_ALLREDUCE_MHC_POST=0`` to keep the separate
    all-reduce plus :func:`hc_post_sqrsum` instead.

    Returns ``(residual, sqrsum)`` exactly like :func:`hc_post_sqrsum`.
    """
    return ixfops.allreduce_mhc_post_with_sqrsum(
        x,
        residual,
        post,
        comb,
        group=group,
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
    post_op = (
        ixfops.mhc_post_partials
        if _fuse_reductions and residual.shape[0] <= 16
        else ixfops.mhc_post_with_sqrsum
    )
    return post_op(
        x.contiguous(),
        residual.contiguous(),
        post.contiguous(),
        comb.contiguous(),
    )
