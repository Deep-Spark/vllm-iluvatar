# SPDX-License-Identifier: Apache-2.0
"""CUDA launchers for Gumbel and speculative rejection sampling."""

from __future__ import annotations

from typing import Any

import torch

from vllm_iluvatar.v1.worker.gpu._kernel_launcher import KernelLauncher


def _gumbel_sample(_grid: Any, *args: Any, **kwargs: Any) -> None:
    block_size = kwargs.pop("BLOCK_SIZE")
    torch.ops._C.triton_gumbel_sample(
        *args,
        block_size,
        kwargs.pop("IS_DRAFTING"),
        kwargs.pop("APPLY_TEMPERATURE"),
        kwargs.pop("USE_FP64"),
        kwargs.pop("PER_TOKEN_COL"),
    )
    if kwargs:
        raise TypeError(f"unexpected Gumbel launch options: {kwargs}")


def _compute_local_logits_stats(
    _grid: Any, *args: Any, **kwargs: Any
) -> None:
    block_size = kwargs.pop("BLOCK_SIZE")
    torch.ops._C.triton_compute_local_logits_stats(
        *args,
        block_size,
        kwargs.pop("HAS_DRAFT_LOGITS"),
    )
    if kwargs:
        raise TypeError(f"unexpected logits-stats launch options: {kwargs}")


def _rejection(_grid: Any, *args: Any, **kwargs: Any) -> None:
    kwargs.pop("PADDED_VOCAB_NUM_BLOCKS")
    kwargs.pop("num_warps", None)
    has_draft = kwargs.pop("HAS_DRAFT_LOGITS")
    synthetic = kwargs.pop("SYNTHETIC_MODE")
    use_block = kwargs.pop("USE_BLOCK_VERIFICATION")
    if kwargs:
        raise TypeError(f"unexpected rejection launch options: {kwargs}")
    # cumulative_log_p and local_residual_mass are consumed only by block
    # verification. The native implementation rejects that unsupported mode
    # explicitly rather than silently changing its algorithm.
    torch.ops._C.triton_rejection(
        *args[:27],
        args[30],
        has_draft,
        synthetic,
        use_block,
    )


def _resample(_grid: Any, *args: Any, **kwargs: Any) -> None:
    block_size = kwargs.pop("BLOCK_SIZE")
    has_draft = kwargs.pop("HAS_DRAFT_LOGITS")
    use_fp64 = kwargs.pop("USE_FP64")
    use_block = kwargs.pop("USE_BLOCK_VERIFICATION")
    kwargs.pop("CONTEXT_WIDTH")
    watermark = kwargs.pop("WATERMARK")
    if watermark:
        raise RuntimeError("native CUDA resampling does not support watermarking")
    if kwargs:
        raise TypeError(f"unexpected resample launch options: {kwargs}")
    # Drop contexts, context stride, watermarking and key words (args 19:24).
    torch.ops._C.triton_resample(
        *args[:19],
        args[24],
        block_size,
        has_draft,
        use_fp64,
        use_block,
    )


def _insert_resampled(_grid: Any, *args: Any, **kwargs: Any) -> None:
    kwargs.pop("PADDED_RESAMPLE_NUM_BLOCKS")
    if kwargs:
        raise TypeError(f"unexpected insert-resampled launch options: {kwargs}")
    torch.ops._C.triton_insert_resampled(
        *args,
        args[5].dtype == torch.float64,
    )


gumbel_sample_kernel = KernelLauncher(_gumbel_sample)
compute_local_logits_stats_kernel = KernelLauncher(_compute_local_logits_stats)
rejection_kernel = KernelLauncher(_rejection)
resample_kernel = KernelLauncher(_resample)
insert_resampled_kernel = KernelLauncher(_insert_resampled)
