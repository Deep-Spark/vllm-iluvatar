# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 FLy verification using the existing MR statistics and recovery kernels."""

import math

import torch
from ixformer.inference.functions.ops.sampling.float.fly_entropy import (
    compute_fly_entropy,
)
from ixformer.inference.functions.ops.sampling.float.fly_verify import fly_verify
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_local_logits_stats_kernel,
    _insert_resampled_kernel,
    _resample_kernel,
)


def rejection_sample(
    # [num_logits, V]
    target_logits: torch.Tensor,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits: torch.Tensor | None,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
    # [num_speculative_steps]
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64: bool = False,
    use_block_verification: bool = False,
    contexts: torch.Tensor | None = None,
    watermarking: torch.Tensor | None = None,
    watermark_key: int | None = None,
    fly_window_size: int = 0,
    fly_entropy_threshold: float = 0.3,
    fly_entropy_top_k: int = 3,
    fly_min_target_probability: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not math.isfinite(fly_min_target_probability)
        or not 0 <= fly_min_target_probability <= 1
    ):
        raise ValueError("FLy target probability floor must be finite and in [0, 1]")
    if fly_min_target_probability > 0 and (
        fly_window_size <= 0 or draft_logits is None
    ):
        raise ValueError(
            "FLy target probability floor requires FLy and full draft logits"
        )
    assert target_logits.ndim == 2 and target_logits.stride(-1) == 1
    assert draft_logits is None or (
        draft_logits.ndim == 3 and draft_logits.stride(-1) == 1
    )
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape

    watermark = contexts is not None
    assert watermark == (watermarking is not None) == (watermark_key is not None), (
        "contexts, watermarking and watermark_key must be set together."
    )
    contexts_stride = 0
    context_width = 1
    watermarking_bytes: torch.Tensor | None = None
    watermark_key_0 = 0
    watermark_key_1 = 0
    if contexts is not None:
        assert watermarking is not None and watermark_key is not None
        assert contexts.ndim == 2 and contexts.shape[0] == num_logits
        # Context words are hashed as uint32, so int32 (the request-state token
        # dtype) and int64 both land on the same PRF stream, -1 included.
        assert contexts.dtype in (torch.int32, torch.int64)
        if contexts.stride(-1) != 1:
            contexts = contexts.contiguous()
        assert watermarking.ndim == 1 and watermarking.dtype == torch.bool
        contexts_stride = contexts.stride(0)
        context_width = contexts.shape[-1]
        watermarking_bytes = watermarking.view(torch.uint8)
        watermark_key_0 = watermark_key & 0xFFFFFFFF
        watermark_key_1 = watermark_key >> 32
    draft_logits_stride_0 = 0
    draft_logits_stride_1 = 0
    if has_draft_logits := draft_logits is not None:
        draft_logits_stride_0 = draft_logits.stride(0)
        draft_logits_stride_1 = draft_logits.stride(1)
        # In some cases (e.g. MiMo v2.5 Pro + DFlash) the target model's
        # vocab size is larger than the draft's due to padding.
        vocab_size = min(vocab_size, draft_logits.size(-1))

    if (
        fly_window_size <= 0
        or synthetic_conditional_rates is not None
        or use_block_verification
    ):
        raise ValueError(
            "FLy needs a positive window and is distinct from synthetic/block verification"
        )
    entropy_result = compute_fly_entropy(
        target_logits[:, :vocab_size],
        fly_entropy_top_k,
        from_logits=True,
        return_logsumexp=fly_min_target_probability > 0,
    )
    fly_entropy, fly_logsumexp = (
        entropy_result if fly_min_target_probability > 0 else (entropy_result, None)
    )

    # Compute the per-vocab-block logits stats, such as target argmax
    # (for greedy requests), and target max + softmax exponential
    # (for non-greedy requests).
    VOCAB_BLOCK_SIZE = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    target_local_argmax = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.int64
    )
    target_local_max = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    target_local_sumexp = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    draft_local_max = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    draft_local_sumexp = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    _compute_local_logits_stats_kernel[(num_logits, vocab_num_blocks)](
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        target_logits,
        target_logits.stride(0),
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_speculative_steps,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
    )

    cumulative_log_p = None

    # Sample up until the first rejected/bonus token, and store
    # the step.
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    draft_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    fly_verify(
        sampled,
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        target_logits,
        target_local_argmax,
        target_local_max,
        target_local_sumexp,
        draft_sampled,
        draft_logits,
        draft_local_max,
        draft_local_sumexp,
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        fly_entropy,
        fly_logsumexp,
        fly_entropy_threshold,
        fly_window_size,
        fly_min_target_probability,
    )

    # Resample the rejected/bonus tokens.
    RESAMPLE_BLOCK_SIZE = 1024
    resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    resampled_local_argmax = target_logits.new_empty(
        num_reqs, resample_num_blocks, dtype=torch.int64
    )
    resampled_local_max = target_logits.new_empty(
        num_reqs,
        resample_num_blocks,
        dtype=torch.float64 if use_fp64 else torch.float32,
    )
    _resample_kernel[(num_reqs, resample_num_blocks)](
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        target_rejected_logsumexp,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_rejected_logsumexp,
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        cumulative_log_p,
        contexts,
        contexts_stride,
        watermarking_bytes,
        watermark_key_0,
        watermark_key_1,
        vocab_size,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        USE_FP64=use_fp64,
        USE_BLOCK_VERIFICATION=use_block_verification,
        CONTEXT_WIDTH=context_width,
        WATERMARK=watermark,
    )

    # Insert the resampled tokens into the output sampled.
    _insert_resampled_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        resample_num_blocks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
    )
    return sampled, num_sampled
