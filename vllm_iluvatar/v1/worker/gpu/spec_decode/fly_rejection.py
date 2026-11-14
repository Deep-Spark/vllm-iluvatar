# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 FLy verification using the existing MR statistics and recovery kernels."""

import math

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_cumulative_log_p_kernel,
    _compute_global_logprobs_and_logsumexp,
    _compute_global_logsumexp,
    _compute_global_residual_mass,
    _compute_global_target_argmax,
    _compute_local_residual_mass_kernel,
)

from vllm_iluvatar.v1.spec_decode.fly import compute_fly_entropy
from vllm_iluvatar.v1.worker.gpu.spec_decode.sampling_kernels import (
    compute_local_logits_stats_kernel as _compute_local_logits_stats_kernel,
)
from vllm_iluvatar.v1.worker.gpu.spec_decode.sampling_kernels import (
    insert_resampled_kernel as _insert_resampled_kernel,
)
from vllm_iluvatar.v1.worker.gpu.spec_decode.sampling_kernels import (
    rejection_kernel as _rejection_kernel,
)
from vllm_iluvatar.v1.worker.gpu.spec_decode.sampling_kernels import (
    rejection_sample_with_aligned_draft,
)
from vllm_iluvatar.v1.worker.gpu.spec_decode.sampling_kernels import (
    resample_kernel as _resample_kernel,
)


@triton.jit
def _fly_rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    rejected_steps_ptr,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    # [num_speculative_steps]
    synthetic_conditional_rates_ptr,
    # [num_logits]
    cumulative_log_p_ptr,
    # [num_logits, num_blocks]
    local_residual_mass_ptr,
    local_residual_mass_stride,
    # [num_logits]
    fly_entropy_ptr,
    fly_entropy_threshold,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
    FLY_WINDOW_SIZE: tl.constexpr,
    FLY_MIN_TARGET_LOGPROB: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0

    accepted_length = tl.zeros((), tl.int64)
    # Freeze the accepted prefix and recovery stats until the window is confirmed.
    pending_rejection = tl.full((), -1, tl.int64)
    rejected_argmax = tl.full((), 0, tl.int64)
    target_lse = 0.0
    draft_lse = 0.0
    verifying = True
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        # -1 is used for placeholder draft token ids that should be rejected.
        is_valid_draft = draft_sampled >= 0
        # Avoid possible OOB ptr access.
        draft_sampled = tl.maximum(0, draft_sampled)
        if not is_greedy and FLY_WINDOW_SIZE == 0:
            # A -1 placeholder ends verification. Greedy is excluded because it
            # stores the target argmax upon first rejection, so it rejects the
            # placeholder via `accepted` instead.
            verifying &= is_valid_draft
        if verifying:
            target_argmax = tl.full((), 0, tl.int64)
            can_defer = False
            accepted = False
            if FLY_MIN_TARGET_LOGPROB is not None:
                target_logprob = tl.full((), float("nan"), tl.float32)
                draft_logprob = tl.full((), float("nan"), tl.float32)
            pos = tl.load(pos_ptr + logit_idx)
            # The MR native verifier uses Philox32 with a uint32 position.
            # Keep that stream: this SDK selects Philox64 for int64 offsets.
            u = tl_rand32(seed, pos.to(tl.uint32), includes_zero=False)
            if is_greedy:
                # Greedy sampling. Accept IFF draft matches target argmax.
                # NOTE: Target argmax is stored directly so that resampling
                # can be skipped upon rejection.
                target_argmax = _compute_global_target_argmax(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_argmax_ptr,
                    target_local_argmax_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                else:
                    accepted = target_argmax == draft_sampled
                accepted &= is_valid_draft
                if FLY_WINDOW_SIZE > 0:
                    target_logit = tl.load(
                        target_logits_ptr
                        + logit_idx * target_logits_stride
                        + draft_sampled,
                        mask=is_valid_draft,
                        other=float("-inf"),
                    ).to(tl.float32)
                    can_defer = is_valid_draft and target_logit > float("-inf")
            elif USE_BLOCK_VERIFICATION:
                # Block verification (Sun et al., 2024): https://arxiv.org/abs/2403.10444
                prefix_joint_ratio = tl.exp(
                    tl.load(cumulative_log_p_ptr + logit_idx).to(tl.float32)
                )
                next_draft_token = tl.load(
                    draft_sampled_ptr + logit_idx + 2,
                    mask=i < num_draft_tokens - 1,
                    other=-1,
                ).to(tl.int64)
                if next_draft_token >= 0:
                    residual_mass = _compute_global_residual_mass(
                        local_residual_mass_ptr,
                        local_residual_mass_stride,
                        prefix_joint_ratio,
                        target_logits_ptr,
                        target_logits_stride,
                        target_local_max_ptr,
                        target_local_max_stride,
                        target_local_sumexp_ptr,
                        target_local_sumexp_stride,
                        next_draft_token,
                        logit_idx + 1,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                        HAS_DRAFT_LOGITS,
                    )
                    denom = residual_mass + 1.0 - prefix_joint_ratio
                    h = tl.where(denom > 0.0, residual_mass / denom, 1.0)
                else:
                    h = prefix_joint_ratio
                accepted_length = tl.where(u <= h, i + 1, accepted_length)
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            else:
                # Speculative decoding (Leviathan et al., 2023): https://arxiv.org/abs/2211.17192
                target_logprob, draft_logprob, step_target_lse, step_draft_lse = (
                    _compute_global_logprobs_and_logsumexp(
                        draft_sampled,
                        is_valid_draft,
                        logit_idx,
                        req_state_idx,
                        i,
                        temp,
                        target_logits_ptr,
                        target_logits_stride,
                        target_local_max_ptr,
                        target_local_max_stride,
                        target_local_sumexp_ptr,
                        target_local_sumexp_stride,
                        draft_logits_ptr,
                        draft_logits_stride_0,
                        draft_logits_stride_1,
                        draft_local_max_ptr,
                        draft_local_max_stride,
                        draft_local_sumexp_ptr,
                        draft_local_sumexp_stride,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                        HAS_DRAFT_LOGITS,
                    )
                )
                if FLY_WINDOW_SIZE == 0 or pending_rejection < 0:
                    target_lse = step_target_lse
                    draft_lse = step_draft_lse
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                else:
                    # Probability ratio test: p(x) > u * q(x)
                    # Equivalent log form: log_p(x) > log(u) + log_q(x)
                    accepted = target_logprob > tl.log(u) + draft_logprob
                if FLY_WINDOW_SIZE > 0:
                    accepted &= is_valid_draft
                    can_defer = (
                        is_valid_draft and target_logprob > float("-inf")
                    ) and draft_logprob > float("-inf")

            if not USE_BLOCK_VERIFICATION or is_greedy:
                if FLY_WINDOW_SIZE > 0:
                    if pending_rejection >= 0:
                        # Only native acceptance can confirm a pending window.
                        verifying = accepted
                        if accepted and i - pending_rejection == FLY_WINDOW_SIZE:
                            accepted_length = i + 1
                            pending_rejection = tl.full((), -1, tl.int64)
                    elif accepted:
                        accepted_length = i + 1
                    else:
                        rejected_argmax = target_argmax
                        entropy = tl.load(fly_entropy_ptr + logit_idx)
                        verifying = (
                            can_defer and i + FLY_WINDOW_SIZE < num_draft_tokens
                        ) and entropy >= fly_entropy_threshold
                        if FLY_MIN_TARGET_LOGPROB is not None:
                            absolute_probability_allowed = (
                                ~is_greedy
                                & is_valid_draft
                                & (target_logprob >= FLY_MIN_TARGET_LOGPROB)
                                & (target_logprob <= 0.0)
                                & (draft_logprob > -float("inf"))
                                & (draft_logprob <= 0.0)
                            )
                            verifying &= absolute_probability_allowed
                        if verifying:
                            pending_rejection = i
                    tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
                else:
                    verifying = accepted
                    accepted_length += accepted
                    tl.store(
                        sampled_ptr + req_idx * sampled_stride + i,
                        target_argmax if is_greedy and not accepted else draft_sampled,
                    )
    if (FLY_WINDOW_SIZE > 0 and is_greedy) and accepted_length < num_draft_tokens:
        tl.store(
            sampled_ptr + req_idx * sampled_stride + accepted_length,
            rejected_argmax,
        )

    tl.store(rejected_steps_ptr + req_idx, accepted_length)
    if (
        USE_BLOCK_VERIFICATION and not is_greedy
    ) and accepted_length < num_draft_tokens:
        # Compute the target and draft log exponential sums for the
        # rejected token.
        rejected_idx = start_idx + accepted_length
        target_lse = _compute_global_logsumexp(
            target_local_max_ptr,
            target_local_max_stride,
            target_local_sumexp_ptr,
            target_local_sumexp_stride,
            rejected_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        if HAS_DRAFT_LOGITS:
            draft_lse = _compute_global_logsumexp(
                draft_local_max_ptr,
                draft_local_max_stride,
                draft_local_sumexp_ptr,
                draft_local_sumexp_stride,
                rejected_idx,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
            )
    tl.store(target_rejected_logsumexp_ptr + req_idx, target_lse)
    tl.store(draft_rejected_logsumexp_ptr + req_idx, draft_lse)


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

    fly_entropy = None
    if fly_window_size > 0:
        assert synthetic_conditional_rates is None and not use_block_verification
        fly_entropy = compute_fly_entropy(
            target_logits[:, :vocab_size], fly_entropy_top_k, from_logits=True
        )

    # Compute the per-vocab-block logits stats, such as target argmax
    # (for greedy requests), and target max + softmax exponential
    # (for non-greedy requests).
    VOCAB_BLOCK_SIZE = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
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

    # Precompute the running joint ratio and residual mass for block
    # verification.
    if use_block_verification:
        assert synthetic_conditional_rates is None, (
            "Block verification is incompatible with synthetic acceptance rates."
        )

        # Compute the log of the running joint ratio, p_i.
        # cumulative_log_p[start + i] = log(p_{i+1}), the cumulative ratio after
        # the (i+1)-th draft token.
        cumulative_log_p = target_logits.new_empty(num_logits, dtype=torch.float32)
        _compute_cumulative_log_p_kernel[(num_reqs,)](
            cumulative_log_p,
            target_logits,
            target_logits.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_sampled,
            draft_logits,
            draft_logits_stride_0,
            draft_logits_stride_1,
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            cu_num_logits,
            idx_mapping,
            temperature,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=has_draft_logits,
            num_warps=1,
        )

        # Compute the per-vocab-block partials of the residual mass, later reduced
        # to the total by _compute_global_residual_mass. Only launched for full
        # draft logits distributions. One-hot drafts used a closed-form residual
        # mass instead.
        if has_draft_logits:
            local_residual_mass = target_logits.new_empty(
                num_logits, vocab_num_blocks, dtype=torch.float32
            )
            _compute_local_residual_mass_kernel[(num_logits, vocab_num_blocks)](
                local_residual_mass,
                local_residual_mass.stride(0),
                cumulative_log_p,
                target_logits,
                target_logits.stride(0),
                target_local_max,
                target_local_max.stride(0),
                target_local_sumexp,
                target_local_sumexp.stride(0),
                draft_logits,
                draft_logits_stride_0,
                draft_logits_stride_1,
                draft_local_max,
                draft_local_max.stride(0),
                draft_local_sumexp,
                draft_local_sumexp.stride(0),
                draft_sampled,
                expanded_idx_mapping,
                expanded_local_pos,
                temperature,
                vocab_size,
                num_speculative_steps,
                vocab_num_blocks,
                BLOCK_SIZE=VOCAB_BLOCK_SIZE,
                PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            )
        else:
            local_residual_mass = None
    else:
        cumulative_log_p = None
        local_residual_mass = None

    # Sample up until the first rejected/bonus token, and store
    # the step.
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    draft_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    if fly_window_size > 0:
        _fly_rejection_kernel[(num_reqs,)](
            sampled,
            sampled.stride(0),
            num_sampled,
            target_rejected_logsumexp,
            draft_rejected_logsumexp,
            target_logits,
            target_logits.stride(0),
            target_local_argmax,
            target_local_argmax.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_sampled,
            draft_logits,
            draft_logits_stride_0,
            draft_logits_stride_1,
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            cu_num_logits,
            idx_mapping,
            temperature,
            seed,
            pos,
            synthetic_conditional_rates,
            cumulative_log_p,
            local_residual_mass,
            local_residual_mass.stride(0) if local_residual_mass is not None else 0,
            fly_entropy,
            fly_entropy_threshold,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=has_draft_logits,
            SYNTHETIC_MODE=synthetic_conditional_rates is not None,
            USE_BLOCK_VERIFICATION=use_block_verification,
            FLY_WINDOW_SIZE=fly_window_size,
            FLY_MIN_TARGET_LOGPROB=(
                math.log(fly_min_target_probability)
                if fly_min_target_probability > 0
                else None
            ),
            num_warps=1,
        )
    else:
        _rejection_kernel[(num_reqs,)](
            sampled,
            sampled.stride(0),
            num_sampled,
            target_rejected_logsumexp,
            draft_rejected_logsumexp,
            target_logits,
            target_logits.stride(0),
            target_local_argmax,
            target_local_argmax.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_sampled,
            draft_logits,
            draft_logits_stride_0,
            draft_logits_stride_1,
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            cu_num_logits,
            idx_mapping,
            temperature,
            seed,
            pos,
            synthetic_conditional_rates,
            cumulative_log_p,
            local_residual_mass,
            local_residual_mass.stride(0) if local_residual_mass is not None else 0,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=has_draft_logits,
            SYNTHETIC_MODE=synthetic_conditional_rates is not None,
            USE_BLOCK_VERIFICATION=use_block_verification,
            num_warps=1,
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


# Preserve the exact BF16-to-FP32 cast required by the native MR kernels.
rejection_sample = rejection_sample_with_aligned_draft(rejection_sample)
