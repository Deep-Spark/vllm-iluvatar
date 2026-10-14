# SPDX-License-Identifier: Apache-2.0
"""Iluvatar override of the GPU spec-decode rejection kernel.

Corresponds to: vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
(``_rejection_kernel``), baseline v0.30.0 (ced6857afa).

Why: two things the Iluvatar Triton build does differently.

* ``tl_rand32`` (``tl.rand``) is already a 0-d scalar when the offset is a
  scalar. Wrapping it in ``tl.sum`` hits ``dtype`` has no ``numel`` inside
  the reduction. Leave ``u`` as returned.
* It does not fold ``constexpr and runtime`` conjunctions. ``USE_BLOCK_VERIFICATION``
  is a ``tl.constexpr``; under the default ``rejection_sample_method='standard'``
  (what DSpark / MTP use here) it is ``False`` and the block-verification
  pointers ``cumulative_log_p`` / ``local_residual_mass`` are ``None``. Upstream
  folds ``False and X`` away so those ``tl.load``s are never compiled; this build
  lowers the conjunction to a runtime branch and compiles the dead body, hitting
  ``AttributeError("'NoneType' object has no attribute 'type'")`` during
  spec-decode warmup.

v0.30.0 already splits the in-loop block-verification arm into a standalone
``elif USE_BLOCK_VERIFICATION:``, so only the tail fixup still needs re-nesting
(0.27.1 needed both).

How: this module mirrors ``_rejection_kernel`` verbatim except the tail guard
becomes ``if USE_BLOCK_VERIFICATION:`` with the runtime tests nested inside.
``u`` is left as the scalar ``tl_rand32`` returns.
Numerics are identical. All ``@triton.jit`` helpers are imported from the
community module so Triton resolves them from this function's globals.

The wiring module ``ilu_patch.general_patches.patch_rejection_sampler_utils``
rebinds ``_rejection_kernel`` onto the community module; the community
``rejection_sample`` launches it through its module globals, so the rebind
reaches it.

Remove: when the Iluvatar Triton build folds ``constexpr and runtime`` guards
and ``tl.rand`` returns a 0-d scalar.
"""

from __future__ import annotations

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32  # noqa: F401

# Helper @triton.jit callables invoked by _rejection_kernel. They must live in
# this module's globals so Triton resolves them when compiling our copy.
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (  # noqa: F401
    _compute_global_logprobs_and_logsumexp,
    _compute_global_logsumexp,
    _compute_global_residual_mass,
    _compute_global_target_argmax,
)


@triton.jit
def _rejection_kernel(
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
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
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
        if not is_greedy:
            # A -1 placeholder ends verification. Greedy is excluded because it
            # stores the target argmax upon first rejection, so it rejects the
            # placeholder via `accepted` instead.
            verifying &= is_valid_draft

        if verifying:
            pos = tl.load(pos_ptr + logit_idx)
            u = tl_rand32(seed, pos, includes_zero=False)
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
                verifying = accepted
                accepted_length += accepted
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    draft_sampled if accepted else target_argmax,
                )
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
                target_logprob, draft_logprob, target_lse, draft_lse = (
                    _compute_global_logprobs_and_logsumexp(
                        draft_sampled,
                        True,  # mask
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
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                else:
                    # Probability ratio test: p(x) > u * q(x)
                    # Equivalent log form: log_p(x) > log(u) + log_q(x)
                    accepted = target_logprob > tl.log(u) + draft_logprob
                verifying = accepted
                accepted_length += accepted
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)

    tl.store(rejected_steps_ptr + req_idx, accepted_length)
    # ILU: same constexpr-guard nesting for the tail block-verification
    # fixup. Upstream relies on `constexpr and runtime` folding; this build
    # lowers that to a runtime branch and compiles the dead body, whose
    # pointers are None under the standard rejection method.
    if USE_BLOCK_VERIFICATION:  # noqa: SIM102
        if not is_greedy and accepted_length < num_draft_tokens:
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
