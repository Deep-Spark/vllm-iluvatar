# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar rejection-sampler overrides.

Corresponds to: vllm/v1/sample/rejection_sampler.py ::
``rejection_sample``, ``generate_uniform_probs``,
``rejection_greedy_sample_kernel``, and ``rejection_random_sample_kernel``

Iluvatar Triton cannot directly use the loaded ``is_greedy`` scalar as a branch
predicate in these kernels because it attempts an invalid bitcast into int1.
CoreX also needs float32 uniform randoms because float64 CUDA tensors are not
reliable on this backend, and deterministic draft sampling avoids the upstream
NO_DRAFT_PROBS Triton recovered-token path that produces wrong tokens here.

Baseline : main_61ac3680210
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.sample import rejection_sampler as _community
from vllm.v1.sample.metadata import SamplingMetadata

GREEDY_TEMPERATURE = _community.GREEDY_TEMPERATURE
PLACEHOLDER_TOKEN_ID = _community.PLACEHOLDER_TOKEN_ID
_community_rejection_sample = _community.rejection_sample


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_logits: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    if draft_probs is not None:
        return _community_rejection_sample(
            draft_token_ids,
            num_draft_tokens,
            max_spec_len,
            cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            synthetic_mode=synthetic_mode,
            synthetic_conditional_rates=synthetic_conditional_rates,
            use_fp64_gumbel=use_fp64_gumbel,
        )

    # >>> ILU MODIFICATION >>>
    # CoreX deterministic draft path: the upstream NO_DRAFT_PROBS Triton branch
    # can produce wrong recovered tokens, so only this path is owned locally.
    assert draft_token_ids.ndim == 1
    assert cu_num_draft_tokens.ndim == 1
    assert target_logits.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    _, vocab_size = target_logits.shape
    device = target_logits.device
    assert draft_token_ids.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)

    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

    uniform_probs: torch.Tensor | None = None
    if synthetic_mode or not sampling_metadata.all_greedy:
        uniform_probs = generate_uniform_probs(
            num_tokens,
            num_draft_tokens,
            sampling_metadata.generators,
            device,
        )

    if not sampling_metadata.all_random:
        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
            uniform_probs,
            synthetic_conditional_rates,
            SYNTHETIC_MODE=synthetic_mode,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    assert target_probs.is_contiguous()

    rejection_sample_deterministic_draft_torch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_probs,
        bonus_token_ids,
        is_greedy,
        num_draft_tokens,
        sampling_metadata,
        device,
    )
    # <<< ILU MODIFICATION <<<
    return output_token_ids


def rejection_sample_deterministic_draft_torch(
    output_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    is_greedy: torch.Tensor | None,
    num_draft_tokens: list[int],
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> None:
    """Rejection sampling for deterministic argmax draft tokens."""
    num_tokens = draft_token_ids.shape[0]
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    recovered_req_indices: list[int] = []
    recovered_pos_indices: list[int] = []
    recovered_target_indices: list[int] = []
    recovered_draft_token_ids: list[int] = []

    for req_idx, n in enumerate(num_draft_tokens):
        if n == 0:
            continue
        if is_greedy is not None and bool(is_greedy[req_idx].item()):
            continue

        start_idx = 0 if req_idx == 0 else int(cu_num_draft_tokens[req_idx - 1].item())
        rejected = False
        for pos in range(n):
            if rejected:
                break
            token_idx = start_idx + pos
            draft_token_id = int(draft_token_ids[token_idx].item())
            target_prob = target_probs[token_idx, draft_token_id]
            uniform_prob = uniform_probs[token_idx]
            if bool(target_prob >= uniform_prob):
                output_token_ids[req_idx, pos] = draft_token_id
            else:
                rejected = True
                recovered_req_indices.append(req_idx)
                recovered_pos_indices.append(pos)
                recovered_target_indices.append(token_idx)
                recovered_draft_token_ids.append(draft_token_id)

        if not rejected and n < output_token_ids.shape[1]:
            output_token_ids[req_idx, n] = bonus_token_ids[req_idx]

    if not recovered_req_indices:
        return

    target_idx_tensor = torch.tensor(
        recovered_target_indices, dtype=torch.long, device=device
    )
    draft_id_tensor = torch.tensor(
        recovered_draft_token_ids, dtype=torch.long, device=device
    )
    probs = target_probs[target_idx_tensor].clone()
    probs.scatter_(1, draft_id_tensor.unsqueeze(1), 0.0)

    q = torch.empty_like(probs)
    q.exponential_()
    for row_idx, req_idx in enumerate(recovered_req_indices):
        generator = sampling_metadata.generators.get(req_idx)
        if generator is not None:
            q[row_idx].exponential_(generator=generator)
    sampled_token_ids = (probs / q).argmax(dim=-1).to(output_token_ids.dtype)

    req_idx_tensor = torch.tensor(
        recovered_req_indices, dtype=torch.long, device=device
    )
    pos_idx_tensor = torch.tensor(
        recovered_pos_indices, dtype=torch.long, device=device
    )
    output_token_ids[req_idx_tensor, pos_idx_tensor] = sampled_token_ids


def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    # >>> ILU MODIFICATION >>>
    # NOTE: Upstream deliberately uses float64 here because float32 has a
    # non-negligible chance of sampling an exact 0.0
    # (https://github.com/pytorch/pytorch/issues/16706), and a uniform_prob
    # of 0.0 makes the `target_prob / draft_prob >= uniform_prob` test always
    # accept. However, on Iluvatar Corex GPUs float64 ("torch.double") is only
    # partially supported: any float64 CUDA tensor reads back as all zeros,
    # which silently forces a 100% draft-acceptance rate regardless of the
    # sampling distribution. We therefore generate in float32 (which is
    # correct on this hardware).
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float32,
        device=device,
    )
    # <<< ILU MODIFICATION <<<
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    max_spec_len,
    uniform_probs_ptr,  # [num_tokens] or None (synthetic mode only)
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    SYNTHETIC_MODE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    # >>> ILU MODIFICATION >>>
    if is_greedy_ptr is None:
        is_greedy = True
    else:
        is_greedy = tl.load(is_greedy_ptr + req_idx).to(tl.int32) != 0
    # <<< ILU MODIFICATION <<<
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(tl.int32)
            if SYNTHETIC_MODE:
                uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                # -1 is used for padded draft token ids that should be rejected.
                accepted = (uniform_prob < rate) and draft_token_id >= 0
                token_id = draft_token_id if accepted else target_argmax_id
                rejected = not accepted
            else:
                token_id = target_argmax_id
                rejected = draft_token_id != target_argmax_id
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    NO_DRAFT_PROBS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    # >>> ILU MODIFICATION >>>
    is_greedy = tl.load(is_greedy_ptr + req_idx).to(tl.int32) != 0
    # <<< ILU MODIFICATION <<<
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            if draft_token_id < 0:
                # -1 is used for padded draft token ids that should be rejected.
                accepted = False
            elif SYNTHETIC_MODE:
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                accepted = uniform_prob < rate
            else:
                if NO_DRAFT_PROBS:
                    draft_prob = 1
                else:
                    draft_prob = tl.load(
                        draft_probs_ptr
                        + (start_idx + pos) * vocab_size
                        + draft_token_id
                    )
                target_prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
                # NOTE(woosuk): While the draft probability should never be 0,
                # we check it to avoid NaNs. If it happens to be 0, we reject.
                accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
            if accepted:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )
