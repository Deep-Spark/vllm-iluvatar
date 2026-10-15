# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar rejection-sampler overrides.

Corresponds to: vllm/v1/sample/rejection_sampler.py ::
``rejection_sample`` and ``generate_uniform_probs``

CoreX needs float32 uniform randoms because float64 CUDA tensors are not
reliable on this backend. The greedy, random, and recovered-token kernels
stay on the community implementations. On Triton 3.6 the bool ``is_greedy``
branch and the NO_DRAFT_PROBS recovered-token kernel both match the
reference.

Baseline : v0.31.0 (db9527a468)
"""

from __future__ import annotations

import torch

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
    # CoreX deterministic draft path. Uniforms are float32. Accept / reject
    # and recovered tokens use the community kernels.
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
        _community.rejection_greedy_sample_kernel[(batch_size,)](
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
    assert uniform_probs is not None
    assert is_greedy is not None

    recovered_token_ids = _community.sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        None,
        target_probs,
        sampling_metadata,
        device,
        use_fp64_gumbel,
    )
    _community.rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        None,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        synthetic_conditional_rates,
        NO_DRAFT_PROBS=True,
        SYNTHETIC_MODE=synthetic_mode,
    )
    # <<< ILU MODIFICATION <<<
    return output_token_ids


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
