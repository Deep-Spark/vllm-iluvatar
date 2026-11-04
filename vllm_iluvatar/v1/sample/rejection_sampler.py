# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar rejection-sampler override.

Corresponds to: vllm/v1/sample/rejection_sampler.py :: generate_uniform_probs

CoreX needs float32 uniform randoms because float64 CUDA tensors are not
reliable on this backend. Upstream ``rejection_sample`` calls
``generate_uniform_probs`` by name, so rebinding that function covers both
the draft-probability path and the deterministic-draft path. An opt-in eager
CQ adapter handles all-random deterministic drafts; other modes keep upstream.

Baseline : v0.31.0 (db9527a468)
"""

from __future__ import annotations

import torch
from vllm.v1.sample import rejection_sampler as _community

_original_rejection_sample = _community.rejection_sample


def _native_rejection_available(logits: torch.Tensor) -> bool:
    if (
        not logits.is_cuda
        or logits.ndim != 2
        or not logits.is_contiguous()
        or not 0 < logits.shape[1] <= 2**24
        or not 0 < logits.numel() <= 2**31 - 1
    ):
        return False
    from ixformer.inference import functions as F

    from vllm_iluvatar.ilu_util.arch import is_ivcore30

    return (
        is_ivcore30()
        and hasattr(F, "rejection_accept")
        and hasattr(F, "rejection_recover")
    )


def rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """V1 deterministic draft fast path; V2 keeps its GPU-only verifier.

    Preserve 0.31's single uniform draw and full-batch exponential draws,
    including generator overrides for nonempty requests. The 0.27 wrapper's
    second uniform draw and rejected-only RNG policy must not be migrated.
    Mixed/greedy, probabilistic drafts, synthetic and FP64 use upstream.
    """
    from vllm import envs

    if not (
        envs.VLLM_ILU_MTP_REJECTION_FUSION
        and draft_probs is None
        and sampling_metadata.all_random
        and not synthetic_mode
        and not use_fp64_gumbel
        and 0 < max_spec_len <= 64
        and 0 < len(num_draft_tokens) <= 65535
        and _native_rejection_available(target_logits)
        and not torch.cuda.is_current_stream_capturing()
    ):
        return _original_rejection_sample(
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

    from ixformer.inference.functions import rejection_accept, rejection_recover

    batch, vocab = len(num_draft_tokens), target_logits.shape[1]
    device = target_logits.device
    uniform = generate_uniform_probs(
        draft_token_ids.numel(), num_draft_tokens, sampling_metadata.generators, device
    )
    probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    output = torch.full((batch, max_spec_len + 1), -1, dtype=torch.int32, device=device)
    rejected = rejection_accept(
        probs, cu_num_draft_tokens, draft_token_ids, bonus_token_ids, uniform, output
    )
    # This is an eager adapter, not a graph-safe verifier. Observe the native
    # sentinel at the existing compact-metadata transfer, before using output.
    host_rejected = rejected.cpu().tolist()
    invalid = [i for i, (row, _) in enumerate(host_rejected) if row == -2]
    if invalid:
        raise RuntimeError(f"Invalid rejection sampling inputs for requests {invalid}")

    # Match upstream sample_recovered_tokens even when every draft is accepted
    # or a request has no draft tokens: default RNG draws cover the whole batch.
    noise = torch.empty((batch, vocab), dtype=torch.float32, device=device)
    noise.exponential_()
    for req, generator in sampling_metadata.generators.items():
        if num_draft_tokens[req] > 0:
            noise[req].exponential_(generator=generator)
    requests = [req for req, (row, _) in enumerate(host_rejected) if row >= 0]
    if requests:
        request_ids = torch.tensor(requests, dtype=torch.int32, device=device)
        rejection_recover(
            probs,
            draft_token_ids,
            rejected,
            request_ids,
            noise.index_select(0, request_ids.to(torch.int64)),
            output,
        )
    return output


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
