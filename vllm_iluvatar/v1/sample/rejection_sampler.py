# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar rejection-sampler override.

Corresponds to: vllm/v1/sample/rejection_sampler.py :: generate_uniform_probs

CoreX needs float32 uniform randoms because float64 CUDA tensors are not
reliable on this backend. Upstream ``rejection_sample`` calls
``generate_uniform_probs`` by name, so rebinding that function covers both
the draft-probability path and the deterministic-draft path. The greedy,
random, and recovered-token kernels stay on the community implementations.

Baseline : v0.31.0 (db9527a468)
"""

from __future__ import annotations

import torch


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
