# SPDX-License-Identifier: Apache-2.0
"""V2 sampling from compact BF16 nucleus distributions on MR and CQ."""

from __future__ import annotations

import numpy as np
import torch
from ixformer.inference.backend_registry import current_cuda_arch
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS

logger = init_logger("vllm.iluvatar.top_p_summary")


def eligible(sampler, logits, idx_mapping_np, *, return_logprobs=False):
    if (
        current_cuda_arch() not in ("ivcore11", "ivcore30")
        or not getattr(sampler, "_ilu_summary_replicated_batch", False)
        or sampler.use_fp64_gumbel
        or sampler.return_sampling_mask
        or not logits.is_cuda
        or logits.dtype != torch.bfloat16
        or logits.ndim != 2
        or not 0 < logits.shape[0] <= 65535
        or not 0 < logits.shape[1] <= 1048576
        or not logits.is_contiguous()
        or idx_mapping_np.size == 0
        or (return_logprobs and sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES)
    ):
        return False
    # Exact types preserve custom processor/subclass semantics.
    processors = sampler.logits_processors
    if [type(p) for p in processors] != [LogitBiasState, PenaltiesState, BadWordsState]:
        return False
    bias, penalties, bad_words = processors
    if (
        np.any(bias.use_logit_bias[idx_mapping_np])
        or np.any(penalties.use_penalty[idx_mapping_np])
        or np.any(bad_words.num_bad_words.np[idx_mapping_np])
    ):
        return False
    budget = sampler.thinking_budget_state
    if budget.enabled and np.any(budget.use_thinking_budget[idx_mapping_np]):
        return False
    s = sampler.sampling_states
    t, p = s.temperature.np[idx_mapping_np], s.top_p.np[idx_mapping_np]
    return bool(
        np.all(np.isfinite(t) & (t >= 0))
        and np.all(np.isfinite(p) & (p > 0) & (p <= 1))
        and np.all(s.min_p.np[idx_mapping_np] == 0)
        and np.all(s.top_k.np[idx_mapping_np] == s.vocab_size)
    )


def verify_eligible(verifier, logits, draft_logits, idx_mapping_np):
    if (
        draft_logits is not None
        or verifier.watermark_key is not None
        or verifier.use_block_verification
        or verifier.synthetic_conditional_rates is not None
        or not 0 <= verifier.num_speculative_steps < 256
    ):
        return False
    sampler = verifier.sampler
    return eligible(
        sampler,
        logits,
        idx_mapping_np,
        return_logprobs=(
            sampler.sampling_states.max_num_logprobs(idx_mapping_np) != NO_LOGPROBS
        ),
    )


def sample_summary(sampler, logits, mapping, cu, draft, positions, width):
    from ixformer.inference.functions.ops.sampling.float.top_p_summary import (
        sample_top_p_summary,
        top_p_summary,
        top_p_summary_buffers,
    )

    group = get_tp_group()
    requests = cu.numel() - 1
    # Outputs must own their storage: upstream may retain several request chunks.
    sampled = torch.empty((requests, width), device=logits.device, dtype=torch.int64)
    counts = torch.empty(requests, device=logits.device, dtype=torch.int32)
    if group.rank_in_group == 0:
        n, v = logits.shape
        cache = getattr(sampler, "_ilu_summary_buffers", None)
        if cache is None or cache[0] < n or cache[1] != v or cache[2] != logits.device:
            capacity = max(n, cache[0] if cache and cache[1] == v else 0)
            output, work = top_p_summary_buffers(capacity, v, logits.device)
            cache = (capacity, v, logits.device, output, work)
            sampler._ilu_summary_buffers = cache
        summary = tuple(x[:n] for x in cache[3])
        scratch = tuple(x[:n] for x in cache[4])
        states = sampler.sampling_states
        top_p_summary(
            logits, states.temperature.gpu, states.top_p.gpu, mapping, summary, scratch
        )
        sample_top_p_summary(
            logits,
            states.temperature.gpu,
            states.seeds.gpu,
            mapping,
            cu,
            draft,
            positions,
            summary,
            width,
            output=(sampled, counts),
        )
    # Weighted FP32 histogram atomics need not reproduce the exact reduction
    # order on TP peers. One owner plus tiny broadcasts guarantees identical
    # accepted prefixes/state updates on every peer. Batch-sharded sampling is
    # deliberately gated out: its TP ranks own different request batches.
    group.broadcast(sampled, src=0)
    group.broadcast(counts, src=0)
    if not getattr(sampler, "_ilu_summary_logged", False):
        logger.info("Using compact BF16 top-p sampling and one-hot verification")
        sampler._ilu_summary_logged = True
    return sampled, counts
