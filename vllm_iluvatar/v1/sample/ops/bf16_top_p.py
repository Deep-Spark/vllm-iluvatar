# SPDX-License-Identifier: Apache-2.0
"""Exact V2 top-p fast path for an unmodified BF16 LM-head output.

Sort the existing BF16 values before temperature, then use the original FP32
temperature, softmax and cumsum. Positive per-row temperature preserves order.
Boundary ties are resolved in original token order, including values merged by
FP32 rounding. Any active processor or unsupported sampling mode falls back.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import ixformer.inference.functions as ixfops
import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.sample.sampler import Sampler

logger = init_logger("vllm.iluvatar.bf16_top_p")


def eligible(
    sampler: Sampler,
    logits: torch.Tensor,
    idx_mapping_np: np.ndarray,
    skip_top_k_top_p: bool,
) -> bool:
    min_rows = 6 if os.environ.get("VLLM_ILU_TOP_P_BF16_HIST", "0") == "1" else 24
    if (
        os.environ.get("VLLM_ILU_TOP_P_BF16_SORT", "1") != "1"
        or skip_top_k_top_p
        or not logits.is_cuda
        or logits.dtype != torch.bfloat16
        or logits.ndim != 2
        or logits.shape[0] < min_rows
        or logits.shape[1] < 65536
        or logits.numel() >= 2**31
        or not logits.is_contiguous()
        or idx_mapping_np.size == 0
    ):
        return False

    # Match exactly the upstream no-op conditions. Never silently bypass a
    # custom processor, a subclass, or an active standard processor.
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
    states = sampler.sampling_states
    temp = states.temperature.np[idx_mapping_np]
    return bool(
        np.all(np.isfinite(temp) & (temp > 0))
        and np.all(states.min_p.np[idx_mapping_np] == 0)
        and np.all(states.top_k.np[idx_mapping_np] == states.vocab_size)
        and np.any(states.top_p.np[idx_mapping_np] != 1)
        and np.any(sampler.needs_logits_processing[idx_mapping_np])
    )


def apply_bf16_top_p(
    sampler: Sampler,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
) -> torch.Tensor:
    states = sampler.sampling_states
    processed = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
    sorted_logits = ixfops.sort_bf16_logits_hist(logits)
    # Reuse the actual V2 temperature arithmetic on both arrays.
    states.apply_temperature(processed, expanded_idx_mapping, idx_mapping_np)
    states.apply_temperature(sorted_logits, expanded_idx_mapping, idx_mapping_np)
    _, top_p = states.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
    probs = sorted_logits.softmax(dim=-1)
    torch.cumsum(probs, dim=-1, out=probs)
    ixfops.mask_top_p_keys(processed, sorted_logits, probs, top_p)
    if not getattr(sampler, "_ilu_bf16_top_p_logged", False):
        logger.info("Using exact BF16 histogram top-p before FP32 temperature")
        sampler._ilu_bf16_top_p_logged = True
    return processed
