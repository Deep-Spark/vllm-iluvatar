# SPDX-License-Identifier: Apache-2.0
"""Compile the PP0 ``post_update`` Triton kernel during warmup.

Community ``warmup_kernels`` only ``sample_tokens`` about six times. Non-last
PP ranks pre-seed the sampled-token queue with ``pp_size`` empty slots, so
those consumes never call ``post_update``. The first user batch then JITs
``_post_update_kernel`` mid-flight.

Do **not** flush extra execute/sample steps to pop that queue: after CUDA
graph capture, another PP broadcast can hit
``ixErrorStreamCaptureUnsupported``. Launch ``post_update`` on the live
``req_states`` tensors with a skipped ``idx_mapping`` so the missing
``output_bin_counts is None`` specialization compiles without NCCL.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import post_update

logger = init_logger("vllm." + __name__)


def compile_pp_post_update_kernel(model_runner: Any) -> bool:
    """JIT the non-last-rank ``post_update`` specialization on live buffers."""
    if getattr(model_runner, "is_pooling_model", False):
        return False
    if getattr(model_runner, "pp_handler", None) is None:
        return False

    device = getattr(model_runner, "device", None)
    req_states = getattr(model_runner, "req_states", None)
    if device is None or req_states is None:
        return False

    num_reqs = min(2, int(getattr(req_states, "max_num_reqs", 2) or 2))
    max_sample = max(int(getattr(model_runner, "num_speculative_steps", 0)), 0) + 1
    # Negative indices make the kernel return before stores.
    idx = torch.full((num_reqs,), -1, device=device, dtype=torch.int32)
    sampled = torch.zeros((num_reqs, max_sample), device=device, dtype=torch.int64)
    num_sampled = torch.ones(num_reqs, device=device, dtype=torch.int32)
    num_rejected = torch.zeros(num_reqs, device=device, dtype=torch.int32)
    query_start = torch.arange(num_reqs + 1, device=device, dtype=torch.int32) * 2
    try:
        post_update(
            idx,
            req_states.num_computed_tokens.gpu,
            req_states.last_sampled_tokens,
            None,
            sampled,
            num_sampled,
            num_rejected,
            query_start,
            req_states.all_token_ids.gpu,
            req_states.total_len.gpu,
        )
        post_update(
            idx,
            req_states.num_computed_tokens.gpu,
            req_states.last_sampled_tokens,
            None,
            sampled,
            num_sampled,
            num_rejected,
            None,
            req_states.all_token_ids.gpu,
            req_states.total_len.gpu,
        )
        if str(device) != "cpu":
            torch.accelerator.synchronize()
    except Exception:
        logger.exception("Failed to compile PP post_update kernel during warmup")
        return False
    logger.info("Compiled PP post_update kernel (output_bin_counts=None) during warmup")
    return True


def make_warmup_kernels(original: Callable[..., None]) -> Callable[..., None]:
    @wraps(original)
    def warmup_kernels(
        model_runner: Any,
        worker_execute_model: Callable[[Any], Any],
        worker_sample_tokens: Callable[[Any], Any],
    ) -> None:
        original(model_runner, worker_execute_model, worker_sample_tokens)
        compile_pp_post_update_kernel(model_runner)

    return warmup_kernels
