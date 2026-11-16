# SPDX-License-Identifier: Apache-2.0
"""Opt-in override for VllmConfig.max_concurrent_batches in async PP serving.

Target   : vllm.config.vllm.VllmConfig.max_concurrent_batches
Baseline : v0.31.0 (341bf7d4)
Why      : With async scheduling + V2 model runner, upstream sets
           max_concurrent_batches = pp_size + 1, leaving only one queued batch
           of slack to hide scheduler / client arrival jitter during PP fill.
How      : VLLM_ILU_PP_MAX_CONCURRENT_BATCHES (>0) raises the queue depth for
           async V2 PP>1 configs without changing other configs or SWA budgets
           when unset (default 0).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from vllm.logger import init_logger

from vllm_iluvatar.ilu_patch import general_patch as p

logger = init_logger(__name__)

# vLLM 341bf7d4, VllmConfig.max_concurrent_batches.fget; normalized source SHA256.
MAX_CONCURRENT_BATCHES_SOURCE_SHA = (
    "fd5570554d04780a12b636702792bd643d1d40ab06f0b9d17796594c0e3906b0"
)


def _apply() -> None:
    import vllm.envs as envs
    from vllm.config import VllmConfig

    depth = int(getattr(envs, "VLLM_ILU_PP_MAX_CONCURRENT_BATCHES", 0) or 0)
    if depth <= 0:
        return

    original = VllmConfig.max_concurrent_batches.fget
    p.check_fingerprint(
        "VllmConfig.max_concurrent_batches",
        inspect.getsource(original),
        MAX_CONCURRENT_BATCHES_SOURCE_SHA,
        baseline_is_hash=True,
        resync_hint="Review VllmConfig.max_concurrent_batches before updating source hash.",
    )

    def max_concurrent_batches(self):
        original_depth = original(self)
        parallel = self.parallel_config
        if (
            parallel.pipeline_parallel_size > 1
            and self.scheduler_config.async_scheduling
            and self.use_v2_model_runner
        ):
            return max(original_depth, depth)
        return original_depth

    wrapped_prop = property(max_concurrent_batches)
    p.check_signatures(
        SimpleNamespace(max_concurrent_batches=original),
        SimpleNamespace(max_concurrent_batches=max_concurrent_batches),
        ["max_concurrent_batches"],
        strict=True,
    )
    VllmConfig.max_concurrent_batches = wrapped_prop  # type: ignore[method-assign]
    logger.info("ILU async PP max_concurrent_batches floor: %d", depth)


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="VllmConfig.max_concurrent_batches opt-in async PP queue floor",
    )
