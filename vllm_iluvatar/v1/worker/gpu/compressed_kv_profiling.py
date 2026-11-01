# SPDX-License-Identifier: Apache-2.0
"""Pair the V2 profiler's throwaway KV cache with temporary scale storage."""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from functools import wraps

_MISSING = object()
logger = logging.getLogger(__name__)


@contextmanager
def _override_runner_attribute(runner, name, value):
    previous = vars(runner).get(name, _MISSING)
    setattr(runner, name, value)
    try:
        yield
    finally:
        if previous is _MISSING:
            delattr(runner, name)
        else:
            setattr(runner, name, previous)


@contextmanager
def compressed_kv_profiling(
    runner, fmt, *, workspace_bytes: int, prefill_producer: bool
):
    """Supply sidecars only while this runner profiles a minimal graph cache.

    The original profiler owns its throwaway graphs and payload teardown.
    Fixed storage is allocated after ordinary memory profiling and before
    capture's initial free-memory sample, so the normal CacheBudget reserves
    the formal sidecars/workspace once, independently of the graph delta.
    """
    original_profile = runner.profile_cudagraph_memory
    if getattr(original_profile, "_ilu_compressed_kv_profiling", False):
        raise RuntimeError("Compressed KV profiling is already active")

    @wraps(original_profile)
    def profile():
        from vllm.v1.worker.gpu.cudagraph_utils import _teardown_profiling_state

        from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
            CompressedKVState,
            plan_scale_layout,
        )

        context = runner.compilation_config.static_forward_context
        original_initialize = runner.initialize_kv_cache
        if getattr(original_initialize, "_ilu_compressed_kv_profiling", False):
            raise RuntimeError("Compressed KV profiling is already active")
        state = None
        index_state = None
        owners = []

        @wraps(original_initialize)
        def initialize(config, is_profiling=False, kv_cache_allocation_context=None):
            nonlocal state, index_state
            if is_profiling:
                if owners:
                    raise RuntimeError("Profiling KV cache was initialized twice")
                if any(
                    getattr(layer, "compressed_kv_workspace", None) is not None
                    or getattr(layer, "kv_cache_scales", None) is not None
                    for layer in context.values()
                ):
                    raise RuntimeError(
                        "Profiling cannot replace a live compressed KV cache"
                    )
                from vllm_iluvatar.models.deepseek_v41.indexer_kv import (
                    IndexerScaleState,
                    plan_indexer_scales,
                )

                index_layout = plan_indexer_scales(config)
                if index_layout.placements:
                    index_state = IndexerScaleState(index_layout, runner.device, context)
                    owners.append(index_state)
                if fmt is not None:
                    state = CompressedKVState(
                        plan_scale_layout(config, fmt),
                        fmt,
                        runner.device,
                        context,
                        workspace_bytes=workspace_bytes,
                        prefill_producer=prefill_producer,
                    )
                    owners.append(state)
            result = original_initialize(
                config,
                is_profiling=is_profiling,
                kv_cache_allocation_context=kv_cache_allocation_context,
            )
            if is_profiling:
                if state is not None:
                    runner.kv_caches.extend(state.scales.values())
                extra_planes = tuple(index_state.scales.values()) if index_state else ()
                runner.kv_caches.extend(extra_planes)
                runner._init_kv_zero_meta()
                if state is not None:
                    state.bind_zeroer(runner.kv_block_zeroer, extra_planes)
                elif index_state is not None:
                    from vllm_iluvatar.models.deepseek_v41.indexer_kv import extend_kv_zeroer

                    extend_kv_zeroer(runner.kv_block_zeroer, config.num_blocks, extra_planes)
            return result

        initialize._ilu_compressed_kv_profiling = True
        # The upstream profiling teardown does not clear kv_block_zeroer.
        # Never leave its address table holding the temporary arena alive.
        with (
            _override_runner_attribute(runner, "initialize_kv_cache", initialize),
            _override_runner_attribute(runner, "kv_block_zeroer", None),
        ):
            failure = None
            try:
                return original_profile()
            except BaseException as error:
                failure = error
                if owners:
                    # Initialization can fail before the upstream profiler
                    # enters the try/finally that normally clears its cache.
                    try:
                        _teardown_profiling_state(runner)
                    except BaseException as cleanup_error:
                        if hasattr(error, "add_note"):
                            error.add_note(
                                f"Profiling teardown also failed: {cleanup_error!r}"
                            )
                        logger.exception("Failed to tear down profiling KV cache")
                raise
            finally:
                if owners:
                    import torch

                    device = runner.device
                    try:
                        try:
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                        finally:
                            runner.kv_block_zeroer = None
                            for owner in reversed(owners):
                                owner.release()
                            gc.collect()
                            if device.type == "cuda":
                                # Upstream empty_cache ran while we still held
                                # the sidecars/arena; release cached segments too.
                                with torch.cuda.device(device):
                                    torch.cuda.empty_cache()
                    except BaseException as cleanup_error:
                        if failure is None:
                            raise
                        if hasattr(failure, "add_note"):
                            failure.add_note(
                                f"Temporary compressed KV cleanup also failed: {cleanup_error!r}"
                            )
                        logger.exception(
                            "Failed to release temporary compressed KV storage"
                        )

    profile._ilu_compressed_kv_profiling = True
    with _override_runner_attribute(runner, "profile_cudagraph_memory", profile):
        yield
