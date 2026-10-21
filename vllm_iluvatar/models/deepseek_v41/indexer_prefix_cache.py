# SPDX-License-Identifier: Apache-2.0
"""Keep compact indexer pages from coarsening producer prefix-cache hits."""

from __future__ import annotations

import os
from functools import wraps

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.request import RequestStatus

from vllm_iluvatar.ilu_patch.dsv41_utils import is_ilu_dsv41_config
from vllm_iluvatar.models.deepseek_v41.compressed_kv import layer_specs
from vllm_iluvatar.models.deepseek_v41.indexer_kv import is_split_indexer_spec

logger = init_logger("vllm.ilu_indexer_prefix_cache")


def _defer_partial_publication(request, num_tokens) -> None:
    """Publish only after a completed producer forward, not during allocation."""


def _publish_completed_tails(manager, publishers) -> None:
    """Prevent a same-batch COW from copying a not-yet-produced partial page.

    The SDK registers full blocks optimistically before their forward. Full
    blocks can be shared because the layer inserts KV before attending, but a
    partial-hit COW copy runs before that insertion. Restrict the additional
    partial entries to successful completed producer requests. Abort/preempt
    paths and in-flight steps must never publish a new partial entry.
    """
    original = manager.free
    for single, _ in publishers:
        single._cache_partial_tail_block = _defer_partial_publication

    @wraps(original)
    def free(request):
        if (
            request.status
            in (
                RequestStatus.FINISHED_STOPPED,
                RequestStatus.FINISHED_LENGTH_CAPPED,
            )
            and request.num_in_flight_tokens == 0
        ):
            for _, publish in publishers:
                publish(request, request.num_computed_tokens)
        return original(request)

    manager.free = free


def configure_indexer_prefix_hits(scheduler) -> bool:
    """Enable the existing partial-hit/COW policy for compatible MLA producers.

    Physical pages and connector block geometry stay unchanged. Only the local
    lookup/publication boundary uses the already negotiated hash granularity.
    The shared tail is copied to an allocator-owned private block before any
    append, through the runner's existing paired payload/scale copy path.

    Decode/standalone, sharded context, speculative producers and cache groups
    with non-append-only state retain their original policies. Replayed SWA
    and compressor rings are scratch and never become shareable cache entries.
    """
    setting = os.getenv("VLLM_ILU_DS41_INDEXER_PREFIX_SUBPAGE", "1")
    if setting not in ("0", "1"):
        raise ValueError("VLLM_ILU_DS41_INDEXER_PREFIX_SUBPAGE must be 0 or 1")
    config = scheduler.vllm_config
    transfer = config.kv_transfer_config
    if (
        setting == "0"
        or not is_ilu_dsv41_config(config)
        or getattr(transfer, "kv_role", None) != "kv_producer"
        or config.speculative_config is not None
    ):
        return False

    manager = scheduler.kv_cache_manager
    coordinator = manager.coordinator
    if (
        not manager.enable_caching
        or not isinstance(coordinator, HybridKVCacheCoordinator)
        or coordinator.enable_partial_hash_hits
    ):
        return False
    specs = layer_specs(coordinator.kv_cache_config.kv_cache_groups)
    if not any(is_split_indexer_spec(spec) for spec in specs.values()):
        return False
    hash_size = coordinator.hash_block_size
    if hash_size <= 0 or hash_size >= coordinator.scheduler_block_size:
        return False
    for spec in specs.values():
        if spec.prefix_cacheable:
            if (
                type(spec) is not MLAAttentionSpec
                or not isinstance(spec.tokens_per_state, int)
                or spec.tokens_per_state <= 0
                or hash_size % spec.tokens_per_state
            ):
                return False
        elif isinstance(spec, SlidingWindowMLASpec) and spec.bounded_replay:
            # The existing SWA allocator retires whole blocks. A replay must
            # start/end at the same boundaries it already supports.
            if (
                hash_size % spec.block_size
                or spec.prefix_replay_tokens % spec.block_size
            ):
                return False
        elif not isinstance(spec, CircularBufferSpec):
            return False
    publishers = []
    for group, single in zip(
        coordinator.kv_cache_config.kv_cache_groups,
        coordinator.single_type_managers,
        strict=True,
    ):
        if single.dcp_world_size != 1 or single.pcp_world_size != 1:
            return False
        if group.kv_cache_spec.prefix_cacheable and (
            not single.supports_fine_grained_hash_lookup
            or single.block_size % hash_size
        ):
            return False
        if group.kv_cache_spec.prefix_cacheable:
            publish = getattr(single, "_cache_partial_tail_block", None)
            if not callable(publish):
                return False
            publishers.append((single, publish))

    _publish_completed_tails(manager, publishers)
    coordinator.enable_partial_hash_hits = True
    for single in coordinator.single_type_managers:
        single.cache_hit_alignment_tokens = hash_size
    logger.info(
        "Compact indexer producer prefix hits: hash_tokens=%d manager_tokens=%d "
        "replay_tokens=%d; existing paired COW enabled",
        hash_size,
        coordinator.scheduler_block_size,
        scheduler.prefix_replay_tokens,
    )
    return True
