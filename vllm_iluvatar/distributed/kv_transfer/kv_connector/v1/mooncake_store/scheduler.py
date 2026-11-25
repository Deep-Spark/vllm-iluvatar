# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side component for Iluvatar MooncakeStoreConnector."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec, UniformTypeKVCacheSpecs
from vllm.v1.outputs import KVConnectorOutput

from .config import parse_bool
from .coordinator import partial_hash_hits_enabled
from .layout import (
    LoadSpec,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreWorkerMetadata,
    ReqMeta,
    RequestTracker,
)
from .worker import LookupKeyClient

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _new_req_prefill_tokens(request: NewRequestData) -> list[int]:
    if request.prefill_token_ids is not None:
        return request.prefill_token_ids
    assert request.prompt_token_ids is not None
    return request.prompt_token_ids


def _unwrap_group_spec(spec: Any) -> Any:
    if isinstance(spec, UniformTypeKVCacheSpecs) and spec.kv_cache_specs:
        return next(iter(spec.kv_cache_specs.values()))
    return spec


class IluMooncakeStoreScheduler:
    """Scheduler-side component for cross-Prefill Mooncake DRAM KV offloading."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        assert vllm_config.kv_transfer_config is not None
        self.vllm_config = vllm_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}

        self.load_async = parse_bool(extra_config.get("load_async", True), default=True)
        self.lookup_async = parse_bool(
            extra_config.get("lookup_async", False), default=False
        )

        is_consumer = self.kv_role == "kv_consumer"
        default_lookup = (
            parse_bool(extra_config.get("consumer_is_to_load", False), default=False)
            if is_consumer
            else True
        )
        self.enable_lookup = parse_bool(
            extra_config.get("enable_lookup", default_lookup), default=default_lookup
        )
        self.consumer_is_to_put = parse_bool(
            extra_config.get("consumer_is_to_put", False), default=False
        )
        default_save_decode = not parse_bool(
            extra_config.get("offload_prompt_only", True), default=True
        )
        self.save_decode_cache = parse_bool(
            extra_config.get("save_decode_cache", default_save_decode),
            default=default_save_decode,
        )
        self.can_put = (
            parse_bool(extra_config.get("enable_store", True), default=True)
            if not is_consumer
            else (self.consumer_is_to_put or self.save_decode_cache)
        )

        kv_event_config = getattr(vllm_config, "kv_events_config", None)
        self.enable_kv_events = bool(
            kv_event_config and kv_event_config.enable_kv_cache_events
        )

        self.client: LookupKeyClient | None = (
            LookupKeyClient(vllm_config) if self.enable_lookup or self.can_put else None
        )
        self.kv_cache_config = kv_cache_config
        self._store_group_ids = kv_cache_config.prefix_cacheable_group_ids
        self._store_group_id_by_kv_cache_group_id = {
            group_id: store_group_id
            for store_group_id, group_id in enumerate(self._store_group_ids)
        }
        store_groups = kv_cache_config.prefix_cacheable_groups

        self._block_size, self._hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self.enable_partial_hash_hits = partial_hash_hits_enabled(
            store_groups,
            self._hash_block_size,
            getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1),
        )

        mamba_groups = {
            group_id: _unwrap_group_spec(group.kv_cache_spec)
            for group_id, group in enumerate(store_groups)
            if isinstance(_unwrap_group_spec(group.kv_cache_spec), MambaSpec)
        }
        assert all(
            spec.mamba_cache_mode == "align" for spec in mamba_groups.values()
        ), "IluMooncakeStoreScheduler requires mamba_cache_mode='align'"
        self._boundary_state_group_ids = frozenset(mamba_groups)

        self._gpu_block_pool: BlockPool | None = None
        self._num_workers = int(vllm_config.parallel_config.world_size)
        self._next_store_job_id = 0
        self._pinned_saves: dict[int, tuple[list[int], int]] = {}

        self.load_specs: dict[str, LoadSpec] = {}
        self._request_trackers: dict[str, RequestTracker] = {}
        self._unfinished_requests: dict[str, tuple[Request, tuple[list[int], ...]]] = {}
        self._unfinished_request_ids: set[str] = set()
        self._finished_partial_tail_metas: dict[str, ReqMeta] = {}

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        self._gpu_block_pool = gpu_block_pool

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def shutdown(self) -> None:
        self.close()

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
        *,
        max_num_new_tokens: int | None = None,
    ) -> tuple[int | None, bool]:
        """Query Mooncake DRAM pool for reusable prefix tokens."""
        self.load_specs.pop(request.request_id, None)
        if not self.enable_lookup or self.client is None:
            return 0, False
        if max_num_new_tokens is not None and max_num_new_tokens <= 0:
            return 0, False

        align = (
            self._hash_block_size if self.enable_partial_hash_hits else self._block_size
        )
        max_total_tokens = request.num_tokens
        if max_num_new_tokens is not None:
            max_total_tokens = min(
                max_total_tokens, num_computed_tokens + max_num_new_tokens + 1
            )
        if max_total_tokens < align:
            return 0, False

        lookup_result = self.client.lookup(
            request.request_id,
            max_total_tokens,
            request.block_hashes,
            non_block=self.lookup_async,
        )
        if lookup_result is None:
            return None, False

        num_external_hit_tokens = lookup_result.hit_length
        if num_external_hit_tokens >= request.num_tokens:
            num_external_hit_tokens = (request.num_tokens - 1) // align * align
        if max_num_new_tokens is not None:
            capped = (num_computed_tokens + max_num_new_tokens) // align * align
            num_external_hit_tokens = min(num_external_hit_tokens, capped)

        if num_external_hit_tokens <= num_computed_tokens:
            return 0, False

        need_to_allocate = num_external_hit_tokens - num_computed_tokens
        logger.debug(
            "Mooncake lookup req=%s total_tokens=%d local_cached=%d "
            "pool_hit=%d need_to_load=%d",
            request.request_id,
            request.num_tokens,
            num_computed_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=num_external_hit_tokens,
            can_load=False,
            tail_key_boundaries=lookup_result.tail_key_boundaries,
        )
        return need_to_allocate, self.load_async

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        local_block_ids: tuple[list[int], ...] = ()
        if num_external_tokens > 0:
            local_block_ids = blocks.get_block_ids(group_ids=self._store_group_ids)

        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        self._unfinished_request_ids.add(request.request_id)

        load_spec = self.load_specs.get(request.request_id)
        if load_spec is None:
            return

        if num_external_tokens == 0:
            load_spec.can_load = False
            return

        # Note: Scheduler.schedule may align ext_tokens down (e.g. SWA bounded
        # replay rounding to scheduler.block_size). Adjust kvpool_cached_tokens
        # to match the scheduler-allocated external prefix cleanly.
        expected_max = load_spec.kvpool_cached_tokens - load_spec.vllm_cached_tokens
        if num_external_tokens > expected_max:
            raise AssertionError(
                f"Allocated external tokens ({num_external_tokens}) exceed "
                f"matched Mooncake tokens ({expected_max}) for {request.request_id}"
            )
        if num_external_tokens < expected_max:
            load_spec.kvpool_cached_tokens = (
                load_spec.vllm_cached_tokens + num_external_tokens
            )
        load_spec.can_load = True

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        is_consumer = self.kv_role == "kv_consumer"
        skip_prefill_save = not self.can_put or (
            is_consumer and not self.consumer_is_to_put
        )
        can_process_cached = self.can_put

        for finished_req_id in scheduler_output.finished_req_ids:
            if self.client is not None:
                self.client.discard(finished_req_id)
            self.load_specs.pop(finished_req_id, None)
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._unfinished_request_ids.discard(finished_req_id)

        preempted_ids = set(scheduler_output.preempted_req_ids or ())
        for req_id in preempted_ids:
            self.load_specs.pop(req_id, None)
            if request_tracker := self._request_trackers.get(req_id):
                request_tracker.reset()
            self._unfinished_requests.pop(req_id, None)

        meta = MooncakeStoreConnectorMetadata(
            self._unfinished_request_ids,
            preempted_ids,
        )

        # 1. Newly scheduled (or MRV2 resumed) requests
        for request in scheduler_output.scheduled_new_reqs:
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            request_tuple = self._unfinished_requests.get(request.req_id)
            assert request_tuple is not None, (
                f"Request {request.req_id} missing from _unfinished_requests"
            )
            request_real = request_tuple[0]

            unfolded_block_ids = tuple(
                request.block_ids[group_id].copy() for group_id in self._store_group_ids
            )
            prefill_tokens = _new_req_prefill_tokens(request)
            initial_saved = (
                load_spec.kvpool_cached_tokens
                if (load_spec is not None and load_spec.can_load)
                else 0
            )
            request_tracker = RequestTracker(
                req_id=request.req_id,
                token_len=num_tokens_to_compute,
                allocated_block_ids=unfolded_block_ids,
                num_saved_tokens=initial_saved,
                token_ids=(
                    prefill_tokens[:num_tokens_to_compute]
                    if self.enable_kv_events
                    else None
                ),
                prefill_end_tokens=len(prefill_tokens),
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                load_spec=load_spec,
                skip_save=skip_prefill_save,
                block_hashes=request_real.block_hashes,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        # 2. Cached (chunked prefill / decode / MRV1 resumed) requests
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if can_process_cached:
            resumed_ids = getattr(cached_reqs, "resumed_req_ids", set()) or set()
            for i, req_id in enumerate(cached_reqs.req_ids):
                raw_new_blocks = cached_reqs.new_block_ids[i]
                new_block_ids: tuple[list[int], ...] | None = None
                if raw_new_blocks:
                    new_block_ids = tuple(
                        raw_new_blocks[group_id] for group_id in self._store_group_ids
                    )

                req_meta = None
                if req_id in resumed_ids:
                    if not new_block_ids:
                        continue
                    copied_blocks = tuple(b.copy() for b in new_block_ids)
                    load_spec = self.load_specs.pop(req_id, None)
                    request_tuple = self._unfinished_requests.get(req_id)
                    assert request_tuple is not None
                    request_real = request_tuple[0]
                    num_tokens_to_compute = (
                        request_real.num_computed_tokens
                        + scheduler_output.num_scheduled_tokens[req_id]
                    )
                    prefill_tokens = list(request_real.all_token_ids)
                    request_tracker = RequestTracker(
                        req_id=req_id,
                        token_len=num_tokens_to_compute,
                        allocated_block_ids=copied_blocks,
                        num_saved_tokens=0,
                        token_ids=(
                            prefill_tokens[:num_tokens_to_compute]
                            if self.enable_kv_events
                            else None
                        ),
                        prefill_end_tokens=len(prefill_tokens),
                    )
                    self._request_trackers[req_id] = request_tracker
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=load_spec,
                        skip_save=skip_prefill_save,
                        block_hashes=request_real.block_hashes,
                    )
                else:
                    request_tracker = self._request_trackers.get(req_id)
                    if request_tracker is None:
                        continue
                    num_computed_token = cached_reqs.num_computed_tokens[i]
                    prefill_end = request_tracker.prefill_end_tokens
                    is_decode = num_computed_token >= prefill_end
                    if is_decode and not self.save_decode_cache:
                        continue

                    num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                    req_tuple = self._unfinished_requests.get(req_id)
                    if req_tuple is None:
                        raise ValueError(
                            f"Request {req_id} is not in _unfinished_requests"
                        )
                    unfinished_req = req_tuple[0]
                    num_current_tokens = request_tracker.token_len
                    new_token_ids = unfinished_req.all_token_ids[
                        num_current_tokens : num_current_tokens + num_new_tokens
                    ]
                    request_tracker.token_len += len(new_token_ids)
                    if request_tracker.token_ids is not None:
                        request_tracker.token_ids.extend(new_token_ids)
                    if new_block_ids:
                        request_tracker.update(new_block_ids)
                    if is_consumer and not is_decode and not self.consumer_is_to_put:
                        continue

                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=None,
                        skip_save=False,
                        block_hashes=unfinished_req.block_hashes,
                    )

                if req_meta is not None:
                    meta.add_request(req_meta)

        # 3. Unscheduled async-load requests waiting for KV transfer before prefill
        scheduled_new_ids = {req.req_id for req in scheduler_output.scheduled_new_reqs}
        cached_id_set = set(cached_reqs.req_ids)
        for request_id, (unfinished_req, block_ids) in self._unfinished_requests.items():
            if request_id not in scheduled_new_ids and request_id not in cached_id_set:
                load_spec = self.load_specs.pop(request_id, None)
                if load_spec is None or not load_spec.can_load:
                    continue
                num_tokens_to_compute = load_spec.kvpool_cached_tokens
                request_tracker = RequestTracker(
                    req_id=request_id,
                    token_len=num_tokens_to_compute,
                    allocated_block_ids=block_ids,
                    num_saved_tokens=num_tokens_to_compute,
                    prefill_end_tokens=len(unfinished_req.prompt_token_ids),
                )
                self._request_trackers[request_id] = request_tracker
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=load_spec,
                    skip_save=True,
                    block_hashes=unfinished_req.block_hashes,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)

        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        if (
            block_state is not None
            and getattr(block_state, "boundary_state_offloads", None)
            and self.can_put
        ):
            self._handle_boundary_state_offloads(
                block_state.boundary_state_offloads, meta
            )

        self._apply_current_save_block_ids(meta, scheduler_output)

        for req_meta in self._finished_partial_tail_metas.values():
            meta.add_request(req_meta)
        self._finished_partial_tail_metas.clear()

        self._reference_save_blocks(meta)
        return meta

    def _apply_current_save_block_ids(
        self,
        meta: MooncakeStoreConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        save_metas = [req_meta for req_meta in meta.requests if req_meta.can_save]
        if not save_metas:
            return
        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        if block_state is None:
            return
        for req_meta in save_metas:
            block_ids = block_state.get_block_ids(req_meta.req_id)
            if block_ids is not None:
                req_meta.block_ids = tuple(
                    block_ids[group_id] for group_id in self._store_group_ids
                )

    def _reference_save_blocks(self, meta: MooncakeStoreConnectorMetadata) -> None:
        pool = self._gpu_block_pool
        for req_meta in meta.requests:
            if not req_meta.can_save:
                continue
            if req_meta.store_job_id is not None:
                continue
            req_meta.store_job_id = store_job_id = self._next_store_job_id
            self._next_store_job_id += 1

            if pool is None:
                continue

            block_ids: list[int] = []
            if req_meta.boundary_state_offloads:
                block_ids.extend(
                    block_id
                    for _, block_id, _ in req_meta.boundary_state_offloads
                    if block_id != NULL_BLOCK_ID
                )
            block_ids.extend(
                block_id
                for group_id, group in enumerate(req_meta.block_ids)
                if group_id not in self._boundary_state_group_ids
                for block_id in group
                if block_id != NULL_BLOCK_ID
            )
            block_ids = list(dict.fromkeys(block_ids))
            if not block_ids:
                continue
            self._pinned_saves[store_job_id] = (block_ids, self._num_workers)
            pool.touch([pool.blocks[block_id] for block_id in block_ids])

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        if not self.can_put or not partial_tail_offloads:
            return False
        tracker = self._request_trackers.get(request.request_id)
        if tracker is None or not any(block_ids):
            return False
        boundaries = {boundary for _, _, boundary in partial_tail_offloads}
        if len(boundaries) != 1:
            raise ValueError(
                "Partial-tail offloads for one request must share a boundary"
            )
        boundary_tokens = next(iter(boundaries))
        if boundary_tokens > tracker.prefill_end_tokens:
            return False

        pinned_block_ids: list[int] = []
        remapped_offloads: list[tuple[int, int, int]] = []
        for group_id, block_id, boundary in partial_tail_offloads:
            store_group_id = self._store_group_id_by_kv_cache_group_id.get(group_id)
            if store_group_id not in self._boundary_state_group_ids:
                return False
            if block_id == NULL_BLOCK_ID:
                return False
            pinned_block_ids.append(block_id)
            remapped_offloads.append((store_group_id, block_id, boundary))
        pinned_block_ids = list(dict.fromkeys(pinned_block_ids))

        store_job_id = self._next_store_job_id
        self._next_store_job_id += 1
        pool = self._gpu_block_pool
        if pool is not None and pinned_block_ids:
            self._pinned_saves[store_job_id] = (pinned_block_ids, self._num_workers)
            pool.touch([pool.blocks[block_id] for block_id in pinned_block_ids])

        self._finished_partial_tail_metas[request.request_id] = ReqMeta(
            req_id=request.request_id,
            token_len_chunk=0,
            block_ids=tuple(
                block_ids[group_id].copy() for group_id in self._store_group_ids
            ),
            block_hashes=list(request.block_hashes),
            can_save=True,
            num_prompt_tokens=tracker.prefill_end_tokens,
            store_job_id=store_job_id,
            boundary_state_offloads=remapped_offloads,
        )
        tracker.has_pending_offload = True
        return False

    def _handle_boundary_state_offloads(
        self,
        offloads: dict[str, list[tuple[int, int, int]]],
        meta: MooncakeStoreConnectorMetadata,
    ) -> None:
        save_metas = {m.req_id: m for m in meta.requests if m.can_save}
        for req_id, entries in offloads.items():
            tracker = self._request_trackers.get(req_id)
            req_tuple = self._unfinished_requests.get(req_id)
            if tracker is None or req_tuple is None:
                continue
            accepted: list[tuple[int, int, int]] = []
            for group_id, block_id, boundary_tokens in entries:
                if boundary_tokens > tracker.prefill_end_tokens:
                    continue
                if block_id == NULL_BLOCK_ID:
                    continue
                store_group_id = self._store_group_id_by_kv_cache_group_id.get(group_id)
                if store_group_id not in self._boundary_state_group_ids:
                    continue
                accepted.append((store_group_id, block_id, boundary_tokens))
            if not accepted:
                continue
            tracker.has_pending_offload = True
            if (req_meta := save_metas.get(req_id)) is not None:
                req_meta.boundary_state_offloads = accepted
                continue
            meta.add_request(
                ReqMeta(
                    req_id=req_id,
                    token_len_chunk=0,
                    block_ids=tracker.allocated_block_ids,
                    block_hashes=list(req_tuple[0].block_hashes),
                    can_save=True,
                    num_prompt_tokens=tracker.prefill_end_tokens,
                    boundary_state_offloads=accepted,
                )
            )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, MooncakeStoreWorkerMetadata):
            return
        pool = self._gpu_block_pool
        if pool is None:
            return
        for store_job_id, count in meta.completed_saves.items():
            pinned = self._pinned_saves.get(store_job_id)
            if pinned is None:
                continue
            block_ids, remaining = pinned
            remaining -= count
            if remaining > 0:
                self._pinned_saves[store_job_id] = (block_ids, remaining)
                continue
            assert remaining == 0, (
                f"store job {store_job_id} reported by too many ranks"
            )
            del self._pinned_saves[store_job_id]
            pool.free_blocks(pool.blocks[bid] for bid in reversed(block_ids))

    def has_pending_push_work(self) -> bool:
        return bool(self._pinned_saves)

    def reset_store(self) -> bool:
        if self._pinned_saves:
            logger.error(
                "Mooncake store reset refused while %d save job(s) hold pinned blocks",
                len(self._pinned_saves),
            )
            return False
        if self.client is None:
            self.client = LookupKeyClient(self.vllm_config)
        try:
            ok = self.client.reset()
            if ok:
                logger.info("Mooncake store reset via remove_all succeeded.")
            else:
                logger.warning("Mooncake store reset returned NACK from worker.")
            return ok
        except Exception as e:
            logger.error("Mooncake reset_store RPC failed: %s", e)
            return False
