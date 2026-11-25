# SPDX-License-Identifier: Apache-2.0
"""Async send/receive threads for Iluvatar MooncakeStoreConnector."""

from __future__ import annotations

import inspect
import queue
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import torch
import vllm.envs as envs
from vllm.distributed.kv_events import BlockStored
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash

from .coordinator import MooncakeStoreCoordinator
from .layout import (
    ChunkedTokenDatabase,
    KeyMetadata,
    ReqMeta,
    StoreShardId,
)

logger = init_logger(__name__)

MOONCAKE_NO_AVAILABLE_HANDLE = -200
DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES = 1280 * 1024 * 1024
_DIRECT_IO_ALIGNMENT = 4096
_DIRECT_IO_PADDING_BYTES = 2 * _DIRECT_IO_ALIGNMENT
_STOP_SENTINEL = object()
_T = TypeVar("_T")


def _rotate_list(values: list[_T], offset: int) -> list[_T]:
    if not values or offset == 0:
        return values
    return values[offset:] + values[:offset]


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _estimate_disk_offload_staging_bytes(size_list: list[int]) -> int:
    return _align_up(sum(size_list), _DIRECT_IO_ALIGNMENT) + _DIRECT_IO_PADDING_BYTES


def _sum_batch_bytes(sizes: list[list[int]]) -> int:
    return sum(sum(size) for size in sizes)


def _get_usable_disk_offload_buffer_budget_bytes(raw_budget_bytes: int) -> int:
    ratio = float(getattr(envs, "VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO", 0.75))
    return max(1, int(raw_budget_bytes * ratio))


def _split_disk_offload_load_batches(
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
    usable_budget_bytes: int,
    raw_budget_bytes: int,
) -> tuple[list[tuple[list[str], list[list[int]], list[list[int]]]], str | None]:
    batches: list[tuple[list[str], list[list[int]], list[list[int]]]] = []
    batch_keys: list[str] = []
    batch_addrs: list[list[int]] = []
    batch_sizes: list[list[int]] = []
    batch_bytes = 0

    for key, addr, size in zip(keys, addrs, sizes, strict=True):
        key_bytes = _estimate_disk_offload_staging_bytes(size)
        if key_bytes > raw_budget_bytes:
            return [], key
        if key_bytes > usable_budget_bytes:
            if batch_keys:
                batches.append((batch_keys, batch_addrs, batch_sizes))
                batch_keys, batch_addrs, batch_sizes = [], [], []
                batch_bytes = 0
            batches.append(([key], [addr], [size]))
            continue
        if batch_keys and batch_bytes + key_bytes > usable_budget_bytes:
            batches.append((batch_keys, batch_addrs, batch_sizes))
            batch_keys, batch_addrs, batch_sizes = [], [], []
            batch_bytes = 0
        batch_keys.append(key)
        batch_addrs.append(addr)
        batch_sizes.append(size)
        batch_bytes += key_bytes

    if batch_keys:
        batches.append((batch_keys, batch_addrs, batch_sizes))
    return batches, None


def _make_mooncake_group_id(metadata: KeyMetadata, chunk_hash: str) -> str:
    prefix = f"{metadata.cache_prefix}@" if metadata.cache_prefix else ""
    return (
        f"vllm-mooncake-store:{prefix}{metadata.model_name}"
        f"{metadata.store_namespace}@{chunk_hash}"
    )


def _store_get_accepts_prefer_same_node(store: Any) -> bool:
    fn = getattr(store, "batch_get_into_multi_buffers", None)
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if any(
            p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in params
        ):
            return True
        return len(params) >= 4
    except (TypeError, ValueError):
        doc = getattr(fn, "__doc__", "") or ""
        return "prefer_alloc_in_same_node" in doc


class KVTransferThread(threading.Thread):
    """Base worker thread with CoreX device-context binding and clean stop."""

    def __init__(
        self,
        store: Any,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        name: str,
        *,
        device_index: int | None = None,
        record_operation: Callable[..., None] | None = None,
        request_queue: queue.Queue[Any] | None = None,
    ) -> None:
        super().__init__(daemon=True, name=name)
        self.store = store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.token_databases = token_databases
        self.device_index = device_index
        self._record_operation_cb = record_operation
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = request_queue or queue.Queue()
        self.finished_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []
        self._stopping = threading.Event()

    def add_request(self, request: ReqMeta) -> None:
        if self._stopping.is_set():
            return
        self.request_queue.put(request)

    def enqueue_stop(self) -> None:
        self._stopping.set()
        self.request_queue.put(_STOP_SENTINEL)

    def get_and_clear_finished_requests(self) -> set[str]:
        with self.done_task_lock:
            finished = self.finished_requests.copy()
            self.finished_requests.clear()
        return finished

    def set_finished_request(self, req_id: str) -> None:
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def run(self) -> None:
        if self.device_index is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_device(self.device_index)
            except Exception as exc:
                logger.warning(
                    "%s could not bind CUDA device %d: %s",
                    self.name,
                    self.device_index,
                    exc,
                )
        self.ready_event.set()
        while True:
            request_data = self.request_queue.get()
            if request_data is _STOP_SENTINEL:
                self.request_queue.task_done()
                break
            if request_data is None:
                self.request_queue.task_done()
                continue
            try:
                self._handle_request(request_data)
            except Exception:
                req_id = getattr(request_data, "req_id", "<unknown>")
                logger.exception("Unhandled error in %s (req=%s)", self.name, req_id)

    def _handle_request(self, req_meta: Any) -> None:
        raise NotImplementedError

    def _record_operation(
        self,
        operation: str,
        start_time: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        if self._record_operation_cb is None:
            return
        self._record_operation_cb(
            operation=operation,
            duration_seconds=time.perf_counter() - start_time,
            num_keys=num_keys,
            num_bytes=num_bytes,
            status=status,
            num_failed_keys=num_failed_keys,
        )

    def update_kv_event(self, events: list[BlockStored]) -> None:
        with self.kv_event_lock:
            self.kv_events.extend(events)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events


class KVCacheStoreSendingThread(KVTransferThread):
    """Background thread storing KV cache blocks into Mooncake DRAM pool."""

    def __init__(
        self,
        store: Any,
        coord: MooncakeStoreCoordinator,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        group_put_steps: Sequence[int],
        kv_role: str,
        ready_event: threading.Event,
        *,
        device_index: int | None = None,
        enable_kv_event: bool = False,
        replicate_config: Any = None,
        enable_group_semantics: bool = False,
        supports_group_ids: bool = False,
        record_operation: Callable[..., None] | None = None,
        group_participates: Sequence[bool] | None = None,
    ) -> None:
        super().__init__(
            store,
            token_databases,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreSendingThread",
            device_index=device_index,
            record_operation=record_operation,
        )
        self.group_put_steps = list(group_put_steps)
        self.coord = coord
        self.kv_role = kv_role
        self.group_participates = (
            list(group_participates)
            if group_participates is not None
            else [True] * len(token_databases)
        )
        self.stored_requests: dict[str, set[int]] = {}
        self._completed_saves: dict[int, int] = {}
        self.enable_kv_event = enable_kv_event
        self.replicate_config = replicate_config
        self.enable_group_semantics = enable_group_semantics
        self.supports_group_ids = supports_group_ids

        self._store_pressure_active = False
        self._skip_store_requests: set[str] = set()
        self._saved_offset: dict[str, int] = {}
        self._retry_token_ids: dict[str, tuple[int, list[int]]] = {}

    def add_request(self, request: ReqMeta) -> None:
        assert request.store_job_id is not None
        if self._stopping.is_set():
            return
        with self.done_task_lock:
            self.stored_requests.setdefault(request.req_id, set()).add(
                request.store_job_id
            )
        super().add_request(request)

    def is_live_store_job(self, req_meta: ReqMeta) -> bool:
        with self.done_task_lock:
            return req_meta.store_job_id in self.stored_requests.get(
                req_meta.req_id, ()
            )

    def delete_finished_stored_request(self, req_id: str) -> None:
        with self.done_task_lock:
            self.stored_requests.pop(req_id, None)
            self._skip_store_requests.discard(req_id)
            self._saved_offset.pop(req_id, None)
            self._retry_token_ids.pop(req_id, None)

    def finish_store_job(self, req_meta: ReqMeta) -> None:
        store_job_id = req_meta.store_job_id
        assert store_job_id is not None
        with self.done_task_lock:
            live = self.stored_requests.get(req_meta.req_id)
            if live is not None:
                live.discard(store_job_id)
            self._completed_saves[store_job_id] = (
                self._completed_saves.get(store_job_id, 0) + 1
            )

    def take_completed_saves(self) -> dict[int, int]:
        with self.done_task_lock:
            completed = self._completed_saves
            self._completed_saves = {}
        return completed

    def _record_saved(self, req_meta: ReqMeta, token_len: int) -> None:
        with self.done_task_lock:
            if req_meta.store_job_id in self.stored_requests.get(req_meta.req_id, ()):
                self._saved_offset[req_meta.req_id] = token_len

    def _get_retry_token_ids(self, req_meta: ReqMeta) -> tuple[int, list[int]] | None:
        with self.done_task_lock:
            if req_meta.store_job_id not in self.stored_requests.get(
                req_meta.req_id, ()
            ):
                return None
            return self._retry_token_ids.get(req_meta.req_id)

    def _update_retry_token_ids(
        self,
        req_meta: ReqMeta,
        save_completed: bool,
        token_ids_start: int,
        event_token_ids: list[int] | None,
    ) -> None:
        with self.done_task_lock:
            if req_meta.store_job_id not in self.stored_requests.get(
                req_meta.req_id, ()
            ):
                return
            if save_completed:
                self._retry_token_ids.pop(req_meta.req_id, None)
            elif event_token_ids is not None:
                self._retry_token_ids[req_meta.req_id] = (
                    token_ids_start,
                    event_token_ids,
                )

    def _should_skip_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            return self._store_pressure_active and req_id in self._skip_store_requests

    def _mark_request_skipped_for_pressure(self, req_meta: ReqMeta) -> bool:
        req_id = req_meta.req_id
        with self.done_task_lock:
            already_skipped = req_id in self._skip_store_requests
            self._store_pressure_active = True
            if req_meta.store_job_id in self.stored_requests.get(req_id, ()):
                self._skip_store_requests.add(req_id)
        return already_skipped

    def _clear_store_pressure(self) -> bool:
        with self.done_task_lock:
            if not self._store_pressure_active and not self._skip_store_requests:
                return False
            self._store_pressure_active = False
            self._skip_store_requests.clear()
        return True

    def _boundary_snapshot_puts(
        self, req_meta: ReqMeta, entries: list[tuple[int, int, int]]
    ) -> list[tuple[str, list[int], list[int], KeyMetadata]]:
        hash_block_size = self.coord.hash_block_size
        puts: list[tuple[str, list[int], list[int], KeyMetadata]] = []
        for group_id, block_id, boundary in entries:
            if not self.group_participates[group_id]:
                continue
            if boundary == 0 or block_id == NULL_BLOCK_ID:
                continue
            hash_idx = boundary // hash_block_size - 1
            if hash_idx >= len(req_meta.block_hashes):
                continue
            db = self.token_databases[group_id]
            put_step = self.group_put_steps[group_id]
            put_step_rank = (self.tp_rank + group_id) % put_step
            if (boundary // db.block_size - 1) % put_step != put_step_rank:
                continue
            addr, size = db.prepare_value_for_block(block_id)
            if addr:
                puts.append(
                    (
                        db.key_for(req_meta.block_hashes[hash_idx]),
                        addr,
                        size,
                        db.metadata,
                    )
                )
        return puts

    def _sub_block_tail_puts(
        self, req_meta: ReqMeta, entries: list[tuple[int, int, int]]
    ) -> list[tuple[str, list[int], list[int], KeyMetadata]]:
        boundaries = {boundary for _, _, boundary in entries}
        if len(boundaries) != 1:
            raise ValueError(
                "Sub-block partial-tail offloads for one request must share a boundary"
            )
        boundary = boundaries.pop()
        hash_block_size = self.coord.hash_block_size
        if boundary == 0 or boundary // hash_block_size - 1 >= len(
            req_meta.block_hashes
        ):
            return []

        mamba_offloads = {group_id: block_id for group_id, block_id, _ in entries}
        saved = self._saved_offset.get(req_meta.req_id, 0)
        puts: list[tuple[str, list[int], list[int], KeyMetadata]] = []
        for g_idx, db in enumerate(self.token_databases):
            if not self.group_participates[g_idx]:
                continue
            group_blocks = req_meta.block_ids[g_idx]
            put_step = self.group_put_steps[g_idx]
            put_step_rank = (self.tp_rank + g_idx) % put_step
            last_block = cdiv(boundary, db.block_size) - 1
            for block_idx in range(
                min(saved // db.block_size, last_block), last_block + 1
            ):
                if block_idx % put_step != put_step_rank:
                    continue
                valid_end = min((block_idx + 1) * db.block_size, boundary)
                key_hash = req_meta.block_hashes[valid_end // hash_block_size - 1]
                if g_idx in mamba_offloads:
                    if valid_end != boundary:
                        continue
                    block_id = mamba_offloads[g_idx]
                elif g_idx in self.coord.mamba_group_ids:
                    continue
                elif block_idx < len(group_blocks):
                    block_id = group_blocks[block_idx]
                else:
                    continue
                if block_id == NULL_BLOCK_ID:
                    continue
                addr, size = db.prepare_value_for_block(block_id)
                if addr:
                    puts.append((db.key_for(key_hash), addr, size, db.metadata))
        return puts

    def _maybe_offload_boundary_states(self, req_meta: ReqMeta) -> bool:
        offloads = req_meta.boundary_state_offloads
        if not offloads or not req_meta.block_hashes:
            return True

        snapshots: list[tuple[int, int, int]] = []
        sub_block: list[tuple[int, int, int]] = []
        for group_id, block_id, boundary in offloads:
            entry = (group_id, block_id, boundary)
            if boundary % self.token_databases[group_id].block_size == 0:
                snapshots.append(entry)
            else:
                sub_block.append(entry)

        puts = self._boundary_snapshot_puts(req_meta, snapshots)
        if sub_block and self.coord.enable_partial_hash_hits:
            puts.extend(self._sub_block_tail_puts(req_meta, sub_block))
        if not puts:
            return True

        keys = [key for key, _, _, _ in puts]
        addrs = [addr for _, addr, _, _ in puts]
        sizes = [size for _, _, size, _ in puts]
        group_ids: list[str] | None = (
            [
                _make_mooncake_group_id(metadata, key.rsplit("@", 1)[-1])
                for key, _, _, metadata in puts
            ]
            if self.enable_group_semantics and self.supports_group_ids
            else None
        )

        exists_start = time.perf_counter()
        try:
            exists = self.store.batch_is_exist(keys)
        except Exception as exc:
            self._record_operation(
                "save_exists",
                exists_start,
                len(keys),
                status="error",
                num_failed_keys=len(keys),
            )
            logger.error(
                "Failed to check boundary-state keys for %s: %s",
                req_meta.req_id,
                exc,
            )
            return False
        self._record_operation("save_exists", exists_start, len(keys))

        missing = [i for i, e in enumerate(exists) if e != 1]
        if not missing:
            return True
        keys = [keys[i] for i in missing]
        addrs = [addrs[i] for i in missing]
        sizes = [sizes[i] for i in missing]
        if group_ids is not None:
            group_ids = [group_ids[i] for i in missing]
        if req_meta.current_event is not None:
            req_meta.current_event.synchronize()
        if (
            self.enable_group_semantics
            and self.supports_group_ids
            and self.replicate_config is not None
        ):
            self.replicate_config.group_ids = group_ids

        batch_bytes = _sum_batch_bytes(sizes)
        put_start = time.perf_counter()
        try:
            res = self.store.batch_put_from_multi_buffers(
                keys, addrs, sizes, self.replicate_config
            )
        except Exception as exc:
            self._record_operation(
                "save_put",
                put_start,
                len(keys),
                num_bytes=batch_bytes,
                status="error",
                num_failed_keys=len(keys),
            )
            logger.error(
                "Failed to put boundary-state keys for %s: %s",
                req_meta.req_id,
                exc,
            )
            return False

        failed = [i for i, val in enumerate(res) if val < 0]
        self._record_operation(
            "save_put",
            put_start,
            len(keys),
            num_bytes=batch_bytes,
            status="partial_failure" if failed else "ok",
            num_failed_keys=len(failed),
        )
        if failed:
            failed_codes = {res[i] for i in failed}
            if MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes:
                self._mark_request_skipped_for_pressure(req_meta)
            return False
        self._clear_store_pressure()
        return True

    def _handle_request(self, req_meta: ReqMeta) -> None:
        save_completed = False
        token_len = 0
        req_id = req_meta.req_id
        event_token_ids = req_meta.token_ids
        token_ids_start = req_meta.token_ids_start
        try:
            lcm_block_size = self.coord.lcm_block_size
            token_len = req_meta.token_len_chunk // lcm_block_size * lcm_block_size
            block_ids_per_group = req_meta.block_ids
            current_event = req_meta.current_event

            if not self.is_live_store_job(req_meta):
                return

            if self.enable_kv_event:
                retry_token_ids = self._get_retry_token_ids(req_meta)
                if retry_token_ids is not None and event_token_ids is not None:
                    retry_start, retry_ids = retry_token_ids
                    if retry_start + len(retry_ids) == token_ids_start:
                        event_token_ids = retry_ids + event_token_ids
                        token_ids_start = retry_start

            if self._should_skip_request(req_id):
                return

            if req_meta.boundary_state_offloads is not None and not (
                self._maybe_offload_boundary_states(req_meta)
            ):
                return

            if token_len == 0:
                return

            save_start = self._saved_offset.get(req_id, 0)
            store_masks = self.coord.store_mask(
                token_len,
                save_start,
                num_prompt_tokens=req_meta.num_prompt_tokens,
            )

            starts: list[int] = []
            ends: list[int] = []
            keys: list[str] = []
            event_specs: list[tuple[int, int, int, BlockHash]] | None = (
                [] if self.enable_kv_event else None
            )
            group_indices: list[int] = []
            store_shard_ids: list[StoreShardId] = []
            for g_idx, db in enumerate(self.token_databases):
                if not self.group_participates[g_idx]:
                    continue
                put_step = self.group_put_steps[g_idx]
                put_step_rank = (self.tp_rank + g_idx) % put_step
                group_blocks = block_ids_per_group[g_idx]
                for start, end, block_hash in db.process_tokens(
                    token_len,
                    req_meta.block_hashes,
                    mask_num=save_start,
                    chunk_mask=store_masks[g_idx],
                    put_step=put_step,
                    put_step_rank=put_step_rank,
                ):
                    block_idx = start // db.block_size
                    if (
                        block_idx >= len(group_blocks)
                        or group_blocks[block_idx] == NULL_BLOCK_ID
                    ):
                        continue
                    for store_shard_id in db.store_layout.local_shard_ids:
                        starts.append(start)
                        ends.append(end)
                        keys.append(db.store_layout.key_for(store_shard_id, block_hash))
                        group_indices.append(g_idx)
                        store_shard_ids.append(store_shard_id)
                        if event_specs is not None:
                            event_specs.append((start, end, g_idx, block_hash))

            if not keys:
                self._record_saved(req_meta, token_len)
                save_completed = True
                return

            save_exists_start = time.perf_counter()
            try:
                exists_states = self.store.batch_is_exist(keys)
            except Exception:
                self._record_operation(
                    "save_exists",
                    save_exists_start,
                    len(keys),
                    status="error",
                    num_failed_keys=len(keys),
                )
                raise
            self._record_operation("save_exists", save_exists_start, len(keys))

            missing_indices = [
                i for i, exists in enumerate(exists_states) if exists != 1
            ]
            if not missing_indices:
                self._record_saved(req_meta, token_len)
                save_completed = True
                return

            if len(missing_indices) != len(keys):
                starts = [starts[i] for i in missing_indices]
                ends = [ends[i] for i in missing_indices]
                keys = [keys[i] for i in missing_indices]
                if event_specs is not None:
                    event_specs = [event_specs[i] for i in missing_indices]
                group_indices = [group_indices[i] for i in missing_indices]
                store_shard_ids = [store_shard_ids[i] for i in missing_indices]

            group_ids = (
                [
                    _make_mooncake_group_id(
                        self.token_databases[g_idx].metadata,
                        key.rsplit("@", 1)[-1],
                    )
                    for key, g_idx in zip(keys, group_indices, strict=True)
                ]
                if self.enable_group_semantics and self.supports_group_ids
                else None
            )

            # Group keys are iterated in ascending g_idx order above, so
            # group-ordered prepare_values stays 1:1 aligned with `keys`.
            addrs: list[list[int]] = []
            sizes: list[list[int]] = []
            chunks_per_group: list[list[tuple[int, int]]] = [
                [] for _ in self.token_databases
            ]
            shards_per_group: list[list[StoreShardId]] = [
                [] for _ in self.token_databases
            ]
            for start, end, g_idx, shard_id in zip(
                starts, ends, group_indices, store_shard_ids, strict=True
            ):
                chunks_per_group[g_idx].append((start, end))
                shards_per_group[g_idx].append(shard_id)
            for g_idx, chunks in enumerate(chunks_per_group):
                if not chunks:
                    continue
                db = self.token_databases[g_idx]
                group_addrs, group_sizes, _ = db.store_layout.prepare_values(
                    chunks,
                    block_ids_per_group[g_idx],
                    shards_per_group[g_idx],
                )
                addrs.extend(group_addrs)
                sizes.extend(group_sizes)

            if current_event is not None:
                current_event.synchronize()
            if (
                self.enable_group_semantics
                and self.supports_group_ids
                and self.replicate_config is not None
            ):
                self.replicate_config.group_ids = group_ids

            failed_indices: set[int] = set()
            put_had_exception = False
            batch_bytes = _sum_batch_bytes(sizes)
            put_start = time.perf_counter()
            try:
                res = self.store.batch_put_from_multi_buffers(
                    keys,
                    addrs,
                    sizes,
                    self.replicate_config,
                )
            except Exception as exc:
                self._record_operation(
                    "save_put",
                    put_start,
                    len(keys),
                    num_bytes=batch_bytes,
                    status="error",
                    num_failed_keys=len(keys),
                )
                logger.error("Mooncake batch_put failed for %s: %s", req_id, exc)
                put_had_exception = True
            else:
                failed_indices = {i for i, val in enumerate(res) if val < 0}
                self._record_operation(
                    "save_put",
                    put_start,
                    len(keys),
                    num_bytes=batch_bytes,
                    status="partial_failure" if failed_indices else "ok",
                    num_failed_keys=len(failed_indices),
                )
                failed_codes = {res[i] for i in failed_indices}
                if failed_indices:
                    logger.warning(
                        "Mooncake batch_put failed: %d/%d keys failed "
                        "(codes=%s, bytes=%d, req=%s, first_key=%s)",
                        len(failed_indices),
                        len(keys),
                        failed_codes,
                        batch_bytes,
                        req_id,
                        keys[0],
                    )
                if (
                    MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes
                    and not self._mark_request_skipped_for_pressure(req_meta)
                ):
                    logger.warning(
                        "Mooncake DRAM/SSD pressure (NO_AVAILABLE_HANDLE); "
                        "skipping further puts for req=%s until pressure clears",
                        req_id,
                    )

            if not put_had_exception and not failed_indices:
                self._record_saved(req_meta, token_len)
                save_completed = True
                self._clear_store_pressure()

            if self.enable_kv_event and not put_had_exception:
                assert event_specs is not None
                indices_by_event: dict[tuple[int, int, int, BlockHash], list[int]] = {}
                for idx, ev_spec in enumerate(event_specs):
                    indices_by_event.setdefault(ev_spec, []).append(idx)
                token_ids_end = token_ids_start + len(event_token_ids or ())
                stored_events: list[BlockStored] = []
                for (
                    s,
                    end,
                    g_idx,
                    block_hash,
                ), ev_indices in indices_by_event.items():
                    if any(i in failed_indices for i in ev_indices):
                        continue
                    db = self.token_databases[g_idx]
                    tok_slice = (
                        event_token_ids[s - token_ids_start : end - token_ids_start]
                        if event_token_ids is not None
                        and token_ids_start <= s
                        and end <= token_ids_end
                        else []
                    )
                    stored_events.append(
                        BlockStored(
                            block_hashes=[maybe_convert_block_hash(block_hash)],
                            parent_block_hash=(
                                maybe_convert_block_hash(
                                    req_meta.block_hashes[s // db.hash_block_size - 1]
                                )
                                if s > 0
                                else None
                            ),
                            token_ids=tok_slice,
                            block_size=db.block_size,
                            lora_id=None,
                            medium="cpu",
                            lora_name=None,
                            group_idx=g_idx,
                        )
                    )
                if stored_events:
                    self.update_kv_event(stored_events)
        finally:
            if self.enable_kv_event and token_len:
                self._update_retry_token_ids(
                    req_meta,
                    save_completed,
                    token_ids_start,
                    event_token_ids,
                )
            self.finish_store_job(req_meta)
            self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    """Background thread loading KV cache blocks from Mooncake DRAM pool."""

    def __init__(
        self,
        store: Any,
        coord: MooncakeStoreCoordinator,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        *,
        device_index: int | None = None,
        prefer_alloc_in_same_node: bool = True,
        disk_offload_buffer_budget_bytes: int | None = None,
        record_operation: Callable[..., None] | None = None,
        request_queue: queue.Queue[Any] | None = None,
        group_participates: Sequence[bool] | None = None,
        is_hma_required: bool = False,
    ) -> None:
        super().__init__(
            store,
            token_databases,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreRecvingThread",
            device_index=device_index,
            record_operation=record_operation,
            request_queue=request_queue,
        )
        self.coord = coord
        self.prefer_alloc_in_same_node = prefer_alloc_in_same_node
        self._accepts_prefer_same_node = _store_get_accepts_prefer_same_node(store)
        self.group_participates = (
            list(group_participates)
            if group_participates is not None
            else [True] * len(token_databases)
        )
        self._invalid_block_ids_lock = threading.Lock()
        self._invalid_block_ids: set[int] = set()
        self._is_hma_required = is_hma_required
        self._failed_requests: set[str] = set()
        self.disk_offload_buffer_budget_bytes = disk_offload_buffer_budget_bytes
        self.usable_disk_offload_buffer_budget_bytes = (
            None
            if disk_offload_buffer_budget_bytes is None
            else _get_usable_disk_offload_buffer_budget_bytes(
                disk_offload_buffer_budget_bytes
            )
        )

    def _add_load_error_block_ids(self, block_ids: list[int]) -> None:
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(block_ids)

    def set_failed_request(self, req_id: str) -> None:
        with self.done_task_lock:
            self._failed_requests.add(req_id)

    def get_and_clear_failed_requests(self) -> set[str]:
        with self.done_task_lock:
            failed = self._failed_requests.copy()
            self._failed_requests.clear()
        return failed

    def _report_load_error(self, req_id: str, block_ids: list[int]) -> None:
        if self._is_hma_required:
            self.set_failed_request(req_id)
        else:
            self._add_load_error_block_ids(block_ids)

    def get_and_clear_block_ids_with_load_errors(self) -> set[int]:
        with self._invalid_block_ids_lock:
            invalid = self._invalid_block_ids.copy()
            self._invalid_block_ids.clear()
        return invalid

    def _batch_get(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
    ) -> list[int]:
        if self._accepts_prefer_same_node:
            return self.store.batch_get_into_multi_buffers(
                keys, addrs, sizes, self.prefer_alloc_in_same_node
            )
        return self.store.batch_get_into_multi_buffers(keys, addrs, sizes)

    def _handle_request(self, req_meta: ReqMeta) -> None:
        assert req_meta.load_spec is not None
        token_len = req_meta.load_spec.token_len
        req_id = req_meta.req_id
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens
            // self.block_size
            * self.block_size
        )

        load_mask_per_group = self.coord.load_mask(req_meta.block_hashes, token_len)
        tail_key_boundaries = {
            boundary.group_id: boundary.num_tokens
            for boundary in req_meta.load_spec.tail_key_boundaries
        }

        addr_list: list[list[int]] = []
        size_list: list[list[int]] = []
        key_list: list[str] = []
        block_id_list: list[int] = []
        for g_idx, db in enumerate(self.token_databases):
            if not self.group_participates[g_idx]:
                continue
            mask = load_mask_per_group[g_idx]
            chunks: list[tuple[int, int]] = []
            store_shard_ids: list[StoreShardId] = []
            for start, end, block_hash in db.process_tokens(
                token_len, req_meta.block_hashes, mask_num
            ):
                chunk_idx = start // db.block_size
                if chunk_idx >= len(mask) or not mask[chunk_idx]:
                    continue
                boundary_tokens = (
                    tail_key_boundaries.get(g_idx) if end == token_len else None
                )
                if (
                    boundary_tokens is not None
                    and start < boundary_tokens <= start + db.block_size
                ):
                    block_hash = req_meta.block_hashes[
                        boundary_tokens // db.hash_block_size - 1
                    ]
                for store_shard_id in db.store_layout.local_shard_ids:
                    key_list.append(db.store_layout.key_for(store_shard_id, block_hash))
                    chunks.append((start, end))
                    store_shard_ids.append(store_shard_id)
            if not chunks:
                continue
            g_addrs, g_sizes, g_block_ids = db.store_layout.prepare_values(
                chunks,
                req_meta.block_ids[g_idx],
                store_shard_ids,
            )
            addr_list.extend(g_addrs)
            size_list.extend(g_sizes)
            block_id_list.extend(g_block_ids)

        if not key_list:
            # Non-owning PP stage (e.g. DeepSeek-V4.1 PP8 stage 0) or SWA pre-window:
            # acknowledge immediately so worker output aggregation completes cleanly.
            self.set_finished_request(req_id)
            self.request_queue.task_done()
            return

        rotation = self.tp_rank % len(key_list)
        key_list_c = _rotate_list(key_list, rotation)
        addr_list_c = _rotate_list(addr_list, rotation)
        size_list_c = _rotate_list(size_list, rotation)
        block_id_list_c = _rotate_list(block_id_list, rotation)

        load_batches = [(key_list_c, addr_list_c, size_list_c, block_id_list_c)]
        if self.usable_disk_offload_buffer_budget_bytes is not None:
            total_staging = sum(
                _estimate_disk_offload_staging_bytes(s) for s in size_list_c
            )
            if total_staging > self.usable_disk_offload_buffer_budget_bytes:
                assert self.disk_offload_buffer_budget_bytes is not None
                split_batches, oversized_key = _split_disk_offload_load_batches(
                    key_list_c,
                    addr_list_c,
                    size_list_c,
                    self.usable_disk_offload_buffer_budget_bytes,
                    self.disk_offload_buffer_budget_bytes,
                )
                if oversized_key is not None:
                    self._report_load_error(req_id, block_id_list_c)
                    logger.warning(
                        "Skipping Mooncake load for req=%s: key %s exceeds "
                        "staging budget %d",
                        req_id,
                        oversized_key,
                        self.disk_offload_buffer_budget_bytes,
                    )
                    self.set_finished_request(req_id)
                    self.request_queue.task_done()
                    return
                load_batches = []
                offset = 0
                for b_keys, b_addrs, b_sizes in split_batches:
                    next_off = offset + len(b_keys)
                    load_batches.append(
                        (b_keys, b_addrs, b_sizes, block_id_list_c[offset:next_off])
                    )
                    offset = next_off

        current_batch_keys = key_list_c
        current_batch_block_ids = block_id_list_c
        batch_bytes = 0
        load_get_start = time.perf_counter()
        try:
            for (
                batch_keys,
                batch_addrs,
                batch_sizes,
                batch_block_ids,
            ) in load_batches:
                current_batch_keys = batch_keys
                current_batch_block_ids = batch_block_ids
                batch_bytes = _sum_batch_bytes(batch_sizes)
                load_get_start = time.perf_counter()
                res = self._batch_get(batch_keys, batch_addrs, batch_sizes)
                failed = [
                    (key, val, bid)
                    for key, val, bid in zip(
                        batch_keys, res, batch_block_ids, strict=True
                    )
                    if val < 0
                ]
                self._record_operation(
                    "load_get",
                    load_get_start,
                    len(batch_keys),
                    num_bytes=batch_bytes,
                    status="partial_failure" if failed else "ok",
                    num_failed_keys=len(failed),
                )
                if failed:
                    self._report_load_error(req_id, [bid for _, _, bid in failed])
                    logger.warning(
                        "Mooncake batch_get failed on %d/%d keys (req=%s, sample=%s)",
                        len(failed),
                        len(batch_keys),
                        req_id,
                        [(k, v) for k, v, _ in failed[:3]],
                    )
                    break
        except Exception as exc:
            self._report_load_error(req_id, current_batch_block_ids)
            self._record_operation(
                "load_get",
                load_get_start,
                len(current_batch_keys),
                num_bytes=batch_bytes,
                status="error",
                num_failed_keys=len(current_batch_keys),
            )
            logger.warning(
                "Mooncake batch_get raised error for req=%s: %s", req_id, exc
            )

        self.set_finished_request(req_id)
        self.request_queue.task_done()
