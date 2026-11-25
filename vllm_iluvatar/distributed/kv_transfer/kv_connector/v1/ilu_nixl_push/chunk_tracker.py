# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""O(1) base-request-ID indexing and non-blocking chunked-bulk state tracker."""

from __future__ import annotations

import contextlib
import queue
import time
from collections import defaultdict
from functools import partial
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_base_request_id,
)
from vllm.logger import init_logger

from ..ilu_nixl_common.types import decode_push_reg_wire
from .metadata import ChunkPushPlan, ChunkPushReqMeta

logger = init_logger(__name__)

_CHUNK_PROGRESS_PREFIX = "__ilu_chunk_progress__:"


def request_id_match_keys(request_id: str) -> tuple[str, ...]:
    """Canonical match keys (exact + base + stripped ``chatcmpl-``) in O(1)."""
    base = get_base_request_id(request_id)
    if request_id.startswith("chatcmpl-"):
        bare = request_id[len("chatcmpl-") :]
        bare_base = get_base_request_id(bare)
        return tuple(dict.fromkeys((request_id, base, bare, bare_base)))
    return (request_id, base) if base != request_id else (request_id,)


class BaseReqIndex:
    """O(1) bidirectional ``match_key -> set[req_id]`` index replacing O(N) scans."""

    __slots__ = ("_key_to_ids", "_id_to_keys")

    def __init__(self) -> None:
        self._key_to_ids: dict[str, set[str]] = defaultdict(set)
        self._id_to_keys: dict[str, tuple[str, ...]] = {}

    def add(self, req_id: str) -> None:
        if req_id in self._id_to_keys:
            return
        keys = request_id_match_keys(req_id)
        self._id_to_keys[req_id] = keys
        for k in keys:
            self._key_to_ids[k].add(req_id)

    def discard(self, req_id: str) -> None:
        keys = self._id_to_keys.pop(req_id, None)
        if keys is None:
            return
        for k in keys:
            bucket = self._key_to_ids.get(k)
            if bucket is not None:
                bucket.discard(req_id)
                if not bucket:
                    del self._key_to_ids[k]

    def match_first(
        self, query_id: str, valid_container: dict[str, Any] | set[str] | None = None
    ) -> str | None:
        if valid_container is not None and query_id in valid_container:
            return query_id
        for k in request_id_match_keys(query_id):
            bucket = self._key_to_ids.get(k)
            if not bucket:
                continue
            for candidate in list(bucket):
                if valid_container is None or candidate in valid_container:
                    return candidate
                self.discard(candidate)
        return None

    def match_all(self, query_id: str) -> set[str]:
        matched: set[str] = set()
        for k in request_id_match_keys(query_id):
            if bucket := self._key_to_ids.get(k):
                matched.update(bucket)
        return matched


class _ChunkBulkTask:
    """One whole-chunk WRITE fenced after the complete forward."""

    __slots__ = ("request_id", "chunk_index", "layer_name", "event", "enqueued_at")

    def __init__(
        self,
        request_id: str,
        chunk_index: int,
        layer_name: str,
        event: Any,
    ) -> None:
        self.request_id = request_id
        self.chunk_index = chunk_index
        self.layer_name = layer_name
        self.event = event
        self.enqueued_at = time.perf_counter()

    @property
    def plan_key(self) -> tuple[str, int]:
        return self.request_id, self.chunk_index


class ChunkTrackerMixin:
    """Chunked-bulk state machine with O(1) lookup, zero-copy reg, and async handshake."""

    def _ensure_req_indices(self) -> None:
        if not hasattr(self, "_pending_reg_index"):
            self._pending_reg_index = BaseReqIndex()
        if not hasattr(self, "_finished_blocks_index"):
            self._finished_blocks_index = BaseReqIndex()
        if not hasattr(self, "_recv_meta_index"):
            self._recv_meta_index = BaseReqIndex()
        if not hasattr(self, "_chunk_live_counts"):
            self._chunk_live_counts: dict[str, int] = defaultdict(int)
            self._chunk_live_index = BaseReqIndex()
        if not hasattr(self, "_chunk_pending_counts"):
            self._chunk_pending_counts: dict[str, int] = defaultdict(int)
            self._chunk_pending_index = BaseReqIndex()
        if not hasattr(self, "_chunk_open_index"):
            self._chunk_open_index = BaseReqIndex()
        if not hasattr(self, "_consumed_reg_bases"):
            self._consumed_reg_bases: dict[str, float] = {}

    def _chunk_inc_pending_locked(self, req_id: str) -> None:
        self._ensure_req_indices()
        if self._chunk_pending_counts[req_id] == 0:
            self._chunk_pending_index.add(req_id)
        self._chunk_pending_counts[req_id] += 1
        if self._chunk_live_counts[req_id] == 0:
            self._chunk_live_index.add(req_id)
        self._chunk_live_counts[req_id] += 1

    def _chunk_dec_pending_locked(self, req_id: str) -> None:
        self._ensure_req_indices()
        cnt = self._chunk_pending_counts.get(req_id, 0)
        if cnt <= 1:
            self._chunk_pending_counts.pop(req_id, None)
            self._chunk_pending_index.discard(req_id)
        else:
            self._chunk_pending_counts[req_id] = cnt - 1
        self._chunk_dec_live_locked(req_id)

    def _chunk_inc_plan_locked(self, req_id: str) -> None:
        self._ensure_req_indices()
        if self._chunk_live_counts[req_id] == 0:
            self._chunk_live_index.add(req_id)
        self._chunk_live_counts[req_id] += 1

    def _chunk_dec_live_locked(self, req_id: str) -> None:
        cnt = self._chunk_live_counts.get(req_id, 0)
        if cnt <= 1:
            self._chunk_live_counts.pop(req_id, None)
            self._chunk_live_index.discard(req_id)
        else:
            self._chunk_live_counts[req_id] = cnt - 1

    def _mark_chunk_open_locked(self, req_id: str) -> None:
        self._ensure_req_indices()
        self._chunk_open.add(req_id)
        self._chunk_open_index.add(req_id)

    def _unmark_chunk_open_locked(self, req_id: str) -> None:
        self._ensure_req_indices()
        self._chunk_open.discard(req_id)
        self._chunk_open_index.discard(req_id)

    @staticmethod
    def _request_ids_match(left: str, right: str) -> bool:
        return left == right or get_base_request_id(left) == get_base_request_id(right)

    def _chunk_matches_any(self, request_id: str, live: set[str]) -> bool:
        if request_id in live:
            return True
        base_id = get_base_request_id(request_id)
        return any(get_base_request_id(rid) == base_id for rid in live)

    def _chunk_matches_send_locked(self, request_id: str) -> bool:
        self._ensure_req_indices()
        if self._chunk_live_index.match_first(request_id) is not None:
            return True
        if self._chunk_pending or self._chunk_plans:
            live = {k[0] for k in self._chunk_pending} | {
                k[0] for k in self._chunk_plans
            }
            return self._chunk_matches_any(request_id, live)
        return False

    def _chunk_should_retain(self, request_id: str) -> bool:
        self._ensure_req_indices()
        with self._chunk_lock:
            if self._chunk_pending_index.match_first(request_id) is not None:
                return True
            if self._chunk_pending and self._chunk_matches_any(
                request_id, {k[0] for k in self._chunk_pending}
            ):
                return True
        with self._sending_transfers_lock:
            if (
                self._chunk_open_index.match_first(request_id, self._chunk_open)
                is not None
            ):
                return True
            return bool(
                self._chunk_open
                and self._chunk_matches_any(request_id, self._chunk_open)
            )

    def _find_matching_registration(
        self, request_id: str
    ) -> tuple[str, dict[str, Any]] | None:
        self._ensure_req_indices()
        chunk_regs = getattr(self, "_chunk_registrations", None)
        if chunk_regs and request_id in chunk_regs:
            return request_id, chunk_regs[request_id]
        pending = self._pending_d_registrations
        if not pending:
            return None
        if request_id in pending:
            return request_id, pending[request_id]
        matched_id = self._pending_reg_index.match_first(request_id, pending)
        if matched_id is not None:
            return matched_id, pending[matched_id]
        # Fallback if a unit test populated _pending_d_registrations directly.
        base_id = get_base_request_id(request_id)
        for reg_id in list(pending):
            self._pending_reg_index.add(reg_id)
            if get_base_request_id(reg_id) == base_id:
                return reg_id, pending[reg_id]
        return None

    def _pop_matching_registration(self, request_id: str) -> dict[str, Any] | None:
        self._ensure_req_indices()
        with self._chunk_lock:
            if self._chunk_matches_send_locked(request_id):
                return None
            pending = self._pending_d_registrations
            if not pending:
                return None
            data = pending.pop(request_id, None)
            if data is not None:
                self._pending_reg_index.discard(request_id)
                self._consumed_reg_bases[get_base_request_id(request_id)] = (
                    time.perf_counter()
                )
                return data
            matched_id = self._pending_reg_index.match_first(request_id, pending)
            if matched_id is not None:
                self._pending_reg_index.discard(matched_id)
                self._consumed_reg_bases[get_base_request_id(matched_id)] = (
                    time.perf_counter()
                )
                return pending.pop(matched_id)
            base_id = get_base_request_id(request_id)
            for reg_id in list(pending):
                if get_base_request_id(reg_id) == base_id:
                    self._pending_reg_index.discard(reg_id)
                    self._consumed_reg_bases[base_id] = time.perf_counter()
                    return pending.pop(reg_id)
            return None

    def _pop_matching_finished_blocks(self, request_id: str):
        self._ensure_req_indices()
        fin_map = self._push_finished_blocks
        if not fin_map:
            return None
        blocks = fin_map.pop(request_id, None)
        if blocks is not None:
            self._finished_blocks_index.discard(request_id)
            return request_id, blocks
        matched_id = self._finished_blocks_index.match_first(request_id, fin_map)
        if matched_id is not None:
            self._finished_blocks_index.discard(matched_id)
            return matched_id, fin_map.pop(matched_id)
        base_id = get_base_request_id(request_id)
        for fin_id in list(fin_map):
            if get_base_request_id(fin_id) == base_id:
                self._finished_blocks_index.discard(fin_id)
                return fin_id, fin_map.pop(fin_id)
        return None

    def _handle_push_reg_notif(self, notif: bytes) -> None:
        self._ensure_req_indices()
        try:
            reg_data = decode_push_reg_wire(notif[len(PUSH_REG_NOTIF_PREFIX) :])
        except Exception:
            logger.exception("Failed to decode PUSH_REG notification payload")
            return
        rid = reg_data.get("request_id") if isinstance(reg_data, dict) else None
        if not isinstance(rid, str):
            logger.warning("PUSH_REG notif missing request_id; dropping")
            return

        base_rid = get_base_request_id(rid)
        with self._chunk_lock:
            # Drop duplicate PUSH_REGs from sibling D TP/PP ranks in O(1).
            if base_rid in self._consumed_reg_bases:
                self._push_obs_stats().skipped_redundant_push_regs += 1
                return
            existing = self._pending_reg_index.match_first(
                rid, self._pending_d_registrations
            )
            if existing is not None:
                self._push_obs_stats().skipped_redundant_push_regs += 1
                return
            if self._chunk_matches_send_locked(rid):
                # Check if already cached in _chunk_registrations.
                if any(
                    get_base_request_id(cid) == base_rid
                    for cid in getattr(self, "_chunk_registrations", ())
                ):
                    self._push_obs_stats().skipped_redundant_push_regs += 1
                    return
                self._pending_d_registrations[rid] = reg_data
                self._pending_reg_index.add(rid)
                return

        match = self._pop_matching_finished_blocks(rid)
        if match is not None:
            self._consumed_reg_bases[base_rid] = time.perf_counter()
            fin_id, blocks = match
            self._do_start_push_kv(fin_id, blocks, reg_data)
        else:
            with self._chunk_lock:
                self._pending_d_registrations[rid] = reg_data
                self._pending_reg_index.add(rid)

    @staticmethod
    def _normalize_immutable_remote_groups(
        raw_groups: Any,
    ) -> tuple[tuple[int, ...], ...]:
        if not raw_groups:
            return ()
        if not isinstance(raw_groups[0], (list, tuple)):
            return (tuple(int(x) for x in raw_groups),)
        return tuple(tuple(int(x) for x in grp) for grp in raw_groups)

    def _arm_chunk_requests(self, metadata: NixlConnectorMetadata) -> None:
        reqs = getattr(metadata, "reqs_to_send_chunked", None)
        if not reqs:
            return
        for req_id, chunk_meta in reqs.items():
            self._arm_one_chunk_request(req_id, chunk_meta)

    def _arm_one_chunk_request(self, req_id: str, chunk_meta: ChunkPushReqMeta) -> None:
        self._ensure_req_indices()
        ordered_layers = tuple(self._ordered_local_layers())
        plan_key = (req_id, chunk_meta.chunk_index)
        self._chunk_current.add(plan_key)
        self._chunk_expected_layers[plan_key] = ordered_layers
        if not hasattr(self, "_chunk_armed_at"):
            self._chunk_armed_at = {}
        self._chunk_armed_at[plan_key] = time.perf_counter()
        if self._host_staging_producer:
            self._chunk_meta[plan_key] = chunk_meta
        with self._sending_transfers_lock:
            self._mark_chunk_open_locked(req_id)
        with self._chunk_lock:
            self._chunk_pending[plan_key] = chunk_meta
            self._chunk_inc_pending_locked(req_id)
            match = self._find_matching_registration(req_id)
            release_only = bool(
                match is not None and match[1].get("notify_release_only")
            )
            empty_skip = bool(
                getattr(self, "_empty_producer_rank", False)
                and self._is_predictable_empty_producer()
            )
            decode_engine_id = (
                match[1].get("decode_engine_id") if match is not None else None
            )
            handshake_ready = (
                release_only
                or empty_skip
                or (
                    decode_engine_id is not None
                    and decode_engine_id in self._remote_agents
                )
            )
            if match is not None and handshake_ready:
                self._commit_chunk_plan_locked(plan_key, chunk_meta, match)
                return

        self._push_writer_wake.set()

    def _commit_chunk_plan_locked(
        self,
        plan_key: tuple[str, int],
        chunk_meta: ChunkPushReqMeta,
        match: tuple[str, dict[str, Any]],
    ) -> None:
        self._ensure_req_indices()
        req_id, chunk_index = plan_key
        reg_id, reg = match
        if not hasattr(self, "_chunk_registrations"):
            self._chunk_registrations = {}
        if reg_id in self._pending_d_registrations:
            self._pending_d_registrations.pop(reg_id, None)
            self._pending_reg_index.discard(reg_id)
            self._consumed_reg_bases[get_base_request_id(reg_id)] = time.perf_counter()
            cached_reg = dict(reg)
            cached_reg["local_block_ids"] = self._normalize_immutable_remote_groups(
                cached_reg.get("local_block_ids", ())
            )
            self._chunk_registrations[req_id] = cached_reg
            reg = cached_reg
        else:
            reg = self._chunk_registrations.get(req_id, reg)

        if plan_key in self._chunk_pending:
            self._chunk_pending.pop(plan_key, None)
            self._chunk_dec_pending_locked(req_id)

        armed_at = getattr(self, "_chunk_armed_at", {}).pop(plan_key, None)
        if armed_at is not None and chunk_index == 0:
            self._push_obs_stats().record_wait_push_reg(
                (time.perf_counter() - armed_at) * 1000.0
            )

        with self._sending_transfers_lock:
            self._mark_chunk_open_locked(req_id)

        start, end = self._pp_local_layer_range()
        ordered_layers = list(
            self._chunk_expected_layers.get(
                plan_key, tuple(self._ordered_local_layers())
            )
        )
        remote_groups = self._normalize_immutable_remote_groups(
            reg.get("local_block_ids", ())
        )
        remote_block_starts = reg.get("remote_block_starts")
        release_only = bool(reg.get("notify_release_only"))
        has_tail_only_offset = any(off < 0 for off in chunk_meta.block_offsets)
        has_hetero_pairings = bool(
            getattr(self, "region_pairings", {}).get(reg["decode_engine_id"])
            or getattr(self, "_mamba_hma_routes", {}).get(reg["decode_engine_id"])
            or getattr(self, "_inactive_remote_keys", {}).get(reg["decode_engine_id"])
            or getattr(self, "_empty_producer_rank", False)
            or has_tail_only_offset
        )

        # P2 Zero-copy chunk_reg: share immutable tuple ``remote_groups`` directly!
        chunk_reg = dict(reg)
        if (
            release_only
            or not any(remote_groups)
            or not any(chunk_meta.local_block_ids)
        ):
            chunk_reg["local_block_ids"] = tuple(() for _ in chunk_meta.local_block_ids)
        elif remote_block_starts is not None or has_hetero_pairings:
            for group_idx, offset in enumerate(chunk_meta.block_offsets):
                if offset < -1:
                    raise RuntimeError(
                        "invalid decode block offset for chunked push transfer "
                        f"{req_id} group={group_idx}: offset={offset}"
                    )
            chunk_reg["local_block_ids"] = remote_groups
            chunk_reg["chunk_block_offsets"] = chunk_meta.block_offsets
        else:
            if len(remote_groups) < len(chunk_meta.local_block_ids):
                raise RuntimeError(
                    "decode PUSH_REG has fewer KV groups than the chunked push "
                    f"transfer for {req_id}: remote={len(remote_groups)}, "
                    f"local={len(chunk_meta.local_block_ids)}"
                )
            sliced_remote: list[tuple[int, ...]] = []
            for group_idx, local_group in enumerate(chunk_meta.local_block_ids):
                offset = (
                    chunk_meta.block_offsets[group_idx]
                    if group_idx < len(chunk_meta.block_offsets)
                    else 0
                )
                if offset < 0:
                    raise RuntimeError(
                        "negative decode block offset for chunked push transfer "
                        f"{req_id} group={group_idx}: offset={offset}"
                    )
                remote_group = (
                    remote_groups[group_idx] if group_idx < len(remote_groups) else ()
                )
                selected_remote = remote_group[offset : offset + len(local_group)]
                if len(selected_remote) != len(local_group):
                    raise RuntimeError(
                        "decode PUSH_REG does not cover the complete chunked "
                        f"block range for {req_id} group={group_idx}: "
                        f"offset={offset}, requested={len(local_group)}, "
                        f"registered={len(remote_group)}, selected={len(selected_remote)}"
                    )
                sliced_remote.append(selected_remote)
            chunk_reg["local_block_ids"] = tuple(sliced_remote)

        if not chunk_meta.is_last_chunk:
            chunk_reg["request_id"] = (
                f"{_CHUNK_PROGRESS_PREFIX}{chunk_index}:{reg['request_id']}"
            )
        plan = ChunkPushPlan(
            request_id=req_id,
            local_block_ids=[list(group) for group in chunk_meta.local_block_ids],
            decode_engine_id=reg["decode_engine_id"],
            remote_request_id=chunk_reg["request_id"],
            prefill_tokens=chunk_meta.prefill_tokens,
            layer_start=start,
            layer_end=end,
            prefill_chunk_index=chunk_index,
            is_last_chunk=chunk_meta.is_last_chunk,
            block_offsets=chunk_meta.block_offsets,
            registration=chunk_reg,
            ordered_layers=ordered_layers,
            armed_at=armed_at or 0.0,
        )
        self._chunk_plans[plan_key] = plan
        self._chunk_inc_plan_locked(req_id)

    def _fail_pending_chunks_for_engine(
        self, decode_engine_id: str, error: BaseException
    ) -> None:
        self._log_failure(
            failure_type="chunk_push_handshake_failed",
            req_id=None,
            error=error,
            remote_engine_id=decode_engine_id,
        )
        failed_reqs: set[str] = set()
        with self._chunk_lock:
            for plan_key in list(self._chunk_pending):
                req_id = plan_key[0]
                match = self._find_matching_registration(req_id)
                target_eid = (
                    match[1].get("decode_engine_id")
                    if match is not None
                    else self._chunk_pending[plan_key].decode_engine_id
                )
                if target_eid == decode_engine_id:
                    self._chunk_pending.pop(plan_key, None)
                    self._chunk_dec_pending_locked(req_id)
                    self._chunk_expected_layers.pop(plan_key, None)
                    failed_reqs.add(req_id)
        if failed_reqs:
            with self._sending_transfers_lock:
                for rid in failed_reqs:
                    self._unmark_chunk_open_locked(rid)
                    self._send_failures.add(rid)
            self._push_writer_wake.set()

    def _activate_pending_chunks(self) -> None:
        """P0-1 Non-blocking chunk promotion: NEVER calls ``fut.result()`` on writer!"""
        with self._chunk_lock:
            if not self._chunk_pending:
                return
            pending_keys = list(self._chunk_pending)

        if not hasattr(self, "_chunk_handshake_watched"):
            self._chunk_handshake_watched: set[str] = set()

        for plan_key in pending_keys:
            req_id = plan_key[0]
            with self._chunk_lock:
                chunk_meta = self._chunk_pending.get(plan_key)
                match = self._find_matching_registration(req_id)
                if chunk_meta is None or match is None:
                    continue
                reg = match[1]
                decode_engine_id = reg.get("decode_engine_id")
                empty_skip = bool(
                    getattr(self, "_empty_producer_rank", False)
                    and self._is_predictable_empty_producer()
                )
                handshake_ready = (
                    bool(reg.get("notify_release_only"))
                    or empty_skip
                    or (
                        decode_engine_id is not None
                        and decode_engine_id in self._remote_agents
                    )
                )
                if handshake_ready:
                    self._commit_chunk_plan_locked(plan_key, chunk_meta, match)
                    continue

            if decode_engine_id is None:
                continue
            fut = self._ensure_handshake(
                decode_engine_id,
                reg["decode_host"],
                int(reg["decode_port"]),
                int(reg["decode_tp_size"]),
                int(reg.get("decode_pp_size", 1)),
                reg.get("decode_pp_split", "") or "",
            )
            if fut is None:
                with self._chunk_lock:
                    chunk_meta = self._chunk_pending.get(plan_key)
                    match = self._find_matching_registration(req_id)
                    if chunk_meta is not None and match is not None:
                        self._commit_chunk_plan_locked(plan_key, chunk_meta, match)
                continue

            if decode_engine_id not in self._chunk_handshake_watched:
                self._chunk_handshake_watched.add(decode_engine_id)

                def _on_chunk_hs_done(f, eid: str = decode_engine_id) -> None:
                    self._chunk_handshake_watched.discard(eid)
                    if (err := f.exception()) is not None:
                        self._fail_pending_chunks_for_engine(eid, err)
                        return
                    self._push_writer_wake.set()

                fut.add_done_callback(_on_chunk_hs_done)

    def finish_chunked_bulk_forward(self) -> None:
        """Queue one whole-chunk WRITE after the full forward is enqueued."""
        if not self._chunk_current:
            return

        event = self._record_chunk_event()
        for req_id, chunk_index in self._chunk_current:
            expected = self._chunk_expected_layers.get((req_id, chunk_index), ())
            if self._host_staging_producer:
                self._enqueue_stage_task(req_id, chunk_index, expected, event)
                continue
            self._chunk_inbox.put(
                _ChunkBulkTask(
                    req_id,
                    chunk_index,
                    expected[-1] if expected else "",
                    event,
                )
            )
        self._push_writer_wake.set()

    def _enqueue_stage_task(
        self,
        req_id: str,
        chunk_index: int,
        layer_names: tuple[str, ...],
        event: Any,
    ) -> None:
        from ..ilu_nixl_common.fp8_hetero.stager import StageTask

        chunk_meta = self._chunk_meta.get((req_id, chunk_index))
        if chunk_meta is None:
            return
        if not layer_names:
            self._chunk_inbox.put(_ChunkBulkTask(req_id, chunk_index, "", event))
            return
        make_writer_task = partial(
            _ChunkBulkTask, req_id, chunk_index, layer_names[-1], None
        )
        self._stage_inbox.put(
            StageTask(
                request_id=req_id,
                chunk_index=chunk_index,
                layer_names=tuple(layer_names),
                ready_event=event,
                quant_block_ids=chunk_meta.quant_block_ids,
                quant_first_pages=chunk_meta.quant_first_pages,
                num_valid_tokens=chunk_meta.num_valid_tokens,
                requantize_block_ids=chunk_meta.requantize_block_ids,
                is_last_chunk=chunk_meta.is_last_chunk,
                make_writer_task=make_writer_task,
            )
        )

    def _drain_chunk_inbox(self) -> None:
        while True:
            try:
                task = self._chunk_inbox.get_nowait()
                self._chunk_deferred.append(task)
            except queue.Empty:
                break
        if not self._chunk_deferred:
            return
        still_deferred: list[_ChunkBulkTask] = []
        blocked_requests: set[str] = set()
        for task in self._chunk_deferred:
            plan_key = task.plan_key
            if task.request_id in blocked_requests:
                still_deferred.append(task)
                continue
            if not self._event_ready(task.event):
                still_deferred.append(task)
                blocked_requests.add(task.request_id)
                continue
            with self._chunk_lock:
                plan = self._chunk_plans.get(plan_key)
                still_pending = plan is None and plan_key in self._chunk_pending
            if plan is None:
                if still_pending:
                    still_deferred.append(task)
                    blocked_requests.add(task.request_id)
                continue
            # Single-step fast path (chunk_index == 0 and is_last_chunk) has no
            # earlier chunks, so only check _sending_transfers when chunk_index > 0.
            if plan.is_last_chunk and plan.prefill_chunk_index > 0:
                with self._sending_transfers_lock:
                    earlier_in_flight = bool(
                        self._sending_transfers.get(task.request_id)
                    )
                if earlier_in_flight:
                    still_deferred.append(task)
                    blocked_requests.add(task.request_id)
                    continue
            try:
                if plan.is_last_chunk:
                    if not hasattr(self, "_tail_chunk_ready_at"):
                        self._tail_chunk_ready_at = {}
                    self._tail_chunk_ready_at[plan.request_id] = time.perf_counter()
                self._do_start_push_kv(
                    plan.request_id,
                    plan.local_block_ids,
                    plan.registration,
                )
                with self._chunk_lock:
                    if self._chunk_plans.pop(plan_key, None) is not None:
                        self._chunk_dec_live_locked(task.request_id)
                    self._chunk_expected_layers.pop(plan_key, None)
                if plan.is_last_chunk:
                    with self._sending_transfers_lock:
                        self._unmark_chunk_open_locked(plan.request_id)
            except Exception as e:
                with self._chunk_lock:
                    if self._chunk_plans.pop(plan_key, None) is not None:
                        self._chunk_dec_live_locked(task.request_id)
                blocked_requests.add(task.request_id)
                self._log_failure(
                    failure_type="chunk_push_xfer_failed",
                    req_id=task.request_id,
                    msg="chunked-bulk WRITE failed; failing closed",
                    error=e,
                )
                self._send_fast_err_notif_from_registration(
                    task.request_id, plan.registration
                )
        self._chunk_deferred = still_deferred

    def _evict_finished_request(self, rid: str) -> None:
        self._ensure_req_indices()
        pending_lock = getattr(self, "_send_done_pending_lock", None)
        if pending_lock is not None:
            with pending_lock:
                self._send_done_pending.pop(rid, None)
            renewals = getattr(self, "_lease_renewals", None)
            if renewals is not None:
                renewals.pop(rid, None)
        if self._push_finished_blocks.pop(rid, None) is not None:
            self._finished_blocks_index.discard(rid)
        if self._pending_d_registrations.pop(rid, None) is not None:
            self._pending_reg_index.discard(rid)
        for reg_id in list(self._pending_reg_index.match_all(rid)):
            self._pending_d_registrations.pop(reg_id, None)
            self._pending_reg_index.discard(reg_id)

        base_id = get_base_request_id(rid)
        self._consumed_reg_bases.pop(base_id, None)
        if len(self._consumed_reg_bases) > 4096:
            cutoff = time.perf_counter() - 120.0
            self._consumed_reg_bases = {
                k: ts for k, ts in self._consumed_reg_bases.items() if ts > cutoff
            }

        with self._chunk_lock:
            all_plan_keys = (
                set(self._chunk_pending)
                | set(self._chunk_plans)
                | set(self._chunk_expected_layers)
            )
            for plan_key in [
                key for key in all_plan_keys if self._request_ids_match(key[0], rid)
            ]:
                if self._chunk_pending.pop(plan_key, None) is not None:
                    self._chunk_dec_pending_locked(plan_key[0])
                if self._chunk_plans.pop(plan_key, None) is not None:
                    self._chunk_dec_live_locked(plan_key[0])
                self._chunk_expected_layers.pop(plan_key, None)
                self._chunk_meta.pop(plan_key, None)
                if hasattr(self, "_chunk_armed_at"):
                    self._chunk_armed_at.pop(plan_key, None)
            for chunk_id in list(self._chunk_registrations):
                if self._request_ids_match(chunk_id, rid):
                    self._chunk_registrations.pop(chunk_id, None)

        if hasattr(self, "_tail_chunk_ready_at"):
            self._tail_chunk_ready_at.pop(rid, None)
        self._chunk_deferred = [
            task
            for task in self._chunk_deferred
            if not self._request_ids_match(task.request_id, rid)
        ]
        lingering: list[int] | None = None
        with self._sending_transfers_lock:
            if hasattr(self, "_inflight_send_engines"):
                self._inflight_send_engines.pop(rid, None)
            if hasattr(self, "_inflight_send_targets"):
                self._inflight_send_targets.pop(rid, None)
            open_ids = [
                open_id
                for open_id in list(self._chunk_open)
                if self._request_ids_match(open_id, rid)
            ]
            for open_id in open_ids:
                self._unmark_chunk_open_locked(open_id)
                if hasattr(self, "_inflight_send_engines"):
                    self._inflight_send_engines.pop(open_id, None)
                if hasattr(self, "_inflight_send_targets"):
                    self._inflight_send_targets.pop(open_id, None)
                leftover = self._sending_transfers.pop(open_id, None)
                if leftover:
                    lingering = (lingering or []) + leftover
        for handle in lingering or []:
            with contextlib.suppress(Exception):
                self.nixl_wrapper.release_xfer_handle(handle)
