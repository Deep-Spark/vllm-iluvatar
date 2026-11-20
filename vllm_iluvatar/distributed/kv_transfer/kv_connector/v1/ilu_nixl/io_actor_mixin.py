# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-owner NIXL IO actor, ZMQ batch handshake, clock sync, and DRAINING sessions."""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import msgspec
import zmq
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import GET_META_MSG
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import _is_ssm_spec
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path

from ..ilu_nixl_common.dsv41 import validate_wire_format_tags
from .metadata import (
    GET_META_BATCH_MSG,
    IluNixlAgentMetadata,
    IluNixlHandshakePayload,
)
from .session import RemoteEngineSession, RemoteSessionState
from .zmq_compat import zmq_ctx

logger = init_logger(__name__)


@dataclass
class _IluCompletionBatch:
    done_sending: set[str] = field(default_factory=set)
    done_recving: set[str] = field(default_factory=set)
    failed_recving: set[str] = field(default_factory=set)
    recv_metadata: dict[str, Any] = field(default_factory=dict)
    enqueued_at: float = 0.0

    def merge(self, other: "_IluCompletionBatch") -> None:
        self.done_sending.update(other.done_sending)
        self.done_recving.update(other.done_recving)
        self.failed_recving.update(other.failed_recving)
        self.recv_metadata.update(other.recv_metadata)


def _query_zmq_payloads_with_clock(
    host: str,
    port: int,
    missing_ranks: list[int],
) -> tuple[dict[int, bytes], float | None]:
    """Fetch rank payloads via single-RTT ``GET_META_BATCH_MSG`` (fallback to pipelined)."""
    if not missing_ranks:
        return {}, None
    from . import base_worker as _bw

    ctx_fn = getattr(_bw, "zmq_ctx", zmq_ctx)
    path = make_zmq_path("tcp", host, port)
    fetched: dict[int, bytes] = {}
    best_rtt = float("inf")
    best_offset: float | None = None

    with ctx_fn(zmq.DEALER, path) as sock:
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        batch_req = msgspec.msgpack.encode((GET_META_BATCH_MSG, missing_ranks))
        send_t = time.perf_counter()
        sock.send_multipart((b"", batch_req))
        try:
            frames = sock.recv_multipart()
            recv_t = time.perf_counter()
            raw_body = frames[1] if len(frames) >= 2 else frames[0]
            decoded = msgspec.msgpack.decode(raw_body)
            if isinstance(decoded, dict) and all(
                int(k) in decoded or k in decoded for k in missing_ranks
            ):
                for r in missing_ranks:
                    fetched[r] = decoded[r] if r in decoded else decoded[int(r)]
                if len(frames) >= 3:
                    remote_ts = msgspec.msgpack.decode(frames[2])
                    if isinstance(remote_ts, (int, float)):
                        best_offset = float(remote_ts) - (send_t + recv_t) / 2.0
                return fetched, best_offset
        except zmq.Again:
            logger.debug(
                "Peer at %s did not answer GET_META_BATCH_MSG; falling back to pipelined GET_META_MSG",
                path,
            )
        except Exception:
            logger.debug("Batch handshake decode fallback on %s", path, exc_info=True)

    # Fallback: pipelined individual GET_META_MSG on a fresh DEALER socket.
    with ctx_fn(zmq.DEALER, path) as sock:
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        send_times: list[float] = []
        for target_rank in missing_ranks:
            msg = msgspec.msgpack.encode((GET_META_MSG, target_rank))
            send_times.append(time.perf_counter())
            sock.send_multipart((b"", msg))
        for idx, target_rank in enumerate(missing_ranks):
            frames = sock.recv_multipart()
            recv_t = time.perf_counter()
            fetched[target_rank] = frames[1] if len(frames) >= 2 else frames[0]
            if len(frames) >= 3:
                with suppress(Exception):
                    remote_ts = msgspec.msgpack.decode(frames[2])
                    rtt = recv_t - send_times[idx]
                    if isinstance(remote_ts, (int, float)) and rtt < best_rtt:
                        best_rtt = rtt
                        best_offset = float(remote_ts) - (send_times[idx] + recv_t) / 2.0

    return fetched, best_offset


class IluNixlIoActorMixin:
    """Handshake orchestration, DRAINING session eviction, and single-owner IO loop."""

    # ------------------------------------------------------------------
    # Session & DRAINING lifecycle (P0-1)
    # ------------------------------------------------------------------

    def _register_active_session(
        self,
        engine_id: str,
        host: str,
        port: int,
        remote_agents: dict[int, str],
    ) -> RemoteEngineSession:
        if not hasattr(self, "_active_sessions"):
            self._active_sessions: dict[str, RemoteEngineSession] = {}
        if not hasattr(self, "_draining_sessions"):
            self._draining_sessions: list[RemoteEngineSession] = []
        if not hasattr(self, "_engine_generations"):
            self._engine_generations: dict[str, int] = {}
        if not hasattr(self, "_engine_by_address"):
            self._engine_by_address: dict[tuple[str, int], str] = {}

        # If a different engine_id previously lived at (host, port), drain it safely!
        addr_key = (host, int(port))
        prior_eid = self._engine_by_address.get(addr_key)
        if prior_eid is not None and prior_eid != engine_id:
            self._retire_remote_engine_to_draining(prior_eid)
        self._engine_by_address[addr_key] = engine_id

        gen = self._engine_generations.get(engine_id, 0) + 1
        self._engine_generations[engine_id] = gen
        dlists = dict(self.dst_xfer_side_handles.get(engine_id, {}))
        session = RemoteEngineSession(
            engine_id=engine_id,
            generation=gen,
            host=host,
            port=int(port),
            remote_agents=dict(remote_agents),
            dlist_handles=dlists,
        )
        self._active_sessions[engine_id] = session
        return session

    def _drop_engine_routing_tables(self, engine_id: str) -> None:
        self._remote_agents.pop(engine_id, None)
        self.dst_xfer_side_handles.pop(engine_id, None)
        self.kv_caches_base_addr.pop(engine_id, None)
        self.dst_num_blocks.pop(engine_id, None)
        self.tp_mappings.pop(engine_id, None)
        if self.transfer_topo is not None and hasattr(
            self.transfer_topo, "unregister_remote_engine"
        ):
            with suppress(Exception):
                self.transfer_topo.unregister_remote_engine(engine_id)
        if hasattr(self, "_engine_clock_offset"):
            self._engine_clock_offset.pop(engine_id, None)
        self._engine_last_active.pop(engine_id, None)

        for attr in (
            "region_pairings",
            "_remote_desc_cumsum",
            "_remote_kv_layer_end_by_engine",
            "_remote_supports_block_starts",
            "_remote_supports_structured_notif",
            "_compiled_desc_buckets",
            "_pending_heartbeats_by_engine",
            "_io_pending_reqs_by_engine",
            "_inactive_remote_keys",
            "local_region_ids_by_group",
            "remote_region_ids_by_group",
            "_mamba_hma_routes",
            "_remote_fa_num_regions",
            "_selected_ssm_remote_region_ids",
            "_remote_pp_size_by_engine",
            "_remote_pp_split_by_engine",
        ):
            mapping = getattr(self, attr, None)
            if isinstance(mapping, dict):
                mapping.pop(engine_id, None)
        if hasattr(self, "_semantic_region_engines"):
            self._semantic_region_engines.discard(engine_id)
        cache = getattr(self, "_handshake_payload_cache", None)
        if isinstance(cache, dict):
            for key in [k for k in cache if k[0] == engine_id]:
                cache.pop(key, None)

    def _release_session_nixl_resources(self, session: RemoteEngineSession) -> None:
        for handle in session.dlist_handles.values():
            with suppress(Exception):
                self.nixl_wrapper.release_dlist_handle(handle)
        session.dlist_handles.clear()
        for agent_name in session.remote_agents.values():
            with suppress(Exception):
                self.nixl_wrapper.remove_remote_agent(agent_name)
        session.remote_agents.clear()
        session.state = RemoteSessionState.CLOSED

    def _retire_remote_engine_to_draining(
        self, engine_id: str, *, log_eviction: bool = True
    ) -> None:
        """Retire ``engine_id`` from routing; defer NIXL release if READs are in-flight."""
        sessions = getattr(self, "_active_sessions", None)
        session = sessions.pop(engine_id, None) if isinstance(sessions, dict) else None
        if session is None:
            if engine_id not in getattr(self, "_remote_agents", {}):
                return
            session = RemoteEngineSession(
                engine_id=engine_id,
                generation=0,
                remote_agents=dict(self._remote_agents.get(engine_id, {})),
                dlist_handles=dict(self.dst_xfer_side_handles.get(engine_id, {})),
            )
            handle_map = getattr(self, "_handle_remote_info", {})
            for h, info in handle_map.items():
                if info[0] == engine_id:
                    session.add_inflight(h)

        self._drop_engine_routing_tables(engine_id)
        grace = float(getattr(self, "_consumer_disconnect_grace", 30.0))
        now = time.perf_counter()
        if not session.has_inflight:
            self._release_session_nixl_resources(session)
            if log_eviction:
                logger.info("Released idle remote engine %s immediately.", engine_id)
            return

        session.start_draining(now, grace)
        if not hasattr(self, "_draining_sessions"):
            self._draining_sessions = []
        self._draining_sessions.append(session)
        logger.info(
            "Remote engine %s entered DRAINING with %d in-flight READ handle(s).",
            engine_id,
            len(session.inflight_handles),
        )

    def _progress_draining_sessions(self, now: float) -> None:
        draining = getattr(self, "_draining_sessions", None)
        if not draining:
            return
        remaining: list[RemoteEngineSession] = []
        for session in draining:
            can_release, newly_stuck = session.check_drain_progress(now)
            if can_release:
                self._release_session_nixl_resources(session)
                logger.info(
                    "Drained and closed remote engine %s (gen %d).",
                    session.engine_id,
                    session.generation,
                )
                continue
            if newly_stuck:
                logger.error(
                    "Remote engine %s (gen %d) still has %d stuck READ handle(s) "
                    "after drain grace timeout; retaining NIXL descriptors safely.",
                    session.engine_id,
                    session.generation,
                    len(session.inflight_handles),
                )
            remaining.append(session)
        self._draining_sessions = remaining

    def _on_handle_settled_for_session(
        self, handle: int, engine_id: str | None
    ) -> None:
        if engine_id is None:
            return
        active = getattr(self, "_active_sessions", None)
        if isinstance(active, dict) and engine_id in active:
            active[engine_id].remove_inflight(handle)
            return
        for session in getattr(self, "_draining_sessions", ()) or ():
            if session.engine_id == engine_id and handle in session.inflight_handles:
                session.remove_inflight(handle)
                break

    def _evict_stale_engines(self) -> None:
        ttl = float(getattr(self, "_engine_ttl", 3600.0))
        if ttl <= 0:
            return
        now = time.perf_counter()
        for eid, last_active in list(getattr(self, "_engine_last_active", {}).items()):
            if now - last_active > ttl:
                self._retire_remote_engine_to_draining(eid, log_eviction=True)

    def _cleanup_remote_engine(
        self, engine_id: str, *, log_eviction: bool = True
    ) -> None:
        self._retire_remote_engine_to_draining(engine_id, log_eviction=log_eviction)

    # ------------------------------------------------------------------
    # Handshake & Clock-Offset Calibration (P0-3, P1-4, P2-3)
    # ------------------------------------------------------------------

    def _decode_and_install_targets(
        self,
        *,
        engine_id: str,
        host: str,
        port: int,
        remote_tp_size: int,
        remote_pp_size: int,
        remote_pp_split: str,
        targets: list[tuple[int, int, int]],
        fetched: dict[int, bytes],
        clock_offset: float | None,
    ) -> dict[int, str]:
        if not self.use_host_buffer:
            current_platform.set_device(self.device_id)
        payload_cache = getattr(self, "_handshake_payload_cache", None)
        if payload_cache is None:
            payload_cache = {}
            self._handshake_payload_cache = payload_cache
        for target_rank, raw_bytes in fetched.items():
            payload_cache[(engine_id, port, target_rank)] = raw_bytes

        if clock_offset is not None:
            if not hasattr(self, "_engine_clock_offset"):
                self._engine_clock_offset = {}
            self._engine_clock_offset[engine_id] = clock_offset
        elif hasattr(self, "_engine_clock_offset"):
            self._engine_clock_offset.setdefault(engine_id, 0.0)

        handshake_decoder = msgspec.msgpack.Decoder(IluNixlHandshakePayload)
        metadata_decoder = msgspec.msgpack.Decoder(IluNixlAgentMetadata)
        is_hetero_key = remote_pp_size != self.pp_world_size or (
            bool(remote_pp_split) and remote_pp_split != self.pp_split
        )
        remote_rank_to_agent_name: dict[int, str] = {}

        for remote_pp_rank, remote_rank, target_rank in targets:
            handshake_bytes = payload_cache[(engine_id, port, target_rank)]
            try:
                handshake_payload = handshake_decoder.decode(handshake_bytes)
            except (msgspec.DecodeError, msgspec.ValidationError) as e:
                raise RuntimeError(
                    "Failed to decode NixlHandshakePayload. This likely "
                    f"indicates an incompatible connector version. Error: {e}"
                ) from e

            wire_staging = getattr(self, "fp8_wire_staging_enabled", False)
            requires_compat = validate_wire_format_tags(
                getattr(self, "dsv41_layout", None),
                remote_engine_id=engine_id,
                remote_main_kv_format=handshake_payload.main_kv_format,
                remote_indexer_kv_layout=handshake_payload.indexer_kv_layout,
                local_main_kv_format="none" if wire_staging else None,
                local_indexer_kv_layout="legacy" if wire_staging else None,
            )
            assert self.compat_hash is not None
            if (
                self.enforce_compat_hash or requires_compat
            ) and handshake_payload.compatibility_hash != self.compat_hash:
                raise RuntimeError(
                    "NIXL compatibility hash mismatch. "
                    f"Local: {self.compat_hash}, Remote: {handshake_payload.compatibility_hash}."
                )
            try:
                metadata = metadata_decoder.decode(
                    handshake_payload.agent_metadata_bytes
                )
            except (msgspec.DecodeError, msgspec.ValidationError) as e:
                raise RuntimeError(
                    f"Failed to decode IluNixlAgentMetadata. Error: {e}"
                ) from e

            if metadata.main_kv_format != handshake_payload.main_kv_format:
                raise RuntimeError(
                    "Remote NIXL main KV format tag disagrees with "
                    f"agent metadata for engine {engine_id}: "
                    f"envelope={handshake_payload.main_kv_format}, agent={metadata.main_kv_format}"
                )
            if metadata.engine_id != engine_id:
                raise RuntimeError(
                    "Remote NIXL agent engine ID mismatch. "
                    f"Expected {engine_id}, received {metadata.engine_id}."
                )
            remote_agent_name = self.add_remote_agent(
                metadata,
                remote_rank,
                remote_tp_size,
                remote_pp_rank,
                remote_pp_size,
            )
            remote_key = target_rank if is_hetero_key else remote_rank
            remote_rank_to_agent_name[remote_key] = remote_agent_name

        return remote_rank_to_agent_name

    def _compute_handshake_targets(
        self, remote_tp_size: int, remote_pp_size: int, remote_pp_split: str
    ) -> list[tuple[int, int, int]]:
        assert self.transfer_topo is not None
        p_remote_ranks = self.transfer_topo.handshake_target_ranks(remote_tp_size)
        if remote_pp_split:
            remote_pp_ranks = self._target_pp_ranks(
                self.pp_split, self.pp_rank, remote_pp_split
            )
        elif remote_pp_size == self.pp_world_size:
            remote_pp_ranks = [self.pp_rank]
        else:
            raise RuntimeError(
                "IluNixl heterogeneous PP handshake requires remote "
                "pp_split in kv_transfer_params."
            )
        return [
            (pp_r, tp_r, pp_r * remote_tp_size + tp_r)
            for pp_r in remote_pp_ranks
            for tp_r in p_remote_ranks
        ]

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        remote_pp_size: int,
        remote_pp_split: str,
        expected_engine_id: str,
    ) -> dict[int, str]:
        t0 = time.perf_counter()
        targets = self._compute_handshake_targets(
            remote_tp_size, remote_pp_size, remote_pp_split
        )
        if not targets:
            return {}
        payload_cache = getattr(self, "_handshake_payload_cache", {})
        missing_ranks = [
            g_r
            for _pp_r, _tp_r, g_r in targets
            if (expected_engine_id, port, g_r) not in payload_cache
        ]
        fetched, clock_offset = _query_zmq_payloads_with_clock(
            host, port, missing_ranks
        )
        agents = self._decode_and_install_targets(
            engine_id=expected_engine_id,
            host=host,
            port=port,
            remote_tp_size=remote_tp_size,
            remote_pp_size=remote_pp_size,
            remote_pp_split=remote_pp_split,
            targets=targets,
            fetched=fetched,
            clock_offset=clock_offset,
        )
        stats = getattr(self, "xfer_stats", None)
        if stats is not None and hasattr(stats, "record_handshake_duration"):
            stats.record_handshake_duration(time.perf_counter() - t0)
        return agents

    def _ensure_handshake(
        self,
        engine_id: str,
        host: str,
        port: int,
        tp_size: int,
        pp_size: int = 1,
        pp_split: str = "",
    ) -> Future[dict[int, str]] | None:
        self._evict_stale_engines()
        with self._handshake_lock:
            if engine_id in self._remote_agents:
                return None
            fut = self._handshake_futures.get(engine_id)
            if fut is not None:
                return fut
            fut = self._handshake_initiation_executor.submit(
                self._nixl_handshake,
                host,
                port,
                tp_size,
                pp_size,
                pp_split,
                engine_id,
            )
            self._handshake_futures[engine_id] = fut

            def done_callback(f: Future[dict[int, str]], eid=engine_id):
                with self._handshake_lock:
                    self._handshake_futures.pop(eid, None)
                    try:
                        agents = f.result()
                        self._remote_agents[eid] = agents
                        self._engine_last_active[eid] = time.perf_counter()
                        self._register_active_session(eid, host, port, agents)
                        self._flush_pending_heartbeats_for_engine(eid)
                    except Exception as e:
                        pending_hb = getattr(
                            self, "_pending_heartbeats_by_engine", None
                        )
                        if isinstance(pending_hb, dict):
                            pending_hb.pop(eid, None)
                        self._log_failure(
                            failure_type="handshake_setup_failed",
                            req_id=None,
                            error=e,
                            remote_engine_id=eid,
                        )

            fut.add_done_callback(done_callback)
            return fut

    def _background_nixl_handshake(self, req_id: str, remote_engine_id, meta):
        assert meta.remote is not None
        fut = self._ensure_handshake(
            remote_engine_id,
            meta.remote.host,
            meta.remote.port,
            meta.tp_size,
            getattr(meta, "remote_pp_size", 1),
            getattr(meta, "remote_pp_split", ""),
        )
        if fut is None:
            self._ready_requests.put((req_id, meta))
            return

        def request_ready(f: Future[Any], entry=(req_id, meta)):
            try:
                f.result()
                self._ready_requests.put(entry)
            except Exception as e:
                self._log_failure(
                    failure_type="handshake_failed",
                    req_id=req_id,
                    error=e,
                    meta=meta,
                )
                self._handle_failed_transfer(req_id, None)

        fut.add_done_callback(request_ready)

    def _pop_done_transfers(self, transfers: dict[str, list[int]]) -> set[str]:
        done_req_ids: set[str] = set()
        handle_info_map = getattr(self, "_handle_remote_info", None)
        for req_id, handles in list(transfers.items()):
            in_progress: list[int] = []
            for handle in handles:
                try:
                    xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                    if xfer_state == "DONE":
                        res = self.nixl_wrapper.get_xfer_telemetry(handle)
                        if hasattr(self, "xfer_stats"):
                            self.xfer_stats.record_transfer(res)
                        self.nixl_wrapper.release_xfer_handle(handle)
                        h_info = (
                            handle_info_map.pop(handle, None)
                            if isinstance(handle_info_map, dict)
                            else None
                        )
                        self._on_handle_settled_for_session(
                            handle, h_info[0] if h_info is not None else None
                        )
                    elif xfer_state == "PROC":
                        in_progress.append(handle)
                        continue
                    else:
                        self._log_failure(
                            failure_type="transfer_failed",
                            msg="Marking blocks as invalid",
                            req_id=req_id,
                            xfer_state=xfer_state,
                        )
                        h_info = (
                            handle_info_map.pop(handle, None)
                            if isinstance(handle_info_map, dict)
                            else None
                        )
                        if h_info is not None:
                            self._notify_remote_read_failed(*h_info)
                            self._on_handle_settled_for_session(handle, h_info[0])
                        self._handle_failed_transfer(req_id, handle)
                except Exception as e:
                    self._log_failure(
                        failure_type="transfer_exception",
                        msg="Marking blocks as invalid",
                        req_id=req_id,
                        error=e,
                    )
                    h_info = (
                        handle_info_map.pop(handle, None)
                        if isinstance(handle_info_map, dict)
                        else None
                    )
                    if h_info is not None:
                        self._notify_remote_read_failed(*h_info)
                        self._on_handle_settled_for_session(handle, h_info[0])
                    self._handle_failed_transfer(req_id, handle)

            if not in_progress:
                done_req_ids.add(req_id)
                del transfers[req_id]
            else:
                transfers[req_id] = in_progress
        return done_req_ids

    def _flush_pending_heartbeats_for_engine(self, engine_id: str) -> None:
        pending_map = getattr(self, "_pending_heartbeats_by_engine", None)
        if not isinstance(pending_map, dict):
            return
        req_ids = pending_map.pop(engine_id, None)
        if not req_ids or engine_id not in getattr(self, "_remote_agents", {}):
            return
        hb_msg = ("HB:" + ",".join(sorted(req_ids))).encode()
        for agent_name in self._remote_agents[engine_id].values():
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=hb_msg)
            except Exception:
                logger.debug(
                    "Failed to flush pending heartbeat to engine %s",
                    engine_id,
                    exc_info=True,
                )

    def _send_heartbeats(self, metadata) -> None:
        pending_map = getattr(self, "_pending_heartbeats_by_engine", None)
        if pending_map is None:
            pending_map = defaultdict(set)
            self._pending_heartbeats_by_engine = pending_map

        for engine_id, hb_info in metadata.heartbeat_by_engine.items():
            if not hb_info.req_ids:
                continue
            if (
                self._ensure_handshake(
                    engine_id,
                    hb_info.host,
                    hb_info.port,
                    hb_info.tp_size,
                    getattr(hb_info, "pp_size", 1),
                    getattr(hb_info, "pp_split", ""),
                )
                is not None
            ):
                pending_map[engine_id].update(hb_info.req_ids)
                continue

            combined_ids = set(hb_info.req_ids)
            queued = pending_map.pop(engine_id, None)
            if queued:
                combined_ids.update(queued)
            hb_msg = ("HB:" + ",".join(sorted(combined_ids))).encode()
            for agent_name in self._remote_agents[engine_id].values():
                try:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=hb_msg)
                except Exception:
                    logger.debug(
                        "Failed to send heartbeat to engine %s",
                        engine_id,
                        exc_info=True,
                    )

    # ------------------------------------------------------------------
    # Single-owner NIXL IO thread (Phase 1 + P0-1 + P2-3)
    # ------------------------------------------------------------------

    def _ensure_io_thread_started(self) -> None:
        if getattr(self, "_io_thread_started", False):
            return
        if not getattr(self, "_io_thread_enabled", False):
            return
        with self._io_shutdown_lock:
            if self._io_thread_started or self._io_stop_event.is_set():
                return
            t = threading.Thread(
                target=self._run_io_loop,
                daemon=True,
                name=f"ilu-nixl-io-{getattr(self, 'engine_id', 'worker')}",
            )
            self._io_thread = t
            self._io_thread_started = True
            t.start()

    def start_load_kv(self, metadata) -> None:
        if not getattr(self, "_io_thread_enabled", False):
            lease_starts = getattr(self, "_send_lease_started_at", None)
            if lease_starts is None:
                lease_starts = {}
                self._send_lease_started_at = lease_starts
            for req_id in metadata.reqs_not_processed:
                lease_starts.pop(req_id, None)
            if metadata.reqs_to_send:
                now_local = time.perf_counter()
                for req_id in metadata.reqs_to_send:
                    if (
                        req_id in self._reqs_to_process
                        or req_id in metadata.reqs_in_batch
                    ):
                        lease_starts.setdefault(req_id, now_local)
            return super().start_load_kv(metadata)
        self._ensure_io_thread_started()
        arrival_local = time.perf_counter()
        self._io_inbox.put(("step", metadata, arrival_local))
        self._io_wake_event.set()

    def _io_process_step_metadata(self, metadata, arrival_local: float) -> None:
        stats = getattr(self, "xfer_stats", None)
        if stats is not None and hasattr(stats, "record_io_queue_wait"):
            stats.record_io_queue_wait(time.perf_counter() - arrival_local)

        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids, self._physical_blocks_per_logical_kv_block
            )
            assert meta.remote is not None
            remote_engine_id = meta.remote.engine_id
            self._recving_metadata[req_id] = meta

            if remote_engine_id not in self._remote_agents:
                fut = self._ensure_handshake_on_io(
                    remote_engine_id,
                    meta.remote.host,
                    meta.remote.port,
                    meta.tp_size,
                    getattr(meta, "remote_pp_size", 1),
                    getattr(meta, "remote_pp_split", ""),
                )
                if fut is not None:
                    self._io_pending_reqs_by_engine[remote_engine_id].append(
                        (req_id, meta)
                    )
                    continue

            self._read_blocks_for_req(req_id, meta)

        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)

        lease_starts = getattr(self, "_send_lease_started_at", None)
        if lease_starts is None:
            lease_starts = {}
            self._send_lease_started_at = lease_starts

        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            self._reqs_to_send.pop(req_id, None)
            lease_starts.pop(req_id, None)

        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                if metadata.scheduler_clock:
                    expiration_time = arrival_local + (
                        expiration_time - metadata.scheduler_clock
                    )
                self._reqs_to_send[req_id] = expiration_time
                lease_starts.setdefault(req_id, arrival_local)

        self._io_send_heartbeats(metadata)

    def _ensure_handshake_on_io(
        self,
        engine_id: str,
        host: str,
        port: int,
        tp_size: int,
        pp_size: int = 1,
        pp_split: str = "",
    ) -> Future[Any] | None:
        self._evict_stale_engines()
        if engine_id in self._remote_agents:
            return None
        fut = self._handshake_futures.get(engine_id)
        if fut is not None:
            return fut

        hs_start = time.perf_counter()
        targets = self._compute_handshake_targets(tp_size, pp_size, pp_split)
        payload_cache = getattr(self, "_handshake_payload_cache", {})
        missing_ranks = [
            g_r
            for _pp_r, _tp_r, g_r in targets
            if (engine_id, port, g_r) not in payload_cache
        ]
        if not missing_ranks:
            self._complete_io_handshake(
                engine_id,
                host,
                port,
                tp_size,
                pp_size,
                pp_split,
                targets,
                {},
                None,
                hs_start,
            )
            return None

        fut = self._handshake_initiation_executor.submit(
            _query_zmq_payloads_with_clock, host, port, missing_ranks
        )
        self._handshake_futures[engine_id] = fut

        def _on_zmq_done(f: Future[tuple[dict[int, bytes], float | None]]) -> None:
            self._io_inbox.put(
                (
                    "handshake_fetched",
                    engine_id,
                    host,
                    port,
                    tp_size,
                    pp_size,
                    pp_split,
                    targets,
                    f,
                    hs_start,
                )
            )
            self._io_wake_event.set()

        fut.add_done_callback(_on_zmq_done)
        return fut

    def _complete_io_handshake(
        self,
        engine_id: str,
        host: str,
        port: int,
        remote_tp_size: int,
        remote_pp_size: int,
        remote_pp_split: str,
        targets: list[tuple[int, int, int]],
        fetched: dict[int, bytes],
        clock_offset: float | None,
        hs_start: float,
    ) -> None:
        agents = self._decode_and_install_targets(
            engine_id=engine_id,
            host=host,
            port=port,
            remote_tp_size=remote_tp_size,
            remote_pp_size=remote_pp_size,
            remote_pp_split=remote_pp_split,
            targets=targets,
            fetched=fetched,
            clock_offset=clock_offset,
        )
        self._remote_agents[engine_id] = agents
        self._engine_last_active[engine_id] = time.perf_counter()
        self._register_active_session(engine_id, host, port, agents)
        self._flush_pending_heartbeats_for_engine(engine_id)

        stats = getattr(self, "xfer_stats", None)
        if stats is not None and hasattr(stats, "record_handshake_duration"):
            stats.record_handshake_duration(time.perf_counter() - hs_start)

        queued_reqs = self._io_pending_reqs_by_engine.pop(engine_id, ())
        for req_id, req_meta in queued_reqs:
            self._read_blocks_for_req(req_id, req_meta)

    def _io_handle_handshake_fetched(
        self,
        engine_id: str,
        host: str,
        port: int,
        tp_size: int,
        pp_size: int,
        pp_split: str,
        targets: list[tuple[int, int, int]],
        fut: Future[tuple[dict[int, bytes], float | None]],
        hs_start: float,
    ) -> None:
        self._handshake_futures.pop(engine_id, None)
        try:
            fetched, clock_offset = fut.result()
            self._complete_io_handshake(
                engine_id,
                host,
                port,
                tp_size,
                pp_size,
                pp_split,
                targets,
                fetched,
                clock_offset,
                hs_start,
            )
        except Exception as e:
            self._pending_heartbeats_by_engine.pop(engine_id, None)
            queued_reqs = self._io_pending_reqs_by_engine.pop(engine_id, ())
            if not queued_reqs:
                self._log_failure(
                    failure_type="handshake_setup_failed",
                    req_id=None,
                    error=e,
                    remote_engine_id=engine_id,
                )
            for req_id, req_meta in queued_reqs:
                self._log_failure(
                    failure_type="handshake_failed",
                    req_id=req_id,
                    error=e,
                    meta=req_meta,
                )
                self._handle_failed_transfer(req_id, None)

    def _io_send_heartbeats(self, metadata) -> None:
        for engine_id, hb_info in metadata.heartbeat_by_engine.items():
            if not hb_info.req_ids:
                continue
            if (
                self._ensure_handshake_on_io(
                    engine_id,
                    hb_info.host,
                    hb_info.port,
                    hb_info.tp_size,
                    getattr(hb_info, "pp_size", 1),
                    getattr(hb_info, "pp_split", ""),
                )
                is not None
            ):
                self._pending_heartbeats_by_engine[engine_id].update(hb_info.req_ids)
                continue

            combined_ids = set(hb_info.req_ids)
            queued = self._pending_heartbeats_by_engine.pop(engine_id, None)
            if queued:
                combined_ids.update(queued)
            hb_msg = ("HB:" + ",".join(sorted(combined_ids))).encode()
            for agent_name in self._remote_agents[engine_id].values():
                try:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=hb_msg)
                except Exception:
                    logger.debug(
                        "Failed to send heartbeat to engine %s",
                        engine_id,
                        exc_info=True,
                    )

    def _io_progress_once(self) -> bool:
        if self.transfer_topo is None:
            return False

        done_sending = self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)

        failed_recv_reqs: set[str] = set()
        while not self._failed_recv_reqs.empty():
            try:
                failed_recv_reqs.add(self._failed_recv_reqs.get_nowait())
            except queue.Empty:
                break
        recv_failures_set = getattr(self, "_recv_failures", None)
        if isinstance(recv_failures_set, set) and recv_failures_set:
            failed_recv_reqs.update(recv_failures_set)
            recv_failures_set.clear()

        done_recving.update(failed_recv_reqs)

        recv_meta_batch: dict[str, Any] = {}
        for req_id in done_recving:
            meta = self._recving_metadata.pop(req_id, None)
            if meta is not None:
                recv_meta_batch[req_id] = meta

        now = time.perf_counter()
        while self._reqs_to_send:
            req_id, expires = next(iter(self._reqs_to_send.items()))
            if now < expires:
                break
            count = self.consumer_notification_counts_by_req.pop(req_id, 0)
            if hasattr(self, "expected_consumer_notifications_by_req"):
                self.expected_consumer_notifications_by_req.pop(req_id, None)
            if hasattr(self, "_notif_seen_senders_by_req"):
                self._notif_seen_senders_by_req.pop(req_id, None)
            if hasattr(self, "xfer_stats"):
                self.xfer_stats.record_kv_expired_req()
            logger.warning(
                "Releasing expired KV blocks for request %s which were "
                "retrieved by %d remote worker(s) before lease expired.",
                req_id,
                count,
            )
            self._reqs_to_process.discard(req_id)
            del self._reqs_to_send[req_id]
            done_sending.add(req_id)

        if done_sending:
            lease_starts = getattr(self, "_send_lease_started_at", None)
            stats = getattr(self, "xfer_stats", None)
            if isinstance(lease_starts, dict):
                for req_id in done_sending:
                    started = lease_starts.pop(req_id, None)
                    if (
                        started is not None
                        and stats is not None
                        and hasattr(stats, "record_lease_hold_duration")
                    ):
                        stats.record_lease_hold_duration(now - started)

        self._progress_draining_sessions(now)

        if done_sending or done_recving or failed_recv_reqs:
            self._io_completion_outbox.put(
                _IluCompletionBatch(
                    done_sending=done_sending,
                    done_recving=done_recving,
                    failed_recving=failed_recv_reqs,
                    recv_metadata=recv_meta_batch,
                    enqueued_at=now,
                )
            )

        return bool(
            self._recving_transfers
            or self._reqs_to_send
            or self._handshake_futures
            or getattr(self, "_draining_sessions", None)
        )

    def _run_io_loop(self) -> None:
        if not getattr(self, "use_host_buffer", True):
            with suppress(Exception):
                current_platform.set_device(self.device_id)

        while not self._io_stop_event.is_set():
            drained_any = False
            while True:
                try:
                    cmd = self._io_inbox.get_nowait()
                except queue.Empty:
                    break
                drained_any = True
                kind = cmd[0]
                if kind == "stop":
                    return
                if kind == "step":
                    _, metadata, arrival_local = cmd
                    try:
                        self._io_process_step_metadata(metadata, arrival_local)
                    except Exception:
                        logger.exception("IluNixl IO thread error in step metadata")
                elif kind == "handshake_fetched":
                    (
                        _,
                        engine_id,
                        host,
                        port,
                        tp_size,
                        pp_size,
                        pp_split,
                        targets,
                        fut,
                        hs_start,
                    ) = cmd
                    self._io_handle_handshake_fetched(
                        engine_id,
                        host,
                        port,
                        tp_size,
                        pp_size,
                        pp_split,
                        targets,
                        fut,
                        hs_start,
                    )

            try:
                has_active_work = self._io_progress_once()
            except Exception:
                logger.exception("IluNixl IO thread error during NIXL progress")
                has_active_work = False

            if self._io_stop_event.is_set():
                break

            timeout = (
                self._io_poll_interval_s
                if (has_active_work or drained_any)
                else self._io_idle_timeout_s
            )
            self._io_wake_event.wait(timeout=timeout)
            self._io_wake_event.clear()

    def _run_main_thread_recv_postprocess(
        self,
        done_recving: set[str],
        failed_recv_reqs: set[str],
        recv_metadata: dict[str, Any],
    ) -> None:
        if not done_recving:
            return
        block_ids_for_blocksize_post_process: dict[
            int, list[tuple[list[int], int]]
        ] = defaultdict(list)
        block_ids_for_heterogeneous_attn_post_process: list[list[int]] = []

        for req_id in done_recving:
            meta = recv_metadata.get(req_id)
            if meta is None or req_id in failed_recv_reqs:
                if req_id in failed_recv_reqs:
                    logger.warning(
                        "Skipping KV post-processing for failed request %s", req_id
                    )
                continue

            assert meta.remote is not None
            if self.use_host_buffer:
                self.sync_recved_kv_to_device(req_id, meta)

            remote_info = self.transfer_topo.get_engine_info(meta.remote.engine_id)
            block_size_ratio = self.transfer_topo.block_size_ratio(
                remote_info.remote_block_size
            )
            hetero_ppl = (
                remote_info.remote_physical_blocks_per_logical
                != self._physical_blocks_per_logical_kv_block
            )
            if (
                (block_size_ratio > 1 or self.enable_permute_local_kv or hetero_ppl)
                and getattr(self, "dsv41_layout", None) is None
                and not getattr(self, "_virtual_packed_regions", False)
            ):
                for g, local_group in enumerate(meta.local_physical_block_ids):
                    if not local_group or (
                        g < len(self._group_spec_types)
                        and _is_ssm_spec(self._group_spec_types[g])
                    ):
                        continue
                    remote_grp = (
                        meta.remote.block_ids[g]
                        if g < len(meta.remote.block_ids)
                        else local_group
                    )
                    remote_starts = getattr(meta, "remote_block_starts", None)
                    start_blk = (
                        remote_starts[g]
                        if remote_starts is not None and g < len(remote_starts)
                        else -1
                    )
                    remote_page_offset = (
                        start_blk
                        * self._physical_blocks_per_logical_kv_block
                        * block_size_ratio
                        if start_blk is not None and start_blk > 0
                        else 0
                    )
                    avail_remote = max(0, len(remote_grp) - remote_page_offset)
                    covered_sub_blocks = min(
                        len(local_group) * block_size_ratio,
                        avail_remote,
                    )
                    block_ids_for_blocksize_post_process[block_size_ratio].append(
                        (local_group, covered_sub_blocks)
                    )
            if self.enable_heterogeneous_attn_post_process:
                block_ids_for_heterogeneous_attn_post_process.append(
                    meta.local_physical_block_ids[0]
                )

        for (
            block_size_ratio,
            block_ids_list,
        ) in block_ids_for_blocksize_post_process.items():
            convert = not self.use_mla and (
                block_size_ratio > 1 or self.enable_permute_local_kv
            )
            self.post_process_device_kv_on_receive(
                block_size_ratio, block_ids_list, convert
            )

        for block_ids in block_ids_for_heterogeneous_attn_post_process:
            self.post_process_device_kv_on_receive_heterogeneous_attn(block_ids)

        if hasattr(self, "_sync_device_after_mamba_recv"):
            self._sync_device_after_mamba_recv(done_recving, failed_recv_reqs)

    def _drain_io_transfer_results(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorTransferResults,
        )

        merged = _IluCompletionBatch()
        now = time.perf_counter()
        stats = getattr(self, "xfer_stats", None)
        while True:
            try:
                batch = self._io_completion_outbox.get_nowait()
            except queue.Empty:
                break
            if (
                batch.enqueued_at > 0
                and stats is not None
                and hasattr(stats, "record_wake_duration")
            ):
                stats.record_wake_duration(now - batch.enqueued_at)
            merged.merge(batch)

        if merged.done_recving:
            self._run_main_thread_recv_postprocess(
                merged.done_recving,
                merged.failed_recving,
                merged.recv_metadata,
            )

        return KVConnectorTransferResults(
            finished_sending=merged.done_sending,
            finished_recving=merged.done_recving,
            failed_recving=merged.failed_recving,
        )

    def shutdown(self) -> None:
        io_thread = getattr(self, "_io_thread", None)
        if io_thread is not None:
            self._io_stop_event.set()
            self._io_inbox.put(("stop",))
            self._io_wake_event.set()
            if threading.current_thread() is not io_thread:
                io_thread.join(timeout=5.0)
            self._io_thread = None
            self._io_thread_started = False
        for session in list(getattr(self, "_draining_sessions", ()) or ()):
            self._release_session_nixl_resources(session)
        if hasattr(self, "_draining_sessions"):
            self._draining_sessions.clear()
        for session in list(getattr(self, "_active_sessions", {}).values()):
            self._drop_engine_routing_tables(session.engine_id)
            self._release_session_nixl_resources(session)
        if hasattr(self, "_active_sessions"):
            self._active_sessions.clear()
        super().shutdown()


