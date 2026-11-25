# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-phase Single-Owner NIXL handshake, leader PUSH_REG election, and peer replacement."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import msgspec
import zmq
from vllm.distributed.kv_transfer.kv_connector.utils import EngineTransferInfo
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    PUSH_REG_NOTIF_PREFIX,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path

from ..ilu_nixl_common.dsv41 import validate_wire_format_tags, wire_dsv41_layout
from ..ilu_nixl_common.semantic_pairing import pp_layer_range
from ..ilu_nixl_common.types import encode_push_reg_wire
from .metadata import (
    IluNixlPushAgentMetadata,
    IluNixlPushHandshakePayload,
)

logger = init_logger(__name__)


@dataclass
class _HandshakeInstallTask:
    """Phase-2 NIXL agent/dlist installation executed exclusively on ``nixl-push-writer``."""

    engine_id: str
    host: str
    port: int
    remote_tp_size: int
    remote_pp_size: int
    remote_pp_split: str
    notif_agents_only: bool
    fetched_ranks: list[tuple[int, int, int, IluNixlPushAgentMetadata]]
    done_future: Future[dict[int, str]]


class HandshakeMixin:
    """Two-phase Single-Owner handshake, leader-only PUSH_REG/HB, and peer eviction."""

    def _writer_is_live(self) -> bool:
        t = getattr(self, "_push_writer_thread", None)
        return isinstance(t, threading.Thread) and t.is_alive()

    # --- P0-2 Leader Election for D -> P Control Plane (PUSH_REG & HB) - #

    def _is_tp_leader_for_remote(
        self, remote_tp_size: int, remote_dcp_size: int = 1
    ) -> bool:
        """Return True iff this D TP rank is the unique sender for its target P TP slice."""
        if remote_dcp_size > 1 or self.world_size <= remote_tp_size:
            return True
        tp_ratio = self.world_size // remote_tp_size
        return (self.tp_rank % tp_ratio) == 0

    def _is_pp_leader_for_remote_stage(
        self, remote_pp_rank: int, remote_pp_split: str
    ) -> bool:
        """When a P PP stage straddles multiple D PP stages, only the first D stage sends."""
        if getattr(self, "pp_world_size", 1) <= 1 or not self.pp_split:
            return True
        if not remote_pp_split:
            return self.pp_rank == 0
        p_start, p_end = pp_layer_range(remote_pp_split, remote_pp_rank)
        d_counts = [int(x) for x in self.pp_split.split(",")]
        d_start = 0
        for d_pp_idx, count in enumerate(d_counts):
            d_end = d_start + count
            if max(p_start, d_start) < min(p_end, d_end):
                return self.pp_rank == d_pp_idx
            d_start = d_end
        return True

    # --- D-side PUSH_REG & Heartbeat (Single-Owner Writer Thread) ------ #

    def _send_registration_to_p(
        self,
        req_id: str,
        reg_data: dict[str, Any],
    ) -> None:
        remote_tp_size = int(reg_data["remote_tp_size"])
        remote_dcp_size = int(reg_data.get("remote_dcp_size", 1))
        if not self._is_tp_leader_for_remote(remote_tp_size, remote_dcp_size):
            self._push_obs_stats().skipped_redundant_push_regs += 1
            return

        remote_engine_id = reg_data["remote_engine_id"]
        remote_pp_size = int(reg_data.get("remote_pp_size", 1))
        remote_pp_split = reg_data.get("remote_pp_split", "") or ""
        fut = self._ensure_handshake(
            remote_engine_id,
            reg_data["remote_host"],
            int(reg_data["remote_port"]),
            remote_tp_size,
            remote_pp_size,
            remote_pp_split,
            notif_agents_only=True,
        )
        if fut is None:
            self._do_send_reg_notif(req_id, reg_data)
            return

        def _on_handshake(f, rid: str = req_id, rd: dict[str, Any] = reg_data):
            try:
                f.result()
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_handshake_failed",
                    req_id=rid,
                    error=e,
                )
                self._failed_recv_reqs.put(rid)
                return
            self._reg_send_inbox.put((rid, rd))
            self._push_writer_wake.set()

        fut.add_done_callback(_on_handshake)

    def _do_send_reg_notif(self, req_id: str, reg_data: dict[str, Any]) -> None:
        engine_id = reg_data["remote_engine_id"]
        remote_tp_size = int(reg_data["remote_tp_size"])
        remote_pp_split = reg_data.get("remote_pp_split", "") or ""
        notif_msg = PUSH_REG_NOTIF_PREFIX + encode_push_reg_wire(reg_data)
        with self._handshake_lock:
            agents = dict(self._remote_agents.get(engine_id) or {})
        if not agents:
            logger.error(
                "No remote agents for engine %s; cannot send registration for %s",
                engine_id,
                req_id,
            )
            self.xfer_stats.record_failed_notification()
            self._failed_recv_reqs.put(req_id)
            return
        for remote_key, agent_name in agents.items():
            remote_pp_rank = remote_key // max(remote_tp_size, 1)
            if not self._is_pp_leader_for_remote_stage(remote_pp_rank, remote_pp_split):
                self._push_obs_stats().skipped_redundant_push_regs += 1
                continue
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_msg)
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_notif_failed",
                    req_id=req_id,
                    error=e,
                    remote_rank=remote_key,
                )
                self.xfer_stats.record_failed_notification()

    def _send_heartbeats(self, metadata) -> None:
        """P1-2 Single-Owner: enqueue heartbeats for ``nixl-push-writer`` (zero main-thread NIXL calls)."""
        hb_map = getattr(metadata, "heartbeat_by_engine", None)
        if not hb_map:
            return
        if not self._writer_is_live():
            for engine_id, hb_info in hb_map.items():
                self._writer_send_heartbeat(engine_id, hb_info)
            return
        for engine_id, hb_info in hb_map.items():
            if not self._is_tp_leader_for_remote(
                int(hb_info.tp_size), int(getattr(hb_info, "dcp_size", 1))
            ):
                continue
            self._hb_send_inbox.put((engine_id, hb_info))
        self._push_writer_wake.set()

    def _writer_send_heartbeat(self, engine_id: str, hb_info: Any) -> None:
        if not hb_info.req_ids:
            return
        if not self._is_tp_leader_for_remote(
            int(hb_info.tp_size), int(getattr(hb_info, "dcp_size", 1))
        ):
            return
        fut = self._ensure_handshake(
            engine_id,
            hb_info.host,
            int(hb_info.port),
            int(hb_info.tp_size),
            int(getattr(hb_info, "pp_size", 1)),
            getattr(hb_info, "pp_split", "") or "",
            notif_agents_only=True,
        )
        if fut is not None:
            # Re-deliver first heartbeat immediately upon handshake completion
            # instead of dropping it until the next 5s heartbeat tick!
            def _on_hb_hs(f, eid: str = engine_id, info: Any = hb_info) -> None:
                if f.exception() is not None:
                    return
                if hasattr(self, "_hb_send_inbox") and self._writer_is_live():
                    self._hb_send_inbox.put((eid, info))
                    self._push_writer_wake.set()

            fut.add_done_callback(_on_hb_hs)
            return

        hb_msg = ("HB:" + ",".join(hb_info.req_ids)).encode()
        remote_tp_size = int(hb_info.tp_size)
        remote_pp_split = getattr(hb_info, "pp_split", "") or ""
        with self._handshake_lock:
            agents = dict(self._remote_agents.get(engine_id) or {})
        for remote_key, agent_name in agents.items():
            remote_pp_rank = remote_key // max(remote_tp_size, 1)
            if not self._is_pp_leader_for_remote_stage(remote_pp_rank, remote_pp_split):
                continue
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=hb_msg)
            except Exception:
                logger.debug(
                    "Failed to send heartbeat to engine %s",
                    engine_id,
                    exc_info=True,
                )

    def _send_pending_recv_notifs(self, req_id: str) -> None:
        """Route any deferred recv notifications through ``nixl-push-writer``."""
        pending = self._pending_recv_notifs.pop(req_id, None)
        if not pending:
            return
        if self._writer_is_live() and hasattr(self, "_raw_notif_send_inbox"):
            for agent_name, notif_id in pending:
                self._raw_notif_send_inbox.put((agent_name, notif_id))
            self._push_writer_wake.set()
            return
        for agent_name, notif_id in pending:
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    req_id=req_id,
                    error=e,
                    remote_agent_name=agent_name,
                )
                self.xfer_stats.record_failed_notification()

    # --- Two-Phase Handshake (ZMQ on Initiator -> NIXL Install on Writer) #

    def _fetch_and_validate_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
        remote_pp_size: int = 1,
        *,
        remote_pp_split: str = "",
    ) -> list[tuple[int, int, int, IluNixlPushAgentMetadata]]:
        """Phase 1 (pure ZMQ + msgpack + validation): zero ``nixl_wrapper`` C++ calls."""
        if not self.use_host_buffer:
            current_platform.set_device(self._worker_device_id)

        assert self.transfer_topo is not None
        remote_tp_ranks = self.transfer_topo.handshake_target_ranks(remote_tp_size)
        if remote_pp_size == 1:
            remote_pp_ranks = [0]
        elif remote_pp_split:
            remote_pp_ranks = self._target_pp_ranks(
                self.pp_split,
                self.pp_rank,
                remote_pp_split,
                getattr(self, "layer_region_ids", None),
            )
        elif remote_pp_size == self.pp_world_size:
            remote_pp_ranks = [self.pp_rank]
        else:
            raise RuntimeError(
                "IluNixlPush heterogeneous PP handshake requires "
                "the remote pp_split metadata."
            )

        fetched: list[tuple[int, int, int, IluNixlPushAgentMetadata]] = []
        path = make_zmq_path("tcp", host, port)
        handshake_decoder = msgspec.msgpack.Decoder(IluNixlPushHandshakePayload)
        metadata_decoder = msgspec.msgpack.Decoder(IluNixlPushAgentMetadata)

        with zmq_ctx(zmq.REQ, path) as sock:
            for remote_pp_rank in remote_pp_ranks:
                for remote_tp_rank in remote_tp_ranks:
                    remote_key = remote_pp_rank * remote_tp_size + remote_tp_rank
                    if self._handshake_uses_pp_tuple_protocol:
                        msg = msgspec.msgpack.encode(
                            (GET_META_MSG, remote_pp_rank, remote_tp_rank)
                        )
                    else:
                        msg = msgspec.msgpack.encode((GET_META_MSG, remote_key))
                    sock.setsockopt(zmq.RCVTIMEO, 5000)
                    sock.send(msg)
                    if self._handshake_uses_pp_tuple_protocol:
                        reply_parts = sock.recv_multipart()
                        if len(reply_parts) != 2:
                            raise RuntimeError(
                                "Expected NIXL metadata and clock-offset reply, "
                                f"received {len(reply_parts)} frame(s)."
                            )
                        handshake_bytes = reply_parts[0]
                        msgspec.msgpack.decode(reply_parts[1], type=float)
                    else:
                        handshake_bytes = sock.recv()

                    try:
                        handshake_payload = handshake_decoder.decode(handshake_bytes)
                    except (msgspec.DecodeError, msgspec.ValidationError) as e:
                        raise RuntimeError(
                            "Failed to decode IluNixlPushHandshakePayload. This "
                            "likely indicates an incompatible connector version. "
                            f"Error: {e}"
                        ) from e

                    requires_compat = validate_wire_format_tags(
                        wire_dsv41_layout(
                            getattr(self, "dsv41_layout", None),
                            host_staging_producer=getattr(
                                self, "_host_staging_producer", False
                            ),
                            hetero_enabled=bool(
                                getattr(
                                    getattr(self, "_hetero_cfg", None),
                                    "enabled",
                                    False,
                                )
                            ),
                        ),
                        remote_engine_id=expected_engine_id,
                        remote_main_kv_format=handshake_payload.main_kv_format,
                        remote_indexer_kv_layout=handshake_payload.indexer_kv_layout,
                    )

                    assert self.compat_hash is not None
                    heterogeneous_pp_hma = (
                        getattr(self, "_is_hma_required", False)
                        and bool(remote_pp_split)
                        and remote_pp_split != self.pp_split
                    ) or self._hetero_cfg.enabled
                    if (
                        (self.enforce_compat_hash or requires_compat)
                        and handshake_payload.compatibility_hash != self.compat_hash
                        and (not heterogeneous_pp_hma or requires_compat)
                    ):
                        raise RuntimeError(
                            "NIXL compatibility hash mismatch. "
                            f"Local: {self.compat_hash}, "
                            f"Remote: {handshake_payload.compatibility_hash}."
                        )

                    try:
                        metadata = metadata_decoder.decode(
                            handshake_payload.agent_metadata_bytes
                        )
                    except (msgspec.DecodeError, msgspec.ValidationError) as e:
                        raise RuntimeError(
                            f"Failed to decode IluNixlPushAgentMetadata. Error: {e}"
                        ) from e

                    if metadata.main_kv_format != handshake_payload.main_kv_format:
                        raise RuntimeError(
                            "Remote NIXL main KV format tag disagrees with "
                            f"agent metadata for engine {expected_engine_id}: "
                            f"envelope={handshake_payload.main_kv_format}, "
                            f"agent={metadata.main_kv_format}"
                        )
                    if metadata.engine_id != expected_engine_id:
                        raise RuntimeError(
                            "Remote NIXL agent engine ID mismatch. "
                            f"Expected {expected_engine_id}, "
                            f"received {metadata.engine_id}."
                        )
                    if hasattr(self, "_validate_remote_parallel_config"):
                        self._validate_remote_parallel_config(metadata)
                    self._validate_wire_compatibility(metadata)
                    fetched.append(
                        (remote_key, remote_pp_rank, remote_tp_rank, metadata)
                    )
        return fetched

    def _install_fetched_handshake(
        self,
        *,
        engine_id: str,
        host: str,
        port: int,
        remote_tp_size: int,
        remote_pp_size: int,
        notif_agents_only: bool,
        fetched_ranks: list[tuple[int, int, int, IluNixlPushAgentMetadata]],
    ) -> dict[int, str]:
        """Phase 2: register NIXL remote agents & prepped dlists (runs on writer when live)."""
        remote_rank_to_agent_name: dict[int, str] = {}
        for (
            remote_key,
            remote_pp_rank,
            remote_tp_rank,
            metadata,
        ) in fetched_ranks:
            if notif_agents_only:
                remote_agent_name = self._add_notif_only_remote_agent(
                    metadata,
                    remote_tp_size,
                    getattr(metadata, "dcp_size", 1),
                )
            else:
                remote_agent_name = self.add_remote_agent(
                    metadata,
                    remote_tp_rank,
                    remote_tp_size,
                    remote_pp_rank,
                    remote_pp_size,
                )
            remote_rank_to_agent_name[remote_key] = remote_agent_name

        with self._handshake_lock:
            self._remote_agents[engine_id] = remote_rank_to_agent_name
            self._engine_last_active[engine_id] = time.perf_counter()
            self._track_remote_engine_replacement(
                engine_id,
                host,
                port,
                notif_agents_only=notif_agents_only,
            )
        return remote_rank_to_agent_name

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
        remote_pp_size: int = 1,
        notif_agents_only: bool = False,
        *,
        remote_pp_split: str = "",
    ) -> dict[int, str]:
        fetched = self._fetch_and_validate_handshake(
            host,
            port,
            remote_tp_size,
            expected_engine_id,
            remote_pp_size=remote_pp_size,
            remote_pp_split=remote_pp_split,
        )
        if self._writer_is_live() and hasattr(self, "_handshake_install_inbox"):
            done_fut: Future[dict[int, str]] = Future()
            self._handshake_install_inbox.put(
                _HandshakeInstallTask(
                    engine_id=expected_engine_id,
                    host=host,
                    port=port,
                    remote_tp_size=remote_tp_size,
                    remote_pp_size=remote_pp_size,
                    remote_pp_split=remote_pp_split,
                    notif_agents_only=notif_agents_only,
                    fetched_ranks=fetched,
                    done_future=done_fut,
                )
            )
            self._push_writer_wake.set()
            return done_fut.result()

        return self._install_fetched_handshake(
            engine_id=expected_engine_id,
            host=host,
            port=port,
            remote_tp_size=remote_tp_size,
            remote_pp_size=remote_pp_size,
            notif_agents_only=notif_agents_only,
            fetched_ranks=fetched,
        )

    def _complete_handshake(
        self,
        engine_id: str,
        host: str,
        port: int,
        tp_size: int,
        pp_size: int,
        pp_split: str,
        notif_agents_only: bool,
    ) -> dict[int, str]:
        return self._nixl_handshake(
            host,
            port,
            tp_size,
            engine_id,
            remote_pp_size=pp_size,
            remote_pp_split=pp_split,
            notif_agents_only=notif_agents_only,
        )

    def _ensure_handshake(
        self,
        engine_id: str,
        host: str,
        port: int,
        tp_size: int,
        pp_size: int = 1,
        pp_split: str = "",
        notif_agents_only: bool = False,
    ):
        self._evict_stale_engines()
        with self._handshake_lock:
            if engine_id in self._remote_agents:
                return None
            fut = self._handshake_futures.get(engine_id)
            if fut is not None:
                return fut
            fut = self._handshake_initiation_executor.submit(
                self._complete_handshake,
                engine_id,
                host,
                port,
                tp_size,
                pp_size,
                pp_split,
                notif_agents_only,
            )
            self._handshake_futures[engine_id] = fut

            def done_callback(f, eid=engine_id):
                with self._handshake_lock:
                    self._handshake_futures.pop(eid, None)
                if (error := f.exception()) is not None:
                    self._log_failure(
                        failure_type="handshake_setup_failed",
                        req_id=None,
                        error=error,
                        remote_engine_id=eid,
                    )
                    if hasattr(self.xfer_stats, "record_failed_handshake"):
                        self.xfer_stats.record_failed_handshake()

            fut.add_done_callback(done_callback)
            return fut

    def _add_notif_only_remote_agent(
        self,
        metadata: IluNixlPushAgentMetadata,
        remote_tp_size: int,
        remote_dcp_size: int = 1,
    ) -> str:
        assert self.transfer_topo is not None
        remote_block_len = (
            metadata.block_lens[0]
            if metadata.block_lens
            else (self.block_len_per_layer[0] if self.block_len_per_layer else 0)
        )
        self.transfer_topo.register_remote_engine(
            metadata.engine_id,
            EngineTransferInfo(
                remote_tp_size=remote_tp_size,
                remote_block_size=metadata.block_size,
                remote_block_len=remote_block_len,
                remote_physical_blocks_per_logical=(
                    metadata.physical_blocks_per_logical_kv_block
                ),
                remote_dcp_size=remote_dcp_size,
            ),
        )
        return self.nixl_wrapper.add_remote_agent(metadata.agent_metadata)

    def _validate_wire_compatibility(
        self, nixl_agent_meta: IluNixlPushAgentMetadata
    ) -> None:
        if not self._hetero_cfg.enabled:
            return
        expected_dtype = self._hetero_cfg.decode_cache_dtype
        if nixl_agent_meta.wire_cache_dtype != expected_dtype:
            raise RuntimeError(
                "Heterogeneous dtype wire mismatch: host pool is laid out for "
                f"{expected_dtype!r} but remote {nixl_agent_meta.engine_id} "
                f"reports {nixl_agent_meta.wire_cache_dtype!r}"
            )

    # --- Peer Replacement & Single-Owner Engine Teardown --------------- #

    def _track_remote_engine_replacement(
        self,
        engine_id: str,
        host: str,
        port: int,
        *,
        notif_agents_only: bool = False,
    ) -> None:
        if not hasattr(self, "_push_engine_by_address"):
            self._push_engine_by_address = {}
        key = (host, int(port), bool(notif_agents_only))
        previous = self._push_engine_by_address.get(key)
        self._push_engine_by_address[key] = engine_id
        if hasattr(self, "_engine_by_address"):
            self._engine_by_address[(host, int(port))] = engine_id
        if previous is None or previous == engine_id:
            return
        logger.info(
            "Remote push peer at %s:%d (notif_only=%s) changed from %s to %s.",
            host,
            int(port),
            notif_agents_only,
            previous,
            engine_id,
        )

    def _engines_with_inflight_transfers(self) -> set[str]:
        busy: set[str] = {
            meta.remote.engine_id
            for meta in getattr(self, "_recving_metadata", {}).values()
            if meta.remote is not None
        }
        reg_inbox = getattr(self, "_reg_send_inbox", None)
        def_inbox = getattr(self, "_deferred_push_inbox", None)
        if (reg_inbox is not None and not reg_inbox.empty()) or (
            def_inbox is not None and not def_inbox.empty()
        ):
            with self._handshake_lock:
                busy.update(self._remote_agents.keys())
            return busy

        send_lock = getattr(self, "_sending_transfers_lock", None)
        if send_lock is not None:
            with send_lock:
                for eid in getattr(self, "_inflight_send_engines", {}).values():
                    if eid:
                        busy.add(eid)

        for reg in list(getattr(self, "_pending_d_registrations", {}).values()):
            if isinstance(reg, dict) and (eid := reg.get("decode_engine_id")):
                busy.add(eid)

        chunk_lock = getattr(self, "_chunk_lock", None)
        if chunk_lock is not None:
            with chunk_lock:
                for cm in self._chunk_pending.values():
                    if cm.decode_engine_id:
                        busy.add(cm.decode_engine_id)
                for plan in self._chunk_plans.values():
                    if plan.decode_engine_id:
                        busy.add(plan.decode_engine_id)
                for reg in getattr(self, "_chunk_registrations", {}).values():
                    if isinstance(reg, dict) and (eid := reg.get("decode_engine_id")):
                        busy.add(eid)
        return busy

    def _schedule_or_run_engine_eviction(
        self, engine_id: str, *, log_eviction: bool
    ) -> None:
        if self._writer_is_live() and hasattr(self, "_evict_engine_inbox"):
            pending = getattr(self, "_pending_engine_evictions", None)
            if pending is not None:
                if engine_id in pending:
                    return
                pending.add(engine_id)
            self._evict_engine_inbox.put((engine_id, log_eviction))
            self._push_writer_wake.set()
            return
        self._cleanup_remote_engine(engine_id, log_eviction=log_eviction)
        if not log_eviction:
            logger.info(
                "Released NIXL state for replaced remote engine %s.",
                engine_id,
            )

    def _evict_stale_engines(self) -> None:
        """P1-2 Single-Owner: route TTL eviction onto ``nixl-push-writer`` when live."""
        ttl = getattr(self, "_engine_ttl", 0.0)
        if ttl <= 0:
            return
        now = time.perf_counter()
        busy = self._engines_with_inflight_transfers()
        for eid, last_active in list(self._engine_last_active.items()):
            if now - last_active > ttl and eid not in busy:
                self._schedule_or_run_engine_eviction(eid, log_eviction=True)

    def _cleanup_replaced_remote_engines(self) -> None:
        push_by_addr = getattr(self, "_push_engine_by_address", None)
        if not push_by_addr:
            return
        with self._handshake_lock:
            if getattr(self, "_handshake_futures", None):
                return
            replaced = set(self._remote_agents.keys()) - set(push_by_addr.values())
        if not replaced:
            return
        busy = self._engines_with_inflight_transfers()
        for engine_id in replaced - busy:
            self._schedule_or_run_engine_eviction(engine_id, log_eviction=False)
