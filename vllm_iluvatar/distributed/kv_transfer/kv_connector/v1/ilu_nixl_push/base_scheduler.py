# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side PP and DeepSeek-V4.1 extensions for the Ilu NIXL Push connector."""

import queue
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    UniformTypeKVCacheSpecs,
)

from ..ilu_nixl_common.dsv41 import (
    compute_prefix_replay_alignment,
    resolve_dsv41_layout,
    validate_decode_pp_split,
)
from .metadata import HeartbeatInfo, IluNixlPushConnectorMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

_TEARDOWN_RESULT_OK = b"teardown_complete"


def _parse_disconnect_payload(payload: object) -> tuple[str, int]:
    """Accept ``vllm-router`` v1 ``[engine_id, teardown_id]`` or a bare engine id."""
    if isinstance(payload, (list, tuple)) and payload:
        engine_id = str(payload[0])
        teardown_id = int(payload[1]) if len(payload) > 1 else 0
        return engine_id, teardown_id
    return str(payload) if payload else "", 0


class IluNixlPushBaseConnectorScheduler(NixlBaseConnectorScheduler):
    """Base scheduler propagating PP, V4.1 topology, and E25 prefix state."""

    prefix_state: Any = None

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self.dsv41_layout = resolve_dsv41_layout(vllm_config)
        validate_decode_pp_split(vllm_config, self.dsv41_layout)
        is_consumer = (
            getattr(vllm_config.kv_transfer_config, "kv_role", None)
            == "kv_consumer"
        )
        self.prefix_replay_alignment = compute_prefix_replay_alignment(
            vllm_config,
            kv_cache_config,
            is_consumer=is_consumer,
            has_mamba=self._has_mamba,
        )
        self._remote_block_starts: dict[str, list[int]] = {}
        self._init_ilu_transfer_clip_rules()

        # Optional E25 DeepSeek-V4.1 prefix-state reuse (attached when platform initialized session).
        self.prefix_state = None
        try:
            from vllm_iluvatar.v1.worker.prefill_state.configuration import (
                SESSION_KEY,
                prefix_transfer_config,
            )
            from vllm_iluvatar.v1.worker.prefill_state.scheduler_state import (
                PrefixSchedulerState,
            )

            transfer = prefix_transfer_config(vllm_config)
            extra_cfg = getattr(transfer, "kv_connector_extra_config", None)
            if isinstance(extra_cfg, dict) and SESSION_KEY in extra_cfg:
                self.prefix_state = PrefixSchedulerState.create(
                    vllm_config, self.kv_cache_config
                )
        except Exception:
            logger.debug(
                "PrefixSchedulerState unavailable; skipping E25 attach",
                exc_info=True,
            )

    def shutdown(self) -> None:
        prefix_state = getattr(self, "prefix_state", None)
        if prefix_state is not None:
            prefix_state.shutdown()
            self.prefix_state = None
        super().shutdown()

    def update_connector_output(self, connector_output) -> None:
        super().update_connector_output(connector_output)
        prefix_state = getattr(self, "prefix_state", None)
        if prefix_state is not None:
            prefix_state.completed(connector_output)

    def _init_ilu_transfer_clip_rules(self) -> None:
        """Compute per-transfer-group V4.1 clip slices while keeping group seats."""
        transfer_groups = getattr(
            self.kv_cache_config,
            "transfer_groups",
            self.kv_cache_config.kv_cache_groups,
        )
        v41_bounded_replay = bool(
            self.dsv41_layout is not None and self.dsv41_layout.swa_bounded_replay
        )
        self._ilu_group_clips: list[slice | None] = []
        for idx, group in enumerate(transfer_groups):
            spec = group.kv_cache_spec
            specs = (
                tuple(spec.kv_cache_specs.values())
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else (spec,)
            )
            is_ssm = (
                idx < len(self._ssm_spec_blocks)
                and self._ssm_spec_blocks[idx] is not None
            )
            if isinstance(spec, CircularBufferSpec):
                # Compressor ring: bounded replay rebuilds it on D; otherwise
                # a single block holds the whole open-group ring state.
                self._ilu_group_clips.append(
                    slice(0, 0) if v41_bounded_replay else slice(0, 1)
                )
            elif not is_ssm and all(
                getattr(s, "bounded_replay", False)
                or (v41_bounded_replay and not getattr(s, "prefix_cacheable", True))
                for s in specs
            ):
                # Bounded-replay SWA groups are rebuilt locally on D; keep an
                # empty list at this group's seat without dropping KpoolTailSpec.
                self._ilu_group_clips.append(slice(0, 0))
            else:
                self._ilu_group_clips.append(None)

    def _is_group_clipped(self, transfer_group_idx: int) -> bool:
        ilu_clips = getattr(self, "_ilu_group_clips", ())
        if (
            transfer_group_idx < len(ilu_clips)
            and ilu_clips[transfer_group_idx] is not None
        ):
            return True
        blocks_per_sw = getattr(self, "blocks_per_sw", ())
        if (
            transfer_group_idx < len(blocks_per_sw)
            and blocks_per_sw[transfer_group_idx]
        ):
            return True
        ssm_blocks = getattr(self, "_ssm_spec_blocks", ())
        if (
            transfer_group_idx < len(ssm_blocks)
            and ssm_blocks[transfer_group_idx] is not None
        ):
            return True
        return False

    def compute_remote_block_starts(self, blocks) -> list[int]:
        """Per D transfer group, index in P's block list of the first block D asks for.

        Under SWA bounded replay, D rounds ``ext_tokens`` down to a block
        multiple and replays the tail locally, while local prefix cache hits
        consume the head. Thus D's unhashed block list sits in the middle of
        P's full prompt list. ``-1`` marks clipped groups (SWA, CircularBuffer,
        Mamba) that keep their existing head/tail alignment.
        """
        if hasattr(blocks, "get_block_ids") and hasattr(
            blocks, "get_unhashed_block_ids_all_groups"
        ):
            full_all = blocks.get_block_ids()
            unhashed_all = blocks.get_unhashed_block_ids_all_groups()
        else:
            raw_groups = getattr(blocks, "blocks", ())
            full_all = [
                [
                    getattr(b, "block_id", i)
                    for i, b in enumerate(grp)
                    if not getattr(b, "is_null", False)
                ]
                for grp in raw_groups
            ]
            unhashed_all = [
                [
                    getattr(b, "block_id", i)
                    for i, b in enumerate(grp)
                    if getattr(b, "block_hash", None) is None
                    and not getattr(b, "is_null", False)
                ]
                for grp in raw_groups
            ]

        full_transfer = self.kv_cache_config.select_transfer_block_ids(full_all)
        unhashed_transfer = self.kv_cache_config.select_transfer_block_ids(
            unhashed_all
        )
        starts: list[int] = []
        for idx, (full_grp, unhashed_grp) in enumerate(
            zip(full_transfer, unhashed_transfer)
        ):
            if self._is_group_clipped(idx):
                starts.append(-1)
            else:
                starts.append(max(0, len(full_grp) - len(unhashed_grp)))
        return starts

    def get_exchange_clipped_blocks(
        self, block_ids: BlockIds, clip_ssm: bool = True
    ) -> BlockIds:
        clipped = super().get_exchange_clipped_blocks(
            block_ids, clip_ssm=clip_ssm
        )
        if not clipped:
            return clipped
        ilu_clips = getattr(self, "_ilu_group_clips", ())
        if not ilu_clips or not any(c is not None for c in ilu_clips):
            return clipped
        out = list(clipped)
        for idx, group_slice in enumerate(ilu_clips):
            if group_slice is not None and idx < len(out):
                out[idx] = out[idx][group_slice]
        return tuple(out)

    def on_new_request(self, request) -> None:
        # Keep in sync with vLLM v0.31.0 NixlBaseConnectorScheduler.on_new_request:
        # truncate prefill requests on P, and carry remote P dcp/pp/pp_split in
        # heartbeat metadata on D.
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_decode"):
            self._truncate_request_for_prefill(request)

        if params is None or not params.get("do_remote_prefill"):
            return

        remote_engine_id = params.get("remote_engine_id")
        remote_request_id = params.get("remote_request_id")
        host = params.get("remote_host")
        port = params.get("remote_port")
        # Remote prefill (P) topology — stock NIXL field names on decode requests.
        tp_size = params.get("tp_size")
        dcp_size = params.get("dcp_size", 1)
        pp_size = params.get("pp_size", 1)
        pp_split = params.get("pp_split", "")
        if (
            remote_engine_id is None
            or remote_request_id is None
            or host is None
            or port is None
            or tp_size is None
        ):
            return

        if remote_engine_id not in self._heartbeat_by_engine:
            self._heartbeat_by_engine[remote_engine_id] = HeartbeatInfo(
                req_ids=set(),
                host=host,
                port=int(port),
                tp_size=int(tp_size),
                dcp_size=int(dcp_size),
                pp_size=int(pp_size),
                pp_split=pp_split or "",
            )
        self._heartbeat_by_engine[remote_engine_id].req_ids.add(remote_request_id)
        self._heartbeat_req_engine[request.request_id] = (
            remote_engine_id,
            remote_request_id,
        )

    def _nixl_handshake_listener(
        self,
        encoded_data: dict[tuple[int, int], Any],
        ready_event: threading.Event,
        stop_event: threading.Event,
        host: str,
        port: int,
    ) -> None:
        """Serve PP-aware NIXL handshakes, disconnect ACKs, and E25 prefix state."""
        path = make_zmq_path("tcp", host, port)
        logger.debug("Starting push handshake listener on path: %s", path)
        get_state_tag = None
        serve_state_fn = None
        try:
            from vllm_iluvatar.v1.worker.prefill_state.source_state import (
                GET_STATE,
                serve_state,
            )

            get_state_tag = GET_STATE
            serve_state_fn = serve_state
        except Exception:
            logger.debug("E25 serve_state import skipped", exc_info=True)

        with zmq_ctx(zmq.ROUTER, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ready_event.set()
            while True:
                try:
                    identity, _, raw = sock.recv_multipart()
                except zmq.Again:
                    if stop_event.is_set():
                        break
                    continue

                try:
                    decoded = msgspec.msgpack.decode(raw)
                except Exception:
                    logger.warning(
                        "Push handshake listener failed to decode ZMQ payload",
                        exc_info=True,
                    )
                    continue

                # Optional E25 DSv4.1 prefix-state query on the shared side-channel socket.
                if (
                    serve_state_fn is not None
                    and isinstance(decoded, (list, tuple))
                    and decoded
                    and decoded[0] == get_state_tag
                ):
                    prefix_state = getattr(self, "prefix_state", None)
                    prefix_wire = (
                        prefix_state.wire
                        if prefix_state is not None and not prefix_state.consumer
                        else None
                    )
                    state_reply, state_payload = serve_state_fn(
                        prefix_wire,
                        getattr(self, "engine_id", ""),
                        decoded,
                    )
                    sock.send_multipart(
                        (
                            identity,
                            b"",
                            msgspec.msgpack.encode(state_reply),
                            state_payload,
                        )
                    )
                    continue

                # ``vllm-router --opt-nixl-proxy`` sends 2-element
                # ``["disconnect", [engine_id, teardown_id]]`` to surviving
                # peers when a registered engine drops its keepalive.
                if (
                    isinstance(decoded, (list, tuple))
                    and decoded
                    and decoded[0] in ("disconnect", b"disconnect")
                ):
                    arg = decoded[1] if len(decoded) > 1 else ""
                    dead_eid, teardown_id = _parse_disconnect_payload(arg)
                    if dead_eid:
                        sink = getattr(self, "_disconnected_peer_engines", None)
                        if sink is not None:
                            sink.put(dead_eid)
                    reply = _TEARDOWN_RESULT_OK if teardown_id else b""
                    sock.send_multipart((identity, b"", reply))
                    continue

                if not isinstance(decoded, (list, tuple)) or len(decoded) != 3:
                    logger.warning(
                        "Connection listener got unexpected message %r", decoded
                    )
                    continue

                msg_type, target_pp_rank, target_tp_rank = decoded
                if msg_type != GET_META_MSG:
                    logger.warning(
                        "Connection listener got unexpected message %s",
                        msg_type,
                    )
                    continue
                rank_payload = encoded_data.get((target_pp_rank, target_tp_rank))
                if rank_payload is None:
                    logger.warning(
                        "Connection listener has no metadata for pp=%s tp=%s",
                        target_pp_rank,
                        target_tp_rank,
                    )
                    continue
                ts = msgspec.msgpack.encode(time.perf_counter())
                sock.send_multipart((identity, b"", rank_payload, ts))

    def _resolve_prefix_state_peers(self, meta) -> dict[str, Any]:
        """Resolve producer ``host:port`` side-channel endpoints for E25 state pulls."""
        prefix_state = getattr(self, "prefix_state", None)
        if prefix_state is None:
            return {}
        peers: dict[str, Any] = {}
        heartbeat_by_engine = getattr(self, "_heartbeat_by_engine", None) or {}
        proxy_cache = getattr(self, "_proxy_meta_cache", None) or {}
        peer_node_metas = getattr(self, "peer_node_metas", None) or {}
        query_proxy = getattr(self, "_query_proxy_node_meta", None)

        for rid, rec in prefix_state.requests.items():
            r_eid = rec.get("remote_engine_id")
            if not r_eid or r_eid in peers:
                continue
            req_obj = rec.get("request")
            params = getattr(req_obj, "kv_transfer_params", None) or {}
            r_host = params.get("remote_host")
            r_port = params.get("remote_port")
            if r_host and r_port is not None and str(r_port) != "":
                peers[r_eid] = SimpleNamespace(addr=f"{r_host}:{r_port}")
                continue
            hb = heartbeat_by_engine.get(r_eid)
            if (
                hb is not None
                and getattr(hb, "host", None)
                and getattr(hb, "port", None)
            ):
                peers[r_eid] = SimpleNamespace(addr=f"{hb.host}:{hb.port}")
                continue
            peer = peer_node_metas.get(r_eid)
            if peer is not None and getattr(peer, "addr", None):
                peers[r_eid] = peer
                continue
            p_meta = proxy_cache.get(r_eid)
            if (
                p_meta is None
                and rid in meta.reqs_to_recv
                and callable(query_proxy)
            ):
                p_meta = query_proxy(r_eid)
            side_addr = (
                str(p_meta.get("side_channel_addr") or "")
                if isinstance(p_meta, dict)
                else ""
            )
            if side_addr:
                peers[r_eid] = SimpleNamespace(addr=side_addr)
        return peers

    def build_connector_meta(self, scheduler_output):
        # Keep in sync with vLLM v0.31.0 NixlBaseConnectorScheduler.build_connector_meta.
        meta = IluNixlPushConnectorMetadata()

        disconnect_q = getattr(self, "_disconnected_peer_engines", None)
        if disconnect_q is not None:
            disconnected: list[str] = []
            while True:
                try:
                    dead_eid = disconnect_q.get_nowait()
                except queue.Empty:
                    break
                if dead_eid:
                    disconnected.append(dead_eid)
                    getattr(self, "_proxy_meta_cache", {}).pop(dead_eid, None)
                    self._heartbeat_by_engine.pop(dead_eid, None)
            if disconnected:
                meta.disconnected_engines = tuple(dict.fromkeys(disconnected))

        for req_id, (
            req,
            block_ids,
            cached,
            awaiting_kvs,
        ) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            prompt_len = getattr(req, "num_prompt_tokens", None)
            if not isinstance(prompt_len, int):
                ptoks = getattr(req, "prompt_token_ids", None)
                prompt_len = len(ptoks) if ptoks is not None else -1
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                local_num_computed_blocks=cached,
                awaiting_kvs=awaiting_kvs,
                remote_block_starts=self._remote_block_starts.pop(req_id, None),
                prompt_token_len=prompt_len,
            )

        if self.use_host_buffer:
            self._build_save_meta(meta, scheduler_output)

        meta.reqs_to_send = self._reqs_need_send
        meta.scheduler_clock = time.perf_counter()
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed

        if self._heartbeat_by_engine:
            now = time.perf_counter()
            if now - self._last_heartbeat_time >= self._heartbeat_interval:
                self._last_heartbeat_time = now
                meta.heartbeat_by_engine = self._heartbeat_by_engine

        prefix_state = getattr(self, "prefix_state", None)
        if prefix_state is not None:
            peers = self._resolve_prefix_state_peers(meta)
            prefix_state.add_metadata(meta, scheduler_output, peers)

        self._reqs_need_recv.clear()
        self._remote_block_starts.clear()
        self._reqs_in_batch = set()
        self._reqs_not_processed = set()
        self._reqs_need_send = {}

        return meta
