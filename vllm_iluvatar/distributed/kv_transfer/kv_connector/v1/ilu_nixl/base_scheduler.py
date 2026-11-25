# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side PP, DeepSeek-V4.1, RLE, and async FP8 staging extensions."""

import threading
import time
from types import SimpleNamespace
from typing import Any

import msgspec
import zmq
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.kv_cache_interface import CircularBufferSpec

from ..ilu_nixl_common.dsv41 import resolve_dsv41_layout, validate_decode_pp_split
from ..ilu_nixl_common.pool_directory import PoolDirectory
from ..ilu_nixl_common.types import resolve_remote_block_ids
from .fp8_layout import use_fp8_ds_mla_wire_staging
from .metadata import (
    GET_META_BATCH_MSG,
    HeartbeatInfo,
    IluNixlConnectorMetadata,
    format_kv_group_page_log,
)

logger = init_logger(__name__)


class IluNixlBaseConnectorScheduler(NixlBaseConnectorScheduler):
    """Base scheduler that propagates PP, V4.1 topology, and E25 prefix state."""

    def __init__(self, *args, **kwargs):
        vllm_config = args[0] if args else kwargs["vllm_config"]
        super().__init__(*args, **kwargs)
        self.dsv41_layout = resolve_dsv41_layout(vllm_config)
        validate_decode_pp_split(vllm_config, self.dsv41_layout)
        self._remote_block_starts: dict[str, list[int]] = {}
        self._urgent_heartbeat_engines: set[str] = set()
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
                "PrefixSchedulerState unavailable; skipping E25 attach", exc_info=True
            )

        self.fp8_wire_staging_enabled = use_fp8_ds_mla_wire_staging(vllm_config)
        if self.fp8_wire_staging_enabled:
            self._fp8_staging_block_ids: dict[str, list[list[int]]] = {}
            self._fp8_exported_block_ids: dict[str, Any] = {}
            self._fp8_pool_directory = PoolDirectory()
            self._fp8_block_hashes: dict[str, list[dict[int, Any]]] = {}
            self._fp8_staged_blocks: dict[str, list[set[int]]] = {}
            transfer_cfg = getattr(vllm_config, "kv_transfer_config", None)
            get_extra = getattr(transfer_cfg, "get_from_extra_config", None)
            chunk_pipe = (
                True
                if get_extra is None
                else str(get_extra("enable_chunk_fp8_staging", True)).lower()
                not in ("0", "false", "no")
            )
            self._fp8_chunk_pipeline_enabled = chunk_pipe and all(
                isinstance(
                    getattr(getattr(g, "kv_cache_spec", None), "block_size", None),
                    int,
                )
                and getattr(g.kv_cache_spec, "block_size", 0) > 0
                for g in getattr(self.kv_cache_config, "kv_cache_groups", ())
            )
            logger.info(
                "[PD-P] scheduler KV groups (chunk_pipeline=%s): %s",
                self._fp8_chunk_pipeline_enabled,
                format_kv_group_page_log(
                    self.kv_cache_config.kv_cache_groups,
                    blocks_per_sw=self.blocks_per_sw,
                ),
            )

    def shutdown(self):
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
        if self.dsv41_layout is None:
            self._ilu_group_clips: list[slice | None] = []
            return
        transfer_groups = getattr(
            self.kv_cache_config,
            "transfer_groups",
            self.kv_cache_config.kv_cache_groups,
        )
        self._ilu_group_clips = []
        for idx, group in enumerate(transfer_groups):
            spec = group.kv_cache_spec
            if isinstance(spec, CircularBufferSpec):
                if self.dsv41_layout.swa_bounded_replay:
                    self._ilu_group_clips.append(slice(0, 0))
                else:
                    self._ilu_group_clips.append(slice(0, 1))
            elif not getattr(spec, "prefix_cacheable", True) and (
                idx >= len(self._ssm_spec_blocks) or self._ssm_spec_blocks[idx] is None
            ):
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
        return (
            transfer_group_idx < len(ssm_blocks)
            and ssm_blocks[transfer_group_idx] is not None
        )

    def compute_remote_block_starts(self, blocks) -> list[int]:
        """Per transfer group, index in P's block list of the first block D asks for."""
        transfer_group_ids = getattr(
            self.kv_cache_config,
            "transfer_group_ids",
            tuple(range(len(blocks.blocks))),
        )
        starts: list[int] = []
        for transfer_idx, group_id in enumerate(transfer_group_ids):
            if self._is_group_clipped(transfer_idx):
                starts.append(-1)
                continue
            group = blocks.blocks[group_id]
            starts.append(
                next(
                    (
                        j
                        for j, block in enumerate(group)
                        if block.block_hash is None and not block.is_null
                    ),
                    len(group),
                )
            )
        return starts

    def get_exchange_clipped_blocks(self, block_ids, clip_ssm: bool = True):
        clipped = super().get_exchange_clipped_blocks(block_ids, clip_ssm=clip_ssm)
        if not clipped or not getattr(self, "_is_hma_required", False):
            return clipped
        ilu_clips = getattr(self, "_ilu_group_clips", None)
        if not ilu_clips or not any(c is not None for c in ilu_clips):
            return clipped
        assert len(clipped) == len(ilu_clips), (
            f"Clipped group count {len(clipped)} != expected {len(ilu_clips)}"
        )
        return tuple(
            list(blocks[clip]) if clip is not None else blocks
            for blocks, clip in zip(clipped, ilu_clips)
        )

    def set_xfer_handshake_metadata(
        self,
        metadata: dict[int | tuple[int, int], KVConnectorHandshakeMetadata],
    ) -> None:
        """Serve handshake metadata by global rank (single, batch, and E25 state)."""
        encoded_data: dict[int, bytes] = {}
        encoder = msgspec.msgpack.Encoder()
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        for rank_key, rank_metadata in metadata.items():
            if isinstance(rank_key, tuple):
                pp_rank, tp_rank = rank_key
                global_rank = pp_rank * tp_size + tp_rank
            else:
                global_rank = rank_key
            if not isinstance(rank_metadata, NixlHandshakePayload):
                raise ValueError(
                    "IluNixlConnectorScheduler expects NixlHandshakePayload "
                    "for handshake metadata."
                )
            encoded_data[global_rank] = encoder.encode(rank_metadata)

        if self._nixl_handshake_listener_t is None:
            prefix_state = getattr(self, "prefix_state", None)
            prefix_wire = (
                prefix_state.wire
                if prefix_state is not None and not prefix_state.consumer
                else None
            )
            ready_event = threading.Event()
            self._nixl_handshake_listener_t = threading.Thread(
                target=self._nixl_handshake_listener,
                args=(
                    encoded_data,
                    ready_event,
                    self._stop_event,
                    self.side_channel_host,
                    self.side_channel_port,
                    tp_size,
                    prefix_wire,
                    self.engine_id,
                ),
                daemon=True,
                name="ilu_nixl_handshake_listener",
            )
            self._nixl_handshake_listener_t.start()
            ready_event.wait()

    @staticmethod
    def _nixl_handshake_listener(
        encoded_data: dict[int, Any],
        ready_event: threading.Event,
        stop_event: threading.Event,
        host: str,
        port: int,
        tp_size: int,
        prefix_wire: Any = None,
        engine_id: str = "",
    ):
        """Background thread serving global-rank handshakes, batch queries, and E25 state."""
        path = make_zmq_path("tcp", host, port)
        logger.info(
            "Starting IluNixl global-rank handshake listener on path: %s, "
            "available ranks=%s",
            path,
            sorted(encoded_data.keys()),
        )
        get_state_tag = None
        serve_state_fn = None
        if prefix_wire is not None:
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
                    identity, _, raw_msg = sock.recv_multipart()
                except zmq.Again:
                    if stop_event.is_set():
                        break
                    continue

                try:
                    decoded_msg = msgspec.msgpack.decode(raw_msg)
                except Exception:
                    logger.exception("IluNixl listener failed to decode message")
                    continue

                # 1. Optional E25 DSv4.1 prefix-state query on the shared side-channel socket.
                if (
                    serve_state_fn is not None
                    and isinstance(decoded_msg, (tuple, list))
                    and decoded_msg
                    and decoded_msg[0] == get_state_tag
                ):
                    state_reply, state_payload = serve_state_fn(
                        prefix_wire, engine_id, decoded_msg
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

                # 2. Single-RTT batch handshake query: (GET_META_BATCH_MSG, [r0, r1, ...])
                if (
                    isinstance(decoded_msg, (tuple, list))
                    and len(decoded_msg) == 2
                    and decoded_msg[0] == GET_META_BATCH_MSG
                    and isinstance(decoded_msg[1], (list, tuple))
                ):
                    requested_ranks = [int(r) for r in decoded_msg[1]]
                    missing = [r for r in requested_ranks if r not in encoded_data]
                    if missing:
                        logger.error(
                            "IluNixl listener: missing batch ranks=%s, available=%s",
                            missing,
                            sorted(encoded_data.keys()),
                        )
                        continue
                    batch_payload = msgspec.msgpack.encode(
                        {r: encoded_data[r] for r in requested_ranks}
                    )
                    ts = msgspec.msgpack.encode(time.perf_counter())
                    sock.send_multipart((identity, b"", batch_payload, ts))
                    continue

                # 3. Single-rank handshake query (2-tuple global rank or 3-tuple (pp, tp))
                if not isinstance(decoded_msg, (tuple, list)):
                    logger.error(
                        "IluNixl listener: invalid handshake message %r", decoded_msg
                    )
                    continue
                is_pp_aware_query = False
                if len(decoded_msg) == 2:
                    msg_tag, target_rank = decoded_msg
                elif len(decoded_msg) == 3:
                    msg_tag, target_pp_rank, target_tp_rank = decoded_msg
                    target_rank = target_pp_rank * tp_size + target_tp_rank
                    is_pp_aware_query = True
                else:
                    logger.error(
                        "IluNixl listener: invalid handshake message %r", decoded_msg
                    )
                    continue

                if msg_tag != GET_META_MSG:
                    logger.warning(
                        "Connection listener got unexpected message %s", msg_tag
                    )
                    continue
                if not isinstance(target_rank, int) or target_rank not in encoded_data:
                    logger.error(
                        "IluNixl listener: no metadata for global rank %r, available=%s",
                        target_rank,
                        sorted(encoded_data.keys()),
                    )
                    continue

                if is_pp_aware_query:
                    # 3-tuple queries use recv_multipart() and expect the 4th clock-sync frame.
                    ts = msgspec.msgpack.encode(time.perf_counter())
                    sock.send_multipart((identity, b"", encoded_data[target_rank], ts))
                else:
                    # Legacy 2-tuple queries may use single-frame sock.recv() on zmq.REQ.
                    sock.send_multipart((identity, b"", encoded_data[target_rank]))

    def on_new_request(self, request) -> None:
        params = request.kv_transfer_params
        if params is not None:
            resolve_remote_block_ids(params)
        if params is not None and params.get("do_remote_decode"):
            if hasattr(self, "_truncate_request_for_prefill"):
                self._truncate_request_for_prefill(request)
            elif getattr(self, "_has_mamba", False) and hasattr(
                self, "_truncate_mamba_request_for_prefill"
            ):
                self._truncate_mamba_request_for_prefill(request)
        if params is None or not params.get("do_remote_prefill"):
            return

        remote_engine_id = params.get("remote_engine_id")
        remote_request_id = params.get("remote_request_id")
        host = params.get("remote_host")
        port = params.get("remote_port")
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
                port=port,
                tp_size=tp_size,
                dcp_size=dcp_size,
                pp_size=pp_size,
                pp_split=pp_split,
            )
        self._heartbeat_by_engine[remote_engine_id].req_ids.add(remote_request_id)
        self._heartbeat_req_engine[request.request_id] = (
            remote_engine_id,
            remote_request_id,
        )
        urgent = getattr(self, "_urgent_heartbeat_engines", None)
        if urgent is not None:
            urgent.add(remote_engine_id)

    def _stop_heartbeat(self, req_id) -> None:
        key = self._heartbeat_req_engine.get(req_id)
        super()._stop_heartbeat(req_id)
        if key is not None:
            engine_id, _ = key
            urgent = getattr(self, "_urgent_heartbeat_engines", None)
            if urgent is not None and engine_id not in self._heartbeat_by_engine:
                urgent.discard(engine_id)

    def _build_save_meta(self, meta, scheduler_output) -> None:
        if not self.fp8_wire_staging_enabled:
            return super()._build_save_meta(meta, scheduler_output)

        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
            yield_req_data,
        )

        pool_dir: PoolDirectory | None = getattr(self, "_fp8_pool_directory", None)
        hashes_map: dict[str, list[dict[int, Any]]] = getattr(
            self, "_fp8_block_hashes", {}
        )
        staged_map: dict[str, list[set[int]]] = getattr(self, "_fp8_staged_blocks", {})
        chunk_pipe = getattr(self, "_fp8_chunk_pipeline_enabled", False)
        n_compute_groups = len(self.kv_cache_config.kv_cache_groups)

        for req_id, new_block_id_groups, _ in yield_req_data(scheduler_output):
            req = self._reqs_need_save.get(req_id)
            if req is None:
                continue

            assert scheduler_output.num_scheduled_tokens is not None
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            end_tokens = req.num_computed_tokens + num_scheduled_tokens
            is_partial = end_tokens < req.num_prompt_tokens

            accumulated = self._fp8_staging_block_ids.setdefault(
                req_id,
                [[] for _ in range(n_compute_groups)],
            )
            if new_block_id_groups is not None:
                if len(accumulated) != len(new_block_id_groups):
                    raise RuntimeError(
                        "FP8 wire staging block group count changed while "
                        f"accumulating request {req_id}: "
                        f"{len(accumulated)} != {len(new_block_id_groups)}"
                    )
                for current, new_ids in zip(accumulated, new_block_id_groups):
                    for block_id in new_ids:
                        if block_id in current:
                            current.remove(block_id)
                        current.append(block_id)

            if is_partial:
                if chunk_pipe:
                    staged_sets = staged_map.setdefault(
                        req_id, [set() for _ in range(n_compute_groups)]
                    )
                    req_hashes = hashes_map.get(req_id)
                    transfer_group_ids = getattr(
                        self.kv_cache_config,
                        "transfer_group_ids",
                        tuple(range(n_compute_groups)),
                    )
                    chunk_export = [[] for _ in range(n_compute_groups)]
                    max_bs = max(
                        int(g.kv_cache_spec.block_size)
                        for g in self.kv_cache_config.kv_cache_groups
                    )
                    sealed_tokens = (end_tokens // max_bs) * max_bs
                    if sealed_tokens > 0:
                        for t_idx, g_id in enumerate(transfer_group_ids):
                            if self._is_group_clipped(t_idx):
                                continue
                            bs = int(
                                self.kv_cache_config.kv_cache_groups[
                                    g_id
                                ].kv_cache_spec.block_size
                            )
                            n_sealed = sealed_tokens // bs
                            candidates = accumulated[g_id][:n_sealed]
                            g_hashes = (
                                req_hashes[g_id]
                                if req_hashes and g_id < len(req_hashes)
                                else None
                            )
                            fresh_filtered = [
                                bid
                                for bid in candidates
                                if bid not in staged_sets[g_id]
                                and (
                                    pool_dir is None
                                    or g_hashes is None
                                    or not pool_dir.is_fresh(
                                        g_id, bid, g_hashes.get(bid)
                                    )
                                )
                            ]
                            if fresh_filtered:
                                chunk_export[g_id] = fresh_filtered
                                staged_sets[g_id].update(fresh_filtered)
                        if any(chunk_export):
                            assert req.kv_transfer_params is not None
                            meta.add_new_req_to_save(
                                request_id=f"{req_id}#chunk@{end_tokens}",
                                local_block_ids=tuple(chunk_export),
                                kv_transfer_params=req.kv_transfer_params,
                                num_tokens=sealed_tokens,
                                sync_save=False,
                            )
                continue

            clipped = self.get_exchange_clipped_blocks(
                tuple(list(group) for group in accumulated)
            )
            exported = [[] for _ in range(n_compute_groups)]
            transfer_group_ids = getattr(
                self.kv_cache_config,
                "transfer_group_ids",
                tuple(range(len(clipped))),
            )
            for group_id, group_blocks in zip(transfer_group_ids, clipped):
                exported[group_id] = list(group_blocks)
            exported_block_ids = tuple(exported)
            self._fp8_exported_block_ids[req_id] = exported_block_ids
            num_tokens = min(end_tokens, req.num_prompt_tokens)

            # Subtract blocks already staged in intermediate chunks or fresh in PoolDirectory.
            staged_sets = staged_map.pop(req_id, None)
            req_hashes = hashes_map.pop(req_id, None)
            delta_export: list[list[int]] = []
            for g_id, grp_blocks in enumerate(exported_block_ids):
                already = staged_sets[g_id] if staged_sets else set()
                g_hashes = (
                    req_hashes[g_id] if req_hashes and g_id < len(req_hashes) else None
                )
                need_ids = [
                    bid
                    for bid in grp_blocks
                    if bid not in already
                    and (
                        pool_dir is None
                        or g_hashes is None
                        or not pool_dir.is_fresh(g_id, bid, g_hashes.get(bid))
                    )
                ]
                delta_export.append(need_ids)

            if pool_dir is not None and req_hashes is not None:
                pool_dir.record_exported(exported_block_ids, req_hashes)

            logger.info(
                "[PD-P] save clip req=%s prompt=%s computed=%s "
                "scheduled=%s export_tokens=%s before=[%s] after=[%s] delta=[%s]",
                req_id,
                req.num_prompt_tokens,
                req.num_computed_tokens,
                num_scheduled_tokens,
                num_tokens,
                ",".join(str(len(group)) for group in accumulated),
                ",".join(str(len(group)) for group in exported_block_ids),
                ",".join(str(len(group)) for group in delta_export),
            )
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_save(
                request_id=req_id,
                local_block_ids=tuple(delta_export),
                kv_transfer_params=req.kv_transfer_params,
                num_tokens=num_tokens,
                sync_save=True,
            )
            self._reqs_need_save.pop(req_id)
            self._fp8_staging_block_ids.pop(req_id, None)

    def build_connector_meta(self, scheduler_output):
        meta = IluNixlConnectorMetadata()
        remote_starts_map = getattr(self, "_remote_block_starts", {})

        for req_id, entry in self._reqs_need_recv.items():
            if len(entry) == 4:
                req, block_ids, cached, awaiting_kvs = entry
            elif len(entry) == 2:
                req, block_ids = entry
                cached, awaiting_kvs = (), False
            else:
                req, block_ids, cached, awaiting_kvs = (
                    entry[0],
                    entry[1],
                    entry[2] if len(entry) > 2 else (),
                    entry[3] if len(entry) > 3 else False,
                )
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
                remote_block_starts=remote_starts_map.pop(req_id, None),
                prompt_token_len=prompt_len,
            )

        if self.use_host_buffer or self.fp8_wire_staging_enabled:
            self._build_save_meta(meta, scheduler_output)

        meta.reqs_to_send = self._reqs_need_send
        meta.scheduler_clock = time.perf_counter()
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed

        if self._heartbeat_by_engine:
            now = time.perf_counter()
            urgent_engines = getattr(self, "_urgent_heartbeat_engines", None)
            if now - self._last_heartbeat_time >= self._heartbeat_interval:
                self._last_heartbeat_time = now
                meta.heartbeat_by_engine = {
                    eid: HeartbeatInfo(
                        req_ids=set(hb.req_ids),
                        host=hb.host,
                        port=hb.port,
                        tp_size=hb.tp_size,
                        dcp_size=hb.dcp_size,
                        pp_size=hb.pp_size,
                        pp_split=hb.pp_split,
                    )
                    for eid, hb in self._heartbeat_by_engine.items()
                }
                if urgent_engines:
                    urgent_engines.clear()
            elif urgent_engines:
                meta.heartbeat_by_engine = {
                    eid: HeartbeatInfo(
                        req_ids=set(hb.req_ids),
                        host=hb.host,
                        port=hb.port,
                        tp_size=hb.tp_size,
                        dcp_size=hb.dcp_size,
                        pp_size=hb.pp_size,
                        pp_split=hb.pp_split,
                    )
                    for eid in urgent_engines
                    if (hb := self._heartbeat_by_engine.get(eid)) is not None
                }
                urgent_engines.clear()

        prefix_state = getattr(self, "prefix_state", None)
        if prefix_state is not None:
            peers = {}
            for rec in prefix_state.requests.values():
                req_obj = rec.get("request")
                p = getattr(req_obj, "kv_transfer_params", None) or {}
                r_eid = rec.get("remote_engine_id")
                r_host = p.get("remote_host")
                r_port = p.get("remote_port")
                if r_eid and r_host is not None and r_port is not None:
                    peers[r_eid] = SimpleNamespace(addr=f"{r_host}:{r_port}")
            prefix_state.add_metadata(meta, scheduler_output, peers)

        self._reqs_need_recv.clear()
        if remote_starts_map:
            remote_starts_map.clear()
        self._reqs_in_batch = set()
        self._reqs_not_processed = set()
        self._reqs_need_send = {}

        return meta
