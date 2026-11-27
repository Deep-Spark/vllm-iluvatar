# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push worker for IluNixlPushConnector (Single-Owner Actor, Write-Once, CompiledXferPlan)."""

from __future__ import annotations

import inspect
import queue
import threading
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
    NixlConnectorMetadata,
    RemoteMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    _is_ssm_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_base_request_id,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec

from ..ilu_nixl_common.dsv41 import (
    resolve_dsv41_layout,
    validate_decode_pp_split,
)
from ..ilu_nixl_common.fp8_hetero.wire_layout import (
    HeteroDtypeConfig,
    WirePlan,
    use_host_staging_wire,
)
from ..ilu_nixl_common.fp8_hetero.xfer_trace import ENABLED as TRACE
from ..ilu_nixl_common.fp8_hetero.xfer_trace import trace
from ..ilu_nixl_common.types import (
    format_push_notif,
    parse_push_notif,
)
from .base_worker import IluNixlPushWorkerMixin
from .chunk_tracker import (
    _CHUNK_PROGRESS_PREFIX,
    ChunkTrackerMixin,
    _ChunkBulkTask,
    request_id_match_keys,
)
from .handshake import HandshakeMixin, _HandshakeInstallTask
from .metadata import (
    ChunkPushPlan,
    ChunkPushReqMeta,
    HmaTransferRoute,
    IluNixlPushTransferRegion,
    PushObservabilityStats,
    RegionPairing,
    ReqMeta,
)
from .registration import RegistrationMixin
from .xfer_plan import (
    CompiledXferPlan,
    align_push_group_blocks,
    concat_descriptor_arrays,
    expand_logical_group,
    nixl_block_size_ratio,
    pair_kernel_pages,
)

logger = init_logger(__name__)


def _infer_pp_split(vllm_config) -> str:
    pp_size = vllm_config.parallel_config.pipeline_parallel_size
    model_config = vllm_config.model_config
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if hf_text_config is not None and hasattr(hf_text_config, "num_hidden_layers"):
        num_hidden_layers = hf_text_config.num_hidden_layers
    else:
        num_hidden_layers = model_config.hf_config.num_hidden_layers

    parts: list[str] = []
    for pp_rank in range(pp_size):
        start, end = get_pp_indices(num_hidden_layers, pp_rank, pp_size)
        parts.append(str(end - start))
    return ",".join(parts)


class IluNixlPushConnectorWorkerMixin(
    ChunkTrackerMixin,
    HandshakeMixin,
    RegistrationMixin,
    IluNixlPushWorkerMixin,
):
    """Single-Owner Push PD worker mixin with Write-Once & CompiledXferPlan."""

    _hetero_cfg = HeteroDtypeConfig()
    _wire_plan: WirePlan | None = None
    _pool = None
    _pipeline = None
    _stager = None
    _host_staging_producer = False
    _cuda_device_id: int | None = None
    _supports_pp_hma = False

    def __init__(self, *args, **kwargs):
        vllm_config = args[0] if args else kwargs.get("vllm_config")
        kv_cache_config = args[2] if len(args) > 2 else kwargs.get("kv_cache_config")
        self.dsv41_layout = resolve_dsv41_layout(vllm_config)
        validate_decode_pp_split(vllm_config, self.dsv41_layout)
        self._virtual_packed_regions = False
        self._empty_producer_rank = False
        self._skipped_layers: list[str] = []
        self._local_desc_cumsum: np.ndarray | None = None
        self._remote_desc_cumsum: dict[str, dict[int, np.ndarray]] = {}
        self.region_pairings: dict[str, dict[int, list[RegionPairing]]] = {}
        self._mamba_hma_routes: dict[str, dict[int, list[HmaTransferRoute]]] = {}
        self._remote_fa_num_regions: dict[str, dict[int, int]] = {}
        self._selected_ssm_remote_region_ids: dict[str, dict[int, list[int]]] = {}
        self._inactive_remote_keys: dict[str, set[int]] = {}
        self._remote_kv_layer_end_by_engine: dict[str, int | None] = {}
        self._remote_supports_block_starts: dict[str, bool] = {}
        self._remote_group_is_ssm_by_engine: dict[str, list[bool] | None] = {}
        self._compiled_xfer_plans: dict[str, dict[int, CompiledXferPlan]] = {}
        self._push_engine_by_address: dict[tuple[str, int, bool], str] = {}
        self._inflight_send_engines: dict[str, str] = {}
        self._inflight_send_targets: dict[str, tuple[str, str, list[int]]] = {}
        self._obs_stats = PushObservabilityStats()

        # Single-Owner NIXL Actor inboxes (main/initiator -> nixl-push-writer)
        self._handshake_install_inbox: queue.Queue[_HandshakeInstallTask] = (
            queue.Queue()
        )
        self._hb_send_inbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._raw_notif_send_inbox: queue.Queue[tuple[str, bytes]] = queue.Queue()
        self._evict_engine_inbox: queue.Queue[tuple[str, bool]] = queue.Queue()
        self._pending_engine_evictions: set[str] = set()

        # V2 bitset notification state on D
        self._recv_notif_bitsets: dict[str, set[tuple[int, int]]] = defaultdict(set)
        self._failed_pre_recvs: set[str] = set()

        parallel_config = getattr(vllm_config, "parallel_config", None)
        original_pp_size = getattr(parallel_config, "pipeline_parallel_size", 1)
        is_consumer = (
            vllm_config is not None
            and getattr(vllm_config, "kv_transfer_config", None) is not None
            and vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        requires_hma = (
            vllm_config is not None
            and kv_cache_config is not None
            and not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(group.kv_cache_spec, FullAttentionSpec)
                for group in kv_cache_config.kv_cache_groups
            )
        )
        should_bypass_upstream_pp_guard = original_pp_size > 1 and (
            is_consumer or requires_hma
        )
        self._bypass_upstream_pp_hma_guard = should_bypass_upstream_pp_guard
        if should_bypass_upstream_pp_guard:
            try:
                parallel_config.pipeline_parallel_size = 1
            except Exception:
                object.__setattr__(parallel_config, "pipeline_parallel_size", 1)
        try:
            super().__init__(*args, **kwargs)
        finally:
            if should_bypass_upstream_pp_guard:
                try:
                    parallel_config.pipeline_parallel_size = original_pp_size
                except Exception:
                    object.__setattr__(
                        parallel_config,
                        "pipeline_parallel_size",
                        original_pp_size,
                    )

        if should_bypass_upstream_pp_guard:
            self.pp_size = original_pp_size

        self._stock_pop_done_transfers_supports_kind = (
            "kind" in inspect.signature(super()._pop_done_transfers).parameters
        )
        parent_handshake = getattr(super(), "_nixl_handshake", None)
        self._handshake_uses_pp_tuple_protocol = bool(
            parent_handshake is not None
            and "notif_agents_only" in inspect.signature(parent_handshake).parameters
        )
        self._hetero_cfg = HeteroDtypeConfig.from_vllm_config(self.vllm_config)
        self._host_staging_producer = use_host_staging_wire(self.vllm_config)
        if self._hetero_cfg.enabled and not is_consumer:
            if not self.use_host_buffer:
                raise ValueError(
                    "hetero_kv_dtype requires kv_buffer_device=cpu: "
                    "PD hardware heterogeneity must transfer through a host buffer"
                )
            if not self._is_hma_required:
                raise ValueError(
                    "hetero_kv_dtype requires the HMA identity-aligned region "
                    "path; do not disable the hybrid KV cache manager"
                )

        self._stage_inbox: queue.Queue[Any] = queue.Queue()
        self._chunk_meta: dict[tuple[str, int], ChunkPushReqMeta] = {}

        self.pp_rank = get_pp_group().rank_in_group
        self.pp_world_size = get_pp_group().world_size
        self.pp_split = envs.VLLM_PP_LAYER_PARTITION or _infer_pp_split(
            self.vllm_config
        )
        assert len(self.pp_split.split(",")) == self.pp_world_size, (
            "PP split count must equal pp_size"
        )
        self.layer_region_ids: dict[str, list[int]] = {}
        self.region_ids_by_group: list[np.ndarray] = []
        self.transfer_regions: list[IluNixlPushTransferRegion] = []
        self.packed_block_stride = 0
        self._region_page_ratios: list[int] = []
        self._region_desc_offsets: list[int] = []
        self.local_region_ids_by_group: dict[str, dict[int, list[np.ndarray]]] = {}
        self.remote_region_ids_by_group: dict[str, dict[int, list[np.ndarray]]] = {}
        self._remote_pp_size_by_engine: dict[str, int] = {}
        self._remote_pp_split_by_engine: dict[str, str] = {}

        self._writer_done_sending: queue.Queue[tuple[str, bool]] = queue.Queue()
        self.pre_recvs: set[str] = set()

        # Completions whose NIXL WRITE is DONE but which cannot be reported yet:
        # the scheduler has not registered the request's lease (``chunk 0`` can
        # complete several prefill steps before ``request_finished``), or later
        # chunks still need those blocks pinned. ``req_id -> buffered_at``
        # mirrors Opt's ``_sends_completed_early``; dropping these (the previous
        # behaviour) stranded every block on ``kv_lease_duration``.
        self._send_done_pending: dict[str, float] = {}
        self._send_done_pending_lock = threading.Lock()
        # Per-request heartbeat renewals, so a decoder that keeps heartbeating
        # cannot pin producer blocks forever.
        self._lease_renewals: dict[str, int] = {}
        self._kv_lease_duration_cfg = int(
            self.kv_transfer_config.get_from_extra_config("kv_lease_duration", 30)
        )
        self._max_lease_renewals = int(
            self.kv_transfer_config.get_from_extra_config("push_max_lease_renewals", 12)
        )

        # Chunked-bulk push state + O(1) request indices
        self._chunk_lock = threading.Lock()
        self._chunk_plans: dict[tuple[str, int], ChunkPushPlan] = {}
        self._chunk_pending: dict[tuple[str, int], ChunkPushReqMeta] = {}
        self._chunk_expected_layers: dict[tuple[str, int], tuple[str, ...]] = {}
        self._chunk_current: set[tuple[str, int]] = set()
        self._chunk_inbox: queue.Queue[_ChunkBulkTask] = queue.Queue()
        self._chunk_deferred: list[_ChunkBulkTask] = []
        self._chunk_open: set[str] = set()
        self._chunk_registrations: dict[str, dict[str, Any]] = {}
        self._ensure_req_indices()

    def _push_obs_stats(self) -> PushObservabilityStats:
        stats = getattr(self, "_obs_stats", None)
        if stats is None:
            stats = PushObservabilityStats()
            self._obs_stats = stats
        return stats

    def get_push_observability_stats(self) -> dict[str, int | float]:
        return self._push_obs_stats().snapshot()

    def _sync_block_size_with_kernel(self) -> None:
        self._scheduler_block_size = self.block_size
        if getattr(self, "dsv41_layout", None) is not None:
            self._logical_num_blocks = self.num_blocks
            self._physical_blocks_per_logical_kv_block = 1
            return
        super()._sync_block_size_with_kernel()

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        if getattr(self, "pcp_rank", 0) > 0 and not getattr(
            self, "pcp_dcp_sharded", False
        ):
            return
        self._ensure_req_indices()
        if metadata.reqs_to_recv:
            for req_id in list(metadata.reqs_to_recv):
                meta = metadata.reqs_to_recv[req_id]
                if not getattr(meta, "awaiting_kvs", True) and not any(
                    meta.local_block_ids
                ):
                    metadata.reqs_to_recv.pop(req_id, None)
                    if (
                        req_id not in metadata.push_registrations
                        and meta.remote is not None
                        and meta.remote.host
                        and meta.remote.port
                    ):
                        metadata.push_registrations[req_id] = {
                            "request_id": req_id,
                            "decode_engine_id": self.engine_id,
                            "decode_host": "",
                            "decode_port": 0,
                            "decode_tp_size": self.world_size,
                            "decode_pp_size": self.pp_world_size,
                            "decode_pp_split": self.pp_split,
                            "local_block_ids": (),
                            "notify_release_only": True,
                            "remote_engine_id": meta.remote.engine_id,
                            "remote_host": meta.remote.host,
                            "remote_port": meta.remote.port,
                            "remote_tp_size": meta.tp_size,
                            "remote_dcp_size": getattr(meta, "dcp_size", 1),
                            "remote_pp_size": getattr(
                                meta, "remote_pp_size", meta.pp_size
                            ),
                            "remote_pp_split": getattr(meta, "remote_pp_split", "")
                            or "",
                        }
                else:
                    self._recv_meta_index.add(req_id)

        super().start_load_kv(metadata)
        if metadata.reqs_not_processed:
            with self._send_done_pending_lock:
                for req_id in metadata.reqs_not_processed:
                    self._send_done_pending.pop(req_id, None)
                    self._lease_renewals.pop(req_id, None)
            for req_id in metadata.reqs_not_processed:
                self._evict_finished_inbox.put(req_id)
            self._push_writer_wake.set()
        for dead_eid in getattr(metadata, "disconnected_engines", ()) or ():
            self._schedule_or_run_engine_eviction(dead_eid, log_eviction=False)
        self._link_pre_recvs()
        self._arm_chunk_requests(metadata)

    def bind_connector_metadata(self, metadata: NixlConnectorMetadata) -> None:
        self._chunk_current = set()

    def _pp_local_layer_range(self) -> tuple[int, int]:
        return self._pp_layer_range(
            self.pp_split, self.pp_rank, getattr(self, "layer_region_ids", None)
        )

    def _ordered_local_layers(self) -> list[str]:
        from vllm.model_executor.models.utils import extract_layer_index

        start, end = self._pp_local_layer_range()
        split = [int(x) for x in self.pp_split.split(",")]
        target_total_layers = sum(split)
        owns_mtp_layers = self._mtp_enabled() and self.pp_rank == len(split) - 1
        target_layers: list[tuple[int, str]] = []
        mtp_layers: list[tuple[int, str]] = []
        for layer_name in self.layer_region_ids:
            try:
                layer_idx = extract_layer_index(layer_name)
            except (AssertionError, ValueError):
                continue
            is_mtp_layer = owns_mtp_layers and (
                layer_idx >= target_total_layers
                or "draft_model" in layer_name
                or ".mtp_block." in layer_name
                or ".mtp.layers." in layer_name
            )
            if start <= layer_idx < end:
                target_layers.append((layer_idx, layer_name))
            elif is_mtp_layer:
                mtp_layers.append((layer_idx, layer_name))
        return [name for _, name in sorted(target_layers)] + [
            name for _, name in sorted(mtp_layers)
        ]

    @staticmethod
    def _record_chunk_event() -> Any:
        if torch.cuda.is_available():
            return torch.cuda.current_stream().record_event()
        return None

    @staticmethod
    def _event_ready(event: Any) -> bool:
        if event is None:
            return True
        return bool(event.query())

    # --- Single-Owner Writer Loop -------------------------------------- #

    def _start_push_writer(self) -> None:
        if self._push_writer_thread is not None:
            return
        self._push_writer_thread = threading.Thread(
            target=self._push_writer_loop,
            daemon=True,
            name="nixl-push-writer",
        )
        self._push_writer_thread.start()
        logger.info("nixl-push-writer thread started (rank=%d)", self.tp_rank)

    def _drain_single_owner_inboxes(self) -> None:
        # Phase-2 handshake NIXL installations
        if hasattr(self, "_handshake_install_inbox"):
            while True:
                try:
                    hs_task = self._handshake_install_inbox.get_nowait()
                except queue.Empty:
                    break
                if hs_task.done_future.cancelled():
                    continue
                try:
                    agents = self._install_fetched_handshake(
                        engine_id=hs_task.engine_id,
                        host=hs_task.host,
                        port=hs_task.port,
                        remote_tp_size=hs_task.remote_tp_size,
                        remote_pp_size=hs_task.remote_pp_size,
                        notif_agents_only=hs_task.notif_agents_only,
                        fetched_ranks=hs_task.fetched_ranks,
                    )
                    hs_task.done_future.set_result(agents)
                except Exception as exc:
                    hs_task.done_future.set_exception(exc)

        # Heartbeat sends
        if hasattr(self, "_hb_send_inbox"):
            while True:
                try:
                    eid, hb_info = self._hb_send_inbox.get_nowait()
                except queue.Empty:
                    break
                self._writer_send_heartbeat(eid, hb_info)

        # Deferred raw notifications
        if hasattr(self, "_raw_notif_send_inbox"):
            while True:
                try:
                    agent_name, notif_id = self._raw_notif_send_inbox.get_nowait()
                except queue.Empty:
                    break
                try:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
                except Exception:
                    self.xfer_stats.record_failed_notification()

    def _push_writer_loop(self) -> None:
        current_platform.set_device(self._worker_device_id)
        self._ensure_req_indices()

        sleep_s = 1.0 / 1000.0
        while not self._push_writer_stop.is_set():
            try:
                # 0. Single-Owner NIXL installs, heartbeats, and raw notifs.
                self._drain_single_owner_inboxes()

                # 1. D registrations to send (leader-filtered + RLE compressed).
                while True:
                    try:
                        rid, rd = self._reg_send_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._send_registration_to_p(rid, rd)

                # 2. Resume P->D pushes after their async handshake completes.
                while True:
                    try:
                        rid, blocks, rd = self._deferred_push_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._do_start_push_kv(rid, blocks, rd)

                # 3. P-side finished blocks; match against pending regs.
                while True:
                    try:
                        rid, blocks = self._finished_blocks_inbox.get_nowait()
                    except queue.Empty:
                        break
                    matched = self._pop_matching_registration(rid)
                    if matched is not None:
                        self._do_start_push_kv(rid, blocks, matched)
                    else:
                        self._push_finished_blocks[rid] = blocks
                        self._finished_blocks_index.add(rid)

                # 3b. Evict finished/expired block state.
                while True:
                    try:
                        rid = self._evict_finished_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._evict_finished_request(rid)

                # 4. NIXL notifs: route PUSH_REG; forward the rest.
                for notifs in self.nixl_wrapper.get_new_notifs().values():
                    for notif in notifs:
                        if notif.startswith(PUSH_REG_NOTIF_PREFIX):
                            self._handle_push_reg_notif(notif)
                        else:
                            self._pending_completion_notifs.put(notif)

                # 4b. Non-blocking chunk promotion + ready chunked-bulk WRITEs.
                self._activate_pending_chunks()
                self._drain_chunk_inbox()

                # 5. Reap completed WRITE transfers & emit fast :ERR on async failure.
                if self._sending_transfers:
                    with self._sending_transfers_lock:
                        done_send, failed_send = self._pop_done_transfers(
                            self._sending_transfers
                        )
                        failed_targets: dict[str, tuple[str, str, list[int]]] = {}
                        for rid in done_send:
                            if hasattr(self, "_inflight_send_engines"):
                                self._inflight_send_engines.pop(rid, None)
                            target = getattr(self, "_inflight_send_targets", {}).pop(
                                rid, None
                            )
                            if rid in failed_send and target is not None:
                                failed_targets[rid] = target
                    for rid, (dst_eid, rem_req_id, rkeys) in failed_targets.items():
                        for rkey in rkeys:
                            self._emit_fast_err_notif(rem_req_id, dst_eid, rkey)
                    for rid in done_send:
                        if (
                            hasattr(self, "_tail_chunk_ready_at")
                            and rid in self._tail_chunk_ready_at
                        ):
                            t_ready = self._tail_chunk_ready_at.pop(rid)
                            self._push_obs_stats().record_tail_chunk_exposed(
                                (time.perf_counter() - t_ready) * 1000.0
                            )
                        self._writer_done_sending.put((rid, rid in failed_send))

                # 6. Tear down replaced / TTL-expired remote engines safely on writer.
                if hasattr(self, "_evict_engine_inbox"):
                    while True:
                        try:
                            item = self._evict_engine_inbox.get_nowait()
                        except queue.Empty:
                            break
                        if isinstance(item, tuple):
                            old_eid, log_eviction = item
                        else:
                            old_eid, log_eviction = item, False
                        if hasattr(self, "_pending_engine_evictions"):
                            self._pending_engine_evictions.discard(old_eid)
                        if (
                            old_eid in self._remote_agents
                            and old_eid not in self._engines_with_inflight_transfers()
                        ):
                            self._cleanup_remote_engine(
                                old_eid, log_eviction=log_eviction
                            )
                            if not log_eviction:
                                logger.info(
                                    "Released NIXL state for replaced remote engine %s.",
                                    old_eid,
                                )
            except Exception:
                logger.exception("nixl-push-writer error; continuing")

            active = bool(
                self._push_finished_blocks
                or self._chunk_pending
                or self._chunk_deferred
                or not self._chunk_inbox.empty()
                or self._sending_transfers
                or (
                    hasattr(self, "_handshake_install_inbox")
                    and not self._handshake_install_inbox.empty()
                )
            )
            if active:
                self._push_writer_stop.wait(timeout=sleep_s)
            else:
                self._push_writer_wake.wait()
                self._push_writer_wake.clear()

    def _pop_done_transfers(
        self, transfers, kind: str | None = None
    ) -> tuple[set[str], set[str]]:
        is_send = transfers is getattr(self, "_sending_transfers", None)
        open_req_ids = set(self._chunk_open) if is_send else set()
        if is_send:
            open_req_ids.update(key[0] for key in self._chunk_pending)

        stock_pop_done = super()._pop_done_transfers
        raw_result = (
            stock_pop_done(transfers, kind)
            if kind is not None and self._stock_pop_done_transfers_supports_kind
            else stock_pop_done(transfers)
        )
        if isinstance(raw_result, tuple):
            done_req_ids, failed_req_ids = raw_result
        else:
            done_req_ids, failed_req_ids = raw_result, set()

        if open_req_ids:
            premature_done = {
                rid
                for rid in done_req_ids
                if rid in open_req_ids or self._chunk_matches_any(rid, open_req_ids)
            }
            # Intermediate chunks complete before the final chunk is submitted;
            # suppress per-chunk completion here and let the terminal chunk emit
            # the request-level completion once all chunks are done.
            done_req_ids -= premature_done
        if TRACE and is_send and done_req_ids:
            now = trace.now_ms()
            for rid in done_req_ids:
                trace.emit(("X", rid, round(now, 3)))
        return done_req_ids, failed_req_ids

    # --- WRITE Execution & V2 Bitset / Fast-:ERR Notifications --------- #

    def _uses_region_mapped_xfer(self, engine_id: str) -> bool:
        return (
            engine_id in self.local_region_ids_by_group
            and engine_id in self.remote_region_ids_by_group
        )

    def _local_xfer_handle(self, remote_block_size: int) -> int:
        block_size = (
            self.block_size if self._wire_plan is not None else remote_block_size
        )
        return self.src_xfer_handles_by_block_size[block_size]

    def _convert_remote_logical_block_ids(
        self, block_ids, remote_physical_per_logical: int
    ):
        legacy_converter = getattr(self, "_logical_to_remote_kernel_block_ids", None)
        if legacy_converter is not None:
            return legacy_converter(block_ids, remote_physical_per_logical)
        return self._logical_to_kernel_block_ids(block_ids, remote_physical_per_logical)

    def _make_pp_aware_notif_id(
        self,
        remote_request_id: str,
        dst_engine_id: str,
        remote_key: int,
        *,
        status: str = "OK",
    ) -> bytes:
        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        remote_pp_rank = remote_key // remote_info.remote_tp_size
        pp_producers = self._effective_pp_producers_for_remote(
            dst_engine_id, remote_pp_rank
        )
        return format_push_notif(
            remote_request_id,
            self.world_size,
            pp_producers,
            src_pp=getattr(self, "pp_rank", 0),
            src_tp=getattr(self, "tp_rank", 0),
            status=status,
        )

    def _emit_fast_err_notif(
        self, remote_request_id: str, dst_engine_id: str, remote_key: int
    ) -> None:
        raw_req_id = (
            remote_request_id.split(":", 1)[1]
            if remote_request_id.startswith(_CHUNK_PROGRESS_PREFIX)
            and ":" in remote_request_id[len(_CHUNK_PROGRESS_PREFIX) :]
            else remote_request_id
        )
        if raw_req_id.startswith(_CHUNK_PROGRESS_PREFIX):
            raw_req_id = raw_req_id[len(_CHUNK_PROGRESS_PREFIX) :].split(":", 1)[-1]
        agent_name = self._remote_agents.get(dst_engine_id, {}).get(remote_key)
        if agent_name is None:
            return
        try:
            err_notif = self._make_pp_aware_notif_id(
                raw_req_id, dst_engine_id, remote_key, status="ERR"
            )
            self.nixl_wrapper.send_notif(agent_name, notif_msg=err_notif)
            self._push_obs_stats().fast_err_notifs_sent += 1
        except Exception:
            logger.debug(
                "Failed to send fast ERR notif to %s rank %s",
                dst_engine_id,
                remote_key,
                exc_info=True,
            )

    def _send_fast_err_notif_from_registration(
        self, request_id: str, reg: dict[str, Any]
    ) -> None:
        dst_eid = reg.get("decode_engine_id")
        rem_req_id = reg.get("request_id")
        if not dst_eid or not rem_req_id:
            return
        for rkey in sorted(self._remote_agents.get(dst_eid, ())):
            self._emit_fast_err_notif(str(rem_req_id), str(dst_eid), rkey)
        with self._sending_transfers_lock:
            self._send_failures.add(request_id)

    def _do_start_push_kv(
        self,
        request_id: str,
        local_block_ids,
        registration_data: dict[str, Any],
    ) -> None:
        if registration_data.get("notify_release_only"):
            decode_req_id = str(registration_data.get("request_id", ""))
            if not decode_req_id.startswith(_CHUNK_PROGRESS_PREFIX):
                self._writer_done_sending.put((request_id, False))
            return None

        # P0-4 Empty/skipped V4.1 producer rank excluded from D's effective_pp_producers:
        # finish immediately with zero network traffic or P->D handshake!
        if (
            getattr(self, "_empty_producer_rank", False)
            and self._is_predictable_empty_producer()
        ):
            decode_req_id = str(registration_data.get("request_id", ""))
            if not decode_req_id.startswith(_CHUNK_PROGRESS_PREFIX):
                self._writer_done_sending.put((request_id, False))
            return None

        decode_engine_id = registration_data["decode_engine_id"]
        fut = self._ensure_handshake(
            decode_engine_id,
            registration_data["decode_host"],
            int(registration_data["decode_port"]),
            int(registration_data["decode_tp_size"]),
            int(registration_data.get("decode_pp_size", 1)),
            registration_data.get("decode_pp_split", "") or "",
        )
        if fut is not None:

            def _on_handshake(
                f,
                rid: str = request_id,
                blocks=local_block_ids,
                rd: dict[str, Any] = registration_data,
            ) -> None:
                if (error := f.exception()) is not None:
                    self._log_failure(
                        failure_type="push_handshake_failed",
                        req_id=rid,
                        error=error,
                    )
                    return
                self._deferred_push_inbox.put((rid, blocks, rd))
                self._push_writer_wake.set()

            fut.add_done_callback(_on_handshake)
            return None

        if not self._uses_region_mapped_xfer(decode_engine_id):
            return super()._do_start_push_kv(
                request_id, local_block_ids, registration_data
            )

        self._engine_last_active[decode_engine_id] = time.perf_counter()
        remote_block_ids = registration_data["local_block_ids"]
        decode_request_id = registration_data["request_id"]
        logical_local = self._as_grouped_block_ids(local_block_ids)
        logical_remote = self._as_grouped_block_ids(remote_block_ids)
        physical_local = (
            self._logical_to_kernel_block_ids(
                logical_local, self._physical_blocks_per_logical_kv_block
            )
            if logical_local
            else ()
        )
        push_meta = ReqMeta(
            local_block_ids=logical_local,
            local_physical_block_ids=physical_local,
            tp_size=self.world_size,
            remote=RemoteMeta(
                block_ids=logical_remote,
                host="",
                port=0,
                engine_id=decode_engine_id,
                request_id=decode_request_id,
            ),
            remote_block_starts=registration_data.get("remote_block_starts"),
        )
        chunk_offsets = registration_data.get("chunk_block_offsets")
        if chunk_offsets is not None:
            push_meta.chunk_block_offsets = list(chunk_offsets)

        t0 = time.perf_counter()
        self._xfer_blocks_for_req(req_id=request_id, meta=push_meta)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > 200.0:
            logger.warning(
                "_do_start_push_kv for %s took %.1fms (slow NIXL submission)",
                request_id,
                elapsed_ms,
            )
        return None

    def _xfer_blocks_for_req(self, req_id: str, meta):
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        if not self._uses_region_mapped_xfer(engine_id):
            return super()._xfer_blocks_for_req(req_id, meta)

        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)
        is_progress_only = meta.remote.request_id.startswith(_CHUNK_PROGRESS_PREFIX)
        inactive_keys = getattr(self, "_inactive_remote_keys", {}).get(engine_id, set())
        remote_agents = self._remote_agents.get(engine_id, {})

        if getattr(self, "_empty_producer_rank", False):
            if not is_progress_only:
                if not self._is_predictable_empty_producer():
                    for remote_key in sorted(remote_agents):
                        notif_id = self._make_pp_aware_notif_id(
                            meta.remote.request_id, engine_id, remote_key
                        )
                        self.nixl_wrapper.send_notif(
                            remote_agents[remote_key], notif_msg=notif_id
                        )
                self._writer_done_sending.put((req_id, False))
            return

        engine_routes = getattr(self, "_mamba_hma_routes", {}).get(engine_id)
        engine_pairings = getattr(self, "region_pairings", {}).get(engine_id)
        remote_block_starts = getattr(meta, "remote_block_starts", None)
        chunk_block_offsets = getattr(meta, "chunk_block_offsets", None)
        uses_logical_pair_routing = bool(
            engine_routes
            or engine_pairings
            or remote_block_starts is not None
            or chunk_block_offsets is not None
        )

        if uses_logical_pair_routing:
            local_logical_ids = list(meta.local_block_ids)
            remote_logical_ids = list(meta.remote.block_ids)
        else:
            meta.remote.block_ids = self._convert_remote_logical_block_ids(
                meta.remote.block_ids,
                remote_info.remote_physical_blocks_per_logical,
            )
            local_logical_ids = list(meta.local_physical_block_ids)
            remote_logical_ids = list(meta.remote.block_ids)

        num_local_groups = len(local_logical_ids)
        group_spec_types = getattr(self, "_group_spec_types", ())
        replicate_mla = self.use_mla and tp_ratio < 0
        skip_replicated_mla_follower = (
            self.use_mla and tp_ratio > 1 and (self.tp_rank % tp_ratio != 0)
        )

        handles: list[int] = []
        active_rkeys: list[int] = []
        for remote_key in sorted(remote_agents):
            if remote_key in inactive_keys:
                if not is_progress_only and not self._is_predictable_empty_producer():
                    notif_id = self._make_pp_aware_notif_id(
                        meta.remote.request_id, engine_id, remote_key
                    )
                    self.nixl_wrapper.send_notif(
                        remote_agents[remote_key], notif_msg=notif_id
                    )
                continue

            active_rkeys.append(remote_key)
            remote_tp_rank = remote_key % remote_info.remote_tp_size

            def _group_targets_rank(
                g: int, *, _remote_tp_rank: int = remote_tp_rank
            ) -> bool:
                is_ssm_grp = g < len(group_spec_types) and _is_ssm_spec(
                    group_spec_types[g]
                )
                if is_ssm_grp:
                    return (
                        g < len(plan.source_ranks_per_group)
                        and _remote_tp_rank in plan.source_ranks_per_group[g]
                    )
                if skip_replicated_mla_follower:
                    return False
                if replicate_mla:
                    return True
                return (
                    g < len(plan.source_ranks_per_group)
                    and _remote_tp_rank in plan.source_ranks_per_group[g]
                )

            if uses_logical_pair_routing:
                read_spec = ReadSpec(
                    remote_rank=remote_key,
                    local_block_ids=[
                        list(local_logical_ids[g]) if _group_targets_rank(g) else []
                        for g in range(num_local_groups)
                    ],
                    remote_block_ids=[list(grp) for grp in remote_logical_ids],
                )
            else:
                read_spec = ReadSpec(
                    remote_rank=remote_key,
                    local_block_ids=[
                        list(local_logical_ids[g]) if _group_targets_rank(g) else []
                        for g in range(num_local_groups)
                    ],
                    remote_block_ids=[
                        list(remote_logical_ids[g])
                        if g < len(remote_logical_ids) and _group_targets_rank(g)
                        else []
                        for g in range(num_local_groups)
                    ],
                )

            if self._needs_split_local_xfer_handles(tp_ratio, plan):
                split_idx = plan.all_source_ranks.index(remote_tp_rank)
                split_key = (tp_ratio, remote_info.remote_block_size)
                handles_by_ratio = self.src_xfer_handles_by_tp_ratio.get(split_key)
                if handles_by_ratio is None:
                    handles_by_ratio = self.src_xfer_handles_by_tp_ratio[tp_ratio]
                local_xfer_side_handle = handles_by_ratio[split_idx]
            else:
                local_xfer_side_handle = self._local_xfer_handle(
                    remote_info.remote_block_size
                )
            remote_xfer_side_handle = self.dst_xfer_side_handles[engine_id][remote_key]
            handle = self._xfer_blocks(
                read_spec=read_spec,
                dst_engine_id=engine_id,
                request_id=req_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
                remote_block_starts=remote_block_starts,
                chunk_block_offsets=chunk_block_offsets,
                uses_logical_pair_routing=uses_logical_pair_routing,
            )
            if handle is not None:
                handles.append(handle)

        if handles:
            with self._sending_transfers_lock:
                self._sending_transfers[req_id].extend(handles)
                if hasattr(self, "_inflight_send_engines"):
                    self._inflight_send_engines[req_id] = engine_id
                if hasattr(self, "_inflight_send_targets"):
                    self._inflight_send_targets[req_id] = (
                        engine_id,
                        meta.remote.request_id,
                        active_rkeys,
                    )
        elif not is_progress_only:
            with self._sending_transfers_lock:
                has_inflight = bool(self._sending_transfers.get(req_id))
                has_failed = req_id in self._send_failures
            if not has_inflight and not has_failed:
                if (
                    hasattr(self, "_tail_chunk_ready_at")
                    and req_id in self._tail_chunk_ready_at
                ):
                    t_ready = self._tail_chunk_ready_at.pop(req_id)
                    self._push_obs_stats().record_tail_chunk_exposed(
                        (time.perf_counter() - t_ready) * 1000.0
                    )
                self._writer_done_sending.put((req_id, False))

    def _get_or_compile_xfer_plan(
        self, dst_engine_id: str, remote_key: int, block_size_ratio: int
    ) -> CompiledXferPlan:
        cached = (
            getattr(self, "_compiled_xfer_plans", {})
            .get(dst_engine_id, {})
            .get(remote_key)
        )
        if cached is not None:
            return cached
        return self._compile_and_cache_xfer_plan(
            dst_engine_id, remote_key, block_size_ratio
        )

    def _xfer_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
        *,
        remote_block_starts: list[int] | None = None,
        chunk_block_offsets: list[int] | None = None,
        uses_logical_pair_routing: bool = False,
    ) -> int | None:
        if not self._uses_region_mapped_xfer(dst_engine_id):
            return super()._xfer_blocks(
                read_spec,
                dst_engine_id,
                request_id,
                remote_request_id,
                local_xfer_side_handle,
                remote_xfer_side_handle,
            )

        assert self.transfer_topo is not None
        remote_key = read_spec.remote_rank
        local_block_ids = list(read_spec.local_block_ids)
        remote_block_ids = list(read_spec.remote_block_ids)
        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = nixl_block_size_ratio(
            self.transfer_topo.block_size, remote_info.remote_block_size
        )
        notif_id = self._make_pp_aware_notif_id(
            remote_request_id, dst_engine_id, remote_key
        )
        is_progress_only = remote_request_id.startswith(_CHUNK_PROGRESS_PREFIX)

        if (
            len(local_block_ids) == 0
            or not any(local_block_ids)
            or not any(remote_block_ids)
        ):
            if not is_progress_only:
                agent_name = self._remote_agents.get(dst_engine_id, {}).get(remote_key)
                if agent_name is not None:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            return None

        local_ratio = (
            1
            if self._wire_plan is not None
            else self._physical_blocks_per_logical_kv_block
        )
        remote_ratio = remote_info.remote_physical_blocks_per_logical

        compiled = self._get_or_compile_xfer_plan(
            dst_engine_id, remote_key, block_size_ratio
        )
        if (
            compiled.mode == "groups"
            and block_size_ratio > 1
            and self._wire_plan is None
        ):
            if uses_logical_pair_routing:
                local_block_ids = [
                    expand_logical_group(grp, local_ratio) for grp in local_block_ids
                ]
                remote_block_ids = [
                    expand_logical_group(grp, remote_ratio) for grp in remote_block_ids
                ]
            local_block_ids, remote_block_ids = (
                self._map_block_ids_for_block_size_ratio(
                    local_block_ids, remote_block_ids, block_size_ratio
                )
            )
            uses_logical_pair_routing = False

        local_descs_list: list[np.ndarray] = []
        remote_descs_list: list[np.ndarray] = []

        for bucket in compiled.buckets:
            l_gid = bucket.local_group_idx
            r_gid = bucket.remote_group_idx
            if l_gid >= len(local_block_ids) or r_gid >= len(remote_block_ids):
                if compiled.mode in ("routes", "pairings"):
                    raise RuntimeError(
                        "Cannot align producer and consumer KV cache groups "
                        f"for {compiled.mode}: local_group={l_gid}/"
                        f"{len(local_block_ids)}, remote_group={r_gid}/"
                        f"{len(remote_block_ids)}"
                    )
                continue

            if uses_logical_pair_routing or compiled.mode in (
                "routes",
                "pairings",
            ):
                p_chunk_off = (
                    chunk_block_offsets[l_gid]
                    if chunk_block_offsets is not None
                    and l_gid < len(chunk_block_offsets)
                    else None
                )
                d_start_off = (
                    remote_block_starts[r_gid]
                    if (
                        not bucket.is_ssm
                        and remote_block_starts is not None
                        and r_gid < len(remote_block_starts)
                    )
                    else None
                )
                bucket_local_ratio = (
                    getattr(bucket, "local_page_ratio", 1)
                    if self._wire_plan is not None
                    else local_ratio
                )
                if not bucket.is_ssm and bucket_local_ratio != remote_ratio:
                    # 4480-token P blocks and 640-token D blocks expand to
                    # different kernel-page counts. Pair the shared token
                    # prefix after expansion; do not zip logical block ids.
                    p_page_off = (
                        p_chunk_off * bucket_local_ratio
                        if p_chunk_off is not None and p_chunk_off >= 0
                        else 0
                    )
                    d_page_off = (
                        d_start_off * remote_ratio
                        if d_start_off is not None and d_start_off >= 0
                        else 0
                    )
                    grp_local, grp_remote = pair_kernel_pages(
                        list(local_block_ids[l_gid]),
                        list(remote_block_ids[r_gid]),
                        local_ratio=bucket_local_ratio,
                        remote_ratio=remote_ratio,
                        local_page_offset=p_page_off,
                        remote_page_offset=d_page_off,
                    )
                else:
                    grp_local, grp_remote = align_push_group_blocks(
                        list(local_block_ids[l_gid]),
                        list(remote_block_ids[r_gid]),
                        p_chunk_offset=p_chunk_off,
                        d_start_offset=d_start_off,
                        is_ssm=bucket.is_ssm,
                        req_id=request_id,
                        local_group_idx=l_gid,
                    )
                    if grp_local and not bucket.is_ssm and uses_logical_pair_routing:
                        grp_local = expand_logical_group(grp_local, bucket_local_ratio)
                        grp_remote = expand_logical_group(grp_remote, remote_ratio)
            else:
                grp_local = list(local_block_ids[l_gid])
                grp_remote = list(remote_block_ids[r_gid])
                if len(grp_local) != len(grp_remote):
                    aligned = min(len(grp_local), len(grp_remote))
                    grp_local = grp_local[-aligned:] if aligned else []
                    grp_remote = grp_remote[-aligned:] if aligned else []

            if not grp_local:
                continue
            pair = bucket.compute_desc_pair(grp_local, grp_remote)
            if pair is not None:
                local_descs_list.append(pair[0])
                remote_descs_list.append(pair[1])

        if not local_descs_list:
            if not is_progress_only:
                agent_name = self._remote_agents.get(dst_engine_id, {}).get(remote_key)
                if agent_name is not None:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            return None

        local_block_descs_ids = concat_descriptor_arrays(local_descs_list)
        remote_block_descs_ids = concat_descriptor_arrays(remote_descs_list)
        assert len(local_block_descs_ids) == len(remote_block_descs_ids), (
            "Heterogeneous PP WRITE descriptor count mismatch: "
            f"local={len(local_block_descs_ids)}, "
            f"remote={len(remote_block_descs_ids)}"
        )

        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=b"" if is_progress_only else notif_id,
            )
            self.nixl_wrapper.transfer(handle)
            self._push_obs_stats().compiled_xfer_submissions += 1
            return handle
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Heterogeneous PP push WRITE submission failed",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_key,
            )
            with self._sending_transfers_lock:
                self._send_failures.add(request_id)
            # P0-4 Fast fail-closed notification to D so D frees blocks immediately!
            self._emit_fast_err_notif(remote_request_id, dst_engine_id, remote_key)
            if not self._handle_failed_transfer(request_id, handle):
                return handle
            return None

    # --- O(1) Recv Resolution & V2 Bitset Notification Accounting ------ #

    def _resolve_recving_req_id(self, notif_req_id: str) -> str | None:
        recv_map = self._recving_metadata
        if not recv_map:
            return None
        if notif_req_id in recv_map:
            return notif_req_id
        self._ensure_req_indices()
        matched = self._recv_meta_index.match_first(notif_req_id, recv_map)
        if matched is not None:
            return matched
        notif_keys = set(request_id_match_keys(notif_req_id))
        for recv_req_id in recv_map:
            self._recv_meta_index.add(recv_req_id)
            if set(request_id_match_keys(recv_req_id)) & notif_keys:
                return recv_req_id
        return None

    def _complete_d_recv_notif(self, recv_req_id: str) -> None:
        self._recving_transfers.setdefault(recv_req_id, [])

    def _fail_d_recv_notif(self, notif_req_id: str) -> None:
        recv_req_id = self._resolve_recving_req_id(notif_req_id)
        if recv_req_id is not None:
            self._failed_recv_reqs.put(recv_req_id)
        else:
            if not hasattr(self, "_failed_pre_recvs"):
                self._failed_pre_recvs = set()
            self._failed_pre_recvs.add(notif_req_id)

    def _link_pre_recvs(self) -> None:
        failed_pre = getattr(self, "_failed_pre_recvs", None)
        if failed_pre:
            still_failed: set[str] = set()
            for pre_id in failed_pre:
                recv_req_id = self._resolve_recving_req_id(pre_id)
                if recv_req_id is None:
                    still_failed.add(pre_id)
                else:
                    self._failed_recv_reqs.put(recv_req_id)
            self._failed_pre_recvs = still_failed

        if self.pre_recvs:
            still_pending: set[str] = set()
            for pre_id in self.pre_recvs:
                recv_req_id = self._resolve_recving_req_id(pre_id)
                if recv_req_id is None:
                    still_pending.add(pre_id)
                    continue
                self._complete_d_recv_notif(recv_req_id)
            self.pre_recvs = still_pending

    def _get_new_notifs(self) -> set[str]:
        assert self.transfer_topo is not None
        if not hasattr(self, "_recv_notif_bitsets"):
            self._recv_notif_bitsets = defaultdict(set)
        notified_req_ids: set[str] = set()

        while True:
            try:
                notif = self._pending_completion_notifs.get_nowait()
            except queue.Empty:
                break

            msg = notif.decode("utf-8")
            if msg.startswith("HB:"):
                self._handle_heartbeat(msg[3:])
                continue

            parsed = parse_push_notif(msg)
            req_id = parsed.req_id
            if req_id.startswith(_CHUNK_PROGRESS_PREFIX):
                continue

            kv_role = self.kv_transfer_config.kv_role
            if kv_role == "kv_consumer":
                is_d_recv = True
            elif kv_role == "kv_producer":
                is_d_recv = False
            elif kv_role == "kv_both":
                is_d_recv = (
                    req_id not in self._reqs_to_send
                    and req_id not in self._reqs_to_process
                )
            else:
                raise RuntimeError(f"Unsupported KV transfer role: {kv_role}")

            if is_d_recv:
                recv_count_key = f"recv:{get_base_request_id(req_id)}"
                # P0-4 Fast circuit breaker on :ERR from any producer writer!
                if parsed.status != "OK":
                    logger.error(
                        "Received fast push ERR notification for request %s "
                        "from producer (pp=%s, tp=%s); failing receive immediately.",
                        req_id,
                        parsed.src_pp,
                        parsed.src_tp,
                    )
                    self._recv_notif_bitsets.pop(recv_count_key, None)
                    self.consumer_notification_counts_by_req.pop(recv_count_key, None)
                    self._fail_d_recv_notif(req_id)
                    continue

                n_producers = parsed.tp_size
                tp_ratio = self.transfer_topo.tp_ratio(n_producers)
                producers_per_consumer = (
                    -tp_ratio if n_producers > self.world_size else 1
                )
                expected_notifications = producers_per_consumer * parsed.pp_producers
                if expected_notifications <= 0:
                    self._recv_notif_bitsets.pop(recv_count_key, None)
                    self.consumer_notification_counts_by_req.pop(recv_count_key, None)
                    recv_req_id = self._resolve_recving_req_id(req_id)
                    if recv_req_id is not None:
                        self._complete_d_recv_notif(recv_req_id)
                    else:
                        self.pre_recvs.add(req_id)
                    continue

                if parsed.src_pp is not None and parsed.src_tp is not None:
                    seen = self._recv_notif_bitsets[recv_count_key]
                    writer_coord = (parsed.src_pp, parsed.src_tp)
                    if writer_coord in seen:
                        self._push_obs_stats().duplicate_notifs_ignored += 1
                        continue
                    seen.add(writer_coord)
                    recv_count = len(seen)
                else:
                    self.consumer_notification_counts_by_req[recv_count_key] += 1
                    recv_count = self.consumer_notification_counts_by_req[
                        recv_count_key
                    ]

                if recv_count < expected_notifications:
                    continue
                if recv_count > expected_notifications:
                    logger.error(
                        "Ignoring excess heterogeneous push recv notification "
                        "request=%s received=%s expected=%s",
                        req_id,
                        recv_count,
                        expected_notifications,
                    )
                    continue
                self._recv_notif_bitsets.pop(recv_count_key, None)
                self.consumer_notification_counts_by_req.pop(recv_count_key, None)
                recv_req_id = self._resolve_recving_req_id(req_id)
                if recv_req_id is not None:
                    self._complete_d_recv_notif(recv_req_id)
                else:
                    self.pre_recvs.add(req_id)
                continue

            n_consumers = parsed.tp_size
            tp_ratio = self.transfer_topo.tp_ratio(n_consumers)
            consumers_per_producer = -tp_ratio if n_consumers > self.world_size else 1
            expected_notifications = consumers_per_producer * parsed.pp_producers
            self.consumer_notification_counts_by_req[req_id] += 1
            if (
                self.consumer_notification_counts_by_req[req_id]
                == expected_notifications
            ):
                notified_req_ids.add(req_id)
                del self.consumer_notification_counts_by_req[req_id]
                self._reqs_to_process.discard(req_id)
                self._reqs_to_send.pop(req_id, None)
        return notified_req_ids

    def _reap_expired_send_leases(self, done_sending: set[str]) -> None:
        before = set(self._reqs_to_send)
        super()._reap_expired_send_leases(done_sending)
        expired_now = before - set(self._reqs_to_send)
        self._last_lease_expired = expired_now
        for req_id in expired_now:
            self._lease_renewals.pop(req_id, None)

    def _send_ready_to_report(self, req_id: str) -> bool:
        """Whether a completed WRITE may be reported as ``finished_sending``.

        All gates must be open:

        1. the scheduler finalized the request and registered its send lease in
           ``_reqs_to_send`` (never ``_reqs_to_process``, which is populated on
           step 0 while the request is still ``RUNNING`` and would trip
           ``assert request.is_finished()`` in ``Scheduler._free_blocks``);
        2. the chunk state machine released the request, i.e. no chunk plan,
           registration or pending chunk still needs those producer blocks;
        3. no NIXL WRITE for this request is still in flight. ``_chunk_open`` is
           cleared as soon as the *last chunk is submitted*, not when it
           completes, so gate 2 alone would free blocks under a live WRITE.
        """
        if req_id not in self._reqs_to_send:
            return False
        if self._chunk_should_retain(req_id):
            return False
        with self._sending_transfers_lock:
            return not self._sending_transfers.get(req_id)

    def _handle_heartbeat(self, payload: str) -> None:
        """Extend leases, but never indefinitely.

        Upstream renews every heartbeat without a ceiling. On the push path the
        decoder heartbeats for as long as it is waiting for KV, so a blocked
        decoder pinned producer blocks forever and the producer's KV pool filled
        up until it could no longer schedule a single request. Cap the number of
        renewals so the lease still eventually fires.
        """
        new_expiry = time.perf_counter() + self._lease_extension
        for req_id in payload.split(","):
            if req_id not in self._reqs_to_send:
                continue
            renewals = self._lease_renewals.get(req_id, 0) + 1
            self._lease_renewals[req_id] = renewals
            if renewals > self._max_lease_renewals:
                logger.warning(
                    "Heartbeat renewal cap (%d) reached for request %s; "
                    "letting its KV lease expire so producer blocks can be freed.",
                    self._max_lease_renewals,
                    req_id,
                )
                self._reqs_to_send[req_id] = time.perf_counter()
                self._push_obs_stats().lease_renewal_capped += 1
                continue
            old = self._reqs_to_send[req_id]
            self._reqs_to_send[req_id] = max(old, new_expiry)
            logger.debug(
                "Heartbeat extended lease for request %s by %ds "
                "(renewal=%d, old_expiry=%.1f, new_expiry=%.1f)",
                req_id,
                self._lease_extension,
                renewals,
                old,
                new_expiry,
            )

    def get_transfer_results(self) -> KVConnectorTransferResults:
        """Collect writer-reaped sends and D-side recv completions (vLLM 0.31.0)."""
        self._push_writer_wake.set()
        self._link_pre_recvs()

        self._last_lease_expired = set()
        results = self._base_push_transfer_results()
        lease_expired: set[str] = getattr(self, "_last_lease_expired", set())
        done_sending = results.finished_sending
        now = time.perf_counter()

        done_pushing: set[str] = set()
        failed_pushing: set[str] = set()
        while True:
            try:
                item = self._writer_done_sending.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                rid, is_failed = item
            else:
                rid, is_failed = item, False
            done_pushing.add(rid)
            if is_failed:
                failed_pushing.add(rid)

        early_buffered = 0
        with self._sending_transfers_lock:
            self._send_failures.update(failed_pushing)
            buffered_now = done_pushing - self._send_failures
            self._send_failures.difference_update(done_pushing | done_sending)
            with self._send_done_pending_lock:
                # A failed WRITE is never reportable: its blocks are reclaimed
                # through the lease path, not through ``finished_sending``.
                for req_id in failed_pushing:
                    self._send_done_pending.pop(req_id, None)
                for req_id in buffered_now:
                    if req_id not in self._send_done_pending:
                        self._send_done_pending[req_id] = now
                        if req_id not in self._reqs_to_send:
                            early_buffered += 1
        if early_buffered:
            self._push_obs_stats().completed_early_buffered += early_buffered

        # Decide outside ``_sending_transfers_lock``: ``_chunk_should_retain``
        # acquires it again internally and the lock is not reentrant.
        successful: set[str] = set()
        with self._send_done_pending_lock:
            pending_snapshot = list(self._send_done_pending.items())
        if pending_snapshot:
            orphan_after = max(4.0 * self._kv_lease_duration_cfg, 60.0)
            reportable: list[str] = []
            orphaned: list[str] = []
            dropped: list[str] = []
            for req_id, buffered_at in pending_snapshot:
                if req_id in done_sending:
                    dropped.append(req_id)
                elif self._send_ready_to_report(req_id):
                    reportable.append(req_id)
                elif (
                    req_id not in self._reqs_to_send
                    and req_id not in self._reqs_to_process
                    and now - buffered_at > orphan_after
                ):
                    # Neither running nor awaiting a lease: request left the
                    # engine without registering a send lease.
                    orphaned.append(req_id)
            with self._send_done_pending_lock:
                for req_id in (*dropped, *orphaned, *reportable):
                    self._send_done_pending.pop(req_id, None)
                for req_id in (*dropped, *orphaned):
                    self._lease_renewals.pop(req_id, None)
            successful.update(reportable)

        for req_id in successful:
            self._reqs_to_send.pop(req_id, None)
            self._reqs_to_process.discard(req_id)
            self._lease_renewals.pop(req_id, None)
            self.consumer_notification_counts_by_req.pop(req_id, None)
            done_sending.add(req_id)

        retained = {
            req_id
            for req_id in done_sending
            if req_id not in lease_expired and self._chunk_should_retain(req_id)
        }
        if retained:
            done_sending -= retained
            # Still owned by the chunk state machine (e.g. bidirectional notif
            # arrived early): re-arm the lease and buffer the completion until
            # the last chunk completes.
            with self._send_done_pending_lock:
                for req_id in retained:
                    self._reqs_to_send.setdefault(
                        req_id, now + self._kv_lease_duration_cfg
                    )
                    self._send_done_pending.setdefault(req_id, now)

        for req_id in done_sending:
            self._evict_finished_inbox.put(req_id)
        if done_sending:
            self._push_writer_wake.set()

        return results

    def get_finished(self) -> tuple[set[str], set[str]]:
        results = self.get_transfer_results()
        return results.finished_sending, results.finished_recving

    def _cleanup_remote_engine(
        self, engine_id: str, *, log_eviction: bool = True
    ) -> None:
        with self._handshake_lock:
            push_by_addr = getattr(self, "_push_engine_by_address", None)
            if isinstance(push_by_addr, dict):
                for addr_key, eid in list(push_by_addr.items()):
                    if eid == engine_id:
                        del push_by_addr[addr_key]
        super()._cleanup_remote_engine(engine_id, log_eviction=log_eviction)
        for attr in (
            "region_pairings",
            "_compiled_xfer_plans",
            "_remote_desc_cumsum",
            "_remote_kv_layer_end_by_engine",
            "_remote_supports_block_starts",
            "_remote_group_is_ssm_by_engine",
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

    def shutdown(self):
        if self._stager is not None:
            self._stager.stop()
            self._stager.join(timeout=5.0)
            self._stager = None
        if self._pipeline is not None:
            self._pipeline.shutdown()
            self._pipeline = None
        super().shutdown()


class IluNixlPushConnectorWorker(
    IluNixlPushConnectorWorkerMixin, NixlPushConnectorWorker
):
    """Iluvatar heterogeneous Push PD worker (Single-Owner chunked-bulk)."""
