# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lean worker mixin composing registration, xfer planning, and single-owner IO."""

from __future__ import annotations

import os
import queue
import threading
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    compute_nixl_compatibility_hash,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.kv_cache_interface import FullAttentionSpec

from ..ilu_nixl_common.dsv41 import resolve_dsv41_layout, validate_decode_pp_split
from ..ilu_nixl_common.types import (
    HmaTransferRoute as _HmaTransferRoute,
)
from ..ilu_nixl_common.types import (
    RegionPairing,
)
from .fp8_layout import use_fp8_ds_mla_wire_staging
from .io_actor_mixin import IluNixlIoActorMixin, _IluCompletionBatch
from .metadata import IluNixlTransferRegion
from .pp_utils import infer_pp_split
from .registration_mixin import (
    _SEMANTIC_COMPAT_BACKEND,
    IluNixlRegistrationMixin,
    _BlocksFirstTopologyBackend,
)
from .stats import IluNixlKVConnectorStats
from .xfer_plan_mixin import IluNixlXferPlanMixin, _CompiledDescBucket
from .zmq_compat import zmq_ctx


class IluNixlWorkerMixin(
    IluNixlRegistrationMixin,
    IluNixlXferPlanMixin,
    IluNixlIoActorMixin,
):
    """Composable PP/HMA/DSv4.1 NIXL worker mixin."""

    def __init__(self, *args, **kwargs):
        vllm_config = args[0] if args else kwargs.get("vllm_config")
        kv_cache_config = args[2] if len(args) > 2 else kwargs.get("kv_cache_config")
        parallel_config = getattr(vllm_config, "parallel_config", None)
        original_pp_size = getattr(parallel_config, "pipeline_parallel_size", 1)

        is_pp = original_pp_size > 1
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
        # Upstream rejects consumer PP>1 and PP+HMA during construction/registration;
        # mask PP temporarily around upstream __init__ / register_kv_caches.
        mask_pp_during_init = is_pp and (is_consumer or requires_hma)
        self._mask_pp_during_register = is_pp and requires_hma

        if mask_pp_during_init:
            try:
                parallel_config.pipeline_parallel_size = 1
            except Exception:
                object.__setattr__(parallel_config, "pipeline_parallel_size", 1)
        try:
            super().__init__(*args, **kwargs)
        finally:
            if mask_pp_during_init:
                try:
                    parallel_config.pipeline_parallel_size = original_pp_size
                except Exception:
                    object.__setattr__(
                        parallel_config, "pipeline_parallel_size", original_pp_size
                    )
        if mask_pp_during_init:
            self.pp_size = original_pp_size

        if hasattr(self, "xfer_stats") and not isinstance(
            self.xfer_stats, IluNixlKVConnectorStats
        ):
            self.xfer_stats = IluNixlKVConnectorStats()

        self.dsv41_layout = resolve_dsv41_layout(vllm_config)
        validate_decode_pp_split(vllm_config, self.dsv41_layout)
        self.fp8_wire_staging_enabled = use_fp8_ds_mla_wire_staging(vllm_config)
        self.fp8_wire_staging = None
        self.compute_device_kv_caches: dict[str, torch.Tensor] = {}
        self._virtual_packed_regions = False
        self._empty_producer_rank = False
        self._skipped_layers: list[str] = []
        self._semantic_region_engines: set[str] = set()

        pp_group = get_pp_group()
        self.pp_rank = pp_group.rank_in_group
        self.pp_world_size = pp_group.world_size
        self.pp_split = envs.VLLM_PP_LAYER_PARTITION or infer_pp_split(self.vllm_config)
        assert len(self.pp_split.split(",")) == self.pp_world_size, (
            "PP split count must equal pp_size"
        )

        self.layer_region_ids: dict[str, list[int]] = {}
        self.region_ids_by_group: list[np.ndarray] = []
        self.transfer_regions: list[IluNixlTransferRegion] = []
        self.region_pairings: dict[str, dict[int, list[RegionPairing]]] = {}
        self._local_desc_cumsum: np.ndarray | None = None
        self._remote_desc_cumsum: dict[str, dict[int, np.ndarray]] = {}
        self._remote_kv_layer_end_by_engine: dict[str, int | None] = {}
        self._remote_supports_block_starts: dict[str, bool] = {}
        self._remote_supports_structured_notif: dict[str, bool] = {}
        self._inactive_remote_keys: dict[str, set[int]] = {}
        self.local_region_ids_by_group: dict[str, dict[int, list[np.ndarray]]] = {}
        self.remote_region_ids_by_group: dict[str, dict[int, list[np.ndarray]]] = {}
        self._mamba_hma_routes: dict[str, dict[int, list[_HmaTransferRoute]]] = {}
        self._remote_fa_num_regions: dict[str, dict[int, int]] = {}
        self._selected_ssm_remote_region_ids: dict[str, dict[int, list[int]]] = {}
        self._remote_pp_size_by_engine: dict[str, int] = {}
        self._remote_pp_split_by_engine: dict[str, str] = {}
        self._compiled_desc_buckets: dict[
            str, dict[int, list[_CompiledDescBucket]]
        ] = {}
        self._pending_heartbeats_by_engine: dict[str, set[str]] = defaultdict(set)
        self._handshake_payload_cache: dict[tuple[str, int, int], bytes] = {}
        self._active_sessions = {}
        self._draining_sessions = []
        self._engine_generations = {}
        self._send_lease_started_at: dict[str, float] = {}

        transfer_cfg = getattr(vllm_config, "kv_transfer_config", None)
        get_extra = getattr(transfer_cfg, "get_from_extra_config", None)
        self._consumer_disconnect_grace = float(
            30.0 if get_extra is None else get_extra("consumer_disconnect_grace", 30.0)
        )
        io_thread_cfg = (
            True if get_extra is None else get_extra("enable_ilu_io_thread", True)
        )
        env_io = os.environ.get("VLLM_ILU_NIXL_IO_THREAD")
        self._io_thread_enabled = (
            str(env_io if env_io is not None else io_thread_cfg).lower()
            not in ("0", "false", "no")
            and getattr(self, "nixl_wrapper", None) is not None
        )
        poll_ms_raw = (
            0.5 if get_extra is None else get_extra("ilu_io_poll_interval_ms", 0.5)
        )
        self._io_poll_interval_s = max(0.0001, float(poll_ms_raw) / 1000.0)
        self._io_idle_timeout_s = 0.01
        self._io_inbox: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._io_wake_event = threading.Event()
        self._io_stop_event = threading.Event()
        self._io_completion_outbox: queue.SimpleQueue[_IluCompletionBatch] = (
            queue.SimpleQueue()
        )
        self._io_pending_reqs_by_engine: dict[str, list[tuple[str, Any]]] = defaultdict(
            list
        )
        self._io_thread: threading.Thread | None = None
        self._io_thread_started = False
        self._io_shutdown_lock = threading.Lock()

        if self.fp8_wire_staging_enabled and getattr(self, "use_host_buffer", False):
            raise ValueError(
                "IluNixl FP8 DS MLA wire staging is incompatible with use_host_buffer"
            )
        if (
            self.fp8_wire_staging_enabled
            and getattr(self.vllm_config, "speculative_config", None) is not None
        ):
            raise ValueError(
                "IluNixl FP8 DS MLA wire staging does not support speculative "
                "prefill-side execution"
            )

    def _compute_compatibility_hash(
        self, _cross_layers_blocks: bool | None = None
    ) -> str:
        from vllm_iluvatar.models.deepseek_v41.compressed_kv import format_factors

        wire_staging = getattr(self, "fp8_wire_staging_enabled", False)
        # Wire staging exposes FP8 DS MLA pages to D; omit P-only compressed
        # / split-indexer factors so the hash matches D's native fp8 layout.
        extra = {} if wire_staging else format_factors()
        backend = (
            _SEMANTIC_COMPAT_BACKEND
            if not extra
            else _SEMANTIC_COMPAT_BACKEND
            + ":"
            + ",".join(f"{k}={v}" for k, v in sorted(extra.items()))
        )
        cfg = self.vllm_config
        spec_cfg = getattr(cfg, "speculative_config", None)
        spec_method = getattr(spec_cfg, "method", None)
        # DSpark runs only on D (no draft KV transferred) and IluNixl skips
        # MTP layers during KV registration, so exclude them from the hash to
        # allow asymmetric P (no spec) <-> D (dspark/mtp) handshakes.
        strip_spec = spec_cfg is not None and spec_method in ("dspark", "mtp")
        if (wire_staging and hasattr(cfg, "cache_config")) or strip_spec:
            cache_cfg = getattr(cfg, "cache_config", None)
            if wire_staging and cache_cfg is not None:
                cache_cfg = SimpleNamespace(
                    cache_dtype="fp8_ds_mla",
                    block_size=getattr(cache_cfg, "block_size", None),
                )
            cfg = SimpleNamespace(
                model_config=getattr(cfg, "model_config", None),
                cache_config=cache_cfg,
                scheduler_config=getattr(cfg, "scheduler_config", None),
                speculative_config=None if (wire_staging or strip_spec) else spec_cfg,
            )
        mode = getattr(self, "_TRANSFER_MODE", "pull")
        try:
            return compute_nixl_compatibility_hash(cfg, backend, transfer_mode=mode)
        except TypeError:
            return compute_nixl_compatibility_hash(
                cfg, backend, bool(_cross_layers_blocks)
            )


__all__ = [
    "IluNixlWorkerMixin",
    "RegionPairing",
    "_BlocksFirstTopologyBackend",
    "_CompiledDescBucket",
    "_HmaTransferRoute",
    "_IluCompletionBatch",
    "compute_nixl_compatibility_hash",
    "get_pp_group",
    "zmq_ctx",
]
