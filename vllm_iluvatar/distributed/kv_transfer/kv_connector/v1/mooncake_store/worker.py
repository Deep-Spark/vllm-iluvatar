# SPDX-License-Identifier: Apache-2.0
"""Worker-side component for Iluvatar MooncakeStoreConnector on CoreX."""

from __future__ import annotations

import dataclasses
import os
import queue
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dcp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    resolve_dcp_kv_cache_spec,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    group_kernel_blocks,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from .config import (
    IluMooncakeStoreConfig,
    get_configured_preferred_segment,
    get_requester_local_hostname,
    parse_bool,
    resolve_pool_dram_bytes,
    setup_ilu_mooncake_store,
)
from .coordinator import ExternalCachedBlockPool, MooncakeStoreCoordinator
from .layout import (
    ChunkedTokenDatabase,
    GroupStoreLayout,
    KeyMetadata,
    LBHNCStoreLayout,
    LBNHCStoreLayout,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreWorkerMetadata,
    PoolKey,
    RankNamespace,
    ReqMeta,
    TPShardedStoreLayout,
    compute_store_topology_namespace,
    plan_group_store_regions,
    resolve_group_pp_owners,
    resolve_store_tp_size,
)
from .lookup_rpc import LookupKeyClient, LookupKeyServer
from .metrics import MooncakeStoreConnectorStats
from .protocol import MooncakeLookupResult, TailKeyBoundary
from .transfer_threads import (
    DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
)

logger = init_logger(__name__)

__all__ = [
    "IluMooncakeStoreWorker",
    "KVCacheStoreRecvingThread",
    "KVCacheStoreSendingThread",
    "KVTransferThread",
    "LookupKeyClient",
    "LookupKeyServer",
    "MooncakeStoreWorker",
]


def _resolve_parallel_ranks(
    parallel_config: Any,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    """Return (dp_rank, tp_rank, tp_size, pp_rank, pp_size, pcp_rank, pcp_size, dcp_rank, dcp_size)."""
    dp_rank = int(getattr(parallel_config, "data_parallel_index", 0))
    rank = int(getattr(parallel_config, "rank", 0))
    tp_size_cfg = max(1, int(getattr(parallel_config, "tensor_parallel_size", 1)))
    pp_size = max(1, int(getattr(parallel_config, "pipeline_parallel_size", 1)))
    pcp_size_cfg = max(
        1, int(getattr(parallel_config, "prefill_context_parallel_size", 1))
    )
    dcp_size_cfg = max(
        1, int(getattr(parallel_config, "decode_context_parallel_size", 1))
    )

    try:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
    except Exception:
        tp_size = tp_size_cfg
        tp_rank = rank % tp_size

    pp_rank = (rank // tp_size) % pp_size

    try:
        pcp_size = get_pcp_group().world_size
        pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
    except Exception:
        pcp_size = pcp_size_cfg
        pcp_rank = (rank // (tp_size * pp_size)) % pcp_size if pcp_size > 1 else 0

    try:
        dcp_size = get_dcp_group().world_size
        dcp_rank = get_dcp_group().rank_in_group if dcp_size > 1 else 0
    except Exception:
        dcp_size = dcp_size_cfg
        dcp_rank = tp_rank % dcp_size if dcp_size > 1 else 0

    return (
        dp_rank,
        tp_rank,
        tp_size,
        pp_rank,
        pp_size,
        pcp_rank,
        pcp_size,
        dcp_rank,
        dcp_size,
    )


def _detect_current_cuda_device(parallel_config: Any) -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            return int(torch.accelerator.current_device_index())
        return int(torch.cuda.current_device())
    except Exception:
        local_rank = int(getattr(parallel_config, "local_rank", 0))
        count = torch.cuda.device_count()
        return local_rank % count if count > 0 else None


def _load_mooncake_store_classes() -> tuple[Callable[[], Any], Callable[[], Any]]:
    try:
        from mooncake.store import (  # type: ignore[import-not-found]
            MooncakeDistributedStore,
            ReplicateConfig,
        )

        return MooncakeDistributedStore, ReplicateConfig
    except ImportError as exc:
        raise ImportError(
            "mooncake-transfer-engine-corex is not installed. Build and install "
            "the CoreX Mooncake wheel (COREX_BUILD=1 bash scripts/build_wheel.sh) "
            "to use IluMooncakeStoreConnector."
        ) from exc


class IluMooncakeStoreWorker:
    """Worker-side component managing CoreX GPU/Host KV registration and transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        *,
        store_factory: Callable[[], Any] | None = None,
        replicate_config_factory: Callable[[], Any] | None = None,
    ) -> None:
        assert vllm_config.kv_transfer_config is not None
        self.vllm_config = vllm_config
        self._kv_cache_config = kv_cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self._extra_config = extra_config

        if store_factory is None or replicate_config_factory is None:
            default_store_cls, default_rep_cls = _load_mooncake_store_classes()
            store_factory = store_factory or default_store_cls
            replicate_config_factory = replicate_config_factory or default_rep_cls

        (
            self.dp_rank,
            self.tp_rank,
            self.tp_size,
            self.pp_rank,
            self.pp_size,
            self.pcp_rank,
            self.pcp_size,
            self.dcp_rank,
            self.dcp_size,
        ) = _resolve_parallel_ranks(parallel_config)
        self.rank = int(getattr(parallel_config, "rank", 0))
        self.world_size = max(1, int(getattr(parallel_config, "world_size", 1)))
        self.device_index = _detect_current_cuda_device(parallel_config)
        if self.device_index is not None:
            torch.cuda.set_device(self.device_index)

        kv_role = vllm_config.kv_transfer_config.kv_role
        assert kv_role is not None
        self.kv_role = kv_role

        is_consumer = self.kv_role == "kv_consumer"
        default_save_decode = not parse_bool(
            extra_config.get("offload_prompt_only", True), default=True
        )
        self.save_decode_cache = parse_bool(
            extra_config.get("save_decode_cache", default_save_decode),
            default=default_save_decode,
        )
        self.consumer_is_to_put = parse_bool(
            extra_config.get("consumer_is_to_put", False), default=False
        )
        self.can_put = (
            parse_bool(extra_config.get("enable_store", True), default=True)
            if not is_consumer
            else (self.consumer_is_to_put or self.save_decode_cache)
        )
        default_lookup = (
            parse_bool(extra_config.get("consumer_is_to_load", False), default=False)
            if is_consumer
            else True
        )
        enable_lookup = parse_bool(
            extra_config.get("enable_lookup", default_lookup), default=default_lookup
        )
        self.load_async = parse_bool(extra_config.get("load_async", True), default=True)
        self._capacity_only = is_consumer and not enable_lookup and not self.can_put

        self.cache_config = vllm_config.cache_config
        self._is_hma_required = len(kv_cache_config.kv_cache_groups) > 1
        self.block_size, self.hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self.use_mla = bool(getattr(model_config, "use_mla", False))
        self.num_kv_head = (
            1
            if self.use_mla
            else int(model_config.get_total_num_kv_heads())
        )

        self._store_group_ids = tuple(kv_cache_config.prefix_cacheable_group_ids)
        self._group_pp_owners = (
            resolve_group_pp_owners(
                vllm_config, kv_cache_config, self._store_group_ids
            )
            if not self._capacity_only
            else ()
        )
        active_pp_ranks = {
            stage for owners in self._group_pp_owners for stage in owners
        }

        store_config = IluMooncakeStoreConfig.load(extra_config)
        self.store_config = store_config
        self._mem_pool: Any = None

        contribute_memory = parse_bool(
            extra_config.get("contribute_memory", True), default=True
        )
        contribute_on_empty = parse_bool(
            extra_config.get("contribute_memory_on_empty_rank", False),
            default=False,
        )
        if not self._capacity_only and self.pp_size > 1 and active_pp_ranks:
            rank_owns_store_kv = (
                self.pp_rank in active_pp_ranks or contribute_on_empty
            )
            participating_ranks = (
                max(1, len(active_pp_ranks)) * self.tp_size * self.pcp_size
            )
        else:
            rank_owns_store_kv = True
            participating_ranks = self.world_size

        pool_dram_bytes = resolve_pool_dram_bytes(extra_config)
        effective_global_seg = store_config.resolve_rank_segment_size(
            participating_ranks=participating_ranks,
            contribute_memory=contribute_memory and rank_owns_store_kv,
            pool_dram_bytes=pool_dram_bytes,
        )
        effective_local_buf = (
            store_config.local_buffer_size
            if (contribute_memory and rank_owns_store_kv)
            or store_config.mode == "standalone-store"
            or store_config.protocol == "tcp"
            else 0
        )

        rank_ssd_path: str | None = None
        if (
            store_config.enable_offload
            and effective_global_seg > 0
            and store_config.ssd_offload_path
        ):
            safe_engine = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                vllm_config.kv_transfer_config.engine_id or "default",
            )
            rank_ssd_path = os.path.join(
                store_config.ssd_offload_path,
                f"{safe_engine}_dp{self.dp_rank}_r{self.rank}",
            )
            os.makedirs(rank_ssd_path, exist_ok=True)

        self.store = store_factory()
        local_hostname = get_requester_local_hostname(get_ip(), extra_config)
        setup_ilu_mooncake_store(
            self.store,
            store_config,
            local_hostname,
            global_segment_size=effective_global_seg,
            local_buffer_size=effective_local_buf,
            rank_ssd_path=rank_ssd_path,
        )

        preferred_segment = get_configured_preferred_segment(extra_config, store_config)
        self.preferred_segment = preferred_segment
        self.store_replicate_config = replicate_config_factory()
        self.enable_group_semantics = parse_bool(
            extra_config.get("enable_group_semantics", False), default=False
        )
        self._supports_group_ids = hasattr(self.store_replicate_config, "group_ids")
        is_loopback_host = local_hostname.split(":", 1)[0] in (
            "127.0.0.1",
            "localhost",
            "::1",
        )
        if preferred_segment is not None:
            self.store_replicate_config.preferred_segment = preferred_segment
        elif (
            store_config.prefer_alloc_in_same_node
            and not is_loopback_host
            and hasattr(self.store_replicate_config, "prefer_alloc_in_same_node")
            and getattr(self.store_replicate_config, "replica_num", 1) == 1
            and getattr(self.store_replicate_config, "nof_replica_num", 0) == 0
        ):
            self.store_replicate_config.prefer_alloc_in_same_node = True

        logger.info(
            "Initialized IluMooncakeStoreWorker rank=%d/%d (pp=%d/%d, tp=%d/%d): "
            "mode=%s protocol=%s device=%r master=%s segment=%dB local_buf=%dB "
            "prefer_same_node=%s",
            self.rank,
            self.world_size,
            self.pp_rank,
            self.pp_size,
            self.tp_rank,
            self.tp_size,
            store_config.mode,
            store_config.protocol,
            store_config.device_name,
            store_config.master_server_address,
            effective_global_seg,
            effective_local_buf,
            store_config.prefer_alloc_in_same_node,
        )

        self.disk_offload_buffer_budget_bytes = (
            DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES
            if store_config.enable_offload
            else None
        )

        kv_event_config = getattr(vllm_config, "kv_events_config", None)
        self.enable_kv_events = bool(
            kv_event_config and kv_event_config.enable_kv_cache_events
        )

        self.kv_send_thread: KVCacheStoreSendingThread | None = None
        self.kv_recv_threads: list[KVCacheStoreRecvingThread] = []
        self.num_recv_threads = max(
            1,
            int(
                extra_config.get(
                    "num_recv_threads",
                    getattr(envs, "VLLM_MOONCAKE_LOAD_RECV_THREADS", 1),
                )
            ),
        )
        self.recv_request_queue: queue.Queue[Any] = queue.Queue()
        self.finished_store_req: set[str] = set()
        self._registered_buffer_addrs: list[int] = []
        self._has_device_regions = False
        self._kv_connector_stats_lock = threading.Lock()
        self.kv_connector_stats = MooncakeStoreConnectorStats()
        self.token_dbs: list[ChunkedTokenDatabase] = []
        self.lookup_server: LookupKeyServer | None = None

        if self._capacity_only:
            logger.info(
                "Mooncake store worker rank=%d running in capacity-only mode "
                "(segment=%d bytes)",
                self.rank,
                effective_global_seg,
            )
            return

        self._kv_cache_groups = [
            dataclasses.replace(
                group,
                kv_cache_spec=resolve_dcp_kv_cache_spec(
                    group.kv_cache_spec,
                    self.dcp_size,
                ),
            )
            for group in kv_cache_config.prefix_cacheable_groups
        ]
        spec_cfg = getattr(vllm_config, "speculative_config", None)
        use_eagle_block_drop = bool(
            spec_cfg.use_eagle_block_drop()
            if spec_cfg is not None
            and callable(getattr(spec_cfg, "use_eagle_block_drop", None))
            else False
        )
        self.coord = MooncakeStoreCoordinator(
            self._kv_cache_groups,
            scheduler_block_size=self.block_size,
            hash_block_size=self.hash_block_size,
            use_eagle=use_eagle_block_drop,
            retention_interval=kv_cache_config.prefix_cache_retention_interval,
            dcp_world_size=self.dcp_size,
        )

        self.store_tp_size, extra_ns, store_layout_cls = self._select_store_layout(
            extra_config
        )
        store_namespace = compute_store_topology_namespace(
            vllm_config,
            kv_cache_config,
            hash_block_size=self.hash_block_size,
            extra_namespace=extra_ns,
        )
        model_name = str(getattr(model_config, "model", "model")).rstrip("/").split("/")[-1]
        metadata = KeyMetadata(
            model_name=model_name,
            tp_rank=self.tp_rank,
            pcp_rank=self.pcp_rank,
            dcp_rank=self.dcp_rank,
            pp_rank=self.pp_rank,
            cache_prefix=str(extra_config.get("cache_prefix", "")),
            store_namespace=store_namespace,
        )
        self._group_tp_replication_factors = (
            self._compute_group_tp_replication_factors()
        )
        self.token_dbs = self._build_token_databases(metadata, store_layout_cls)
        self._init_lookup_key_prefixes()

        if self.rank == 0:
            self.lookup_server = LookupKeyServer(self, vllm_config)

    def _supports_tp_sharded_store_layout(
        self,
        layout_cls: type[TPShardedStoreLayout] | None,
        extra_config: Mapping[str, Any],
    ) -> bool:
        if (
            layout_cls is None
            or self.pcp_size != 1
            or self.dcp_size != 1
            or len(self._kv_cache_groups) != 1
        ):
            return False
        return (
            type(self._kv_cache_groups[0].kv_cache_spec) is FullAttentionSpec
            and not parse_bool(
                extra_config.get("enable_cross_layers_blocks", False), default=False
            )
        )

    def _select_store_layout(
        self, extra_config: Mapping[str, Any]
    ) -> tuple[int | None, str, type[TPShardedStoreLayout] | None]:
        lcm_enabled = extra_config.get("enable_store_tp_lcm") is True
        if not lcm_enabled and extra_config.get("store_tp_size") is None:
            return None, "", None

        requested_tp = resolve_store_tp_size(extra_config)
        cache_layout = self.cache_config.get_resolved_kv_cache_layout()
        layout_cls: type[TPShardedStoreLayout] | None = {
            KVCacheLayout.LBHNC: LBHNCStoreLayout,
            KVCacheLayout.LBNHC: LBNHCStoreLayout,
        }.get(cache_layout)

        if (
            requested_tp is not None
            and requested_tp >= self.tp_size
            and requested_tp % self.tp_size == 0
            and self._supports_tp_sharded_store_layout(layout_cls, extra_config)
        ):
            assert layout_cls is not None
            if self.num_kv_head % requested_tp == 0:
                return (
                    requested_tp,
                    layout_cls.shared_namespace(requested_tp, self.pp_size),
                    layout_cls,
                )
            if self.num_kv_head == 1:
                return (
                    None,
                    f"@store_pp:{self.pp_size}@store_format:tp_shared_mqa",
                    None,
                )

        layout_name = getattr(cache_layout, "name", str(cache_layout))
        logger.warning(
            "TP-sharded Mooncake store layout unavailable for layout=%s; "
            "falling back to rank-local GroupStoreLayout.",
            layout_name,
        )
        return (
            None,
            f"@store_pp:{self.pp_size}@store_format:rank_local_tp{self.tp_size}_{layout_name}",
            None,
        )

    def _spec_tp_replication_factor(self, spec: KVCacheSpec) -> int:
        if self.dcp_size > 1:
            return 1
        inner_specs = (
            tuple(spec.kv_cache_specs.values())
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else (spec,)
        )
        if any(isinstance(inner, MambaSpec) for inner in inner_specs):
            return 1
        if all(
            isinstance(inner, (MLAAttentionSpec, SlidingWindowMLASpec))
            for inner in inner_specs
        ):
            return self.tp_size
        return max(1, self.tp_size // max(1, self.num_kv_head))

    def _compute_group_tp_replication_factors(self) -> tuple[int, ...]:
        return tuple(
            self._spec_tp_replication_factor(group.kv_cache_spec)
            for group in self._kv_cache_groups
        )

    def _build_token_databases(
        self,
        metadata: KeyMetadata,
        layout_cls: type[TPShardedStoreLayout] | None,
    ) -> list[ChunkedTokenDatabase]:
        token_dbs: list[ChunkedTokenDatabase] = []
        for g_idx, group in enumerate(self._kv_cache_groups):
            hbs = (
                self.hash_block_size
                if group.kv_cache_spec.prefix_cacheable
                else group.kv_cache_spec.block_size
            )
            group_tp_rank = self.tp_rank
            if layout_cls is None:
                group_tp_rank //= self._group_tp_replication_factors[g_idx]
            group_meta = dataclasses.replace(
                metadata,
                group_id=g_idx,
                tp_rank=group_tp_rank,
            )
            store_layout: TPShardedStoreLayout | GroupStoreLayout
            if layout_cls is not None:
                assert self.store_tp_size is not None
                store_layout = layout_cls(
                    group_meta,
                    group.kv_cache_spec.block_size,
                    hbs,
                    local_tp_size=self.tp_size,
                    store_tp_size=self.store_tp_size,
                    tp_rank=self.tp_rank,
                    num_kv_heads=self.num_kv_head,
                )
            else:
                store_layout = GroupStoreLayout(
                    group_meta, group.kv_cache_spec.block_size, hbs
                )
            token_dbs.append(
                ChunkedTokenDatabase(
                    group_meta,
                    group.kv_cache_spec.block_size,
                    hash_block_size=hbs,
                    store_layout=store_layout,
                )
            )
        return token_dbs

    def _init_lookup_key_prefixes(self) -> None:
        def rank_namespaces(
            factor: int, pp_owners: Sequence[int]
        ) -> tuple[RankNamespace, ...]:
            if not pp_owners:
                return ()
            if self.dcp_size > 1:
                return tuple(
                    (tp_r, pcp_r, tp_r % self.dcp_size, pp_r)
                    for pcp_r in range(self.pcp_size)
                    for tp_r in range(self.tp_size)
                    for pp_r in pp_owners
                )
            return tuple(
                (shard_r, pcp_r, 0, pp_r)
                for pcp_r in range(self.pcp_size)
                for shard_r in range(max(1, self.tp_size // factor))
                for pp_r in pp_owners
            )

        self._lookup_key_prefixes = tuple(
            db.store_layout.lookup_key_prefixes(
                rank_namespaces(
                    self._group_tp_replication_factors[g_idx],
                    self._group_pp_owners[g_idx],
                )
            )
            for g_idx, db in enumerate(self.token_dbs)
        )
        self._group_storing_workers = tuple(
            max(1, self.tp_size // self._group_tp_replication_factors[g_idx])
            * self.pcp_size
            * len(self._group_pp_owners[g_idx])
            for g_idx in range(len(self.token_dbs))
        )

    @property
    def group_tp_replication_factors(self) -> tuple[int, ...]:
        return self._group_tp_replication_factors

    @property
    def group_storing_workers(self) -> tuple[int, ...]:
        return self._group_storing_workers

    def get_mem_pool_context(self) -> AbstractContextManager | None:
        if self._mem_pool is None:
            return None
        return torch.cuda.use_mem_pool(self._mem_pool)

    def register_kv_caches(
        self,
        kv_caches: Mapping[str, Any],
        kv_cache_scales: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if self._capacity_only:
            return

        num_blocks = int(self._kv_cache_config.num_blocks)
        uses_tp_sharded = bool(
            self.token_dbs
            and isinstance(self.token_dbs[0].store_layout, TPShardedStoreLayout)
        )

        if uses_tp_sharded and kv_caches:
            store_layers = {
                name for group in self._kv_cache_groups for name in group.layer_names
            }
            cache_tensors = [
                group_kernel_blocks(
                    cache if isinstance(cache, torch.Tensor) else cache[0],
                    num_blocks,
                )
                for name, cache in kv_caches.items()
                if name in store_layers
            ]
            seen_addrs: set[int] = set()
            for cache in cache_tensors:
                storage = cache.untyped_storage()
                s_addr = int(storage.data_ptr())
                s_len = int(storage.nbytes())
                if s_addr in seen_addrs or s_len <= 0:
                    continue
                seen_addrs.add(s_addr)
                ret = self.store.register_buffer(s_addr, s_len)
                if ret != 0:
                    raise RuntimeError(
                        f"Mooncake register_buffer failed at {s_addr:#x} "
                        f"len={s_len}: ret={ret}"
                    )
                self._registered_buffer_addrs.append(s_addr)
                if cache.is_cuda:
                    self._has_device_regions = True
            for db in self.token_dbs:
                assert isinstance(db.store_layout, TPShardedStoreLayout)
                if cache_tensors:
                    db.store_layout.register_kv_caches(cache_tensors, num_blocks)
        elif kv_caches:
            comp_cfg = getattr(self.vllm_config, "compilation_config", None)
            forward_context = (
                getattr(comp_cfg, "static_forward_context", None)
                if comp_cfg is not None
                else None
            )
            coalesce = parse_bool(
                self._extra_config.get("coalesce_regions", True), default=True
            )
            honor_skip = parse_bool(
                self._extra_config.get("honor_skip_layers", False), default=False
            )
            per_group_regions, registrations = plan_group_store_regions(
                self._kv_cache_config,
                self._store_group_ids,
                kv_caches,
                forward_context=forward_context,
                kv_cache_scales=kv_cache_scales,
                coalesce=coalesce,
                honor_skip_layers=honor_skip,
            )
            for s_addr, s_len in registrations:
                if s_len <= 0:
                    continue
                ret = self.store.register_buffer(s_addr, s_len)
                if ret != 0:
                    raise RuntimeError(
                        f"Mooncake register_buffer failed at {s_addr:#x} "
                        f"len={s_len}: ret={ret}"
                    )
                self._registered_buffer_addrs.append(s_addr)

            for db, regions in zip(self.token_dbs, per_group_regions, strict=True):
                assert isinstance(db.store_layout, GroupStoreLayout)
                db.store_layout.bind_regions(regions)

            for value in kv_caches.values():
                t = (
                    value
                    if isinstance(value, torch.Tensor)
                    else (value[0] if isinstance(value, (list, tuple)) and value else None)
                )
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    self._has_device_regions = True
                    break

        group_participates = [
            group.kv_cache_spec.prefix_cacheable and db.store_layout.has_regions
            for group, db in zip(self._kv_cache_groups, self.token_dbs, strict=True)
        ]
        total_regions = sum(
            len(db.store_layout.regions)
            if isinstance(db.store_layout, GroupStoreLayout)
            else int(db.store_layout.has_regions)
            for db in self.token_dbs
        )
        logger.info(
            "Registered Mooncake KV caches on rank=%d (pp=%d, tp=%d): "
            "groups=%d participating=%s regions=%d buffers=%d",
            self.rank,
            self.pp_rank,
            self.tp_rank,
            len(self.token_dbs),
            group_participates,
            total_regions,
            len(self._registered_buffer_addrs),
        )

        # Always start send/recv threads even on zero-region PP stages so global
        # store_job_id fencing and async-load finished_recving aggregation complete.
        if self.can_put:
            ready_send = threading.Event()
            group_put_steps = [
                1 if isinstance(db.store_layout, TPShardedStoreLayout) else factor
                for db, factor in zip(
                    self.token_dbs,
                    self._group_tp_replication_factors,
                    strict=True,
                )
            ]
            self.kv_send_thread = KVCacheStoreSendingThread(
                self.store,
                self.coord,
                self.token_dbs,
                self.block_size,
                self.tp_rank,
                group_put_steps,
                self.kv_role,
                ready_send,
                device_index=self.device_index,
                enable_kv_event=self.enable_kv_events,
                replicate_config=self.store_replicate_config,
                enable_group_semantics=self.enable_group_semantics,
                supports_group_ids=self._supports_group_ids,
                record_operation=self._record_kv_connector_operation,
                group_participates=group_participates,
            )
            self.kv_send_thread.start()
            ready_send.wait()

        self.kv_recv_threads = []
        ready_recvs: list[threading.Event] = []
        for idx in range(self.num_recv_threads):
            ev = threading.Event()
            t = KVCacheStoreRecvingThread(
                self.store,
                self.coord,
                self.token_dbs,
                self.block_size,
                self.tp_rank,
                ev,
                device_index=self.device_index,
                prefer_alloc_in_same_node=self.store_config.prefer_alloc_in_same_node,
                disk_offload_buffer_budget_bytes=self.disk_offload_buffer_budget_bytes,
                record_operation=self._record_kv_connector_operation,
                request_queue=self.recv_request_queue,
                group_participates=group_participates,
                is_hma_required=self._is_hma_required,
            )
            t.name = f"KVCacheStoreRecvingThread-{idx}"
            t.start()
            self.kv_recv_threads.append(t)
            ready_recvs.append(ev)
        for ev in ready_recvs:
            ev.wait()

    def start_load_kv(self, metadata: MooncakeStoreConnectorMetadata) -> None:
        if self._capacity_only:
            return
        enqueued = 0
        for request in metadata.requests:
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:
                continue
            load_spec.token_len = load_spec.kvpool_cached_tokens
            self.recv_request_queue.put(request)
            enqueued += 1
        if enqueued > 0 and not self.load_async:
            self.recv_request_queue.join()

    def wait_for_save(self, metadata: MooncakeStoreConnectorMetadata) -> None:
        if self._capacity_only or not self.can_put:
            return
        current_event = None
        if self._has_device_regions and torch.cuda.is_available():
            for request in metadata.requests:
                if request.can_save:
                    current_event = torch.cuda.Event()
                    current_event.record()
                    break
        for request in metadata.requests:
            if not request.can_save:
                continue
            request.current_event = current_event
            assert self.kv_send_thread is not None
            self.kv_send_thread.add_request(request)

    def get_finished(
        self, finished_req_ids: set[str], meta: MooncakeStoreConnectorMetadata
    ) -> tuple[set[str], set[str]]:
        if self._capacity_only:
            return set(), set()
        if self.can_put and self.kv_send_thread is not None:
            self._close_ended_store_requests(finished_req_ids, meta)

        done_sending: set[str] = set()
        done_recving: set[str] = set()
        for recv_thread in self.kv_recv_threads:
            done_recving |= recv_thread.get_and_clear_finished_requests()
        if not self.load_async:
            done_recving.clear()
        return done_sending, done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        block_ids: set[int] = set()
        for recv_thread in self.kv_recv_threads:
            block_ids |= recv_thread.get_and_clear_block_ids_with_load_errors()
        return block_ids

    def get_transfer_results(
        self, finished_req_ids: set[str], meta: MooncakeStoreConnectorMetadata
    ) -> KVConnectorTransferResults:
        done_sending, done_recving = self.get_finished(finished_req_ids, meta)
        if self._capacity_only:
            return KVConnectorTransferResults(done_sending, done_recving)
        failed_recving: set[str] = set()
        for recv_thread in self.kv_recv_threads:
            failed_recving |= recv_thread.get_and_clear_failed_requests()
        return KVConnectorTransferResults(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed_recving,
        )

    def _record_kv_connector_operation(
        self,
        operation: str,
        duration_seconds: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        with self._kv_connector_stats_lock:
            self.kv_connector_stats.record_operation(
                operation=operation,
                duration_seconds=duration_seconds,
                num_keys=num_keys,
                num_bytes=num_bytes,
                status=status,
                num_failed_keys=num_failed_keys,
            )

    def get_kv_connector_stats(self) -> MooncakeStoreConnectorStats | None:
        with self._kv_connector_stats_lock:
            if self.kv_connector_stats.is_empty():
                return None
            stats = self.kv_connector_stats
            self.kv_connector_stats = MooncakeStoreConnectorStats()
            return stats

    def handle_preemptions(self, meta: MooncakeStoreConnectorMetadata) -> None:
        if self._capacity_only or not self.can_put or self.kv_send_thread is None:
            return
        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(req_id)
        setattr(meta, "_preemptions_handled", True)

    def _close_ended_store_requests(
        self,
        finished_req_ids: set[str],
        meta: MooncakeStoreConnectorMetadata,
    ) -> None:
        assert self.kv_send_thread is not None
        if not getattr(meta, "_preemptions_handled", False):
            for req_id in meta.preempted_req_ids:
                self.kv_send_thread.delete_finished_stored_request(req_id)
        for req_id in finished_req_ids | self.finished_store_req:
            if self.kv_send_thread.stored_requests.get(req_id):
                self.finished_store_req.add(req_id)
            else:
                self.finished_store_req.discard(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)

    def build_connector_worker_meta(self) -> MooncakeStoreWorkerMetadata | None:
        if self.kv_send_thread is None:
            return None
        completed = self.kv_send_thread.take_completed_saves()
        if not completed:
            return None
        return MooncakeStoreWorkerMetadata(completed_saves=completed)

    def lookup(
        self, num_tokens: int, block_hashes: Sequence[BlockHash]
    ) -> MooncakeLookupResult:
        if self._capacity_only or self.store is None:
            return MooncakeLookupResult(0)

        token_len = self.coord.align_lookup_length(num_tokens)
        if not block_hashes or token_len <= 0:
            return MooncakeLookupResult(0)

        candidate_keys: list[str] = []
        candidate_meta: list[tuple[int, bytes, int]] = []
        fine_grained = self.coord.enable_partial_hash_hits
        lookup_masks = None if fine_grained else self.coord.lookup_mask(token_len)

        for g_idx, db in enumerate(self.token_dbs):
            if not self._kv_cache_groups[g_idx].kv_cache_spec.prefix_cacheable:
                continue
            key_prefixes = self._lookup_key_prefixes[g_idx]
            if not key_prefixes:
                continue
            spec_block_size = db.block_size
            if fine_grained:
                max_units = min(len(block_hashes), token_len // self.hash_block_size)
                unit_ids: range | list[int] = range(max_units)
                group_hashes: Sequence[BlockHash] = block_hashes
            else:
                assert lookup_masks is not None
                lookup_mask = lookup_masks[g_idx]
                group_hashes = self.coord.block_hashes_for_spec(
                    block_hashes, self._kv_cache_groups[g_idx].kv_cache_spec
                )
                max_chunks = min(len(group_hashes), cdiv(token_len, spec_block_size))
                mask_limit = (
                    max_chunks
                    if lookup_mask is None
                    else min(max_chunks, len(lookup_mask))
                )
                unit_ids = [
                    cid
                    for cid in range(mask_limit)
                    if lookup_mask is None or lookup_mask[cid]
                ]
            for cid in unit_ids:
                h = group_hashes[cid]
                hash_hex = h.hex()
                for prefix in key_prefixes:
                    candidate_keys.append(PoolKey.build_key_string(prefix, hash_hex))
                candidate_meta.append((g_idx, bytes(h), len(key_prefixes)))

        if not candidate_keys:
            return MooncakeLookupResult(0)

        lookup_start = time.perf_counter()
        try:
            res = self.store.batch_is_exist(candidate_keys)
            self._record_kv_connector_operation(
                "lookup_exists",
                time.perf_counter() - lookup_start,
                len(candidate_keys),
            )
        except Exception as exc:
            self._record_kv_connector_operation(
                "lookup_exists",
                time.perf_counter() - lookup_start,
                len(candidate_keys),
                status="error",
                num_failed_keys=len(candidate_keys),
            )
            logger.error("Mooncake lookup batch_is_exist failed: %s", exc)
            return MooncakeLookupResult(0)

        exists_set: set[tuple[int, bytes]] = set()
        pos = 0
        for g_idx, hash_bytes, count in candidate_meta:
            if all(res[pos + j] == 1 for j in range(count)):
                exists_set.add((g_idx, hash_bytes))
            pos += count

        cached_block_pool = ExternalCachedBlockPool(self.hash_block_size, exists_set)
        _, hit_length = self.coord.find_longest_cache_hit(
            block_hashes,
            token_len,
            cached_block_pool,
        )
        if hit_length >= num_tokens:
            usable = self.coord.align_lookup_length(num_tokens - 1)
            if usable <= 0:
                return MooncakeLookupResult(0)
            _, hit_length = self.coord.find_longest_cache_hit(
                block_hashes,
                usable,
                cached_block_pool,
            )
        return MooncakeLookupResult(
            hit_length,
            self._tail_key_boundaries(block_hashes, hit_length, cached_block_pool),
        )

    def _tail_key_boundaries(
        self,
        block_hashes: Sequence[BlockHash],
        hit_length: int,
        cached_block_pool: ExternalCachedBlockPool,
    ) -> tuple[TailKeyBoundary, ...]:
        if hit_length <= 0 or not self.coord.enable_partial_hash_hits:
            return ()
        boundaries: list[TailKeyBoundary] = []
        hit_hash_idx = hit_length // self.hash_block_size - 1
        for group_id, db in enumerate(self.token_dbs):
            if not self._kv_cache_groups[group_id].kv_cache_spec.prefix_cacheable:
                continue
            if not self._lookup_key_prefixes[group_id]:
                continue
            chunk_id = cdiv(hit_length, db.block_size) - 1
            boundary_tokens = hit_length
            contains_hit = cached_block_pool.contains(
                group_id, block_hashes[hit_hash_idx]
            )
            if not contains_hit:
                next_chunk_hash_idx = min(
                    (chunk_id + 1) * db.block_size // self.hash_block_size,
                    len(block_hashes),
                )
                for hash_idx in range(hit_hash_idx + 1, next_chunk_hash_idx):
                    if cached_block_pool.contains(group_id, block_hashes[hash_idx]):
                        boundary_tokens = (hash_idx + 1) * self.hash_block_size
                        break
                else:
                    raise AssertionError(
                        f"No tail key found for cache group {group_id} at "
                        f"hit_length={hit_length}"
                    )
            boundaries.append(TailKeyBoundary(group_id, boundary_tokens))
        return tuple(boundaries)

    def reset_store(self) -> bool:
        if self.store is None:
            return False
        try:
            if self.kv_send_thread is not None:
                self.kv_send_thread.request_queue.join()
            ret = self.store.remove_all(force=True)
            return ret is None or int(ret) >= 0
        except Exception as exc:
            logger.error("Mooncake remove_all failed: %s", exc)
            return False

    def get_kv_events(self) -> list[BlockStored]:
        if self.enable_kv_events and self.kv_send_thread is not None:
            return self.kv_send_thread.get_kv_events()
        return []

    def close(self) -> None:
        lookup_server = getattr(self, "lookup_server", None)
        if lookup_server is not None:
            self.lookup_server = None
            lookup_server.close()

        store = getattr(self, "store", None)
        if store is None:
            return
        self.store = None

        deadline = time.monotonic() + 5.0
        if self.kv_send_thread is not None:
            while (
                self.kv_send_thread.request_queue.unfinished_tasks
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            self.kv_send_thread.enqueue_stop()
            self.kv_send_thread.join(timeout=2.0)

        if self.kv_recv_threads:
            while (
                self.recv_request_queue.unfinished_tasks
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            for recv_t in self.kv_recv_threads:
                recv_t.enqueue_stop()
            for recv_t in self.kv_recv_threads:
                recv_t.join(timeout=2.0)

        for addr in reversed(getattr(self, "_registered_buffer_addrs", [])):
            try:
                store.unregister_buffer(addr)
            except Exception as exc:
                logger.debug("Mooncake unregister_buffer(%#x) ignored: %s", addr, exc)
        self._registered_buffer_addrs.clear()

        try:
            store.close()
        except Exception as exc:
            logger.warning("Error closing MooncakeDistributedStore: %s", exc)


MooncakeStoreWorker = IluMooncakeStoreWorker
