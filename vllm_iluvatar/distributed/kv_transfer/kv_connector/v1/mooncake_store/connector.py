# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MooncakeStoreConnector for cross-Prefill DRAM KV cache offloading.

Supports both standard vLLM attention/hybrid models and DeepSeek-V4.1-Flash
(paired compressed MLA KV + FP16 scales, split INT8 indexer K + FP32 scales,
bounded-replay SWA, non-uniform PP=8 partitions, and MultiConnector composition
with OptNixlConnector on Prefill nodes).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
from functools import cached_property
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    BlockStored,
    KVCacheEvent,
    KVConnectorKVEvents,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    CrossAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .config import parse_bool
from .layout import MooncakeStoreConnectorMetadata
from .metrics import MooncakeStoreConnectorStats, MooncakeStorePromMetrics
from .scheduler import IluMooncakeStoreScheduler
from .worker import IluMooncakeStoreWorker

logger = init_logger(__name__)


def _unwrap_group_spec(spec: Any) -> Any:
    if isinstance(spec, UniformTypeKVCacheSpecs) and spec.kv_cache_specs:
        return next(iter(spec.kv_cache_specs.values()))
    return spec


class MooncakeStoreKVEvents(KVConnectorKVEvents):
    """KV event aggregation across TP/PP workers (including non-uniform PP)."""

    def __init__(
        self,
        num_workers: int,
        group_tp_replication_factors: Sequence[int] = (1,),
        *,
        world_size: int | None = None,
        group_storing_workers: Sequence[int] | None = None,
    ) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be greater than zero.")
        if any(factor <= 0 for factor in group_tp_replication_factors):
            raise ValueError("TP replication factors must be greater than zero.")
        self._event_counter: Counter[KVCacheEvent] = Counter()
        self._num_workers = num_workers
        self._group_tp_replication_factors = tuple(group_tp_replication_factors)
        self._world_size = world_size
        self._group_storing_workers = (
            tuple(group_storing_workers) if group_storing_workers is not None else None
        )

    def add_events(self, events: list[KVCacheEvent]) -> None:
        if not isinstance(events, list):
            raise TypeError("events must be a list of KVCacheEvent.")
        self._event_counter.update(events)

    def _is_common_event(self, event: KVCacheEvent, count: int) -> bool:
        if isinstance(event, BlockStored) and event.group_idx is not None:
            g_idx = event.group_idx
            if (
                self._world_size is not None
                and self._group_storing_workers is not None
                and g_idx < len(self._group_storing_workers)
            ):
                expected = self._group_storing_workers[g_idx]
                if expected <= 0:
                    return False
                return count * self._world_size >= self._num_workers * expected
            factor = self._group_tp_replication_factors[g_idx]
            return count * factor >= self._num_workers
        return count >= self._num_workers

    def aggregate(self) -> MooncakeStoreKVEvents:
        common_events = self.pop_common_events()
        self._event_counter.clear()
        self._event_counter.update(common_events)
        self._num_workers = 1
        self._world_size = None
        self._group_storing_workers = None
        return self

    def pop_common_events(self) -> list[KVCacheEvent]:
        common_events = [
            event
            for event, count in self._event_counter.items()
            if self._is_common_event(event, count)
        ]
        for event in common_events:
            del self._event_counter[event]
        return common_events

    def has_events(self) -> bool:
        return bool(self._event_counter)

    def increment_workers(self, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("count must be positive.")
        self._num_workers += count

    def get_all_events(self) -> list[KVCacheEvent]:
        return list(self._event_counter.elements())

    def get_number_of_workers(self) -> int:
        return self._num_workers

    def clear_events(self) -> None:
        self._event_counter.clear()
        self._num_workers = 1

    def __repr__(self) -> str:
        return f"<MooncakeStoreKVEvents events={self.get_all_events()}>"


class MooncakeStoreConnector(KVConnectorBase_V1, SupportsHMA):
    """KV connector using MooncakeDistributedStore for cross-Prefill DRAM pooling."""

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: Any) -> bool:
        del extra_config
        return False

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        attn_cfg = getattr(vllm_config, "attention_config", None)
        if attn_cfg is not None and getattr(attn_cfg, "hisparse_config", None) is not None:
            return "BLHNC"
        model_cfg = getattr(vllm_config, "model_config", None)
        if model_cfg is None or getattr(model_cfg, "use_mla", False):
            return None
        return "LBHNC"

    @property
    def requires_kv_delivery(self) -> bool:
        """Best-effort prefix cache (matches OffloadingConnector): a dropped save
        is just a future cache miss and must not force preemption recompute."""
        return False

    @cached_property
    def _bounding_group_ids(self) -> tuple[int, ...]:
        """Prefix-cacheable groups not stored in Mooncake (bounds external hits)."""
        offloaded = set(self._kv_cache_config.prefix_cacheable_group_ids)
        return tuple(
            group_id
            for group_id, group in enumerate(self._kv_cache_config.kv_cache_groups)
            if group.kv_cache_spec.prefix_cacheable and group_id not in offloaded
        )

    @staticmethod
    def _validate_kv_cache_config(
        vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
    ) -> None:
        unsupported: list[str] = []
        store_group_ids = kv_cache_config.prefix_cacheable_group_ids
        if not store_group_ids:
            raise ValueError(
                "MooncakeStoreConnector requires at least one prefix-cacheable "
                "KV cache group"
            )
        for group_id in store_group_ids:
            spec = _unwrap_group_spec(
                kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
            )
            if isinstance(spec, CrossAttentionSpec):
                unsupported.append(f"group {group_id}: CrossAttentionSpec")
            if isinstance(spec, MambaSpec) and spec.mamba_cache_mode != "align":
                unsupported.append(
                    f"group {group_id}: mamba_cache_mode="
                    f"{spec.mamba_cache_mode!r} != 'align'"
                )
        pcp = int(getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1))
        if len(store_group_ids) > 1 and pcp > 1:
            unsupported.append(f"PCP > 1 (pcp={pcp}) with hybrid attention")
        if unsupported:
            raise ValueError(
                "MooncakeStoreConnector does not support: " + "; ".join(unsupported)
            )

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        assert kv_cache_config is not None, "kv_cache_config is required"
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}

        default_save_decode = not parse_bool(
            extra_config.get("offload_prompt_only", True), default=True
        )
        save_decode_cache = parse_bool(
            extra_config.get("save_decode_cache", default_save_decode),
            default=default_save_decode,
        )
        consumer_is_to_put = parse_bool(
            extra_config.get("consumer_is_to_put", False), default=False
        )
        default_lookup = (
            parse_bool(extra_config.get("consumer_is_to_load", False), default=False)
            if self.kv_role == "kv_consumer"
            else True
        )
        enable_lookup = parse_bool(
            extra_config.get("enable_lookup", default_lookup), default=default_lookup
        )
        self._capacity_only = (
            self.kv_role == "kv_consumer"
            and not enable_lookup
            and not save_decode_cache
            and not consumer_is_to_put
        )
        if not self._capacity_only:
            self._validate_kv_cache_config(vllm_config, kv_cache_config)

        self._kv_cache_events: MooncakeStoreKVEvents | None = None
        self._kv_cache_manager: KVCacheManager | None = None

        self.connector_scheduler: IluMooncakeStoreScheduler | None = None
        self.connector_worker: IluMooncakeStoreWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = IluMooncakeStoreScheduler(
                vllm_config, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = IluMooncakeStoreWorker(
                vllm_config, kv_cache_config
            )
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")

    def shutdown(self) -> None:
        worker = getattr(self, "connector_worker", None)
        if worker is not None:
            worker.close()
        scheduler = getattr(self, "connector_scheduler", None)
        if scheduler is not None:
            scheduler.close()

    def __del__(self) -> None:
        self.shutdown()

    # ============================================================
    # MultiConnector PP/P2P Handshake Compatibility
    # ============================================================

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """Ignore sibling OptNixlConnector handshake metadata in MultiConnector."""
        del metadata

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        """Ignore sibling OptNixlConnector PP handshake metadata in MultiConnector."""
        del metadata

    # ============================================================
    # Scheduler-side methods
    # ============================================================

    def bind_kv_cache_manager(self, kv_cache_manager: KVCacheManager) -> None:
        self._kv_cache_manager = kv_cache_manager

    def _max_loadable_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> int | None:
        if not self._bounding_group_ids or self._kv_cache_manager is None:
            return None
        coordinator = self._kv_cache_manager.coordinator
        assert isinstance(coordinator, HybridKVCacheCoordinator)
        _, per_group_hits = coordinator.find_longest_cache_hit_per_group(
            request.block_hashes, request.num_tokens - 1
        )
        bound = min(per_group_hits[gid] for gid in self._bounding_group_ids)
        return max(0, bound - num_computed_tokens)

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request,
            num_computed_tokens,
            max_num_new_tokens=self._max_loadable_tokens(request, num_computed_tokens),
        )

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.bind_gpu_block_pool(gpu_block_pool)

    def has_pending_push_work(self) -> bool:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.has_pending_push_work()

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        assert self.connector_worker is not None
        return self.connector_worker.build_connector_worker_meta()

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        del request, block_ids
        return False, None

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.register_finished_partial_tail(
            request, block_ids, partial_tail_offloads
        )

    def reset_cache(self) -> bool | None:
        if self.role == KVConnectorRole.SCHEDULER:
            assert self.connector_scheduler is not None
            self.connector_scheduler.load_specs.clear()
            self._kv_cache_events = None
            return self.connector_scheduler.reset_store()
        return None

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

        kv_cache_events = connector_output.kv_cache_events
        if not kv_cache_events or not isinstance(
            kv_cache_events, MooncakeStoreKVEvents
        ):
            return

        if self._kv_cache_events is None:
            self._kv_cache_events = kv_cache_events
        else:
            self._kv_cache_events.add_events(kv_cache_events.get_all_events())

    def take_events(self) -> Iterable[KVCacheEvent]:
        if self._kv_cache_events is not None:
            events = self._kv_cache_events.pop_common_events()
            if not self._kv_cache_events.has_events():
                self._kv_cache_events = None
            yield from events

    # ============================================================
    # Worker-side methods
    # ============================================================

    def get_mem_pool_context(self) -> AbstractContextManager | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_mem_pool_context()

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_scales: dict[str, torch.Tensor] | None = None,
    ) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches, kv_cache_scales)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        if self.connector_worker is not None and isinstance(
            kv_connector_metadata, MooncakeStoreConnectorMetadata
        ):
            self.connector_worker.handle_preemptions(kv_connector_metadata)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        del forward_context, kwargs
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeStoreConnectorMetadata)
        self.connector_worker.start_load_kv(metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs

    def wait_for_save(self) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeStoreConnectorMetadata)
        self.connector_worker.wait_for_save(metadata)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeStoreConnectorMetadata)
        return self.connector_worker.get_finished(finished_req_ids, metadata)

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeStoreConnectorMetadata)
        return self.connector_worker.get_transfer_results(finished_req_ids, metadata)

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def get_kv_connector_kv_cache_events(
        self,
    ) -> MooncakeStoreKVEvents | None:
        assert self.connector_worker is not None
        if (
            not self.connector_worker.enable_kv_events
            or self.connector_worker.kv_send_thread is None
        ):
            return None
        events = self.connector_worker.get_kv_events()
        kv_events = MooncakeStoreKVEvents(
            num_workers=1,
            group_tp_replication_factors=(
                self.connector_worker.group_tp_replication_factors
            ),
            world_size=self.connector_worker.world_size,
            group_storing_workers=self.connector_worker.group_storing_workers,
        )
        kv_events.add_events(events)
        return kv_events

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return (
            MooncakeStoreConnectorStats(data=data)
            if data is not None
            else MooncakeStoreConnectorStats()
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return MooncakeStorePromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )


class IluMooncakeStoreConnector(MooncakeStoreConnector):
    """Public alias registered as ``kv_connector=IluMooncakeStoreConnector``.

    Subclassed so ``connector.__class__.__name__`` matches the factory name
    during ``MultiConnector`` stats/metrics serialization.
    """
