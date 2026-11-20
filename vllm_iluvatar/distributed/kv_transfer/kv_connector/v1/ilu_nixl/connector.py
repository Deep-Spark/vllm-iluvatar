# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ilu NIXL pull connector facade inheriting upstream ``NixlBaseConnector``."""

from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
)
from vllm.forward_context import ForwardContext

from .pull_scheduler import IluNixlPullConnectorScheduler
from .pull_worker import IluNixlPullConnectorWorker
from .stats import IluNixlKVConnectorStats, IluNixlPromMetrics

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


class IluNixlBaseConnector(NixlBaseConnector):
    """PP/HMA/DSv4.1-aware NIXL pull connector reusing upstream base facade."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self.prefix_state = None

    @property
    def prefix_replay_alignment(self) -> int | None:
        """Optional token alignment for complete remote-prefix replay."""
        if self.connector_scheduler is None:
            return None
        return getattr(self.connector_scheduler, "prefix_replay_alignment", None)

    def attach_prefix_state(self, context: Any) -> None:
        """Attach DeepSeek-V4.1 E25 ProducerPrefixState / ConsumerPrefixState."""
        self.prefix_state = context

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self.prefix_state is not None:
            self.prefix_state.bind_metadata(connector_metadata)

    def set_xfer_handshake_metadata(
        self, metadata: dict[int | tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_scales: dict[str, torch.Tensor] | None = None,
    ) -> None:
        assert self.connector_worker is not None
        if kv_cache_scales is not None:
            self.connector_worker.register_kv_caches(kv_caches, kv_cache_scales)
        else:
            self.connector_worker.register_kv_caches(kv_caches)

    def wait_for_save(self):
        assert self.connector_worker is not None
        if self.connector_worker.fp8_wire_staging_enabled:
            self.connector_worker.export_fp8_wire_staging(self._connector_metadata)
        return super().wait_for_save()

    def get_transfer_results(self, finished_req_ids: set[str]):
        assert self.connector_worker is not None
        if hasattr(self.connector_worker, "get_transfer_results"):
            result = self.connector_worker.get_transfer_results()
        else:
            result = super().get_transfer_results(finished_req_ids)
        if self.prefix_state is not None:
            self.prefix_state.receive_results(result)
        return result

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return (
            IluNixlKVConnectorStats(data=data)
            if data is not None
            else IluNixlKVConnectorStats()
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return IluNixlPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )


class IluNixlPullConnector(IluNixlBaseConnector):
    """Pull-based Ilu NIXL connector."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = IluNixlPullConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = IluNixlPullConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        self.connector_worker.start_load_kv(self._connector_metadata)


class IluNixlConnector(IluNixlPullConnector):
    """Compatibility name for the default Ilu pull-based NIXL connector."""
