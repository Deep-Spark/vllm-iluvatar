# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ilu NIXL connector facade.

This package keeps PP-aware NIXL changes out of the official ``v1.nixl``
implementation while reusing its transfer logic.
"""

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
)
from vllm.forward_context import ForwardContext

from .pull_scheduler import IluNixlPullConnectorScheduler
from .pull_worker import IluNixlPullConnectorWorker

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


class IluNixlBaseConnector(NixlBaseConnector):
    def wait_for_save(self):
        assert self.connector_worker is not None
        if self.connector_worker.fp8_wire_staging_enabled:
            self.connector_worker.export_fp8_wire_staging(self._connector_metadata)
        return super().wait_for_save()


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
