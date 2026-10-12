# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ilu NIXL connector facade.

This package keeps PP-aware NIXL changes out of the official ``v1.nixl``
implementation while reusing its transfer logic.
"""

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger

from .pull_scheduler import IluNixlPullConnectorScheduler
from .pull_worker import IluNixlPullConnectorWorker

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


class IluNixlBaseConnector(NixlBaseConnector):
    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        assert self.connector_scheduler is not None
        logger.info(
            "IluNixlConnector setting global-rank handshake metadata for ranks=%s",
            sorted(metadata.keys()),
        )
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        tp_size = self._vllm_config.parallel_config.tensor_parallel_size
        global_rank_metadata = {
            pp_rank * tp_size + tp_rank: rank_metadata
            for (pp_rank, tp_rank), rank_metadata in metadata.items()
        }
        logger.info(
            "IluNixlConnector converted PP-aware handshake metadata keys=%s "
            "to global ranks=%s",
            sorted(metadata.keys()),
            sorted(global_rank_metadata.keys()),
        )
        self.set_xfer_handshake_metadata(global_rank_metadata)


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
