# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar NIXL push connector facade (chunked-bulk / non-layerwise)."""

from typing import TYPE_CHECKING

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.forward_context import ForwardContext

from .push_scheduler import IluNixlPushConnectorScheduler
from .push_worker import IluNixlPushConnectorWorker

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


class IluNixlPushBaseConnector(NixlBaseConnector):
    @property
    def prefix_replay_alignment(self) -> int | None:
        """Optional token alignment for complete remote-prefix replay (V4.1 CR2)."""
        if self.connector_scheduler is None:
            return None
        return getattr(self.connector_scheduler, "prefix_replay_alignment", None)

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

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        if "set_xfer_handshake_metadata_pp_aware" in NixlBaseConnector.__dict__:
            return super().set_xfer_handshake_metadata_pp_aware(metadata)

        tp_size = self._vllm_config.parallel_config.tensor_parallel_size
        global_rank_metadata = {
            pp_rank * tp_size + tp_rank: rank_metadata
            for (pp_rank, tp_rank), rank_metadata in metadata.items()
        }
        self.set_xfer_handshake_metadata(global_rank_metadata)


class IluNixlPushConnector(IluNixlPushBaseConnector):
    """Push-based Iluvatar NIXL connector (chunked-bulk / non-layerwise)."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        if vllm_config.parallel_config.decode_context_parallel_size > 1:
            raise ValueError(
                "IluNixlPushConnector does not support decode_context_parallel_size > 1."
            )
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = IluNixlPushConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = IluNixlPushConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self.connector_worker, IluNixlPushConnectorWorker)
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def bind_connector_metadata(
        self, connector_metadata: KVConnectorMetadata
    ) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self.connector_worker is not None:
            self.connector_worker.bind_connector_metadata(connector_metadata)

    def wait_for_save(self) -> None:
        # vLLM calls this after model.forward has enqueued the entire prefill
        # chunk. Use that boundary as chunked-bulk's readiness fence.
        worker = self.connector_worker
        if worker is None:
            return
        if not worker._host_staging_producer:
            super().wait_for_save()
        worker.finish_chunked_bulk_forward()
