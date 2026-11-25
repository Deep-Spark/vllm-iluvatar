# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar NIXL push connector facade (chunked-bulk / non-layerwise)."""

from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
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
    prefix_state: Any = None

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
        """Optional token alignment for complete remote-prefix replay (V4.1 CR2)."""
        if self.connector_scheduler is None:
            return None
        return getattr(self.connector_scheduler, "prefix_replay_alignment", None)

    def attach_prefix_state(self, context: Any) -> None:
        """Attach DeepSeek-V4.1 E25 ProducerPrefixState / ConsumerPrefixState."""
        self.prefix_state = context

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        worker = getattr(self, "connector_worker", None)
        if worker is not None:
            worker.bind_connector_metadata(connector_metadata)
        if self.prefix_state is not None:
            self.prefix_state.bind_metadata(connector_metadata)

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        assert self.connector_worker is not None
        if hasattr(self.connector_worker, "get_transfer_results"):
            result = self.connector_worker.get_transfer_results()
            if (
                getattr(self, "kv_transfer_config", None) is not None
                and self.kv_transfer_config.kv_role == "kv_producer"
                and getattr(self.connector_worker, "pcp_rank", 0) > 0
                and not getattr(self.connector_worker, "pcp_dcp_sharded", False)
            ):
                result.finished_sending.clear()
        else:
            result = super().get_transfer_results(finished_req_ids)
        if self.prefix_state is not None:
            self.prefix_state.receive_results(result)
        return result

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

    def wait_for_save(self) -> None:
        # vLLM calls this after model.forward has enqueued the entire prefill
        # chunk. Use that boundary as chunked-bulk's readiness fence.
        worker = self.connector_worker
        if worker is None:
            return
        if not worker._host_staging_producer:
            super().wait_for_save()
        worker.finish_chunked_bulk_forward()
