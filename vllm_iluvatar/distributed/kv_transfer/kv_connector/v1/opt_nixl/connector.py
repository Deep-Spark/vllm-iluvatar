# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NixlConnector – thin facade that delegates to scheduler / worker."""

from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import EngineId
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
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
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

from .metadata import (
    NixlConnectorMetadata,
)
from .scheduler import (
    NixlConnectorScheduler,
)
from .stats import (
    NixlKVConnectorStats,
    NixlPromMetrics,
)
from .worker import (
    NixlConnectorWorker,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger("vllm." + __name__)


class NixlConnector(KVConnectorBase_V1, SupportsHMA):
    """Iluvatar NIXL push connector.

    Cross-layer KV registration was removed in the 0.30 port.  vLLM 0.30 no
    longer declares ``prefer_cross_layer_blocks`` or
    ``register_cross_layers_kv_cache`` on ``KVConnectorBase_V1``, and nothing in
    the 0.30 tree calls either hook, so the whole CROSS path was unreachable.
    Upstream replaced it with shared-storage detection inside
    ``register_kv_caches`` (``nixl/base_worker.py``:
    ``_share_storage_and_block_stride`` plus the ``storage_is_block_major``
    branch), which arrives at the same one-region-per-slab registration from the
    tensors themselves.

    To bring cross-layer KV back:

    1. Restore ``prefer_cross_layer_blocks`` (property) and
       ``register_cross_layers_kv_cache`` here, and re-add
       ``NixlConnectorWorker.register_cross_layers_kv_caches``,
       ``_register_cross_kv_caches``, ``_handshake_cross_remote_xfer`` and
       ``_write_blocks_cross`` from the 0.27.1 tree, plus
       ``CROSS_KV_SLAB_KEY``, the 0.27.1 ``_select_cross`` /
       ``validate_cross_peer``, the ``_use_cross_layer_slab_xfer`` /
       ``_cross_layer_tensor_shape`` / ``_cross_layer_count`` worker state and
       the CROSS branches in ``register_kv_caches`` and ``_nixl_handshake``.
       Since v12 there is no registration kind to advertise: the slab is just
       one more view for ``_register_kv_views`` (storage base, offset 0,
       ``len`` = ``stride`` = slab bytes per block).
    2. Bring the restored code up to the v11 registration model: the slab
       entry must fill every ``layer_*_mapping`` (one entry, one pointer),
       including ``layer_kernel_blocks_mapping`` / ``layer_block_tokens_mapping``.
       ``_select_cross`` becomes a ``LayerMappingPlanner`` special case that
       returns one ``LayerSelection`` holding one ``PointerPair`` per group
       the slab serves (same pointer and region, a different local/remote
       group id each); ``build_xfer_plan`` then dedups the region and emits
       one ``XferRegionGroup`` per group, so ``_write_blocks`` needs no CROSS
       branch.  The slab spans all groups, so ``_layer_group_id`` has no
       entry for ``CROSS_KV_SLAB_KEY``.
    3. Add a caller: 0.30 has none, so the model runner has to invoke the
       registration hook itself, or the slab has to be detected from shared
       storage the way upstream now does it.
    4. ``get_kv_cache_layout()`` is gone; the layout gate becomes
       ``cache_config.get_resolved_kv_cache_layout().name`` and the layout names
       are ``LBNHC`` / ``LBHNC``, not ``NHD`` / ``HND``.
    """

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: Any) -> bool:
        # The argument is ignored: this vLLM passes kv_connector_extra_config,
        # newer ones pass the whole KVTransferConfig, and neither changes the
        # answer here.  A pure consumer never pushes (save_kv_layer_cuda() and
        # save_kv_layer_host() return immediately on is_consumer, and
        # wait_for_layer_load() is a no-op), so full decode graphs are safe on
        # D.  A producer pushes from save_kv_layer() between layers and cannot
        # be captured as one full graph, so P has to run with --enforce-eager.
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id
        self.kv_transfer_config = vllm_config.kv_transfer_config
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: NixlConnectorScheduler | None = (
                NixlConnectorScheduler(vllm_config, kv_cache_config, self.engine_id)
            )
            self.connector_worker: NixlConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlConnectorWorker(
                vllm_config, kv_cache_config, self.engine_id
            )
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")

    ############################################################
    # Class Methods
    ############################################################
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        if vllm_config.model_config is None:
            logger.warning_once(
                "Unable to detect current VLLM config. "
                "Fallback to default kv cache layout."
            )
            return None
        use_mla = vllm_config.model_config.use_mla
        if use_mla:
            # return None when we have mla
            # as the layout should not matter in that case,
            # which fallback to the default behavior.
            return None
        logger.info_once(
            "NixlConnector setting KV cache layout to LBHNC for better xfer "
            "performance."
        )
        # 0.30 renamed the layouts: HND -> LBHNC, NHD -> LBNHC.
        return "LBHNC"

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def on_new_request(self, request: "Request") -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(
            request, (block_ids,)
        )

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(request, block_ids)

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        tp_size = self._vllm_config.parallel_config.tensor_parallel_size
        self.set_xfer_handshake_metadata(
            {
                pp_rank * tp_size + tp_rank: meta
                for (pp_rank, tp_rank), meta in metadata.items()
            }
        )

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_scales: dict[str, tuple[torch.Tensor, ...]] | None = None,
    ):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches, kv_cache_scales)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        assert self.connector_worker is not None
        super().bind_connector_metadata(connector_metadata)
        self.connector_worker.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self.connector_worker is not None:
            self.connector_worker.clear_connector_metadata()

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        """Report completions plus receive failures.

        This is the hook 0.30 actually calls (``v1/worker/gpu/kv_connector.py``
        and ``kv_connector_model_runner_mixin.py``).  The base class would
        otherwise wrap ``get_finished`` and leave ``failed_recving`` empty, so
        a failed load would look successful to the scheduler.
        """
        assert self.connector_worker is not None
        return self.connector_worker.get_transfer_results()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get block IDs that failed to load via NIXL."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return (
            NixlKVConnectorStats(data=data)
            if data is not None
            else NixlKVConnectorStats()
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return NixlPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """NixlConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        assert self.connector_worker is not None
        # TODO(gyf) :remove "if" when mamba support pd, dummpy run do not set NixlConnectorMetadata, so pass for now
        if self._connector_metadata is not None:
            assert isinstance(self._connector_metadata, NixlConnectorMetadata)
            self.connector_worker.save_kv_layer(
                layer_name, kv_layer, attn_metadata, **kwargs
            )

    def wait_for_save(self):
        """Direct VRAM NIXL writes complete asynchronously via get_finished()."""
        pass

    def has_pending_push_work(self) -> bool:
        if self.connector_scheduler is None:
            return False
        return self.connector_scheduler.has_pending_push_work()

    def shutdown(self):
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get the KVConnector handshake metadata for this connector.
        This metadata is used for out-of-band connector handshake
        between P/D workers.

        Returns:
            KVConnectorHandshakeMetadata: the handshake metadata.
            None if no handshake metadata is available.
        """
        assert self.connector_worker is not None
        return self.connector_worker.xfer_handshake_metadata


class IluNixlPushConnector(NixlConnector):
    """Iluvatar push-based NIXL connector."""


class OptNixlConnector(IluNixlPushConnector):
    """Public name used by ``--kv-transfer-config kv_connector=OptNixlConnector``.

    Same implementation as ``IluNixlPushConnector``; the extra subclass keeps
    ``connector_cls.__name__`` equal to the registered config name.
    """
