# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side logic for the NIXL connector."""

import asyncio
import contextlib
import os
import queue
import threading
import time
import traceback
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import msgspec
import numpy as np
import torch
import uvloop
import zmq
from nixl._api import nixl_agent as NixlWrapper
from nixl._api import nixl_agent_config, nixl_prepped_dlist_handle
from vllm import envs
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    get_current_attn_backends,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    MambaConvSplitInfo,
    derive_mamba_conv_split,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_iluvatar.v1.worker.prefill_state.configuration import config_pipeline_bounds

from ..skip_layers import get_skip_layers, producer_last_layer
from .handshake_validation import (
    validate_block_id_groups,
    validate_compatibility_hash,
    validate_descriptor_geometry,
    validate_main_kv_format,
    validate_parallel_topology,
)
from .host_kv_stub import (
    host_kv_enabled,
    may_override_kv_cache,
    may_override_kv_cache_config,
)
from .lifecycle import (
    CompletionBatch,
    LayerReadyBatch,
    NixlDrainIncomplete,
    RequestOwnershipState,
    RequestTombstoneStore,
    RemoteEngineSession,
    RemoteSessionState,
    SendRequestState,
)
from .mapping import (
    AdjustedXferLayout,
    LayerMappingPlanner,
    LayerSelection,
    NixlTpKVTopology,
    PipelineMappingPlanner,
    RegistrationView,
    TpLayoutAdjuster,
)
from .memory_registration import (
    MemoryRegistrationPlanner,
    TensorRegistration,
    validate_registered_allocation_alias,
)
from .metadata import (
    ENGINE_REACHABILITY_MSG,
    GET_META_MSG,
    OWNERSHIP_CANCEL_MSG,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    NodeMeta,
    ReqId,
    ReqMeta,
    TEARDOWN_COMPLETE_MSG,
    TransferIntent,
    TransferNotificationStatus,
    compute_nixl_compatibility_hash,
)
from .stats import (
    NixlKVConnectorStats,
)
from .utils import (
    _NIXL_SUPPORTED_DEVICE,
    NotifyPayload,
    TransferStatusChecker,
    TransferWrapper,
    dp_scoped_addr,
    find_number_on_layer_name,
    make_zmq_socket,
    zmq_comm2proxy,
    zmq_ctx,
)
from .xfer_plan import (
    RegionGeometry,
    RegionPair,
    RemoteXfer,
    build_xfer_plan,
)

_PROXY_ADDR = os.getenv("VLLM_ILU_NIXL_PROXY_ADDR", "tcp://172.31.0.123:8079")
_SCHEDULER_ADDR = os.getenv("VLLM_ILU_NIXL_SCHEDULER_ADDR", "tcp://127.0.0.1:19001")

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

BufferedRecvNotification = tuple[TransferNotificationStatus, NotifyPayload]

logger = init_logger("vllm." + __name__)


def byte_compatible_kv_layouts(
    local_layout: str,
    remote_layout: str,
    registered_specs: list[KVCacheSpec],
) -> bool:
    """Return whether registered blocks have identical bytes in two layouts.

    The listed mixed layout pairs are equivalent only for singleton-head MLA:
    descriptor offsets and strides account for outer layer/block packing, and
    the in-block head dimension has length one.
    """
    if local_layout == remote_layout:
        return True
    layout_pair = {local_layout, remote_layout}
    if layout_pair not in (
        {"LBNHC", "BLHNC"},
        {"LBNHC", "LBHNC"},
        {"LBHNC", "BLHNC"},
    ):
        return False
    # A PP stage with no registered NIXL tensors transfers no layout-sensitive
    # bytes, so the certified layout mismatch is vacuously safe there.
    return all(
        isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec))
        and spec.num_kv_heads == 1
        for spec in registered_specs
    )


# A status that can only come from the peer's endpoint being gone.
_SESSION_TERMINAL_STATUSES = ("REMOTEDISCONNECT",)
# Ambiguous between an unknown agent and an unknown descriptor.  Only a call
# that carries no descriptors at all -- a notification -- proves the former.
_AGENT_SCOPED_TERMINAL_STATUSES = ("NOTFOUND",)


def _is_session_terminal_error(error: BaseException, *, agent_scoped: bool) -> bool:
    """Whether a NIXL status proves the whole session is unusable.

    ``_is_terminal_xfer_query_error`` classifies one handle, which is a much
    weaker statement: INVALID_PARAM, MISMATCH or the CANCELED this worker asked
    for are request-scoped and must never tear a session down, or one bad
    request would fail every other request mapped to the same pair.
    """

    text = f"{type(error).__name__}{error}".upper().replace("_", "")
    statuses = _SESSION_TERMINAL_STATUSES
    if agent_scoped:
        statuses += _AGENT_SCOPED_TERMINAL_STATUSES
    return any(status in text for status in statuses)


def _is_terminal_xfer_query_error(error: BaseException) -> bool:
    """Whether a NIXL status exception means the transfer cannot progress."""

    # check_xfer_state returns only IN_PROG/SUCCESS; every negative NIXL
    # status is raised as a typed nixl*Error and is permanent for that handle.
    error_name = type(error).__name__.upper()
    if error_name.startswith("NIXL") and error_name.endswith("ERROR"):
        return True
    return "NIXL_ERR_" in str(error).upper()


class NixlConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        engine_id: str,
    ):
        kv_cache_config = may_override_kv_cache_config(kv_cache_config)
        nixl_wrapper_cls = NixlWrapper
        if nixl_wrapper_cls is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        logger.info("Initializing NIXL wrapper")
        logger.info("Initializing NIXL worker %s", engine_id)

        # ---- Config ----
        self.vllm_config = vllm_config
        # mypy will complain on re-assignment otherwise.
        if vllm_config.kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set for NixlConnector")
        self.kv_transfer_config = vllm_config.kv_transfer_config

        # ---- Agent ----
        self.nixl_backends = vllm_config.kv_transfer_config.get_from_extra_config(
            "backends", ["UCX"]
        )
        non_ucx_backends = [b for b in self.nixl_backends if b != "UCX"]
        # Configure NIXL num_threads to avoid UAR exhaustion on Mellanox NICs.
        # Each UCX thread allocates UARs (doorbell pages) via DevX, and
        # excessive NIXL UAR usage can exhaust NIC UAR space. This can cause
        # components like NVSHMEM (used by DeepEP kernels) to fail during RDMA
        # initialization with "mlx5dv_devx_alloc_uar" errors.
        # Ref: https://network.nvidia.com/files/doc-2020/ethernet-adapters-programming-manual.pdf#page=63
        num_threads = vllm_config.kv_transfer_config.get_from_extra_config(
            "num_threads", 4
        )
        if nixl_agent_config is None:
            config = None
        else:
            # Enable telemetry by default for NIXL 0.7.1 and above.
            config = (
                nixl_agent_config(backends=self.nixl_backends, capture_telemetry=True)
                if len(non_ucx_backends) > 0
                else nixl_agent_config(num_threads=num_threads, capture_telemetry=True)
            )
        # KV Caches and nixl tracking data.
        self.device_type = current_platform.device_type
        self.kv_buffer_device: str = vllm_config.kv_transfer_config.kv_buffer_device
        if self.device_type != "cuda":
            raise NotImplementedError(
                "NIXL push currently supports CUDA workers with direct VRAM "
                f"transfer only; device_type={self.device_type!r} is not supported"
            )
        if self.kv_buffer_device != "cuda":
            raise NotImplementedError(
                "NIXL push currently supports direct CUDA/VRAM KV transfer only; "
                f"kv_buffer_device={self.kv_buffer_device!r} is not supported"
            )
        if self.device_type not in _NIXL_SUPPORTED_DEVICE:
            raise RuntimeError(f"{self.device_type} is not supported.")
        elif self.kv_buffer_device not in _NIXL_SUPPORTED_DEVICE[self.device_type]:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )

        # support for oot platform which can't register nixl memory
        # type based on kv_buffer_device
        nixl_memory_type = current_platform.get_nixl_memory_type()
        if nixl_memory_type is None:
            nixl_memory_type = "VRAM"
        if nixl_memory_type != "VRAM":
            raise NotImplementedError(
                "NIXL push currently supports direct CUDA/VRAM KV transfer only; "
                f"platform memory type is {nixl_memory_type!r}"
            )
        self.nixl_memory_type = nixl_memory_type
        if host_kv_enabled():
            self.nixl_memory_type = "DRAM"

        self.nixl_wrapper = nixl_wrapper_cls(str(uuid.uuid4()), config)
        # Map of engine_id -> [remote_agent0, remote_agent1, ...]
        self._remote_agents: dict[EngineId, list[str]] = {}
        # Local dlist layout (tuple of regions) -> prepped handle.  Shared by
        # every remote target whose deduplicated local layout is identical.
        self.src_xfer_side_handles: dict[
            tuple[RegionGeometry, ...], nixl_prepped_dlist_handle
        ] = {}
        # Map of EngineId -> [remote nixl_prepped_dlist_handle]
        self.dst_xfer_side_handles: dict[EngineId, list[nixl_prepped_dlist_handle]] = (
            defaultdict(list)
        )
        # EngineId -> [local layer][remote tp] = transfer.  Every layer of this
        # PP rank has a row; a cell is None when the layer is not sent to that
        # remote tp rank (another layer of the overlap carries its notify).
        self._remote_agents_per_layer: dict[EngineId, list[list[RemoteXfer | None]]] = (
            defaultdict(list)
        )
        # EngineId -> [remote pp][remote tp] = transfer.
        self._remote_agents_per_rank: dict[EngineId, list[list[RemoteXfer | None]]] = (
            defaultdict(list)
        )
        # all local registered data
        self._registered_descs: list[Any] = []
        self._registered_cache_owners: dict[int, torch.Tensor] = {}
        self.xfer_handshake_metadata: NixlHandshakePayload | None = None
        # infos about registered kv cache
        self.layer_ptr_mapping = defaultdict(list[int])
        self.layer_desc_mapping = defaultdict(list[tuple[int, int]])
        self.layer_desc_num_mapping = defaultdict(list[tuple[int, int]])
        # Per registered pointer: block count and owning KV group.  These two
        # make the descriptor space explicit rather than implied by a single
        # global num_blocks: descriptor id = cumsum(num_blocks)[region] +
        # block_id, and the block id is taken from the group that owns the
        # region.  Both are advertised in the handshake so the peer can index
        # its own block-id tuple with its own group numbering.
        self.layer_num_blocks_mapping = defaultdict(list[int])
        self.layer_group_mapping = defaultdict(list[int])
        # Per registered pointer: kernel blocks per scheduler block and the
        # owning group's scheduler block size (tokens).  Both must match the
        # peer pointer for pointer, since block ids are exchanged unmodified.
        self.layer_kernel_blocks_mapping = defaultdict(list[int])
        self.layer_block_tokens_mapping = defaultdict(list[int])
        self.layer_backend_mapping: dict[str, str] = {}
        # Layers the model asked to leave unregistered (see skip_layers).
        self._skipped_layers: set[str] = set()
        # device kv caches
        self.device_kv_caches: dict[str, torch.Tensor] = {}

        # ---- hybrid kv ----
        # [layer_index_local, layer_index_global, layer_name, kv_spec]
        self.kv_group_infos: list[list[tuple[int, int, str, KVCacheSpec]]] = []
        self.kv_group_infos_for_check: list[str] = []
        # layer name -> owning KV cache group id.  Registration records this
        # per registered pointer so that a transfer can index the block-id
        # tuple with the group that actually owns the region, instead of
        # assuming the peer numbers its groups the same way.
        self._layer_group_id: dict[str, int] = {}
        self._layer_specs: dict[str, KVCacheSpec] = {}
        self._all_specs = set()
        self.num_groups = len(kv_cache_config.kv_cache_groups)
        self.kv_cache_config = kv_cache_config
        self._memory_registration_planner = MemoryRegistrationPlanner(kv_cache_config)

        for group_index, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            layer_names = kv_cache_group.layer_names
            if isinstance(kv_cache_group.kv_cache_spec, UniformTypeKVCacheSpecs):
                self._all_specs.update(
                    set(kv_cache_group.kv_cache_spec.kv_cache_specs.values())
                )
                specs = next(iter(kv_cache_group.kv_cache_spec.kv_cache_specs.values()))
                kv_cache_specs = kv_cache_group.kv_cache_spec.kv_cache_specs
            else:
                self._all_specs.add(kv_cache_group.kv_cache_spec)
                specs = kv_cache_group.kv_cache_spec
                kv_cache_specs = {
                    layer_name: specs for layer_name in kv_cache_group.layer_names
                }

            self.kv_group_infos_for_check.append(specs.__class__.__name__)
            for layer_name, layer_spec in kv_cache_specs.items():
                self._layer_specs[layer_name] = layer_spec
                self._layer_group_id[layer_name] = group_index

            try:
                layer_index_mapping = {
                    layer_name: int(
                        find_number_on_layer_name(layer_name)[0].replace(".", "")
                    )
                    for layer_name in layer_names
                }
            except Exception as e:
                logger.error(
                    "error when create kv cache info from kv_cache_config.kv_cache_groups, error: %s",
                    e,
                )
                raise RuntimeError("Create NixlConnectorScheduler error!") from e
            current_group_info = [
                (layer_index_mapping[layer_name], layer_name, layer_kv_spec)
                for layer_name, layer_kv_spec in kv_cache_specs.items()
            ]
            current_group_info = [
                (index_in_group, global_index, layer_name, layer_spec_name)
                for index_in_group, (
                    global_index,
                    layer_name,
                    layer_spec_name,
                ) in enumerate(current_group_info)
            ]

            self.kv_group_infos.append(current_group_info)

        logger.info("kv_group_infos : %s", self.kv_group_infos)
        self._has_normmal = any(
            isinstance(spec, AttentionSpec) for spec in self._all_specs
        )
        self._has_mamba = any(isinstance(spec, MambaSpec) for spec in self._all_specs)
        _is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )

        self._layer_global_index: dict[str, int] = {
            layer_name: global_index
            for group_info in self.kv_group_infos
            for _, global_index, layer_name, _ in group_info
        }

        # ---- Model state (derived from model config) ----
        mamba_ssm_size = (0, 0)
        # Conv state sub-projection decomposition (None when no Mamba).
        # The 3-read transfer requires DS (dim, state_len) conv layout so
        # that x/B/C sub-projections are contiguous in memory.
        self._conv_decomp: MambaConvSplitInfo | None = None
        if self._has_mamba:
            assert _is_hma_required
            from vllm.model_executor.layers.mamba.mamba_utils import (
                is_conv_state_dim_first,
            )

            assert is_conv_state_dim_first(), (
                "3-read Mamba conv transfer requires DS conv state layout. "
                "Set VLLM_SSM_CONV_STATE_LAYOUT=DS"
            )
            mamba_spec = next(
                spec
                for spec in self._layer_specs.values()
                if isinstance(spec, MambaSpec)
            )
            self._conv_decomp = derive_mamba_conv_split(
                mamba_spec,
                vllm_config.parallel_config.tensor_parallel_size,
            )
            mamba_ssm_size = self._conv_decomp.ssm_sizes
        self._mamba_ssm_size = mamba_ssm_size

        # ---- Metadata ----
        self.engine_id: EngineId = engine_id
        self.block_size: int = cast(int, vllm_config.cache_config.block_size)
        self._logical_num_blocks: int = None
        self.num_blocks = kv_cache_config.num_blocks
        self.is_consumer = vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        self.enable_heterogeneous_attn_post_process = False
        self.tp_world_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_world_size = vllm_config.parallel_config.pipeline_parallel_size
        self.tp_rank = 0 if self.tp_world_size == 1 else get_tp_group().rank_in_group
        self.pp_rank = 0 if self.pp_world_size == 1 else get_pp_group().rank_in_group
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.dp_world_size = vllm_config.parallel_config.data_parallel_size
        # Must match the port this rank's own scheduler binds.
        self.scheduler_addr = dp_scoped_addr(
            _SCHEDULER_ADDR, vllm_config.parallel_config.data_parallel_index
        )
        side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_index
        )
        self.side_channel_addr = (
            f"{envs.VLLM_NIXL_SIDE_CHANNEL_HOST}:{side_channel_port}"
        )
        self.global_rank = self.pp_rank * self.tp_world_size + self.tp_rank
        self.enable_permute_local_kv = False
        self.use_quant_to_fp8 = False
        self.device_id: int = 0
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.compat_hash = None
        self.enforce_handshake_compat = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "enforce_handshake_compat", True
            )
        )

        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.use_mla = self.model_config.use_mla
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(vllm_config, layer_type, None)
        if not layers:
            raise RuntimeError("can not find an appropriate attention backend ")

        self.attn_backends = get_current_attn_backends(vllm_config)
        self.attn_backend = self.attn_backends[0]
        self.backend_name = self.attn_backend.get_name()
        self._layer_backend_names = {
            name: layer.get_attn_backend().get_name() for name, layer in layers.items()
        }
        # 0.30 resolves one layout in the engine core and hands it to every
        # worker; the free function get_kv_cache_layout() is gone and the
        # vocabulary changed from NHD/HND to LBNHC/LBHNC.
        self.kv_cache_layout = (
            vllm_config.cache_config.get_resolved_kv_cache_layout().name
        )
        self._sync_block_size_with_kernel()
        logger.info("Detected attention backend %s", self.backend_name)
        logger.info("Detected kv cache layout %s", self.kv_cache_layout)
        self.transfer_topo: NixlTpKVTopology | None = None

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_world_size}
        self._pp_size: dict[EngineId, int] = {self.engine_id: self.pp_world_size}
        self._block_size: dict[EngineId, int] = {self.engine_id: self.block_size}

        self.num_hidden_layers = (
            vllm_config.model_config.hf_text_config.num_hidden_layers
            if vllm_config.model_config.hf_text_config is not None
            else vllm_config.model_config.hf_config.num_hidden_layers
        )
        pp_bounds = config_pipeline_bounds(vllm_config)
        self.pp_layer_start, self.pp_layer_end = pp_bounds[self.pp_rank]
        # A PD prefiller stops at the last kv source (producer_last_layer), so
        # only that many layers run their save hook. Leaving the stage bounds
        # at the full depth would make un_send_layer_nums never reach 0 and
        # pp_layer_end_closed name a layer that never fires.
        if (last_layer := producer_last_layer(vllm_config)) is not None:
            self.pp_layer_start = min(self.pp_layer_start, last_layer + 1)
            self.pp_layer_end = min(self.pp_layer_end, last_layer + 1)
        self.pp_layer_end_closed = self.pp_layer_end - 1
        self.pp_split = [start for start, _ in pp_bounds]
        self.num_layers = self.pp_layer_end - self.pp_layer_start
        self.num_layers_end_index = self.num_layers - 1
        self.total_num_kv_heads = self.model_config.get_total_num_kv_heads()

        # ---- transferring ----
        self.kv_lease_duration: int = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_lease_duration", 480
            )
        )
        role = vllm_config.kv_transfer_config.kv_role
        if role not in ("kv_producer", "kv_consumer"):
            raise ValueError(
                "NIXL push worker only supports kv_producer or kv_consumer, "
                f"got {role!r}"
            )
        self._session_role = "consumer" if role == "kv_consumer" else "producer"
        self.consumer_disconnect_grace = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "consumer_disconnect_grace", 0.0
            )
        )
        self.producer_disconnect_drain_timeout = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "producer_disconnect_drain_timeout", 45.0
            )
        )
        # Extra window for a send whose local metadata and remote blocks never
        # joined, e.g. because the client killed the request before the
        # consumer disclosed its destination.  Must exceed the worst-case delay
        # between this engine finishing a prefill and the consumer allocating.
        self.producer_orphan_timeout = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "producer_orphan_timeout", 600.0
            )
        )
        # A peer whose transport died answers neither the proxy nor the
        # handshake, so every rebuild attempt costs this thread one proxy
        # timeout per remote rank.  Rate-limit the retry per engine.
        self.rehandshake_retry_interval = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "rehandshake_retry_interval", 5.0
            )
        )
        self._connector_metadata = None

        # thread and related resources
        self.ctx = zmq.asyncio.Context()
        self._async_loop = uvloop.new_event_loop()
        self.stop_event = threading.Event()
        self._completion_outbox: queue.SimpleQueue[CompletionBatch] = (
            queue.SimpleQueue()
        )
        # The following lifecycle objects are owned exclusively by _async_loop.
        self.remote_sessions: dict[EngineId, RemoteEngineSession] = {}
        self._engine_generations: dict[EngineId, int] = {}
        # Connect metadata of every engine handshaken with, so a session the
        # transport invalidated can be rebuilt without a new connect: the agent
        # metadata itself comes from the proxy, not from the peer.
        self._remote_node_metas: dict[EngineId, NodeMeta] = {}
        # Engines whose NIXL mappings this worker tore down after a terminal
        # transport status.  The peer cannot learn of such a failure -- the only
        # producer-to-consumer channel is the notification that just proved
        # unusable -- so recovery is driven from here.
        self._engines_needing_rehandshake: set[EngineId] = set()
        self._rehandshake_retry_at: dict[EngineId, float] = {}
        self._rehandshake_failures: dict[EngineId, int] = {}
        self._reachability_outbox: asyncio.Queue[
            tuple[EngineId, bool]
        ] = asyncio.Queue()
        self._reported_reachability: dict[EngineId, bool] = {}
        # Engine -> reason, applied on the next progress tick.  Invalidating in
        # place would tear down the dlists a running _write_blocks still reads.
        self._pending_transport_failures: dict[EngineId, str] = {}
        self._loop_tasks: set[asyncio.Task[Any]] = set()
        self._loop_exit_event = asyncio.Event()
        self._teardown_outbox: asyncio.Queue[
            tuple[str, int, int, int, str]
        ] = (
            asyncio.Queue()
        )
        self._accepting_work = True
        self._shutdown_task: asyncio.Task[None] | None = None
        self._shutdown_future = None
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = False
        self._send_failures: dict[ReqId, str] = {}
        self.worker_thread: threading.Thread = threading.Thread(
            target=self.connecter_loop,
            daemon=True,
            name="nixl_connector_worker_loop",
        )
        self.save_kv_layer = (
            self.save_kv_layer_cuda
            if not host_kv_enabled()
            else self.save_kv_layer_host
        )
        # Receive state is owned exclusively by the uvloop actor.
        self._recv_requests: dict[ReqId, ReqMeta] = {}
        self._recv_generations: dict[ReqId, str] = {}
        self._aborted_recv_requests: set[ReqId] = set()
        self._invalid_block_ids: set[int] = set()
        self._recv_notification_status: dict[ReqId, TransferStatusChecker] = {}
        self.use_async_scheduling = vllm_config.scheduler_config.async_scheduling
        # The scheduler sends block_ids to the producer while scheduling, but
        # this worker only learns of the request with the next model-runner
        # metadata batch.  A producer that writes fast (e.g. a prefix-cache
        # hit) can notify inside that gap, with or without async scheduling,
        # and a dropped notification leaves the request waiting until its
        # lease expires.  Buffered entries that no request ever claims are
        # discarded after the same lease.
        self._early_recv_notifications: dict[
            tuple[ReqId, str], list[BufferedRecvNotification]
        ] = defaultdict(list)
        self._early_recv_deadlines: dict[tuple[ReqId, str], float] = {}
        tombstone_ttl = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "ownership_tombstone_ttl", 86400.0
            )
        )
        tombstone_limit = int(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "ownership_tombstone_limit", 262144
            )
        )
        self._recv_tombstones = RequestTombstoneStore(
            tombstone_ttl, tombstone_limit
        )
        self._send_tombstones = RequestTombstoneStore(
            tombstone_ttl, tombstone_limit
        )

        # Step-scoped producer metadata is main-thread-owned.
        self._step_layerwise_sends: dict[ReqId, ReqMeta] = {}
        self._step_rank_sends: dict[ReqId, ReqMeta] = {}
        # Persistent producer state and transfer handles are uvloop-owned.
        self._send_requests: dict[ReqId, SendRequestState] = {}
        self._send_transfers: dict[ReqId, TransferWrapper] = {}
        self._send_ready_queue: asyncio.Queue[LayerReadyBatch] = asyncio.Queue()
        self._host_send_ready_queue: asyncio.Queue[LayerReadyBatch] | None = (
            asyncio.Queue() if host_kv_enabled() else None
        )
        self.d2h_manager: Any | None = None
        # send status
        self.xfer_stats = NixlKVConnectorStats()
        # int helper to per-layer transferring
        self.layer_index = self.pp_layer_start
        self.un_send_layer_nums = self.num_layers
        # Loop-owned request leases and disconnect lifecycle.
        self._kv_lease_duration: int = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_lease_duration", 480
            )
        )

        self.worker_thread.start()

    def _sync_block_size_with_kernel(self) -> None:
        # Upstream turns a kernel/scheduler block size mismatch into one
        # worker-wide ratio and rewrites block ids to kernel ids.  vLLM 0.30
        # allocates every group in units of its own spec.block_size, so one
        # rank can hold groups with different ratios (V4.1: 1/2/4/8).  NIXL
        # therefore registers scheduler blocks and reads the per-view ratio at
        # registration time; block_size and num_blocks stay logical.
        self._logical_num_blocks = self.num_blocks
        logger.info(
            "NIXL KV block geometry: block_size=%s, num_blocks=%s, attn_backends=%s",
            self.block_size,
            self.num_blocks,
            [backend.get_name() for backend in self.attn_backends],
        )

    def _log_failure(
        self,
        failure_type: str,
        req_id: str | None,
        msg: str = "",
        error: Exception | None = None,
    ):
        """Log transfer failure with structured context for easier debugging."""
        context: dict[str, Any] = {
            "failure_type": failure_type,
            "request_id": req_id,
            "engine_id": self.engine_id,
        }

        if msg:
            failure_type = f"{failure_type}. {msg}"

        logger.error(
            "NIXL transfer failure: %s | Context: %s",
            failure_type,
            context,
            exc_info=error is not None,
            stacklevel=2,
        )

    ############################################################
    # register kv caches
    ############################################################
    # Layers listed in skip_layers (filled by the model) are never registered
    # and never appear in any layer_* mapping.  On DeepSeek-V4.1 only the
    # compressed MLA KV and the indexer K cache of kv sources are sent:
    #
    # * SWA cache and compressor state ring, under SWA bounded replay: D
    #   replays the hit's last window and rebuilds both.  Symmetric.
    # * PP forwarder replicas (5/10/15/25/30/35 at TP2/PP8): they only relay
    #   inside P.  P-only; the model refuses a forwarder on D.

    def _reset_registration_mappings(self) -> None:
        # Per layer, one entry per registered pointer: pointer index,
        # inter-block stride, payload segments within one block, block count
        # and owning KV group.  For ordinary layouts stride and payload length
        # are equal.  Packed layouts intentionally differ.
        self.layer_block_len_mapping = defaultdict(list[int])
        self.layer_ptr_mapping = defaultdict(list[int])
        self.layer_desc_mapping = defaultdict(list[list[tuple[int, int]]])
        self.layer_num_blocks_mapping = defaultdict(list[int])
        self.layer_group_mapping = defaultdict(list[int])
        self.layer_kernel_blocks_mapping = defaultdict(list[int])
        self.layer_block_tokens_mapping = defaultdict(list[int])
        self.layer_backend_mapping: dict[str, str] = {}
        self._skipped_layers = set()

    def _record_registered_kv_region(
        self,
        *,
        layer_name: str,
        cache: torch.Tensor,
        layer_spec: KVCacheSpec,
        registration: TensorRegistration,
        segments: list[tuple[int, int]],
        caches_data: list,
        seen_base_addresses: list,
        seen_base_addresses_check: dict[int, int],
        registered_allocation_sizes: dict[int, int],
    ) -> None:
        base_addr = registration.registration_base
        registration_nbytes = registration.registration_nbytes
        # Native descriptors contain addresses, so they cannot retain either
        # the payload or a private scale allocation by themselves.
        if not hasattr(self, "_registered_cache_owners"):
            self._registered_cache_owners = {}
        self._registered_cache_owners[base_addr] = cache
        if base_addr in seen_base_addresses_check:
            # if exist, record the index for remote to get the correct data_ptr.
            current_index = seen_base_addresses_check[base_addr]
            validate_registered_allocation_alias(
                base_address=base_addr,
                registered_nbytes=registered_allocation_sizes[base_addr],
                expected_nbytes=registration_nbytes,
                layer_name=layer_name,
            )
        else:
            current_index = len(seen_base_addresses_check)
            seen_base_addresses_check[base_addr] = current_index
            registered_allocation_sizes[base_addr] = registration_nbytes
            seen_base_addresses.append(base_addr)
            caches_data.append(
                (base_addr, registration_nbytes, registration.device_id, "")
            )

        logger.info(
            "NIXL layer geometry: layer=%s shape=%s spec=%s group=%s "
            "block_tokens=%s num_blocks=%s kernel_blocks=%s block_stride=%s "
            "payload=%s registration_nbytes=%s",
            layer_name,
            tuple(cache.shape),
            type(layer_spec).__name__,
            self._layer_group_id[layer_name],
            layer_spec.block_size,
            registration.num_blocks,
            registration.kernel_blocks,
            registration.block_stride,
            segments,
            registration_nbytes,
        )
        # Need to make sure the device ID is non-negative for NIXL,
        # Torch uses -1 to indicate CPU tensors.
        if not host_kv_enabled():
            self.device_id = registration.device_id
        else:
            self.device_id = torch.distributed.get_rank()

        self.layer_block_len_mapping[layer_name].append(registration.block_stride)
        self.layer_ptr_mapping[layer_name].append(current_index)
        self.layer_desc_mapping[layer_name].append(segments)
        self.layer_num_blocks_mapping[layer_name].append(registration.num_blocks)
        self.layer_group_mapping[layer_name].append(self._layer_group_id[layer_name])
        self.layer_kernel_blocks_mapping[layer_name].append(registration.kernel_blocks)
        self.layer_block_tokens_mapping[layer_name].append(layer_spec.block_size)
        self.layer_backend_mapping[layer_name] = self._layer_backend_names.get(
            layer_name, ""
        )

    def _transfer_segments(
        self, layer_spec: KVCacheSpec, registration: TensorRegistration
    ) -> list[tuple[int, int]]:
        """Payload segments of one registered view.

        Hybrid Mamba models also keep split segments (conv/ssm for state, K/V
        halves for attention) in front of the whole-block segment:
        ``LayerMappingPlanner`` moves the whole block under equal TP and the
        split segments under heterogeneous TP.
        """

        ((offset, payload),) = registration.payload_segments
        if not self._has_mamba or isinstance(layer_spec, MLAAttentionSpec):
            return [(offset, payload)]
        whole = registration.block_stride - offset
        if isinstance(layer_spec, MambaSpec):
            assert self._conv_decomp is not None
            conv_size, ssm_size = self._mamba_ssm_size
            split = [*self._conv_decomp.local_conv_offsets, (conv_size, ssm_size)]
        else:
            split = [(0, whole // 2), (whole // 2, whole // 2)]
        return [(offset + start, size) for start, size in split] + [(offset, whole)]

    def _register_kv_views(
        self, kv_caches: dict[str, torch.Tensor],
        kv_cache_scales: dict[str, torch.Tensor] | None = None,
    ) -> tuple[list, list]:
        """Register every KV view with NIXL.

        One path for plain, packed and Mamba views: the registration unit is
        the backing storage and each view contributes ``(ptr, offset, len,
        stride, num_blocks)`` at scheduler-block granularity, all read off the
        view (``MemoryRegistrationPlanner.describe``).
        """
        logger.info(
            "Registering direct VRAM KV caches. use_mla: %s, "
            "kv_buffer_device: %s, packed layers: %s",
            self.use_mla,
            self.kv_buffer_device,
            self._memory_registration_planner.has_packed_layers,
        )

        caches_data: list = []
        # With hybrid allocator, layers can share a kv cache tensor
        seen_base_addresses: list = []
        seen_base_addresses_check: dict[int, int] = {}
        registered_allocation_sizes: dict[int, int] = {}
        plain_storage_nbytes: int | None = None
        self._reset_registration_mappings()
        from ixformer.inference.functions.deepseek.quantized_kv_format import (
            resolve_quant_format,
        )

        from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
            is_quantized_main_spec,
        )
        from vllm_iluvatar.models.deepseek_v41.indexer_kv import is_split_indexer_spec

        forward_context = self.vllm_config.compilation_config.static_forward_context
        skip_layers = get_skip_layers()
        unknown_skip_layers = skip_layers - kv_caches.keys()
        if unknown_skip_layers:
            logger.warning(
                "skip layers without a KV cache on this rank: %s",
                sorted(unknown_skip_layers),
            )

        for layer_name, cache_or_caches in kv_caches.items():
            if "mtp" in layer_name:
                continue

            layer_spec = self._layer_specs.get(layer_name, None)
            # None means shared with other tensor, skip register this tensor
            if layer_spec is None:
                continue
            if layer_name in skip_layers:
                self._skipped_layers.add(layer_name)
                continue

            if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                # UniformTypeKVCacheSpecs merges kv_cache_specs
                layer_spec = layer_spec.kv_cache_specs[layer_name]

            for cache in self.transfer_topo.get_transfer_cache_regions(
                cache_or_caches, layer_spec
            ):
                registration = self._memory_registration_planner.describe(
                    layer_name=layer_name,
                    tensor=cache,
                    spec=layer_spec,
                    num_blocks=self._logical_num_blocks,
                )
                if (
                    not self.use_mla
                    and self._memory_registration_planner.get_packing(layer_name)
                    is None
                ):
                    # Different kv cache shape is not supported by HeteroTP.
                    # This must also hold true for Mamba-like models.
                    if plain_storage_nbytes is None:
                        plain_storage_nbytes = registration.registration_nbytes
                    assert plain_storage_nbytes == registration.registration_nbytes, (
                        "All kv cache tensors must have the same size"
                    )
                self._record_registered_kv_region(
                    layer_name=layer_name,
                    cache=cache,
                    layer_spec=layer_spec,
                    registration=registration,
                    segments=self._transfer_segments(layer_spec, registration),
                    caches_data=caches_data,
                    seen_base_addresses=seen_base_addresses,
                    seen_base_addresses_check=seen_base_addresses_check,
                    registered_allocation_sizes=registered_allocation_sizes,
                )
                if is_quantized_main_spec(layer_spec) or is_split_indexer_spec(layer_spec):
                    scales = (kv_cache_scales or {}).get(layer_name)
                    if scales is None:
                        scales = getattr(forward_context[layer_name], "kv_cache_scales", None)
                    if not isinstance(scales, torch.Tensor):
                        raise ValueError(
                            f"{layer_name}: compressed KV registration requires its scale plane"
                        )
                    if is_split_indexer_spec(layer_spec):
                        scale_registration = (
                            self._memory_registration_planner.describe_indexer_scale_plane(
                                layer_name=layer_name, scales=scales, payload=registration,
                            )
                        )
                    else:
                        fmt = resolve_quant_format(layer_spec.cache_dtype_str)
                        scale_registration = self._memory_registration_planner.describe_scale_plane(
                            layer_name=layer_name,
                            scales=scales,
                            payload=registration,
                            scale_page_bytes=fmt.scale_page_bytes(layer_spec.head_size),
                        )
                    self._record_registered_kv_region(
                        layer_name=layer_name, cache=scales, layer_spec=layer_spec,
                        registration=scale_registration,
                        segments=list(scale_registration.payload_segments),
                        caches_data=caches_data, seen_base_addresses=seen_base_addresses,
                        seen_base_addresses_check=seen_base_addresses_check,
                        registered_allocation_sizes=registered_allocation_sizes,
                    )

        return caches_data, seen_base_addresses

    def _register_nixl_memory(
        self, caches_data: list, seen_base_addresses: list
    ) -> None:
        descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
        logger.info("Registering descs: %s", seen_base_addresses)
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            logger.info(
                "CUDA memory before NIXL register_memory: "
                "free=%.2f GiB total=%.2f GiB "
                "allocated=%.2f GiB reserved=%.2f GiB",
                free_bytes / (1 << 30),
                total_bytes / (1 << 30),
                torch.cuda.memory_allocated() / (1 << 30),
                torch.cuda.memory_reserved() / (1 << 30),
            )
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            logger.info(
                "CUDA memory after NIXL register_memory: "
                "free=%.2f GiB total=%.2f GiB "
                "allocated=%.2f GiB reserved=%.2f GiB",
                free_bytes / (1 << 30),
                total_bytes / (1 << 30),
                torch.cuda.memory_allocated() / (1 << 30),
                torch.cuda.memory_reserved() / (1 << 30),
            )
        logger.info("Done registering descs")
        self._registered_descs.append(descs)

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_scales: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Register the KV Cache data in nixl."""
        from vllm_iluvatar.models.deepseek_v41.compressed_kv import main_kv_format

        main_format = main_kv_format()
        if main_format is not None and host_kv_enabled():
            raise NotImplementedError("Compressed main KV requires paired direct-VRAM registration")
        self.gpu_kv_cache = kv_caches
        kv_caches = may_override_kv_cache(kv_caches)
        self.cpu_kv_cache = kv_caches
        if host_kv_enabled() and not self.is_consumer:
            from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.host_kv import (
                get_host_layer_plans,
            )
            from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.host_kv.manager import (
                D2HManager,
            )

            self.d2h_manager = D2HManager(
                kv_cache_config=self.kv_cache_config,
                host_kv_caches=kv_caches,
                static_forward_context=(
                    self.vllm_config.compilation_config.static_forward_context
                ),
                layer_plans=get_host_layer_plans(self.vllm_config),
            )

        self.transfer_topo = NixlTpKVTopology(
            tp_rank=self.tp_rank,
            tp_size=self.tp_world_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=[
                self.attn_backend,
            ],
            # SSM States come in tuples (ssm, conv)
            tensor_shape=next(iter(kv_caches.values())).shape
            if not self._has_mamba
            else None,
            is_mamba=self._has_mamba,
        )
        self.compat_hash = compute_nixl_compatibility_hash(self.vllm_config)

        caches_data, seen_base_addresses = self._register_kv_views(kv_caches, kv_cache_scales)
        if caches_data:
            self._register_nixl_memory(caches_data, seen_base_addresses)
        else:
            # Every KV of this rank is skipped (only forwarder / SWA / ring
            # layers).  The rank still handshakes and sends notify-only.
            logger.info("No KV to transfer on this rank; skip NIXL memory registration")

        # Register local/src descr for NIXL xfer.
        self.seen_base_addresses = seen_base_addresses

        if self._has_mamba:
            logger.info(
                "Hybrid SSM registration: num_blocks=%s, mamba_ssm_size=%s",
                self.num_blocks,
                self._mamba_ssm_size,
            )
        logger.info(
            "layer_ptr_mapping for all register kv cache: %s", self.layer_ptr_mapping
        )
        logger.info(
            "layer_block_len_mapping for all register kv cache: %s",
            self.layer_block_len_mapping,
        )
        logger.info(
            "layer_desc_mapping for all register kv cache: %s", self.layer_desc_mapping
        )
        logger.info("cache dates for register kv cache: %s", caches_data)
        logger.info(
            "layer_group_mapping for all register kv cache: %s; "
            "layer_kernel_blocks_mapping: %s; skipped: %s",
            dict(self.layer_group_mapping),
            dict(self.layer_kernel_blocks_mapping),
            sorted(self._skipped_layers),
        )

        # After KV Caches registered, listen for new connections.
        agent_metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=self.seen_base_addresses,
            num_blocks=self.num_blocks,
            kv_cache_layout=self.kv_cache_layout,
            block_size=self.block_size,
            ssm_sizes=self._mamba_ssm_size,
            attn_backend_name=self.backend_name,
            # Block ids are exchanged as scheduler block ids; kernel blocks
            # live inside a registered block (layer_kernel_blocks_mapping).
            physical_blocks_per_logical_kv_block=1,
            # NOTE: extra infos pass to remote.
            physical_block_size=self.block_size,
            logical_num_blocks=self._logical_num_blocks,
            layer_ptr_mapping=self.layer_ptr_mapping,
            layer_block_len_mapping=self.layer_block_len_mapping,
            layer_desc_mapping=self.layer_desc_mapping,
            layer_num_blocks_mapping=self.layer_num_blocks_mapping,
            layer_group_mapping=self.layer_group_mapping,
            layer_kernel_blocks_mapping=self.layer_kernel_blocks_mapping,
            layer_block_tokens_mapping=self.layer_block_tokens_mapping,
            layer_backend_mapping=self.layer_backend_mapping,
            global_rank=self.global_rank,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            main_kv_format=main_format.name if main_format is not None else "none",
        )
        # Wrap metadata in payload with hash for defensive decoding
        assert self.compat_hash is not None
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
            main_kv_format=agent_metadata.main_kv_format,
        )

    ############################################################
    # Add remote and prepare resources for sending kv cache
    ############################################################
    def _build_xfer_dlist(
        self,
        *,
        agent_name: str,
        regions: tuple[RegionGeometry, ...],
        base_addresses: list[int],
        device_id: int,
        memory_type: str,
    ) -> nixl_prepped_dlist_handle:
        """Prepare a dlist laid out [region][block], matching the XferPlan offsets."""

        blocks_data: list[tuple[int, int, int]] = []
        for region in regions:
            start = base_addresses[region.ptr] + region.offset
            addresses = (
                start
                + np.arange(region.num_blocks, dtype=np.int64) * region.block_stride
            )
            blocks_data.extend(
                (address, region.length, device_id) for address in addresses.tolist()
            )
        assert blocks_data
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, memory_type)
        return self.nixl_wrapper.prep_xfer_dlist(agent_name, descs)

    def _local_xfer_dlist(
        self, regions: tuple[RegionGeometry, ...]
    ) -> nixl_prepped_dlist_handle:
        handle = self.src_xfer_side_handles.get(regions)
        if handle is None:
            # NIXL_INIT_AGENT to be used for preparations of local descs.
            handle = self._build_xfer_dlist(
                agent_name="NIXL_INIT_AGENT",
                regions=regions,
                base_addresses=self.seen_base_addresses,
                device_id=self.device_id,
                memory_type=self.nixl_memory_type,
            )
            self.src_xfer_side_handles[regions] = handle
            logger.info("Prepared local NIXL dlist with regions %s", regions)
        return handle

    def _remote_xfer_dlist(
        self,
        metadata: NixlAgentMetadata,
        remote_agent_name: str,
        regions: tuple[RegionGeometry, ...],
    ) -> nixl_prepped_dlist_handle:
        handle = self._build_xfer_dlist(
            agent_name=remote_agent_name,
            regions=regions,
            base_addresses=metadata.kv_caches_base_addr,
            device_id=metadata.device_id,
            memory_type="VRAM",
        )
        self.dst_xfer_side_handles[metadata.engine_id].append(handle)
        return handle

    def _remember_remote_engine(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_size: int,
        remote_pp_size: int,
    ) -> None:
        engine_id = nixl_agent_meta.engine_id
        if engine_id not in self._tp_size:
            self._tp_size[engine_id] = remote_tp_size
            self._pp_size[engine_id] = remote_pp_size
            self._block_size[engine_id] = nixl_agent_meta.block_size
            return
        if self._tp_size[engine_id] != remote_tp_size:
            raise ValueError("remote TP size changed after engine registration")
        if self._pp_size[engine_id] != remote_pp_size:
            raise ValueError("remote PP size changed after engine registration")

    def _adjust_remote_xfer_layout(
        self,
        *,
        remote_tp_rank: int,
        remote_tp_size: int,
        remote_dp_rank: int,
        local_ptr_ids: list[int],
        remote_ptr_ids: list[int],
        local_descs: list[list[tuple[int, int]]],
        remote_descs: list[list[tuple[int, int]]],
        local_block_strides: list[int],
        remote_block_strides: list[int],
        kv_specs: list[KVCacheSpec],
    ) -> AdjustedXferLayout:
        return TpLayoutAdjuster(
            local_tp_rank=self.tp_rank,
            local_tp_size=self.tp_world_size,
            remote_tp_rank=remote_tp_rank,
            remote_tp_size=remote_tp_size,
            dp_rank=remote_dp_rank,
            total_num_kv_heads=self.total_num_kv_heads,
            use_mla=tuple(
                isinstance(kv_spec, (MLAAttentionSpec, SlidingWindowMLASpec))
                for kv_spec in kv_specs
            ),
            is_mamba=tuple(isinstance(kv_spec, MambaSpec) for kv_spec in kv_specs),
        ).adjust(
            local_ptr_ids=local_ptr_ids,
            remote_ptr_ids=remote_ptr_ids,
            local_desc=local_descs,
            remote_desc=remote_descs,
            local_block_len=local_block_strides,
            remote_block_len=remote_block_strides,
        )

    def _ensure_remote_nixl_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_agent_name: str | None,
    ) -> str:
        if remote_agent_name is not None:
            return remote_agent_name
        engine_id = nixl_agent_meta.engine_id
        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )
        if engine_id not in self._remote_agents:
            self._remote_agents[engine_id] = [remote_agent_name]
        else:
            assert remote_agent_name not in self._remote_agents[engine_id]
            self._remote_agents[engine_id].append(remote_agent_name)
        return remote_agent_name

    @staticmethod
    def _check_remote_group_id(
        remote_group_id: int, remote_block_ids: list[np.ndarray]
    ) -> None:
        # The peer's block-id tuple follows the peer's own group numbering,
        # which need not match ours (P under PP has extra forwarder groups).
        if not 0 <= remote_group_id < len(remote_block_ids):
            raise ValueError(
                f"remote group id {remote_group_id} out of range for "
                f"{len(remote_block_ids)} remote block-id groups"
            )

    def _fetch_remote_agent_metadata(
        self,
        sock: zmq.Socket,
        *,
        engine_id: EngineId,
        global_remote_rank: int,
        handshake_decoder: msgspec.msgpack.Decoder,
        decoder: msgspec.msgpack.Decoder,
    ) -> NixlAgentMetadata:
        msg = msgspec.msgpack.encode((GET_META_MSG, engine_id, global_remote_rank))
        metadata_bytes = zmq_comm2proxy(sock, msg, need_recv=True, logger=logger)
        try:
            handshake_payload = handshake_decoder.decode(metadata_bytes)
        except (msgspec.DecodeError, msgspec.ValidationError) as e:
            raise RuntimeError(
                f"Failed to decode NixlHandshakePayload. This likely indicates "
                f"an incompatibility between connector version. Error: {e}"
            ) from e

        from vllm_iluvatar.models.deepseek_v41.compressed_kv import main_kv_format

        main_format = main_kv_format()
        local_format = main_format.name if main_format is not None else "none"
        requires_compat = validate_main_kv_format(
            local_format,
            handshake_payload.main_kv_format,
            remote_engine_id=engine_id,
        )
        validate_compatibility_hash(
            self.compat_hash,
            handshake_payload.compatibility_hash,
            remote_engine_id=engine_id,
            enforce=self.enforce_handshake_compat or requires_compat,
        )
        metadata: NixlAgentMetadata = decoder.decode(
            handshake_payload.agent_metadata_bytes
        )
        if metadata.main_kv_format != handshake_payload.main_kv_format:
            raise RuntimeError(
                f"Remote NIXL main KV format tag disagrees with agent metadata "
                f"for engine {engine_id}: envelope={handshake_payload.main_kv_format}, "
                f"agent={metadata.main_kv_format}"
            )
        if metadata.engine_id != engine_id:
            raise RuntimeError(
                f"Remote NIXL agent engine ID mismatch. "
                f"Expected {engine_id},"
                f"received {metadata.engine_id}."
            )
        return metadata

    def _check_remote_agent(
        self,
        metadata: NixlAgentMetadata,
        *,
        remote_tp_size: int,
        remote_pp_size: int,
    ) -> None:
        """Rank-level checks, once per remote rank.

        Everything about a single kv (pointer geometry, kernel blocks, group
        block size) is checked per layer by ``LayerMappingPlanner`` and
        ``_check_layer_xfer``.
        """

        validate_parallel_topology(
            local_tp_size=self.tp_world_size,
            local_pp_size=self.pp_world_size,
            remote_tp_size=remote_tp_size,
            remote_pp_size=remote_pp_size,
        )
        # Block ids travel unmodified, so the block size (and every group's
        # kernel split, checked per pointer by LayerMappingPlanner) must match.
        if metadata.block_size != self.block_size:
            raise RuntimeError(
                "P/D block sizes must match: "
                f"local={self.block_size}, remote={metadata.block_size}"
            )
        if metadata.physical_blocks_per_logical_kv_block != 1:
            raise RuntimeError(
                "remote advertises kernel block ids "
                f"(physical_blocks_per_logical_kv_block="
                f"{metadata.physical_blocks_per_logical_kv_block}); this "
                "connector exchanges scheduler block ids only"
            )
        registered_specs = [
            self._layer_specs[layer_name] for layer_name in self.layer_ptr_mapping
        ]
        if not byte_compatible_kv_layouts(
            self.kv_cache_layout,
            metadata.kv_cache_layout,
            registered_specs,
        ):
            raise RuntimeError(
                "P/D KV cache layouts must be byte-compatible: "
                f"local={self.kv_cache_layout}, "
                f"remote={metadata.kv_cache_layout}"
            )
        if self.kv_cache_layout != metadata.kv_cache_layout:
            logger.info_once(
                "Allowing byte-compatible P/D KV cache layouts: "
                "local=%s, remote=%s, registered_specs=%d",
                self.kv_cache_layout,
                metadata.kv_cache_layout,
                len(registered_specs),
            )
        self._remember_remote_engine(metadata, remote_tp_size, remote_pp_size)

    def _check_layer_xfer(
        self,
        layer: LayerSelection,
        *,
        remote_tp_rank: int,
        remote_tp_size: int,
        remote_dp_rank: int,
    ) -> AdjustedXferLayout:
        """Check one layer against the peer and fix its TP-sliced geometry.

        Returns the adjusted ``(offset, len, stride)`` of every pointer of the
        layer plus the per-pointer writer choice; block counts are fixed by
        the planner (both sides register scheduler blocks).
        """

        pointers = layer.pointers
        kv_specs = [pointer.spec for pointer in pointers]
        local_descs = [list(pointer.local_desc) for pointer in pointers]
        remote_descs = [list(pointer.remote_desc) for pointer in pointers]
        local_block_strides = [pointer.local_block_stride for pointer in pointers]
        remote_block_strides = [pointer.remote_block_stride for pointer in pointers]
        if self.tp_world_size != remote_tp_size:
            for pointer in pointers:
                if pointer.kernel_blocks > 1 and not isinstance(
                    pointer.spec, (MLAAttentionSpec, SlidingWindowMLASpec)
                ):
                    # A scheduler block of r kernel blocks interleaves r head
                    # layouts; head slicing would need r segments per block.
                    raise NotImplementedError(
                        f"{pointer.kv_name}: heterogeneous TP with "
                        f"{pointer.kernel_blocks} kernel blocks per block is "
                        "only supported for MLA / replicated KV"
                    )
        validate_descriptor_geometry(
            local_ptr=[pointer.local_ptr for pointer in pointers],
            remote_ptr=[pointer.remote_ptr for pointer in pointers],
            local_desc=local_descs,
            remote_desc=remote_descs,
            local_block_len=local_block_strides,
            remote_block_len=remote_block_strides,
            layer_desc=kv_specs,
        )
        return self._adjust_remote_xfer_layout(
            remote_tp_rank=remote_tp_rank,
            remote_tp_size=remote_tp_size,
            remote_dp_rank=remote_dp_rank,
            local_ptr_ids=[pointer.local_ptr for pointer in pointers],
            remote_ptr_ids=[pointer.remote_ptr for pointer in pointers],
            local_descs=local_descs,
            remote_descs=remote_descs,
            local_block_strides=local_block_strides,
            remote_block_strides=remote_block_strides,
            kv_specs=kv_specs,
        )

    @staticmethod
    def _layer_region_pairs(
        layer: LayerSelection, layout: AdjustedXferLayout
    ) -> list[RegionPair]:
        pairs: list[RegionPair] = []
        for index, pointer in enumerate(layer.pointers):
            if not layout.pointer_write[index]:
                continue
            for (l_offset, l_len), (r_offset, r_len) in zip(
                layout.local.desc[index], layout.remote.desc[index], strict=True
            ):
                pairs.append(
                    RegionPair(
                        layer_index=layer.layer_index,
                        local=RegionGeometry(
                            ptr=pointer.local_ptr,
                            offset=l_offset,
                            length=l_len,
                            block_stride=layout.local.block_len[index],
                            num_blocks=pointer.local_num_blocks,
                        ),
                        remote=RegionGeometry(
                            ptr=pointer.remote_ptr,
                            offset=r_offset,
                            length=r_len,
                            block_stride=layout.remote.block_len[index],
                            num_blocks=pointer.remote_num_blocks,
                        ),
                        local_group_id=pointer.local_group_id,
                        remote_group_id=pointer.remote_group_id,
                        tail_aligned=not isinstance(pointer.spec, MambaSpec),
                    )
                )
        return pairs

    def _source_tp_range(
        self, remote_tp_rank: int, remote_tp_size: int
    ) -> tuple[int, int]:
        """Local TP ranks that feed ``remote_tp_rank``, as a half-open range."""

        if self.tp_world_size >= remote_tp_size:
            ratio = self.tp_world_size // remote_tp_size
            return remote_tp_rank * ratio, (remote_tp_rank + 1) * ratio
        return self.tp_rank, self.tp_rank + 1

    def _notify_msg(
        self,
        pp_range: tuple[int, int],
        pp_layer: int,
        pp_layer_range: tuple[int, int],
        tp_range: tuple[int, int],
        count: int,
    ) -> str:
        return NotifyPayload(
            src_pp=self.pp_rank,
            pp_start=pp_range[0],
            pp_end=pp_range[1],
            pp_layer=pp_layer,
            pp_layer_start=pp_layer_range[0],
            pp_layer_end=pp_layer_range[1],
            src_tp=self.tp_rank,
            tp_start=tp_range[0],
            tp_end=tp_range[1],
            count=count,
        ).encode()

    def _rank_notify_msg(
        self, pp_range: tuple[int, int], tp_range: tuple[int, int]
    ) -> str:
        return self._notify_msg(pp_range, self.pp_rank, pp_range, tp_range, 1)

    def _nixl_handshake(
        self,
        meta: NodeMeta,
    ) -> tuple[list[list[RemoteXfer | None]], list[list[RemoteXfer | None]]]:
        """建立到一个 remote engine 的全部 PP/TP 传输映射。

        对每个需要写入的 remote TP rank、每个 PP overlap：

        1. 拉取 remote agent metadata，做 rank 级检查（block size、kind），
           并 add_remote_agent（不写的副本 rank 也要，用于发 notify）；
        2. 按层号、kv 名排序遍历本端层，逐层 ``_check_layer_xfer`` 修正
           ``(offset, len, stride)`` 并选出 writer pointer；
        3. 所有 writer region 去重后拼成一对 local/remote dlist，每个 region
           带自己的 local/remote group id，一次 transfer 跨所有 group；
        4. 生成 per-layer ``[layer][remote tp]`` 与 per-rank
           ``[remote pp][remote tp]`` 两套 mapping。

        计数（D 端按 ``[source tp][source pp]`` 汇总）：有写入时 per-layer
        count = 写入层数，per-rank count = 1；本 rank 不写或 overlap 内没有可传
        层时退化为一条 notify-only，count = 1，per-layer 挂在 overlap 首层。
        notify 的 pp 段是本端 pp rank 和写这个 remote stage 的本端 pp 区间；
        pp_layer 段 per-layer 为层号 + remote stage 的层区间，per-rank 与 pp
        段相同。
        """

        remote_expire_time = meta.expire_time
        engine_id = meta.engine_id
        assert remote_expire_time >= self._kv_lease_duration, (
            f"the remote engine's: {engine_id}  expire time: {remote_expire_time}, must >= local expire time: {self._kv_lease_duration}, otherwise may get dirty data after remote expired."
        )
        remote_pp_size = meta.pp_size
        remote_tp_size = meta.tp_size
        remote_pp_split = meta.pp_split
        remote_dp_rank = meta.dp_rank
        # Group lists legitimately differ between P and D (PP on P adds
        # forwarder groups, skip_register drops layers), so they are only
        # logged.  Compatibility is enforced per layer and per pointer.
        logger.info(
            "NIXL handshake with %s: local KV groups %s, remote KV groups %s",
            engine_id,
            self.kv_group_infos_for_check,
            meta.kv_group_infos,
        )

        pipeline_planner = PipelineMappingPlanner(
            local_stage_start=self.pp_layer_start,
            local_stage_end=self.pp_layer_end,
            local_pp_split=self.pp_split,
            num_hidden_layers=self.num_hidden_layers,
        )
        pp_slices = pipeline_planner.target_slices(remote_pp_split)
        source_pp_ranges = pipeline_planner.remote_notification_ranges(remote_pp_split)
        remote_layer_bounds = [*remote_pp_split, self.num_hidden_layers]
        tp_num_remotes = max(1, remote_tp_size // self.tp_world_size)
        init_remote_tp_rank = self.transfer_topo.get_target_remote_rank(remote_tp_size)
        layer_planner = LayerMappingPlanner(
            local=RegistrationView(
                ptr=self.layer_ptr_mapping,
                block_stride=self.layer_block_len_mapping,
                desc=self.layer_desc_mapping,
                num_blocks=self.layer_num_blocks_mapping,
                group=self.layer_group_mapping,
                kernel_blocks=self.layer_kernel_blocks_mapping,
                block_tokens=self.layer_block_tokens_mapping,
                backend=self.layer_backend_mapping,
            ),
            layer_specs=self._layer_specs,
            layer_global_index=self._layer_global_index,
            pp_layer_start=self.pp_layer_start,
            num_layers=self.num_layers,
            local_tp_size=self.tp_world_size,
            has_mamba=self._has_mamba,
        )
        decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
        handshake_decoder = msgspec.msgpack.Decoder(NixlHandshakePayload)

        # [remote tp][local layer] and [remote tp][pp slice]; transposed below.
        per_layer_by_tp: list[list[RemoteXfer | None]] = []
        per_rank_by_tp: list[list[RemoteXfer | None]] = []
        with zmq_ctx(zmq.REQ, _PROXY_ADDR) as sock:
            # Set receive timeout to 5 seconds to avoid hanging on dead server
            sock.setsockopt(zmq.RCVTIMEO, 5000)  # milliseconds

            # each local rank may write to multi remote tp rank
            for i in range(tp_num_remotes):
                remote_tp_rank = init_remote_tp_rank + i
                tp_start, tp_end = self._source_tp_range(remote_tp_rank, remote_tp_size)
                per_layer_row: list[RemoteXfer | None] = [None] * self.num_layers
                per_rank_row: list[RemoteXfer | None] = [None] * len(pp_slices)

                # then, this rank may write to multi remote pp rank
                for pp_index, pp_slice in enumerate(pp_slices):
                    remote_pp_rank = pp_slice.remote_pp_rank
                    metadata = self._fetch_remote_agent_metadata(
                        sock,
                        engine_id=engine_id,
                        global_remote_rank=remote_pp_rank * remote_tp_size
                        + remote_tp_rank,
                        handshake_decoder=handshake_decoder,
                        decoder=decoder,
                    )
                    assert metadata.tp_rank == remote_tp_rank
                    assert metadata.pp_rank == remote_pp_rank
                    self._check_remote_agent(
                        metadata,
                        remote_tp_size=remote_tp_size,
                        remote_pp_size=remote_pp_size,
                    )
                    remote_agent_name = self._ensure_remote_nixl_agent(metadata, None)

                    layers = layer_planner.select(
                        local_start=pp_slice.local_start,
                        local_end=pp_slice.local_end,
                        remote=RegistrationView.from_agent_metadata(metadata),
                        remote_tp_size=remote_tp_size,
                    )
                    region_pairs: list[RegionPair] = []
                    for layer in layers:
                        layout = self._check_layer_xfer(
                            layer,
                            remote_tp_rank=remote_tp_rank,
                            remote_tp_size=remote_tp_size,
                            remote_dp_rank=remote_dp_rank,
                        )
                        region_pairs.extend(self._layer_region_pairs(layer, layout))

                    pp_range = source_pp_ranges[remote_pp_rank]
                    tp_range = (tp_start, tp_end)
                    remote_layer_range = (
                        remote_layer_bounds[remote_pp_rank],
                        remote_layer_bounds[remote_pp_rank + 1],
                    )
                    rank_notify_msg = self._rank_notify_msg(pp_range, tp_range)

                    if not region_pairs:
                        first_layer = self.pp_layer_start + pp_slice.local_start
                        per_layer_row[pp_slice.local_start] = RemoteXfer(
                            remote_agent_name=remote_agent_name,
                            notify_msg=self._notify_msg(
                                pp_range, first_layer, remote_layer_range, tp_range, 1
                            ),
                        )
                        per_rank_row[pp_index] = RemoteXfer(
                            remote_agent_name=remote_agent_name,
                            notify_msg=rank_notify_msg,
                        )
                        logger.info(
                            "NIXL mapping to %s tp %s pp %s: nothing to write "
                            "(%s registered layers), notify only",
                            engine_id,
                            remote_tp_rank,
                            remote_pp_rank,
                            len(layers),
                        )
                        continue

                    plan = build_xfer_plan(region_pairs)
                    local_handle = self._local_xfer_dlist(plan.local_regions)
                    remote_handle = self._remote_xfer_dlist(
                        metadata, remote_agent_name, plan.remote_regions
                    )
                    for layer_index, groups in plan.per_layer.items():
                        per_layer_row[layer_index - self.pp_layer_start] = RemoteXfer(
                            remote_agent_name=remote_agent_name,
                            notify_msg=self._notify_msg(
                                pp_range,
                                layer_index,
                                remote_layer_range,
                                tp_range,
                                len(plan.per_layer),
                            ),
                            local_handle=local_handle,
                            remote_handle=remote_handle,
                            region_groups=groups,
                        )
                    per_rank_row[pp_index] = RemoteXfer(
                        remote_agent_name=remote_agent_name,
                        notify_msg=rank_notify_msg,
                        local_handle=local_handle,
                        remote_handle=remote_handle,
                        region_groups=plan.per_rank,
                    )
                    logger.info(
                        "NIXL mapping to %s tp %s pp %s: %s layers write, "
                        "%s region pairs -> %s distinct regions "
                        "(local %s descriptors, remote %s descriptors)",
                        engine_id,
                        remote_tp_rank,
                        remote_pp_rank,
                        len(plan.per_layer),
                        len(region_pairs),
                        len(plan.local_regions),
                        sum(region.num_blocks for region in plan.local_regions),
                        sum(region.num_blocks for region in plan.remote_regions),
                    )

                per_layer_by_tp.append(per_layer_row)
                per_rank_by_tp.append(per_rank_row)

        # [layer][remote tp]: transfer order is layer -> remote TP.
        per_layer_mapping = [
            [row[layer_index] for row in per_layer_by_tp]
            for layer_index in range(self.num_layers)
        ]
        # [pp target][remote tp]: transfer order is PP target -> remote TP.
        per_rank_mapping = [
            [row[pp_index] for row in per_rank_by_tp]
            for pp_index in range(len(pp_slices))
        ]
        logger.info(
            "per_layer notify: %s\nper_rank notify: %s",
            [
                [cell.notify_msg if cell else None for cell in row]
                for row in per_layer_mapping
            ],
            [
                [cell.notify_msg if cell else None for cell in row]
                for row in per_rank_mapping
            ],
        )
        return per_layer_mapping, per_rank_mapping

    ############################################################
    # Common funs for repare/do send/recv kv
    ############################################################
    def bind_connector_metadata(
        self, connector_metadata: NixlConnectorMetadata
    ) -> None:
        self._connector_metadata = connector_metadata
        for _, meta in self._connector_metadata.reqs_to_send.items():
            meta.local_block_ids_np = [
                np.array(block_id, dtype=np.int32)[None, :]
                for block_id in meta.local_block_ids
            ]
        for _, meta in self._connector_metadata.reqs_to_send_layerwise.items():
            meta.local_block_ids_np = [
                np.array(block_id, dtype=np.int32)[None, :]
                for block_id in meta.local_block_ids
            ]
        self.layer_index = self.pp_layer_start
        self.un_send_layer_nums = self.num_layers
        # Step-scoped and main-thread-owned: the save hooks of this step's
        # forward consume them.  The 0.30 V2 runner calls start_load_kv only
        # after the forward (unless there are sync loads), so they must be
        # set here, at bind time, which always precedes the forward.
        self._step_layerwise_sends = connector_metadata.reqs_to_send_layerwise
        self._step_rank_sends = connector_metadata.reqs_to_send

    def clear_connector_metadata(self) -> None:
        """Drop step-scoped metadata after ModelRunner post-forward cleanup."""

        self._connector_metadata = None
        self._step_layerwise_sends = {}
        self._step_rank_sends = {}

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        # Persistent request state belongs to the uvloop actor.  This may run
        # after the forward's save hooks already queued layer readiness;
        # SendRequestState joins metadata, readiness and remote blocks in any
        # order.
        self._async_loop.call_soon_threadsafe(
            self._apply_start_load,
            dict(metadata.reqs_to_recv),
            dict(metadata.reqs_to_send),
            dict(metadata.reqs_to_send_layerwise),
            set(metadata.reqs_to_abort),
            dict(metadata.reqs_to_send_time_record),
            dict(metadata.reqs_to_release),
        )

    def _put_completion(self, batch: CompletionBatch) -> None:
        """Publish an immutable-by-convention result to the engine thread."""
        if not batch.is_empty:
            self._completion_outbox.put(batch)

    def _release_xfer_handle(self, handle: Any) -> bool:
        try:
            self.nixl_wrapper.release_xfer_handle(handle)
            return True
        except Exception as e:
            self._log_failure("release_xfer_handle_failed", None, error=e)
            return False

    def _release_dlist_handle(self, handle: Any) -> bool:
        try:
            self.nixl_wrapper.release_dlist_handle(handle)
            return True
        except Exception as e:
            self._log_failure("release_dlist_handle_failed", None, error=e)
            return False

    def _remove_remote_agent(self, agent_name: str) -> bool:
        try:
            self.nixl_wrapper.remove_remote_agent(agent_name)
            return True
        except Exception as e:
            self._log_failure("remove_remote_agent_failed", None, error=e)
            return False

    def _deregister_memory(self, desc: Any) -> bool:
        try:
            self.nixl_wrapper.deregister_memory(desc)
            return True
        except Exception as e:
            self._log_failure("deregister_memory_failed", None, error=e)
            return False

    def _session_accepts_request(
        self, req_id: ReqId, meta: ReqMeta, aborted: bool = False
    ) -> bool:
        if not meta.engine_id:
            reason = f"request {req_id} has no remote engine id"
            batch = CompletionBatch(failures={req_id: reason})
            if self.is_consumer:
                batch.done_recving.add(req_id)
                if not aborted:
                    for block_ids in meta.local_block_ids[:1]:
                        batch.invalid_block_ids.update(block_ids)
            else:
                self._finish_send(req_id, reason)
                logger.error(reason)
                return False
            self._put_completion(batch)
            logger.error(reason)
            return False

        if (
            self._accepting_work
            and not self.is_consumer
            and meta.engine_id in self._engines_needing_rehandshake
        ):
            # A placeholder would accept this request and then wait for an
            # ACTIVE session that only a handshake can produce, so rebuild the
            # session here or reject the request rather than let it hang.
            self._rehandshake_remote_engine(meta.engine_id)
        needs_rehandshake = meta.engine_id in self._engines_needing_rehandshake

        session = self.remote_sessions.get(meta.engine_id)
        if (
            self._accepting_work
            and not needs_rehandshake
            and (session is None or session.state == RemoteSessionState.CLOSED)
        ):
            # Request metadata and the scheduler-side connect message travel on
            # independent paths.  Create a placeholder so either may arrive
            # first.  Consumers never perform a remote handshake themselves,
            # so their request is also the ACTIVE-session signal.
            generation = self._engine_generations.get(meta.engine_id, 0) + 1
            self._engine_generations[meta.engine_id] = generation
            session = RemoteEngineSession(
                engine_id=meta.engine_id,
                generation=generation,
                role=self._session_role,
            )
            self.remote_sessions[meta.engine_id] = session
            if self.is_consumer:
                session.activate()
        if (
            self._accepting_work
            and not needs_rehandshake
            and session is not None
            and session.accepts_work
        ):
            return True
        reason = (
            "worker is shutting down"
            if not self._accepting_work
            else f"remote engine {meta.engine_id!r} transport failed and is "
            "waiting for a new handshake"
            if needs_rehandshake
            else f"remote engine {meta.engine_id!r} is unavailable"
            if session is None
            else f"remote engine {meta.engine_id!r} is {session.state.name}"
        )
        batch = CompletionBatch(failures={req_id: reason})
        if self.is_consumer:
            batch.done_recving.add(req_id)
            if not aborted:
                for block_ids in meta.local_block_ids[:1]:
                    batch.invalid_block_ids.update(block_ids)
        else:
            self._finish_send(req_id, reason)
            logger.warning("Rejecting request %s: %s", req_id, reason)
            return False
        self._put_completion(batch)
        logger.warning("Rejecting request %s: %s", req_id, reason)
        return False

    def _apply_start_load(
        self,
        reqs_to_recv: dict[ReqId, ReqMeta],
        reqs_to_send: dict[ReqId, ReqMeta],
        reqs_to_send_layerwise: dict[ReqId, ReqMeta],
        reqs_to_abort: set[ReqId],
        send_expirations: dict[ReqId, float],
        safe_releases: dict[ReqId, tuple[str, str]],
    ) -> None:
        """Apply model-runner metadata on the uvloop owner thread."""
        if not self._accepting_work:
            all_reqs = {
                **reqs_to_recv,
                **reqs_to_send,
                **reqs_to_send_layerwise,
            }
            for req_id, meta in all_reqs.items():
                self._session_accepts_request(
                    req_id, meta, aborted=req_id in reqs_to_abort
                )
            return

        for req_id, meta in reqs_to_recv.items():
            if self._recv_tombstones.get(req_id, meta.generation) is not None:
                continue
            aborted = req_id in reqs_to_abort
            if not self._session_accepts_request(req_id, meta, aborted=aborted):
                continue
            self._recv_requests[req_id] = meta
            self._recv_generations[req_id] = meta.generation
            session = self.remote_sessions[meta.engine_id]
            session.pending_recv.add(req_id)
            if aborted:
                self._aborted_recv_requests.add(req_id)
            early_key = (req_id, meta.generation)
            self._early_recv_deadlines.pop(early_key, None)
            for notification in self._early_recv_notifications.pop(
                early_key, []
            ):
                self._apply_recv_notification(
                    req_id, meta.generation, *notification
                )

        all_sends = {**reqs_to_send, **reqs_to_send_layerwise}
        for req_id, meta in all_sends.items():
            state = self._send_state_for_generation(
                req_id, meta.generation
            )
            if state is None:
                continue
            state.install_metadata(meta, send_expirations.get(req_id))
            if not self._session_accepts_request(req_id, meta):
                continue
            state.session = self.remote_sessions[meta.engine_id]
            state.session.pending_send.add(req_id)
            self._maybe_start_or_continue_send(state)

        for req_id, (generation, reason) in safe_releases.items():
            if self._recv_generations.get(req_id) != generation:
                continue
            self._fail_recv(req_id, reason)

    def _send_state_for_generation(
        self, req_id: ReqId, generation: str
    ) -> SendRequestState | None:
        """Return generation-matched producer state or reject fail-closed."""

        if not generation:
            logger.error("Rejecting request %s without ownership generation", req_id)
            self._put_completion(
                CompletionBatch(
                    done_sending={req_id},
                    failures={req_id: "missing ownership generation"},
                )
            )
            return None
        if self._send_tombstones.get(req_id, generation) is not None:
            # Duplicate or reordered PREPARED/CANCEL after terminal. Re-emit
            # completion so a peer retry can recover a lost proof.
            self._put_completion(CompletionBatch(done_sending={req_id}))
            return None
        state = self._send_requests.get(req_id)
        if state is None:
            state = SendRequestState(req_id=req_id, generation=generation)
            self._send_requests[req_id] = state
            return state
        if state.generation and state.generation != generation:
            self._send_tombstones.record(
                req_id, generation, RequestOwnershipState.CANCELED
            )
            self._put_completion(
                CompletionBatch(
                    done_sending={req_id},
                    failures={
                        req_id: (
                            "stale ownership generation "
                            f"{generation}; active={state.generation}"
                        )
                    },
                )
            )
            return None
        state.generation = generation
        return state

    def _cancel_send_request(
        self, req_id: ReqId, generation: str, reason: str
    ) -> None:
        """Stop new WRITEs and publish completion only after native terminal."""

        state = self._send_state_for_generation(req_id, generation)
        if state is None:
            return
        state.cancel_requested = True
        state.ready_layers.clear()
        if not state.started:
            state.ownership_state = RequestOwnershipState.CANCELED
            self._finish_send(req_id, reason)
            return
        state.ownership_state = RequestOwnershipState.WRITING
        wrapper = self._send_transfers.get(req_id)
        if wrapper is None:
            self._finish_send(req_id, reason)
            return
        wrapper.ready_for_check = True
        self._poll_send_transfers(include_unready=True)

    def _maybe_start_or_continue_send(self, state: SendRequestState) -> None:
        """Join metadata, ACTIVE handshake, remote blocks and local readiness."""

        if state.metadata is None:
            return
        req_id = state.req_id
        meta = state.metadata
        session = state.session
        if state.cancel_requested:
            state.ready_layers.clear()
            if not state.started:
                self._finish_send(req_id, "ownership canceled before WRITE")
            return
        if not self._accepting_work or not state.is_ready():
            logger.info(
                "NIXL send %s waiting: accepting=%s metadata=%s session=%s "
                "intent=%s remote_blocks=%s ready_layers=%d",
                req_id,
                self._accepting_work,
                state.metadata is not None,
                None if session is None else session.state.name,
                state.transfer_intent,
                state.has_remote_blocks,
                len(state.ready_layers),
            )
            return

        try:
            if not state.started:
                state.started = True
                state.ownership_state = RequestOwnershipState.WRITING
                assert session is not None
                session.pending_send.discard(req_id)
                session.inflight_send.add(req_id)
                logger.debug("start nixl send req=%s", req_id)

            while state.ready_layers:
                layer_index, _ready_meta = state.ready_layers.pop(0)
                self._write_blocks(
                    meta.local_block_ids_np,
                    state.remote_block_ids,
                    meta.engine_id,
                    req_id,
                    state.generation,
                    layer_index,
                    TransferIntent(state.transfer_intent),
                    state.remote_block_starts,
                )
        except Exception as e:
            reason = f"failed to start NIXL transfer: {e}"
            self._log_failure("transfer_start_failed", req_id, error=e)
            self._note_transport_failure(meta.engine_id, e, "transfer start failed")
            self._send_failures[req_id] = reason
            wrapper = self._send_transfers.get(req_id)
            if wrapper is None:
                self._send_failures.pop(req_id, None)
                self._finish_send(req_id, reason)
            else:
                wrapper.ready_for_check = True

    @staticmethod
    def _build_notification(
        request_id: str,
        generation: str,
        notify_msg: str,
        status: TransferNotificationStatus,
    ) -> bytes:
        return (
            f"{request_id}:{generation}:{notify_msg}:{status.value}".encode()
        )

    def _cell_descriptor_ids(
        self,
        cell: RemoteXfer,
        local_block_ids: list[np.ndarray],
        remote_block_ids: list[np.ndarray],
        remote_block_starts: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Concatenate the descriptor ids of every region group of one cell.

        None means nothing to move for this request (a notify-only cell, or
        every group of the cell has no blocks on one side, e.g. the local
        request was aborted); the notify is still sent so D can count it.
        """

        local_parts: list[np.ndarray] = []
        remote_parts: list[np.ndarray] = []
        for group in cell.region_groups:
            self._check_remote_group_id(group.remote_group_id, remote_block_ids)
            ids = group.descriptor_ids(
                local_block_ids, remote_block_ids, remote_block_starts
            )
            if ids is None:
                continue
            local_parts.append(ids[0])
            remote_parts.append(ids[1])
        if not local_parts:
            return None
        local_ids = np.concatenate(local_parts)
        remote_ids = np.concatenate(remote_parts)
        assert len(local_ids) == len(remote_ids)
        return local_ids, remote_ids

    def _write_blocks(
        self,
        local_block_ids: list[np.ndarray],
        remote_block_ids: list[np.ndarray],
        dst_engine_id: str,
        request_id: str,
        generation: str,
        layer_index: int,
        transfer_intent: TransferIntent,
        remote_block_starts: list[int] | None = None,
    ):
        validate_block_id_groups(
            local_block_ids=local_block_ids,
            remote_block_ids=remote_block_ids,
            expected_groups=self.num_groups,
        )
        final_layer_or_rank = (
            layer_index == -1 or layer_index == self.pp_layer_end_closed
        )
        has_local_blocks = any(block_ids.shape[1] > 0 for block_ids in local_block_ids)
        has_remote_blocks = any(
            block_ids.shape[1] > 0 for block_ids in remote_block_ids
        )
        if not has_remote_blocks:
            logger.info(
                "NIXL send %s layer=%d intent=%s: no remote blocks, nothing sent",
                request_id,
                layer_index,
                transfer_intent.value,
            )
            if transfer_intent == TransferIntent.RECEIVE_KV:
                raise ValueError(
                    f"request {request_id} asked to receive KV without remote blocks"
                )
            if transfer_intent == TransferIntent.NO_TRANSFER:
                # remote got a full cache, no transfer needed, but local blocks must be non-empty
                assert has_local_blocks
            # for transfer_intent == TransferIntent.CANCEL, local blocks can be empty or non-empty
            if request_id not in self._send_transfers:
                self._send_transfers[request_id] = TransferWrapper(
                    final_layer_or_rank,
                    [[None, True, None, dst_engine_id]],
                )
            else:
                self._send_transfers[request_id].ready_for_check = final_layer_or_rank
            return

        if layer_index != -1:
            cells = self._remote_agents_per_layer[dst_engine_id][
                layer_index - self.pp_layer_start
            ]
        else:
            cells = [
                cell
                for pp_cells in self._remote_agents_per_rank[dst_engine_id]
                for cell in pp_cells
            ]

        new_handles = []
        # (notify msg, WRITE issued, local descs, remote descs) per sent cell
        sent: list[tuple[str, bool, int, int]] = []
        for cell in cells:
            if cell is None:
                continue
            success_notif = self._build_notification(
                request_id,
                generation,
                cell.notify_msg,
                TransferNotificationStatus.SUCCESS,
            )
            ids = (
                None
                if cell.notify_only
                else self._cell_descriptor_ids(
                    cell, local_block_ids, remote_block_ids, remote_block_starts
                )
            )
            if ids is None:
                self.nixl_wrapper.send_notif(cell.remote_agent_name, success_notif)
                new_handles.append([None, True, time.perf_counter(), dst_engine_id])
                sent.append((cell.notify_msg, False, 0, 0))
                continue

            local_descs_id, remote_descs_id = ids
            sent.append(
                (cell.notify_msg, True, len(local_descs_id), len(remote_descs_id))
            )
            handle = None
            try:
                handle = self.nixl_wrapper.make_prepped_xfer(
                    "WRITE",
                    cell.local_handle,
                    local_descs_id,
                    cell.remote_handle,
                    remote_descs_id,
                    notif_msg=success_notif,
                )
                self.nixl_wrapper.transfer(handle)
                new_handles.append([handle, False, time.perf_counter(), dst_engine_id])
                logger.debug("transfer with notify_id: %s", success_notif)
            except Exception as e:
                self._log_failure(
                    failure_type="transfer_setup_failed",
                    req_id=request_id,
                    msg="Marking blocks as invalid",
                    error=e,
                )
                self.xfer_stats.record_failed_transfer()
                self._send_failures[request_id] = f"transfer setup failed: {e}"
                self._note_transport_failure(
                    dst_engine_id, e, "transfer setup failed"
                )
                try:
                    self.nixl_wrapper.send_notif(
                        cell.remote_agent_name,
                        self._build_notification(
                            request_id,
                            generation,
                            cell.notify_msg,
                            TransferNotificationStatus.FAILED,
                        ),
                    )
                except Exception as notify_error:
                    self.xfer_stats.record_failed_notification()
                    self._log_failure(
                        "failure_notification_failed",
                        request_id,
                        error=notify_error,
                    )
                    # D is now unreachable even for a one-byte notification,
                    # so it cannot be told about this or any later failure.
                    self._note_transport_failure(
                        dst_engine_id,
                        notify_error,
                        "failure notification failed",
                        agent_scoped=True,
                    )
                if handle is not None:
                    try:
                        xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                    except Exception:
                        xfer_state = "PROC"
                    if xfer_state == "PROC":
                        new_handles.append(
                            [handle, False, time.perf_counter(), dst_engine_id]
                        )
                    else:
                        self._release_xfer_handle(handle)

        if sent:
            logger.info(
                "NIXL send %s layer=%d intent=%s: %d notifies, %d with WRITE; "
                "(notify, write, local_descs, remote_descs)=%s",
                request_id,
                layer_index,
                transfer_intent.value,
                len(sent),
                sum(write for _, write, _, _ in sent),
                sent,
            )
        if request_id not in self._send_transfers:
            headles = TransferWrapper(False, [])
            self._send_transfers[request_id] = headles
        else:
            headles = self._send_transfers[request_id]
        headles.transfer_list.extend(new_handles)
        headles.ready_for_check = final_layer_or_rank

    ############################################################
    # Main fun for put save tasks to async loop
    ############################################################
    def save_kv_layer_cuda(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        if self.is_consumer:
            return

        # save kv layer will be called after bind_connector_metadata
        if not self._connector_metadata:
            return

        layer_index = self.layer_index
        if reqs := self._step_layerwise_sends:
            self._async_loop.call_soon_threadsafe(
                self._send_ready_queue.put_nowait,
                LayerReadyBatch(
                    requests=reqs,
                    layer_index=layer_index,
                    cuda_event=torch.cuda.current_stream().record_event(),
                ),
            )

        self.layer_index += 1
        self.un_send_layer_nums -= 1
        if not self.un_send_layer_nums and (reqs := self._step_rank_sends):
            self._async_loop.call_soon_threadsafe(
                self._send_ready_queue.put_nowait,
                LayerReadyBatch(
                    requests=reqs,
                    layer_index=-1,
                    cuda_event=torch.cuda.current_stream().record_event(),
                ),
            )

    def save_kv_layer_host(
        self,
        layer_name: str,
        kv_layer: Any,
        slot_mapping: torch.Tensor,
        **kwargs,
    ) -> None:
        if self.is_consumer:
            return

        # save kv layer will be called after start_load_kv
        if not self._connector_metadata:
            return

        if not (reqs := self._step_rank_sends):
            return
        if self._host_send_ready_queue is None or self.d2h_manager is None:
            raise RuntimeError("host D2H manager was not initialized")
        self._async_loop.call_soon_threadsafe(
            self._host_send_ready_queue.put_nowait,
            LayerReadyBatch(
                requests=reqs,
                layer_index=layer_name,
                cuda_event=torch.cuda.current_stream().record_event(),
                slot_mapping=slot_mapping,
            ),
        )
        qsize = self._host_send_ready_queue.qsize()
        if qsize >= 16:
            logger.debug(
                "host d2h enqueue layer=%s qsize=%s reqs=%s",
                layer_name,
                qsize,
                tuple(reqs),
            )

    async def _wait_host_pred(
        self, pred, *, what: str, layer: str, interval: float
    ) -> None:
        waits = 0
        while not pred():
            waits += 1
            if waits == 10 or waits % 100 == 0:
                logger.debug(
                    "host %s waiting layer=%s n=%s qsize=%s",
                    what,
                    layer,
                    waits,
                    self._host_send_ready_queue.qsize()
                    if self._host_send_ready_queue is not None
                    else -1,
                )
            await asyncio.sleep(interval)

    async def _run_d2h(self):
        if self._host_send_ready_queue is None:
            raise RuntimeError("host D2H queue was not initialized")
        while True:
            batch = await self._host_send_ready_queue.get()
            qsize = self._host_send_ready_queue.qsize()
            if qsize >= 8:
                logger.debug(
                    "host d2h backlog qsize=%s layer=%s reqs=%s",
                    qsize,
                    batch.layer_index,
                    tuple(batch.requests),
                )
            manager = self.d2h_manager
            if manager is None:
                raise RuntimeError("host D2H manager was not initialized")
            if batch.slot_mapping is None or not isinstance(batch.layer_index, str):
                raise RuntimeError("invalid host layer-ready batch")
            await self._wait_host_pred(
                batch.cuda_event.query,
                what="cuda_event",
                layer=batch.layer_index,
                interval=0.0005,
            )

            event = manager.transfer_d2h_once(
                batch.slot_mapping,
                batch.layer_index,
            )

            await self._wait_host_pred(
                event.query,
                what="d2h",
                layer=batch.layer_index,
                interval=0.001,
            )
            manager.transfer_h2h_once(batch.layer_index)

            if manager.is_last_tensor_on_rank(batch.layer_index):
                await self._wait_host_pred(
                    manager.all_h2h_done,
                    what="h2h",
                    layer=batch.layer_index,
                    interval=0.001,
                )
                logger.debug(
                    "host h2h done reqs=%s qsize=%s",
                    tuple(batch.requests),
                    self._host_send_ready_queue.qsize(),
                )
                manager.reset()
                for req_id, meta in batch.requests.items():
                    state = self._send_state_for_generation(
                        req_id, meta.generation
                    )
                    if state is None:
                        continue
                    state.ready_layers.append((-1, meta))
                    self._maybe_start_or_continue_send(state)

    ############################################################
    # Get results for sending and recving
    ############################################################
    def _complete_recv(self, req_id: ReqId) -> None:
        meta = self._recv_requests.get(req_id)
        if meta is None:
            logger.warning("Dropping completion for unknown receive %s", req_id)
            return
        generation = self._recv_generations.get(req_id, meta.generation)
        try:
            self._recv_tombstones.record(
                req_id, generation, RequestOwnershipState.TERMINAL
            )
        except RuntimeError as error:
            logger.critical(
                "Unable to retain receive tombstone for %s/%s: %s",
                req_id,
                generation,
                error,
            )
            return
        self._recv_requests.pop(req_id, None)
        self._recv_generations.pop(req_id, None)
        self._aborted_recv_requests.discard(req_id)
        self._recv_notification_status.pop(req_id, None)
        session = self.remote_sessions.get(meta.engine_id)
        if session is not None:
            session.pending_recv.discard(req_id)
        self._put_completion(
            CompletionBatch(
                done_recving={req_id},
                recv_metadata={req_id: meta},
            )
        )

    def _fail_recv(self, req_id: ReqId, reason: str) -> None:
        meta = self._recv_requests.get(req_id)
        if meta is None:
            return
        generation = self._recv_generations.get(req_id, meta.generation)
        try:
            self._recv_tombstones.record(
                req_id, generation, RequestOwnershipState.TERMINAL
            )
        except RuntimeError as error:
            logger.critical(
                "Unable to retain receive tombstone for %s/%s: %s",
                req_id,
                generation,
                error,
            )
            return
        self._recv_requests.pop(req_id, None)
        self._recv_generations.pop(req_id, None)
        self._recv_notification_status.pop(req_id, None)
        aborted = req_id in self._aborted_recv_requests
        self._aborted_recv_requests.discard(req_id)
        session = self.remote_sessions.get(meta.engine_id)
        if session is not None:
            session.pending_recv.discard(req_id)
        invalid_block_ids: set[int] = set()
        # With several KV groups block ids are only unique per group and the
        # 0.30 scheduler rejects them; failed_recving already covers the request.
        if not aborted and self.num_groups == 1:
            # Preserve the old policy: invalidate only the first Attention group.
            for block_ids in meta.local_block_ids[:1]:
                invalid_block_ids.update(block_ids)
        self._put_completion(
            CompletionBatch(
                done_recving={req_id},
                invalid_block_ids=invalid_block_ids,
                failures={req_id: reason},
            )
        )

    def _apply_recv_notification(
        self,
        req_id: ReqId,
        generation: str,
        status: TransferNotificationStatus,
        payload: NotifyPayload,
        *,
        allow_buffer: bool = False,
    ) -> None:
        if self._recv_tombstones.get(req_id, generation) is not None:
            return
        if req_id not in self._recv_requests:
            if allow_buffer and self._accepting_work:
                early_key = (req_id, generation)
                self._early_recv_notifications[early_key].append((status, payload))
                self._early_recv_deadlines.setdefault(
                    early_key, time.perf_counter() + self._kv_lease_duration
                )
            else:
                logger.warning(
                    "Dropping late or unknown NIXL notification for request %s",
                    req_id,
                )
            return
        if self._recv_generations.get(req_id) != generation:
            logger.warning(
                "Dropping stale NIXL notification for request %s generation %s",
                req_id,
                generation,
            )
            return

        if status == TransferNotificationStatus.FAILED:
            # Setup failure does not prove a partially-created native WRITE is
            # terminal. Scheduler lease expiry drives generation-scoped CANCEL.
            logger.warning(
                "Remote reported NIXL transfer failure for %s/%s; retaining "
                "destination blocks pending producer terminal proof",
                req_id,
                generation,
            )
            return

        transfer_helper = self._recv_notification_status.get(req_id)
        if transfer_helper is None:
            transfer_helper = TransferStatusChecker.for_payload(payload)
            self._recv_notification_status[req_id] = transfer_helper
        elif not transfer_helper.matches(payload):
            logger.error(
                "Dropping NIXL notification for request %s with a producer "
                "table different from the first one "
                "tp [%s, %s) x pp [%s, %s) x pp_layer [%s, %s): %s",
                req_id,
                transfer_helper.tp_start,
                transfer_helper.tp_end,
                transfer_helper.pp_start,
                transfer_helper.pp_end,
                transfer_helper.pp_layer_start,
                transfer_helper.pp_layer_end,
                payload,
            )
            return
        transfer_helper.update(payload, logger)
        if transfer_helper.is_done():
            self._complete_recv(req_id)

    def _poll_new_notifs(self) -> None:
        """Poll and account notifications on the uvloop owner thread."""
        try:
            notifications = self.nixl_wrapper.get_new_notifs().values()
        except Exception as e:
            self._log_failure("notification_poll_failed", None, error=e)
            return

        for notifs in notifications:
            for notif in notifs:
                try:
                    (
                        req_id,
                        generation,
                        pp_text,
                        layer_text,
                        tp_text,
                        status_text,
                    ) = notif.decode("utf-8").rsplit(":", 5)
                    status = TransferNotificationStatus(status_text)
                    payload = NotifyPayload.decode(pp_text, layer_text, tp_text)
                except Exception as e:
                    logger.warning(
                        "Dropping malformed NIXL notification %r: %s", notif, e
                    )
                    continue

                self._apply_recv_notification(
                    req_id,
                    generation,
                    status,
                    payload,
                    allow_buffer=True,
                )

    def _finish_send(self, req_id: ReqId, failure: str | None = None) -> bool:
        if failure:
            self._send_failures[req_id] = failure
        state = self._send_requests.get(req_id)
        wrapper = self._send_transfers.get(req_id)
        engine_id = None
        if wrapper is not None:
            release_failed = False
            for item in wrapper.transfer_list:
                handle, _done, _, remote_engine_id = item
                engine_id = remote_engine_id
                if handle is None:
                    continue
                if self._release_xfer_handle(handle):
                    # Do not poll or release this native handle a second time
                    # if another handle in the same transfer fails cleanup.
                    item[0] = None
                    item[1] = True
                else:
                    release_failed = True
            if release_failed:
                self._send_failures.setdefault(
                    req_id, "unable to release terminal NIXL transfer handle"
                )
                return False
            self._send_transfers.pop(req_id, None)
        if state is not None and state.generation:
            terminal_state = (
                RequestOwnershipState.TERMINAL
                if state.started
                else RequestOwnershipState.CANCELED
            )
            try:
                self._send_tombstones.record(
                    req_id, state.generation, terminal_state
                )
            except RuntimeError as error:
                # Forgetting this generation would permit a late PREPARED to
                # start a new WRITE. Stop accepting work and withhold proof.
                self._accepting_work = False
                state.cancel_requested = True
                logger.critical(
                    "Unable to retain ownership tombstone for %s/%s: %s",
                    req_id,
                    state.generation,
                    error,
                )
                return False
        state = self._send_requests.pop(req_id, None)
        if state is not None and state.metadata is not None:
            engine_id = state.metadata.engine_id
        if engine_id is not None:
            session = self.remote_sessions.get(engine_id)
            if session is not None:
                session.pending_send.discard(req_id)
                session.inflight_send.discard(req_id)
        batch = CompletionBatch(done_sending={req_id})
        failure = self._send_failures.pop(req_id, None)
        if failure:
            batch.failures[req_id] = failure
        self._put_completion(batch)
        return True

    def _poll_send_transfers(self, include_unready: bool = False) -> None:
        for req_id, wrapper in list(self._send_transfers.items()):
            if not include_unready and not wrapper.ready_for_check:
                continue
            in_progress = False
            for item in wrapper.transfer_list:
                handle, done, _xfer_stime, _remote_engine_id = item
                if done:
                    continue
                try:
                    xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                except Exception as e:
                    self._note_transport_failure(
                        _remote_engine_id, e, "transfer state query failed"
                    )
                    if _is_terminal_xfer_query_error(e):
                        item[1] = True
                        if req_id not in self._send_failures:
                            self._send_failures[req_id] = (
                                "terminal NIXL transfer state: "
                                f"{type(e).__name__}: {e}"
                            )
                            self.xfer_stats.record_failed_transfer()
                            logger.error(
                                "NIXL transfer for request %s reached terminal "
                                "disconnect state %s; releasing handle",
                                req_id,
                                self._send_failures[req_id],
                            )
                    else:
                        in_progress = True
                        if req_id not in self._send_failures:
                            self._send_failures[req_id] = (
                                "unable to query NIXL transfer state: "
                                f"{type(e).__name__}: {e}"
                            )
                            logger.error(
                                "Unable to query NIXL transfer state for request %s; "
                                "retaining handle: %s",
                                req_id,
                                self._send_failures[req_id],
                            )
                    continue
                if xfer_state == "DONE":
                    try:
                        self.xfer_stats.record_transfer(
                            self.nixl_wrapper.get_xfer_telemetry(handle)
                        )
                    except Exception as e:
                        logger.warning(
                            "Unable to collect telemetry for request %s: %s",
                            req_id,
                            e,
                        )
                    item[1] = True
                elif xfer_state == "PROC":
                    in_progress = True
                else:
                    item[1] = True
                    self._send_failures[req_id] = f"NIXL transfer state {xfer_state}"
                    self.xfer_stats.record_failed_transfer()
                    logger.error(
                        "NIXL transfer failed for request %s with state %s",
                        req_id,
                        xfer_state,
                    )
            if not in_progress:
                self._finish_send(req_id, self._send_failures.get(req_id))

    def _expire_requests(self, now: float) -> None:
        for early_key, expires in list(self._early_recv_deadlines.items()):
            if now >= expires:
                del self._early_recv_deadlines[early_key]
                self._early_recv_notifications.pop(early_key, None)
                logger.warning(
                    "Discarding NIXL notification for request %s/%s that no "
                    "receive ever claimed",
                    *early_key,
                )

        for req_id, state in list(self._send_requests.items()):
            if state.deadline is None or now < state.deadline:
                continue
            if not state.started:
                if state.generation and state.metadata is not None and (
                    state.has_remote_blocks
                ):
                    # Both halves joined, so a WRITE still follows once the
                    # local layers are ready.  Keep source ownership until D
                    # sends CANCEL, the remote session disconnects, or WRITE
                    # reaches terminal.
                    state.deadline = None
                    logger.warning(
                        "Producer lease expired for request %s generation %s "
                        "before WRITE; retaining ownership "
                        "until consumer cancel or disconnect",
                        req_id,
                        state.generation,
                    )
                    continue
                if state.generation and not state.orphan_grace_armed:
                    # A consumer queued behind other work can disclose its
                    # destination blocks after the first lease, so grant one
                    # more window before calling this ownerless.
                    state.orphan_grace_armed = True
                    state.deadline = now + self.producer_orphan_timeout
                    logger.warning(
                        "Producer send %s generation %s is still unjoined "
                        "(metadata=%s remote_blocks=%s); holding it for %.0fs "
                        "more before reclaiming the source blocks",
                        req_id,
                        state.generation,
                        state.metadata is not None,
                        state.has_remote_blocks,
                        self.producer_orphan_timeout,
                    )
                    continue
                # Nobody ever claimed both halves: an aborted client leaves the
                # consumer with nothing to send and this engine would otherwise
                # hold the source blocks for its whole lifetime.  No WRITE was
                # issued, so _finish_send's tombstone also fences a late
                # PREPARED instead of letting it start one.
                self.xfer_stats.record_kv_expired_req()
                self._finish_send(
                    req_id,
                    "ownerless send reclaimed: metadata="
                    f"{state.metadata is not None} "
                    f"remote_blocks={state.has_remote_blocks}",
                )

    def _progress_sessions(self, now: float) -> None:
        for session in list(self.remote_sessions.values()):
            if (
                session.role == "consumer"
                and session.state == RemoteSessionState.DRAINING
                and session.drain_deadline is not None
                and now >= session.drain_deadline
            ):
                if session.pending_recv and not session.stuck_reported:
                    session.stuck_reported = True
                    logger.warning(
                        "Remote producer %s disconnected with %d destination "
                        "ownership(s); fencing native peer before release",
                        session.engine_id,
                        len(session.pending_recv),
                    )
                self._cleanup_session_if_drained(
                    session, fence_pending_recv=True
                )
                continue
            elif (
                session.role == "producer"
                and session.state == RemoteSessionState.DRAINING
                and session.drain_deadline is not None
                and now >= session.drain_deadline
                and session.inflight_send
            ):
                session.state = RemoteSessionState.STUCK
                if not session.stuck_reported:
                    session.stuck_reported = True
                    logger.error(
                        "Remote engine %s generation %s still has active transfer "
                        "handles after disconnect drain timeout; retaining NIXL "
                        "resources (last failure=%s)",
                        session.engine_id,
                        session.generation,
                        next(
                            (
                                failure
                                for req_id, failure in self._send_failures.items()
                                if req_id in session.inflight_send
                            ),
                            "none",
                        ),
                    )
            self._cleanup_session_if_drained(session)

    def _progress_once(self) -> None:
        if self.is_consumer:
            self._poll_new_notifs()
        else:
            self._poll_send_transfers()
        self._apply_transport_failures()
        now = time.perf_counter()
        self._expire_requests(now)
        self._progress_sessions(now)

    async def _run_progress(self) -> None:
        while not self.stop_event.is_set():
            self._progress_once()
            await asyncio.sleep(0.005)

    # NOTE(gyf): this is a extra fun for dsv32 nv use
    def quant_device_kv(self, block_ids: list[int]):
        indices = torch.tensor(block_ids, dtype=torch.int32, device=self.device_for_ops)
        torch.ops._Extra.indexer_k_quant_and_cache_with_block(
            self.indexer_tensor_ptrs,
            self.quant_indexer_tensor_ptrs,
            indices,
            128,
            self.block_size,
            self.index_head_dim,
            "ue8m0",
        )

    def permute_device_kv(self, block_ids: list[int]):
        """Transforms the layout of received KV cache blocks to the local format.

        This method corrects layout mismatches from direct memory copies by
        permuting the tensor dimensions.

        - **Source Layout:** `[num_blocks, n_kv_head, block_size, head_dim]`
        - **Target Layout:** `[num_blocks, block_size, n_kv_head, head_dim]`

        Args:
            block_ids: A list of block IDs to update and permute.

        Implementation:
        - x = blocks_to_update.reshape(src_shape) # view local kv with sender layout
        - permuted_blocks = x.permute(*inv_order) # transpose n_kv_heads, block_size
        - cache.index_copy_(0, indices, permuted_blocks) # copy permuted kv back

        """
        split_k_and_v = self.transfer_topo.split_k_and_v
        inv_order = [0, 2, 1, 3]
        sample_cache = list(self.device_kv_caches.values())[0][0]
        target_shape = list(sample_cache.shape)
        target_shape[0] = -1
        src_shape = tuple(target_shape[i] for i in inv_order)
        indices = torch.tensor(block_ids, device=sample_cache.device)

        for _, cache_or_caches in self.device_kv_caches.items():
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
            for cache in cache_list:
                blocks_to_update = cache.index_select(0, indices)
                permuted_blocks = blocks_to_update.reshape(src_shape).permute(
                    *inv_order
                )
                cache.index_copy_(0, indices, permuted_blocks)

    def get_transfer_results(self) -> KVConnectorTransferResults:
        """Non-blockingly drain uvloop completions and run CUDA postprocess.

        0.30 calls this instead of ``get_finished`` so that receive failures
        reach the scheduler: a failed receive appears in both
        ``finished_recving`` (to release the transfer wait) and
        ``failed_recving`` (to mark the request as unusable).  The drain is
        destructive, so ``get_finished`` delegates here rather than draining
        a second time.
        """
        merged = CompletionBatch()
        while True:
            try:
                merged.merge(self._completion_outbox.get_nowait())
            except queue.Empty:
                break

        block_ids_to_permute: list[int] = []
        block_ids_to_quant: list[int] = []
        for meta in merged.recv_metadata.values():
            if self.enable_permute_local_kv and meta.need_permute_after_recv:
                block_ids_to_permute.extend(meta.local_block_ids[0])
            if self.use_quant_to_fp8:
                block_ids_to_quant.extend(meta.local_block_ids[0])
        if block_ids_to_permute:
            self.permute_device_kv(block_ids_to_permute)
        if block_ids_to_quant:
            self.quant_device_kv(block_ids_to_quant)
        self._invalid_block_ids.update(merged.invalid_block_ids)

        if merged.failures:
            logger.warning(
                "NIXL completed %s request(s) with failure: %s",
                len(merged.failures),
                merged.failures,
            )
        # A send failure is surfaced through stats and logs only; the scheduler
        # has no failed_sending channel.
        failed_recving = {
            req_id for req_id in merged.failures if req_id in merged.done_recving
        }
        return KVConnectorTransferResults(
            finished_sending=merged.done_sending,
            finished_recving=merged.done_recving,
            failed_recving=failed_recving,
        )

    def get_finished(self) -> tuple[set[str], set[str]]:
        results = self.get_transfer_results()
        return results.finished_sending, results.finished_recving

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the connector.
        """
        # Clear stats for next iteration
        if not self.xfer_stats.is_empty():
            return self.xfer_stats.clone_and_reset()
        return None

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Return and clear the set of block IDs that failed to load.

        This is called by the scheduler to identify blocks that need
        to be retried after a NIXL transfer failure.
        """
        # Drain the queue (thread-safe, no lock needed).
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    ############################################################
    # Exit for clear resources
    ############################################################
    def __del__(self):
        # Destructors must tolerate partially initialized interpreters.
        with contextlib.suppress(Exception):
            self.shutdown()

    def shutdown(self):
        """Synchronously ask the uvloop owner to drain and release resources."""
        shutdown_lock = getattr(self, "_shutdown_lock", None)
        if shutdown_lock is None:
            return
        with shutdown_lock:
            if self._shutdown_complete:
                return
            loop = getattr(self, "_async_loop", None)
            worker_thread = getattr(self, "worker_thread", None)
            if loop is None or worker_thread is None or not worker_thread.is_alive():
                if any(
                    getattr(self, name, None)
                    for name in ("_send_transfers", "_recv_requests", "_registered_descs")
                ):
                    raise NixlDrainIncomplete("Owner loop stopped with live registrations")
                self._shutdown_complete = True
                return
            if threading.current_thread() is worker_thread:
                loop.create_task(self._shutdown_on_loop())
                return
            future = getattr(self, "_shutdown_future", None)
            if future is None or future.done():
                future = asyncio.run_coroutine_threadsafe(self._shutdown_on_loop(), loop)
                self._shutdown_future = future
            drain_bound = min(self.producer_disconnect_drain_timeout, 5.0)
            try:
                future.result(timeout=max(1.0, drain_bound + 5.0))
            except Exception as e:
                # A timeout does not cancel the live owner-loop drain.
                raise NixlDrainIncomplete("NIXL shutdown is not quiescent") from e
            worker_thread.join(timeout=5.0)
            if worker_thread.is_alive():
                raise NixlDrainIncomplete("NIXL owner loop has not exited")
            self._shutdown_complete = True

    def _report_teardown(
        self, session: RemoteEngineSession, status: str
    ) -> None:
        """Report one terminal teardown result per local worker and generation."""
        if session.teardown_id is None:
            return
        if session.reported_teardown_id == session.teardown_id:
            return
        session.reported_teardown_id = session.teardown_id
        self._teardown_outbox.put_nowait(
            (
                session.engine_id,
                session.generation,
                session.teardown_id,
                self.global_rank,
                status,
            )
        )

    async def _deliver_to_scheduler(self, path: str, payload: bytes) -> bool:
        """Send one report to this rank's scheduler, retrying a lost ACK."""
        for attempt in range(5):
            try:
                with make_zmq_socket(
                    path=path,
                    ctx=self.ctx,
                    socket_type=zmq.REQ,
                    bind=False,
                    linger=0,
                ) as socket:
                    socket.setsockopt(zmq.SNDTIMEO, 3000)
                    socket.setsockopt(zmq.RCVTIMEO, 3000)
                    await socket.send(payload)
                    await socket.recv()
                return True
            except Exception as error:
                logger.warning(
                    "NIXL worker report retry %d/5 failed: %s", attempt + 1, error
                )
                await asyncio.sleep(0.2 * (attempt + 1))
        return False

    async def _run_reachability_reports(self) -> None:
        """Tell this rank's scheduler which remote engines it can still reach.

        The scheduler answers the consumer's commands, so this is the only way
        a consumer can learn about a dead pair before its own lease expires:
        the producer cannot reach it over NIXL by definition.
        """
        path = make_zmq_path("tcp", self.side_channel_addr)
        while True:
            engine_id, reachable = await self._reachability_outbox.get()
            payload = msgspec.msgpack.encode(
                (
                    ENGINE_REACHABILITY_MSG,
                    (engine_id, self.global_rank, bool(reachable)),
                )
            )
            if not await self._deliver_to_scheduler(path, payload):
                logger.error(
                    "Unable to report remote engine %s as %s from rank %s",
                    engine_id,
                    "reachable" if reachable else "unreachable",
                    self.global_rank,
                )
            self._reachability_outbox.task_done()

    async def _run_teardown_reports(self) -> None:
        """Deliver worker teardown completion to this rank's scheduler."""
        path = make_zmq_path("tcp", self.side_channel_addr)
        while True:
            engine_id, generation, teardown_id, global_rank, status = (
                await self._teardown_outbox.get()
            )
            payload = msgspec.msgpack.encode(
                (
                    TEARDOWN_COMPLETE_MSG,
                    (
                        engine_id,
                        generation,
                        teardown_id,
                        global_rank,
                        status,
                    ),
                )
            )
            if not await self._deliver_to_scheduler(path, payload):
                # Router remains fenced when an ACK is lost; never fake safety.
                logger.error(
                    "Unable to report teardown for %s generation %s rank %s",
                    engine_id,
                    generation,
                    global_rank,
                )
            self._teardown_outbox.task_done()

    def _cleanup_session_if_drained(
        self,
        session: RemoteEngineSession,
        *,
        fence_pending_recv: bool = False,
    ) -> None:
        can_fence_recv = (
            fence_pending_recv
            and session.role == "consumer"
            and session.state
            in (RemoteSessionState.DRAINING, RemoteSessionState.STUCK)
            and not session.pending_send
            and not session.inflight_send
        )
        if not session.can_release and not can_fence_recv:
            return
        engine_id = session.engine_id
        # Remove mappings first so no later queued work can discover a dlist.
        self._remote_agents_per_layer.pop(engine_id, None)
        self._remote_agents_per_rank.pop(engine_id, None)
        failed_dlists = [
            handle
            for handle in self.dst_xfer_side_handles.get(engine_id, [])
            if not self._release_dlist_handle(handle)
        ]
        if failed_dlists:
            self.dst_xfer_side_handles[engine_id] = failed_dlists
            session.state = RemoteSessionState.STUCK
            logger.error(
                "Remote engine %s generation %s cleanup is STUCK: %d remote "
                "dlist handle(s) could not be released",
                engine_id,
                session.generation,
                len(failed_dlists),
            )
            return
        self.dst_xfer_side_handles.pop(engine_id, None)

        failed_agents = [
            agent_name
            for agent_name in self._remote_agents.get(engine_id, [])
            if not self._remove_remote_agent(agent_name)
        ]
        if failed_agents:
            self._remote_agents[engine_id] = failed_agents
            session.state = RemoteSessionState.STUCK
            logger.error(
                "Remote engine %s generation %s cleanup is STUCK: %d remote "
                "agent(s) could not be removed",
                engine_id,
                session.generation,
                len(failed_agents),
            )
            return
        self._remote_agents.pop(engine_id, None)
        self._tp_size.pop(engine_id, None)
        self._pp_size.pop(engine_id, None)
        self._block_size.pop(engine_id, None)
        if can_fence_recv:
            # Native peer teardown is the proof that the failed producer can no
            # longer issue a late WRITE. Only now may D release failed receives.
            for req_id in list(session.pending_recv):
                self._fail_recv(
                    req_id,
                    "remote producer disconnected after native teardown",
                )
            if session.request_count:
                session.state = RemoteSessionState.STUCK
                logger.error(
                    "Remote producer %s generation %s native teardown "
                    "completed but %d ownership(s) could not be released",
                    engine_id,
                    session.generation,
                    session.request_count,
                )
                return
        session.state = RemoteSessionState.CLOSED
        logger.info(
            "Closed remote engine %s generation %s",
            engine_id,
            session.generation,
        )
        self._report_teardown(session, "closed")

    def _new_connect_session(self, engine_id: EngineId) -> RemoteEngineSession | None:
        current = self.remote_sessions.get(engine_id)
        if current is not None and current.state == RemoteSessionState.CONNECTING:
            # Producer request metadata may arrive before the D-side connect.
            return current
        if current is not None and current.state != RemoteSessionState.CLOSED:
            logger.warning(
                "Rejecting connect for engine %s while generation %s is %s",
                engine_id,
                current.generation,
                current.state.name,
            )
            return None
        generation = self._engine_generations.get(engine_id, 0) + 1
        self._engine_generations[engine_id] = generation
        session = RemoteEngineSession(
            engine_id=engine_id,
            generation=generation,
            role=self._session_role,
        )
        self.remote_sessions[engine_id] = session
        return session

    def _note_transport_failure(
        self,
        engine_id: EngineId | None,
        error: BaseException,
        context: str,
        *,
        agent_scoped: bool = False,
    ) -> None:
        """Queue a session invalidation for a session-terminal status.

        ``agent_scoped`` marks a call that names only the remote agent, so an
        unknown-object status there proves the agent itself is gone rather than
        one descriptor of this request.
        """

        if engine_id is None or not _is_session_terminal_error(
            error, agent_scoped=agent_scoped
        ):
            return
        self._pending_transport_failures.setdefault(
            engine_id, f"{context}: {type(error).__name__}: {error}"
        )

    def _apply_transport_failures(self) -> None:
        while self._pending_transport_failures:
            engine_id, reason = self._pending_transport_failures.popitem()
            self._invalidate_remote_engine(engine_id, reason)

    def _invalidate_remote_engine(self, engine_id: EngineId, reason: str) -> None:
        """Stop feeding work to a session whose transport is terminal."""

        session = self.remote_sessions.get(engine_id)
        if session is not None and not session.accepts_work:
            # Already draining or closed; its requests fail through the
            # disconnect path and a rebuild is attempted when they are done.
            self._engines_needing_rehandshake.add(engine_id)
            return
        logger.error(
            "Invalidating NIXL session for remote engine %s: %s", engine_id, reason
        )
        self._engines_needing_rehandshake.add(engine_id)
        self._rehandshake_retry_at.pop(engine_id, None)
        self._report_reachability(engine_id, reachable=False)
        self._mark_engine_disconnected(engine_id)

    def _report_reachability(self, engine_id: EngineId, *, reachable: bool) -> None:
        """Publish this rank's view of a remote engine to its own scheduler."""

        if self.is_consumer:
            # Only a producer's scheduler answers a peer's ownership commands.
            return
        if self._reported_reachability.get(engine_id) == reachable:
            return
        self._reported_reachability[engine_id] = reachable
        self._reachability_outbox.put_nowait((engine_id, reachable))

    def _note_handshake_failure(
        self, engine_id: EngineId, error: BaseException
    ) -> None:
        """Keep a failed connect recoverable instead of silently dead.

        D caches its peer on the first connect and never re-sends one, so a
        request arriving after this would otherwise sit behind a CONNECTING
        placeholder that nothing can ever activate.  Flagging the engine makes
        the next request either rebuild the session or fail fast with a reason.
        """

        self._log_failure(
            "handshake_failed",
            None,
            msg=f"remote engine {engine_id}",
            error=error,
        )
        self._report_reachability(engine_id, reachable=False)
        self._mark_engine_disconnected(engine_id)
        self._engines_needing_rehandshake.add(engine_id)
        failures = self._rehandshake_failures.get(engine_id, 0)
        self._rehandshake_failures[engine_id] = failures + 1
        # The handshake just failed; do not let the next request retry it
        # immediately.
        self._rehandshake_retry_at[engine_id] = (
            time.perf_counter() + self.rehandshake_retry_interval
        )

    def _retract_teardown_if_unsettled(
        self, session: RemoteEngineSession | None
    ) -> None:
        """Tell the scheduler a teardown this rank reported no longer holds.

        Re-activating an engine undoes a teardown already reported as closed.
        Re-reporting the same id overwrites this rank's ack, which turns a
        pending result into a failure so the Router retries instead of
        believing the pair is gone.  A teardown the scheduler already finished
        is out of reach; fencing that needs an epoch on connect itself.
        """

        if session is None or session.teardown_id is None:
            return
        if session.reported_teardown_id != session.teardown_id:
            return
        session.reported_teardown_id = None
        self._report_teardown(session, "reconnected")

    def _rehandshake_remote_engine(self, engine_id: EngineId) -> bool:
        """Rebuild an invalidated session from the cached connect metadata.

        D never re-sends connect for a session only this side knows is dead, so
        the rebuild has to start here.  Returns whether the session is ACTIVE
        again; the caller rejects the request when it is not.
        """

        node_meta = self._remote_node_metas.get(engine_id)
        if node_meta is None or not self._accepting_work:
            return False
        now = time.perf_counter()
        retry_at = self._rehandshake_retry_at.get(engine_id)
        if retry_at is not None and now < retry_at:
            return False
        session = self._new_connect_session(engine_id)
        if session is None:
            # Old generation still owns live handles.  No handshake ran, so no
            # cooldown is owed: retry as soon as it drains.
            return False
        # The handshake blocks this thread, so arm the cooldown around the
        # attempt itself and back off while the peer stays unreachable.
        failures = self._rehandshake_failures.get(engine_id, 0)
        self._rehandshake_retry_at[engine_id] = now + min(
            self.rehandshake_retry_interval * (2**failures),
            self.rehandshake_retry_interval * 8,
        )
        try:
            per_layer_infos, per_rank_info = self._nixl_handshake(node_meta)
        except Exception as e:
            self._rehandshake_failures[engine_id] = failures + 1
            self._log_failure(
                "rehandshake_failed",
                None,
                msg=f"remote engine {engine_id}",
                error=e,
            )
            self._mark_engine_disconnected(engine_id)
            self._engines_needing_rehandshake.add(engine_id)
            return False
        self._remote_agents_per_layer[engine_id] = per_layer_infos
        self._remote_agents_per_rank[engine_id] = per_rank_info
        session.activate()
        self._engines_needing_rehandshake.discard(engine_id)
        self._rehandshake_retry_at.pop(engine_id, None)
        self._rehandshake_failures.pop(engine_id, None)
        self._report_reachability(engine_id, reachable=True)
        logger.info(
            "Rebuilt NIXL session for remote engine %s generation %s",
            engine_id,
            session.generation,
        )
        return True

    def _mark_engine_disconnected(
        self, engine_id: EngineId, teardown_id: int | None = None
    ) -> None:
        now = time.perf_counter()
        session = self.remote_sessions.get(engine_id)
        if session is None:
            generation = self._engine_generations.get(engine_id, 0) + 1
            self._engine_generations[engine_id] = generation
            session = RemoteEngineSession(
                engine_id=engine_id,
                generation=generation,
                role=self._session_role,
            )
            self.remote_sessions[engine_id] = session
        if teardown_id is not None:
            session.teardown_id = teardown_id
            if session.state == RemoteSessionState.CLOSED:
                self._report_teardown(session, "closed")
                return
        timeout = (
            self.consumer_disconnect_grace
            if self.is_consumer
            else self.producer_disconnect_drain_timeout
        )
        session.start_draining(now, timeout)
        if not self.is_consumer:
            for req_id, wrapper in self._send_transfers.items():
                state = self._send_requests.get(req_id)
                if (
                    state is not None
                    and state.metadata is not None
                    and state.metadata.engine_id == engine_id
                ):
                    # Poll already-started writes until UCX reports terminal;
                    # never free an in-progress native request.
                    wrapper.ready_for_check = True
            pending = [
                req_id
                for req_id, state in self._send_requests.items()
                if state.metadata is not None
                and state.metadata.engine_id == engine_id
                and not state.started
            ]
            for req_id in pending:
                self._finish_send(req_id, "remote engine disconnected before transfer")
        self._cleanup_session_if_drained(session)

    async def _shutdown_on_loop(self) -> None:
        task = getattr(self, "_shutdown_task", None)
        if task is None or task.done():
            task = asyncio.create_task(self._drain_and_cleanup_on_loop())
            self._shutdown_task = task
        # Preserve the one live drain if the synchronous caller times out.
        await asyncio.shield(task)
        self.stop_event.set()
        current = asyncio.current_task()
        tasks = [item for item in self._loop_tasks if item is not current]
        for item in tasks:
            item.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._loop_exit_event.set()

    async def _drain_and_cleanup_on_loop(self) -> None:
        self._accepting_work = False
        for engine_id in list(self.remote_sessions):
            self._mark_engine_disconnected(engine_id)
        for req_id in list(self._send_requests):
            if req_id not in self._send_transfers:
                self._finish_send(req_id, "worker shutdown before transfer")

        deadline = time.perf_counter() + min(
            self.producer_disconnect_drain_timeout, 5.0
        )
        while True:
            self._poll_send_transfers(include_unready=True)
            # SDK generation-scoped teardown fences remote WRITEs before it
            # releases a failed receive. Never bypass that proof on shutdown.
            self._progress_sessions(time.perf_counter())
            if not self._send_transfers and not self._recv_requests:
                break
            if time.perf_counter() >= deadline:
                raise NixlDrainIncomplete("Native transfers still own registered pages")
            await asyncio.sleep(0.005)

        for session in self.remote_sessions.values():
            if session.state not in (
                RemoteSessionState.CLOSED,
                RemoteSessionState.DRAINING,
                RemoteSessionState.STUCK,
            ):
                session.start_draining(time.perf_counter(), 0.0)
            self._cleanup_session_if_drained(session)

        # Keep every failed native release in its registry so a retry does not
        # forget ownership. Cache tensors outlive all registration descriptors.
        for key, handle in list(self.src_xfer_side_handles.items()):
            if self._release_dlist_handle(handle):
                del self.src_xfer_side_handles[key]
        for key, handles in list(self.dst_xfer_side_handles.items()):
            remaining = [handle for handle in handles if not self._release_dlist_handle(handle)]
            if remaining:
                self.dst_xfer_side_handles[key] = remaining
            else:
                del self.dst_xfer_side_handles[key]
        for key, agents in list(self._remote_agents.items()):
            remaining = [agent for agent in agents if not self._remove_remote_agent(agent)]
            if remaining:
                self._remote_agents[key] = remaining
            else:
                del self._remote_agents[key]
        if self.src_xfer_side_handles or self.dst_xfer_side_handles or self._remote_agents:
            raise NixlDrainIncomplete("Native descriptors or peers remain registered")
        self._registered_descs = [
            desc for desc in self._registered_descs if not self._deregister_memory(desc)
        ]
        if self._registered_descs:
            raise NixlDrainIncomplete("Native memory deregistration is incomplete")
        reports = getattr(self, "_teardown_outbox", None)
        if reports is not None:
            try:
                await asyncio.wait_for(reports.join(), timeout=5.0)
            except TimeoutError as error:
                raise NixlDrainIncomplete("Teardown reports are still pending") from error
        getattr(self, "_registered_cache_owners", {}).clear()

    ############################################################
    # async loop for recv info from scheduler, run kv cache sending...
    ############################################################
    def connecter_loop(self):
        asyncio.set_event_loop(self._async_loop)
        torch.cuda.set_device(torch.distributed.get_rank())

        async def tasks():
            coroutines = [
                self._scheduler_listener(),
                self._run_progress(),
                self._run_write_kv_cache(),
                self._run_teardown_reports(),
                self._run_reachability_reports(),
            ]
            if host_kv_enabled():
                coroutines.append(self._run_d2h())
            self._loop_tasks = {
                asyncio.create_task(coroutine) for coroutine in coroutines
            }
            await self._loop_exit_event.wait()
            await asyncio.gather(*self._loop_tasks, return_exceptions=True)

        try:
            self._async_loop.run_until_complete(tasks())
        except Exception:
            logger.fatal("An exception occurred in the connecter background thread")
            traceback.print_exc()

    async def _scheduler_listener(self) -> None:
        scheduler_addr = self.scheduler_addr
        with make_zmq_socket(
            path=scheduler_addr, ctx=self.ctx, socket_type=zmq.XSUB, linger=0
        ) as recv_from_scheduler:
            # subscribe to scheduler
            await recv_from_scheduler.send(b"\x01")
            if await recv_from_scheduler.recv() != b"READY":
                raise RuntimeError(
                    "NIXL worker listener failed to connect to scheduler"
                )
            recv_from_scheduler.setsockopt(zmq.RCVTIMEO, 1000)

            while True:
                try:
                    msg = await recv_from_scheduler.recv()
                except zmq.Again:
                    if self.stop_event.is_set():
                        break
                    continue
                try:
                    op_type, remote_infos = msgspec.msgpack.decode(msg)
                    if op_type == "block_ids":
                        (
                            req_id,
                            generation,
                            intent_text,
                            block_ids,
                            remote_engine_id,
                        ) = remote_infos[:5]
                        req_id = str(req_id)
                        generation = str(generation)
                        block_starts: list[int] | None = (
                            list(remote_infos[5]) if len(remote_infos) > 5 else None
                        )
                        intent = TransferIntent(intent_text)
                        logger.info(
                            "NIXL recv block_ids req=%s generation=%s "
                            "intent=%s groups=%s "
                            "starts=%s from=%s session=%s",
                            req_id,
                            generation,
                            intent.value,
                            [len(x) for x in block_ids],
                            block_starts,
                            remote_engine_id,
                            remote_engine_id in self.remote_sessions,
                        )
                        if (
                            not self.is_consumer
                            and remote_engine_id
                            in self._engines_needing_rehandshake
                        ):
                            # block_ids normally arrives before this request's
                            # local prefill ends, so rebuilding here lets the
                            # first request after a transport failure still
                            # transfer instead of being rejected.
                            self._rehandshake_remote_engine(remote_engine_id)
                        # D-side block_ids and local prefill metadata travel on
                        # independent paths. Preserve unmatched block_ids in a
                        # placeholder until metadata and handshake also arrive.
                        remote_block_ids = [
                            np.array(block_id, dtype=np.int32)[None, :]
                            for block_id in block_ids
                        ]
                        state = self._send_state_for_generation(
                            req_id, generation
                        )
                        if state is None:
                            continue
                        fingerprint = msgspec.msgpack.encode(
                            (
                                intent.value,
                                block_ids,
                                remote_engine_id,
                                block_starts,
                            )
                        )
                        if (
                            state.command_fingerprint is not None
                            and state.command_fingerprint != fingerprint
                        ):
                            self._cancel_send_request(
                                req_id,
                                generation,
                                "conflicting duplicate PREPARED command",
                            )
                            continue
                        state.command_fingerprint = fingerprint
                        state.transfer_intent = intent.value
                        state.remote_block_ids = remote_block_ids
                        state.remote_block_starts = block_starts
                        state.session = self.remote_sessions.get(remote_engine_id)
                        if (
                            state.ownership_state
                            == RequestOwnershipState.PREPARED
                        ):
                            state.ownership_state = (
                                RequestOwnershipState.ACCEPTED
                            )
                        if state.deadline is None:
                            state.deadline = (
                                time.perf_counter() + self._kv_lease_duration
                            )
                        if intent == TransferIntent.CANCEL:
                            self._cancel_send_request(
                                req_id,
                                generation,
                                "consumer canceled before allocation",
                            )
                        else:
                            self._maybe_start_or_continue_send(state)
                    elif op_type == OWNERSHIP_CANCEL_MSG:
                        req_id, generation, _remote_engine_id, reason = remote_infos
                        self._cancel_send_request(
                            str(req_id), str(generation), str(reason)
                        )
                    elif op_type == "disconnect":
                        if isinstance(remote_infos, (list, tuple)):
                            remote_engine_id = remote_infos[0]
                            teardown_id = int(remote_infos[1])
                        else:
                            remote_engine_id = remote_infos
                            teardown_id = None
                        # A router teardown is authoritative and gets ACKed, so
                        # drop the cached connect metadata: a queued request
                        # must not silently rebuild a session the router has
                        # been told is gone.  Only a fresh connect re-enables it.
                        self._remote_node_metas.pop(remote_engine_id, None)
                        self._engines_needing_rehandshake.discard(remote_engine_id)
                        self._rehandshake_retry_at.pop(remote_engine_id, None)
                        self._rehandshake_failures.pop(remote_engine_id, None)
                        self._mark_engine_disconnected(
                            remote_engine_id, teardown_id
                        )
                    elif op_type == "connect":
                        node_meta = NodeMeta(**remote_infos)
                        if not self._accepting_work:
                            logger.warning(
                                "Dropping connect for %s during shutdown",
                                node_meta.engine_id,
                            )
                            continue
                        # Kept so a session invalidated by a terminal transport
                        # status can be rebuilt without another connect.
                        self._remote_node_metas[node_meta.engine_id] = node_meta
                        previous = self.remote_sessions.get(node_meta.engine_id)
                        session = self._new_connect_session(node_meta.engine_id)
                        if session is not None:
                            try:
                                # Handshake is intentionally synchronous and
                                # rare. Blocking here guarantees this XSUB
                                # listener does not consume the following
                                # block_ids message before all NIXL dlists and
                                # transfer mappings are ready.
                                per_layer_infos, per_rank_info = self._nixl_handshake(
                                    node_meta
                                )
                                self._remote_agents_per_layer[node_meta.engine_id] = (
                                    per_layer_infos
                                )
                                self._remote_agents_per_rank[node_meta.engine_id] = (
                                    per_rank_info
                                )
                                session.activate()
                                if previous is not session:
                                    self._retract_teardown_if_unsettled(previous)
                                self._engines_needing_rehandshake.discard(
                                    node_meta.engine_id
                                )
                                self._rehandshake_retry_at.pop(
                                    node_meta.engine_id, None
                                )
                                self._rehandshake_failures.pop(
                                    node_meta.engine_id, None
                                )
                                self._report_reachability(
                                    node_meta.engine_id, reachable=True
                                )
                                for req_id, state in list(self._send_requests.items()):
                                    if (
                                        state.metadata is not None
                                        and state.metadata.engine_id
                                        == node_meta.engine_id
                                    ):
                                        # connect is sent before block_ids, so
                                        # binding the ACTIVE session is enough;
                                        # the later block/layer event performs
                                        # the readiness check.
                                        state.session = session
                                logger.info(
                                    "Connected remote engine %s generation %s",
                                    node_meta.engine_id,
                                    session.generation,
                                )
                            except Exception as e:
                                self._note_handshake_failure(
                                    node_meta.engine_id, e
                                )
                    else:
                        logger.info(
                            "get a unsupported op: %s, supported ops: block_ids, "
                            "connect, disconnect, check_helper",
                            op_type,
                        )
                except Exception as e:
                    logger.info("error on fun _scheduler_listener: %s", e)
                    traceback.print_exc()

    async def _run_write_kv_cache(self):
        while True:
            batch = await self._send_ready_queue.get()
            while not batch.cuda_event.query():
                await asyncio.sleep(0.0005)

            for req_id, meta in batch.requests.items():
                state = self._send_state_for_generation(
                    req_id, meta.generation
                )
                if state is None:
                    continue
                state.ready_layers.append((batch.layer_index, meta))
                self._maybe_start_or_continue_send(state)
