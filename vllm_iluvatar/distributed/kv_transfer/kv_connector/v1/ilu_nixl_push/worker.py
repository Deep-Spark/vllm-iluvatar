# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side logic for the NIXL connector."""

import asyncio
import copy
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
from torch.cuda import Event
from vllm import envs
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import CopyBlocksOp
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    MambaConvSplitInfo,
    derive_mamba_conv_split,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import select_common_block_size

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_push.metadata import (
    GET_META_MSG,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    NodeMeta,
    ReqId,
    ReqMeta,
    compute_nixl_compatibility_hash,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_push.stats import (
    NixlKVConnectorStats,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_push.utils import (
    _NIXL_SUPPORTED_DEVICE,
    D2HManger,
    NixlTpKVTopology,
    TransferStatusChecker,
    TransferWrapper,
    find_number_on_layer_name,
    make_zmq_socket,
    zmq_comm2proxy,
    zmq_ctx,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

from nixl._api import nixl_agent as NixlWrapper
from nixl._api import nixl_agent_config, nixl_prepped_dlist_handle

logger = init_logger(__name__)


class NixlConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        engine_id: str,
    ):
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
        if self.device_type not in _NIXL_SUPPORTED_DEVICE:
            raise RuntimeError(f"{self.device_type} is not supported.")
        elif self.kv_buffer_device not in _NIXL_SUPPORTED_DEVICE[self.device_type]:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )

        if self.device_type == "cpu":
            assert False, "iluvatar nixl only support cuda device"

        # support for oot platform which can't register nixl memory
        # type based on kv_buffer_device
        nixl_memory_type = current_platform.get_nixl_memory_type()
        if nixl_memory_type is None:
            if self.kv_buffer_device in ["cuda"]:
                nixl_memory_type = "VRAM"
            else:
                raise RuntimeError(
                    f"kv_buffer_device with {self.kv_buffer_device} kv_buffer "
                    "is not supported."
                )
        if nixl_memory_type is None:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )
        self.nixl_memory_type = nixl_memory_type
        self.use_host_buffer = envs.VLLM_ILU_NIXL_WRITE_USE_HOST_BUFFER
        if self.use_host_buffer:
            logger.warning(
                "KV Cache use Cuda memory, but VLLM_ILU_NIXL_WRITE_USE_HOST_BUFFER is set, we will create a host tensor same as "
                "need registered kv cache, and use D2H + Host to remote Cuda RDMA transferring"
            )
            self.nixl_memory_type = "DRAM"

        self.nixl_wrapper = nixl_wrapper_cls(str(uuid.uuid4()), config)
        # Map of engine_id -> [remote_agent0, remote_agent1, ...]
        self._remote_agents: dict[EngineId, list[str]] = {}
        # Map of block_size -> {key -> local nixl_prepped_dlist_handle}, for now only one block size
        self.src_xfer_side_handles: dict[
            int, dict[tuple[int], nixl_prepped_dlist_handle]
        ] = {}
        # Map of EngineId -> [remote nixl_prepped_dlist_handle]
        self.dst_xfer_side_handles: dict[EngineId, list[nixl_prepped_dlist_handle]] = (
            defaultdict(list)
        )
        # Map of EngineId -> [info for write kv cache], size equals the num layers of this rank
        # infO: local_descs, remote_descs, notify_msg, local_blocks, remote_blocks, write, group_index
        self._remote_agents_per_layer: dict[
            EngineId,
            list[
                tuple[
                    nixl_prepped_dlist_handle,
                    nixl_prepped_dlist_handle,
                    str,
                    str,
                    np.ndarray,
                    np.ndarray,
                    bool,
                    int,
                    bool,
                ]
            ],
        ] = defaultdict(list)
        # Map of EngineId -> [info for write kv cache], size equals the num kv group of this model
        self._remote_agents_per_rank: dict[
            EngineId,
            list[
                tuple[
                    nixl_prepped_dlist_handle,
                    nixl_prepped_dlist_handle,
                    str,
                    str,
                    np.ndarray,
                    np.ndarray,
                    bool,
                    int,
                    bool,
                ]
            ],
        ] = defaultdict(list)
        # Map of engine_id -> num_blocks. All ranks in the same deployment will have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}
        # all local registered data
        self._registered_descs: list[Any] = []
        self.xfer_handshake_metadata: NixlAgentMetadata | None = None
        # infos about registered kv cache
        self.layer_ptr_mapping = defaultdict(list[int])
        self.layer_desc_mapping = defaultdict(list[tuple[int, int]])
        self.layer_desc_num_mapping = defaultdict(list[tuple[int, int]])
        # device kv caches
        self.device_kv_caches: dict[str, torch.Tensor] = {}

        # ---- hybrid kv ----
        # [layer_index_local, layer_index_global, layer_name, kv_spec]
        self.kv_group_infos: list[list[tuple[int, int, str, KVCacheSpec]]] = []
        self.kv_group_infos_for_check: list[str] = []
        self._layer_specs: dict[str, KVCacheSpec] = {}
        self._all_specs = set()
        self.num_groups = len(kv_cache_config.kv_cache_groups)

        for group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
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
                raise RuntimeError("Create NixlConnectorScheduler error!")
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

        logger.debug("kv_group_infos : %s", self.kv_group_infos)
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

        self.remap_to_kernel_size = []
        for g in kv_cache_config.kv_cache_groups:
            if isinstance(g.kv_cache_spec, MambaSpec):
                self.remap_to_kernel_size.append(0)
            else:
                self.remap_to_kernel_size.append(1)

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
        self.global_rank = self.pp_rank * self.tp_world_size + self.tp_rank
        self.enable_permute_local_kv = False
        # self.device_id_calculated = (self.pp_rank * self.tp_world_size + self.tp_rank) if self.use_host_buffer else 0
        self.device_id: int = 0
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.compat_hash = None

        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.use_mla = self.model_config.use_mla
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(vllm_config, layer_type, None)
        if layers:
            for backend_cls in layers.values():
                backend = backend_cls.get_attn_backend()
                try:
                    backend.get_kv_cache_shape(
                        num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
                    )
                    self.attn_backend = backend
                except:
                    pass
        else:
            raise RuntimeError("can not find an appropriate attention backend ")

        self.backend_name = self.attn_backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()
        self.host_buffer_kv_cache_layout = self.kv_cache_layout
        self._physical_blocks_per_logical_kv_block = 1
        self._sync_block_size_with_kernel()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)
        self.transfer_topo: NixlTpKVTopology | None = None

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_world_size}
        self._pp_size: dict[EngineId, int] = {self.engine_id: self.pp_world_size}
        self._block_size: dict[EngineId, int] = {self.engine_id: self.block_size}
        self._block_size_ratio: dict[EngineId, int] = {}

        self.num_hidden_layers = (
            vllm_config.model_config.hf_text_config.num_hidden_layers
            if vllm_config.model_config.hf_text_config is not None
            else vllm_config.model_config.hf_config.num_hidden_layers
        )
        self.pp_layer_start, self.pp_layer_end = get_pp_indices(
            self.num_hidden_layers, self.pp_rank, self.pp_world_size
        )
        self.pp_layer_end_closed = self.pp_layer_end - 1
        self.pp_split = [
            get_pp_indices(self.num_hidden_layers, i, self.pp_world_size)[0]
            for i in range(self.pp_world_size)
        ]
        self.num_layers = self.pp_layer_end - self.pp_layer_start
        self.num_layers_end_index = self.num_layers - 1
        self.total_num_kv_heads = self.model_config.get_total_num_kv_heads()

        # ---- transferring ----
        self.copy_blocks: CopyBlocksOp | None = None
        self.kv_lease_duration: int = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_lease_duration", 480
            )
        )
        self.net_info_lock = threading.Lock()
        self._connector_metadata = None

        # thread and related resources
        self.finished_send_reqs = set()
        self.ctx = zmq.asyncio.Context()
        self._async_loop = uvloop.new_event_loop()
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread = threading.Thread(
            target=self.connecter_loop,
            daemon=True,
            name="nixl_connector_worker_loop",
        )
        self.worker_thread.start()
        self.save_kv_layer = (
            self.save_kv_layer_cuda
            if not self.use_host_buffer
            else self.save_kv_layer_host
        )

        # recv usesd
        self._reqs_to_recv: dict[str, ReqMeta] = {}
        self._reqs_ready_to_recv_time_record: dict[ReqId, float] = {}
        self._reqs_to_abort_recv: set[str] = set()
        self._invalid_block_ids: set[int] = set()
        # helper class to make sure this seqs is recv done
        self.notification_class = TransferStatusChecker
        self.notification_class.init_pp_start = self.pp_layer_start
        self.notification_class.init_pp_end = self.pp_layer_end
        self.consumer_notification_counts_by_req: dict[
            ReqId, TransferStatusChecker
        ] = {}
        self.use_async_scheduling = vllm_config.scheduler_config.async_scheduling
        self.pre_recvs: set[str] = set()
        self.pre_recvs_to_this_step: set[str] = set()

        # send used
        self._reqs_to_send_record: dict[ReqId, ReqMeta] = {}
        self._reqs_to_send_layerwise: dict[ReqId, ReqMeta] = {}
        self._reqs_to_send_per_rank: dict[ReqId, ReqMeta] = {}
        self._sending_transfers: dict[ReqId, TransferWrapper] = {}
        # the req that model forward paas to, will be wait then put to _reqs_ready_to_send
        self._reqs_to_send: asyncio.Queue[tuple[dict[ReqId, ReqMeta], int, Event]] = (
            asyncio.Queue() if not self.use_host_buffer else None
        )
        self._reqs_to_send_d2h: asyncio.Queue[
            tuple[dict[ReqId, ReqMeta], int, Event]
        ] = asyncio.Queue() if self.use_host_buffer else None
        self._reqs_to_send_host: asyncio.Queue[
            tuple[dict[ReqId, ReqMeta], int, Event]
        ] = asyncio.Queue() if self.use_host_buffer else None
        # the req that can be process(in _reqs_to_blocks(from other node) and _reqs_ready_to_send).
        self._reqs_to_blocks: dict[ReqId, list[list[np.ndarray], bool]] = defaultdict(
            list
        )
        self._reqs_ready_to_send: dict[ReqId, list[tuple[int, ReqMeta]]] = defaultdict(
            list
        )
        self._reqs_ready_to_send_time_record: dict[ReqId, float] = {}
        # send status
        self.xfer_stats = NixlKVConnectorStats()
        # int helper to per-layer transferring
        self.layer_index = self.pp_layer_start
        self.un_send_layer_nums = self.num_layers
        # disconnected
        self._kv_lease_duration: int = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_lease_duration", 480
            )
        )
        self._reqs_disconnected_need_check: set[str] = set()
        self._reqs_check_done_after_disconnected: set[str] = set()
        self._engines_check_done_after_disconnected: set[str] = set()
        self._reqs_to_checked: asyncio.Queue[dict[str, set[str]]] = asyncio.Queue()
        self.disconnected_engine: dict[str, float] = {}

        # ---- extra host buffer ----
        self.use_quant_to_fp8 = envs.VLLM_ILU_NIXL_DS32_QUANT_AFTER_TRANSFER
        self.host_xfer_buffers: dict[str, torch.Tensor] = {}
        if self.use_host_buffer:
            assert self.use_mla and len(self._all_specs) == 1, (
                "host buffer only support mla only model for now."
            )
            self.index_head_dim = vllm_config.model_config.hf_config.index_head_dim
            self.new_allocated_indexer_tensors: list[torch.Tensor] = []
            self.origin_indexer_tensors: list[torch.Tensor] = []
            self.indexer_tensor_ptrs: torch.Tensor = None
            self.quant_indexer_tensor_ptrs: torch.Tensor = None
            self.device_for_ops = None
            self.h2d_manager = None
            self.mla_shape = [
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.model_config.hf_config.kv_lora_rank
                + vllm_config.model_config.hf_config.qk_rope_head_dim,
            ]
            self.mla_indexer_shape = [
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.model_config.hf_config.index_head_dim,
            ]
            self.host_dtype = {
                "auto": torch.bfloat16,
                "half": torch.half,
                "bf16": torch.bfloat16,
                "int8": torch.int8,
                "fp8": torch.int8,
            }[envs.VLLM_ILU_NIXL_HOST_DTYPE]

        assert not all([self.use_host_buffer, self.use_quant_to_fp8]), (
            "use host buffer for P and use quant to fp8 for D"
        )

    def _sync_block_size_with_kernel(self) -> None:
        kernel_block_size = select_common_block_size(
            self.block_size,
            [
                self.attn_backend,
            ],
        )

        # Number of blocks not accounting for kernel block mismatches
        self._logical_num_blocks = self.num_blocks
        if self.block_size != kernel_block_size:
            logger.info_once(
                "User-specified logical block size (%s) does not match"
                " physical kernel block size (%s). Using the latter.",
                self.block_size,
                kernel_block_size,
            )
            assert self.block_size > kernel_block_size
            self._physical_blocks_per_logical_kv_block = (
                self.block_size // kernel_block_size
            )
            self.block_size = kernel_block_size
            self.num_blocks *= self._physical_blocks_per_logical_kv_block

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Assign copy (d2h, h2d) operations when host buffer is used."""
        return

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
    # Add remote and prepare resources for sending kv cache
    ############################################################
    def register_local_xfer_handler(
        self,
        local_ptr: list[int],
        local_desc: list[list[tuple[int, int]]],
        local_block_lens: list[int],
        local_block_num_ratios: list[int],
        layer_desc: list[KVCacheSpec],
    ) -> nixl_prepped_dlist_handle:
        """
        Function used for register local xfer handler with local block_size or
        Remote block_size.

        When local block_size is same as remote block_size, we use local block_size
        to register local_xfer_handler during init.

        When remote block size is less than local block size, we need to use
        register another local_xfer_handler using remote block len to ensure
        data copy correctness.
        """
        blocks_data = []
        for l_index, l_desc_per_ptr, l_block_len, ratio, kv_spec in zip(
            local_ptr, local_desc, local_block_lens, local_block_num_ratios, layer_desc
        ):
            base_addr = self.seen_base_addresses[l_index]
            num_blocks = (
                self.num_blocks * ratio
                if not isinstance(kv_spec, MambaSpec)
                else self._logical_num_blocks
            )

            for block_id in range(num_blocks):
                block_offset = block_id * l_block_len
                # a ptr may register multi xfer
                for offset, lens in l_desc_per_ptr:
                    addr = base_addr + block_offset + offset
                    # (addr, len, device id)
                    blocks_data.append((addr, lens, self.device_id))

        logger.debug(
            "Created %s blocks for src engine %s and rank %s on device id %s",
            len(blocks_data),
            self.engine_id,
            self.global_rank,
            self.device_id,
        )

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        # NIXL_INIT_AGENT to be used for preparations of local descs.
        return self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)

    def _validate_remote_agent_handshake(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int,
        remote_tp_size: int,
        remote_pp_size: int,
        dp_rank: int,
        local_desc: list[list[tuple[int, int]]],
        remote_desc: list[list[tuple[int, int]]],
        local_block_len: list[int],
        remote_block_len: list[int],
        layer_desc: list[KVCacheSpec],
    ):
        """
        Validate the remote agent handshake metadata ensuring the
        invariants hold true.
        """
        remote_engine_id = nixl_agent_meta.engine_id

        assert self._tp_size[remote_engine_id] == remote_tp_size
        assert self._pp_size[remote_engine_id] == remote_pp_size

        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        local_block_bigger, block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )

        kv_cache_layout = self.kv_cache_layout
        assert kv_cache_layout == nixl_agent_meta.kv_cache_layout, (
            "only support P/D kv cache layout are the same for now"
        )

        is_local_kv_replicated = self.tp_world_size >= self.total_num_kv_heads
        is_remote_kv_replicated = remote_tp_size >= self.total_num_kv_heads

        write = True
        # for some cases, some local tp do not write kv to remote, use resize_size to adjust the tp notify msg
        tp_msg_resize = 1
        remote_tp_notify_msg = "{}-{}"
        new_local_desc: list[list[tuple[int, int]]] = [None] * len(local_desc)
        new_local_block_len: list[int] = [None] * len(local_desc)
        new_remote_desc: list[list[tuple[int, int]]] = [None] * len(local_desc)
        new_remote_block_len: list[int] = [None] * len(local_desc)
        local_block_num_ratios: list[int] = [1] * len(local_desc)
        remote_block_num_ratios: list[int] = [1] * len(local_desc)

        for index, (
            l_desc,
            r_desc,
            local_block_len,
            remote_block_len,
            kv_spec,
        ) in enumerate(
            zip(local_desc, remote_desc, local_block_len, remote_block_len, layer_desc)
        ):
            assert len(l_desc) == len(r_desc) and len(l_desc) > 0
            # for kv:       num_block, num_kv_head, block_size, head_dim
            # for scales:   num_block, num_kv_head, block_size
            # so the validate codes are the same for them.
            new_l_desc: list[tuple[int, int]] = []
            new_r_desc: list[tuple[int, int]] = []
            if (
                isinstance(kv_spec, MLAAttentionSpec)
                or isinstance(kv_spec, SlidingWindowMLASpec)
                or (is_local_kv_replicated and is_remote_kv_replicated)
            ):
                if local_block_bigger:
                    assert local_block_len // block_size_ratio == remote_block_len, (
                        "KV cache sizes must match between P and D when replicated kv head or MLA model"
                    )
                    new_local_block_len[index] = local_block_len // block_size_ratio
                    new_remote_block_len[index] = remote_block_len
                    local_block_num_ratios[index] = block_size_ratio
                    for l_desc_per_ptr, r_desc_per_ptr in zip(l_desc, r_desc):
                        l_start, l_len = l_desc_per_ptr
                        r_start, r_len = r_desc_per_ptr
                        assert l_start == 0 and r_start == 0
                        assert l_len // block_size_ratio == r_len
                        new_l_desc.append((l_start, l_len // block_size_ratio))
                        new_r_desc.append((r_start, r_len))
                else:
                    assert local_block_len == remote_block_len // block_size_ratio, (
                        "KV cache sizes must match between P and D when replicated kv head or MLA model"
                    )
                    new_local_block_len[index] = local_block_len
                    new_remote_block_len[index] = remote_block_len // block_size_ratio
                    remote_block_num_ratios[index] = block_size_ratio
                    for l_desc_per_ptr, r_desc_per_ptr in zip(l_desc, r_desc):
                        l_start, l_len = l_desc_per_ptr
                        r_start, r_len = r_desc_per_ptr
                        assert l_start == 0 and r_start == 0
                        assert l_len == r_len // block_size_ratio
                        new_l_desc.append((l_start, l_len))
                        new_r_desc.append((r_start, r_len // block_size_ratio))
                if self.tp_world_size >= remote_tp_size:
                    write = (self.tp_rank + dp_rank) % tp_ratio == 0
                    if write:
                        remote_tp_notify_msg = remote_tp_notify_msg.format(
                            self.tp_rank, self.tp_rank + 1
                        )
                else:
                    remote_tp_notify_msg = remote_tp_notify_msg.format(
                        self.tp_rank, self.tp_rank + 1
                    )
            else:
                assert block_size_ratio == 1
                if self.tp_world_size >= remote_tp_size:
                    assert not is_remote_kv_replicated
                    local_repeat_times = max(
                        self.tp_world_size // self.total_num_kv_heads, 1
                    )
                    new_local_block_len[index] = local_block_len
                    new_remote_block_len[index] = remote_block_len
                    if not is_local_kv_replicated:
                        for l_desc_per_ptr, r_desc_per_ptr in zip(l_desc, r_desc):
                            l_start, l_len = l_desc_per_ptr
                            r_start, r_len = r_desc_per_ptr
                            assert l_len == r_len // tp_ratio
                            assert l_start == r_start // tp_ratio
                            new_l_desc.append((l_start, l_len))
                            new_r_desc.append(
                                (r_start + (self.tp_rank % tp_ratio) * l_len, l_len)
                            )
                        remote_tp_notify_msg = remote_tp_notify_msg.format(
                            remote_tp_rank * tp_ratio, (remote_tp_rank + 1) * tp_ratio
                        )
                    else:
                        # Local kv is replicated, rare happen
                        write = (self.tp_rank + dp_rank) % local_repeat_times == 0
                        if write:
                            for l_desc_per_ptr, r_desc_per_ptr in zip(l_desc, r_desc):
                                l_start, l_len = l_desc_per_ptr
                                r_start, r_len = r_desc_per_ptr
                                assert l_len == r_len // (
                                    tp_ratio // local_repeat_times
                                )
                                assert l_start == r_start // (
                                    tp_ratio // local_repeat_times
                                )
                                new_l_desc.append((l_start, l_len))
                                new_r_desc.append(
                                    (
                                        r_start
                                        + (
                                            self.tp_rank
                                            // local_repeat_times
                                            % (tp_ratio // local_repeat_times)
                                        )
                                        * l_len,
                                        l_len,
                                    )
                                )
                            # some additional logice here, because only some local tp will write date to remote
                            tp_msg_resize = local_repeat_times
                            remote_tp_notify_msg = remote_tp_notify_msg.format(
                                remote_tp_rank * (tp_ratio // local_repeat_times),
                                (remote_tp_rank + 1) * (tp_ratio // local_repeat_times),
                            )
                        else:
                            pass
                    assert remote_block_len == (
                        local_block_len * (tp_ratio // local_repeat_times)
                    ), (
                        "Remote P worker KV layer cache must be of shape [2, N, "
                        "local_kv_heads*tp_ratio, block_size, head_dim] and same dtype."
                    )
                else:
                    assert not is_local_kv_replicated
                    remote_repeat_times = max(
                        remote_tp_size // self.total_num_kv_heads, 1
                    )
                    new_remote_block_len[index] = remote_block_len
                    new_local_block_len[index] = local_block_len
                    for l_desc_per_ptr, r_desc_per_ptr in zip(l_desc, r_desc):
                        l_start, l_len = l_desc_per_ptr
                        r_start, r_len = r_desc_per_ptr
                        assert l_len // tp_ratio == r_len
                        assert l_start // tp_ratio == r_start
                        new_r_desc.append((r_start, r_len))
                        if not is_remote_kv_replicated:
                            new_l_desc.append(
                                (l_start + (remote_tp_rank % tp_ratio) * r_len, r_len)
                            )
                        else:
                            new_l_desc.append(
                                (
                                    l_start
                                    + (
                                        remote_tp_rank
                                        // remote_repeat_times
                                        % (tp_ratio // remote_repeat_times)
                                    )
                                    * r_len,
                                    r_len,
                                )
                            )

                    remote_tp_notify_msg = remote_tp_notify_msg.format(
                        self.tp_rank, self.tp_rank + 1
                    )
                    assert remote_block_len == (
                        local_block_len // (tp_ratio // remote_repeat_times)
                    ), (
                        "Remote P worker KV layer cache must be of shape [2, N, "
                        "local_kv_heads//tp_ratio, block_size, head_dim] and same dtype."
                    )

            new_local_desc[index] = new_l_desc
            new_remote_desc[index] = new_r_desc

        # TP workers have same #blocks.
        assert self.dst_num_blocks[remote_engine_id] == nixl_agent_meta.num_blocks

        return (
            remote_tp_notify_msg,
            tp_msg_resize,
            write,
            new_local_desc,
            new_local_block_len,
            local_block_num_ratios,
            new_remote_desc,
            new_remote_block_len,
            remote_block_num_ratios,
        )

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        local_ptr: list[int],
        remote_ptr: list[int],
        local_desc: list[list[tuple[int, int]]],
        remote_desc: list[list[tuple[int, int]]],
        local_block_len: list[int],
        remote_block_len: list[int],
        layer_desc: list[KVCacheSpec],
        remote_tp_rank: int,
        remote_tp_size: int,
        remote_pp_rank: int,
        remote_pp_size: int,
        remote_dp_rank: int,
        group_id: int,
        remote_agent_name,
    ) -> tuple[Any]:
        assert nixl_agent_meta.tp_rank == remote_tp_rank
        assert nixl_agent_meta.pp_rank == remote_pp_rank

        engine_id = nixl_agent_meta.engine_id
        # Register remote agent metadata
        if engine_id not in self._tp_size:
            self._tp_size[engine_id] = remote_tp_size
            self._pp_size[engine_id] = remote_pp_size
            self._block_size[engine_id] = nixl_agent_meta.block_size
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

            local_block_bigger, block_size_ratio = self.transfer_topo.block_size_ratio(
                nixl_agent_meta.block_size
            )
            if local_block_bigger:
                self._block_size_ratio[engine_id] = block_size_ratio
            else:
                self._block_size_ratio[engine_id] = 1

        (
            remote_tp_notify_msg,
            tp_msg_resize,
            write,
            new_local_desc,
            new_local_block_len,
            local_block_num_ratios,
            new_remote_desc,
            new_remote_block_len,
            remote_block_num_ratios,
        ) = self._validate_remote_agent_handshake(
            nixl_agent_meta,
            remote_tp_rank,
            remote_tp_size,
            remote_pp_size,
            remote_dp_rank,
            local_desc,
            remote_desc,
            local_block_len,
            remote_block_len,
            layer_desc,
        )
        if not write:
            return None, None, None, False, None, None
        if remote_agent_name is None:
            remote_agent_name = self.nixl_wrapper.add_remote_agent(
                nixl_agent_meta.agent_metadata
            )
            with self.net_info_lock:
                if engine_id not in self._remote_agents:
                    self._remote_agents[engine_id] = [remote_agent_name]
                else:
                    self._remote_agents[engine_id].append(remote_agent_name)

        # Register remote agent memory regions
        blocks_data = []
        for r_index, r_desc_per_ptr, r_block_len, ratio, kv_spec in zip(
            remote_ptr,
            new_remote_desc,
            new_remote_block_len,
            remote_block_num_ratios,
            layer_desc,
        ):
            base_addr = nixl_agent_meta.kv_caches_base_addr[r_index]
            num_blocks = (
                nixl_agent_meta.num_blocks * ratio
                if not isinstance(kv_spec, MambaSpec)
                else nixl_agent_meta.logical_num_blocks
            )
            for block_id in range(num_blocks):
                block_offset = block_id * r_block_len
                # a ptr may register multi xfer
                for offset, lens in r_desc_per_ptr:
                    addr = base_addr + block_offset + offset
                    # (addr, len, device id)
                    blocks_data.append((addr, lens, nixl_agent_meta.device_id))

        assert len(blocks_data)
        logger.debug(
            "Created %s blocks for dst engine %s with remote tp rank %s , remote pp rank %s , remote dp rank %s and local tp rank %s local pp rank %s",
            len(blocks_data),
            engine_id,
            remote_tp_rank,
            remote_pp_rank,
            remote_dp_rank,
            self.tp_rank,
            self.pp_rank,
        )

        # Register with NIXL. For now, remote type is always be vram on write mode.
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")
        remote_xfer_handle = self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)

        with self.net_info_lock:
            self.dst_xfer_side_handles[engine_id].append(remote_xfer_handle)

        remote_block_size = nixl_agent_meta.block_size
        local_desc_tuple = tuple[int]()
        for l_desc_per_ptr in new_local_desc:
            for decs in l_desc_per_ptr:
                local_desc_tuple += decs
        key = (
            (remote_tp_rank, remote_pp_rank, remote_dp_rank)
            + local_desc_tuple
            + tuple(new_local_block_len)
        )

        if remote_block_size not in self.src_xfer_side_handles:
            self.src_xfer_side_handles[remote_block_size] = {}
        if key not in self.src_xfer_side_handles[remote_block_size]:
            logger.debug(
                "register_local_xfer_handler with: local decs: %s, local block lens: %s.",
                local_desc_tuple,
                new_local_block_len,
            )
            self.src_xfer_side_handles[remote_block_size][key] = (
                self.register_local_xfer_handler(
                    local_ptr,
                    new_local_desc,
                    new_local_block_len,
                    local_block_num_ratios,
                    layer_desc,
                )
            )
        else:
            logger.debug(
                "a key: %s with block size: %s already exists, please make sure the used local_xfer_handle is corrected.",
                remote_block_size,
                key,
            )

        local_xfer_handle = self.src_xfer_side_handles[remote_block_size][key]
        return (
            remote_tp_notify_msg,
            tp_msg_resize,
            remote_agent_name,
            write,
            local_xfer_handle,
            remote_xfer_handle,
        )

    def _nixl_handshake(
        self,
        meta: NodeMeta,
    ) -> dict[int, str]:
        """Do a NIXL handshake with a remote instance."""

        # given the local start end, and remote pp_split, return the info this region contains,
        def _get_pp_target_rank_infos(start, end, pp_split):
            remote_agent_layer_record = []
            init_start = start
            index = 0
            local_start = 0
            while start != end:
                layer_start = pp_split[index]
                if start == layer_start:
                    cur_pp_start = 0
                    cur_pp_end = cur_pp_start + (min(pp_split[index + 1], end) - start)
                    local_end = local_start + (min(pp_split[index + 1], end) - start)
                    remote_agent_layer_record.append(
                        (index, cur_pp_start, cur_pp_end, local_start, local_end)
                    )
                    start = min(pp_split[index + 1], end)
                    local_start = local_end
                    index += 1
                elif start > layer_start:
                    index += 1
                else:
                    cur_pp_start = start - pp_split[index - 1]
                    cur_pp_end = cur_pp_start + (min(pp_split[index], end) - start)
                    local_end = local_start + (min(pp_split[index], end) - start)
                    remote_agent_layer_record.append(
                        (index - 1, cur_pp_start, cur_pp_end, local_start, local_end)
                    )
                    start = min(pp_split[index], end)
                    local_start = local_end
            assert len(set([x[0] for x in remote_agent_layer_record])) == len(
                remote_agent_layer_record
            )
            assert (end - init_start) == remote_agent_layer_record[-1][4]
            return remote_agent_layer_record

        # given the local pp split and remote pp split, return the notify info str for remote each pp rank
        def _get_pp_notify_infos(local_pp_split, remote_pp_split):
            results = {}
            start_index = 1
            start = 0
            for rank, remote_end in enumerate(remote_pp_split[1:]):
                cur_rank_peer_nums = 0
                for index, local_end in enumerate(local_pp_split[start_index:]):
                    cur_rank_peer_nums += 1
                    if local_end == remote_end:
                        start_index += index + 1
                        break
                    elif local_end > remote_end:
                        start_index += index
                        break
                results[rank] = (start, start + cur_rank_peer_nums)
                start += cur_rank_peer_nums
                if local_end > remote_end:
                    start -= 1

            return results

        # given the local start/end and remote layer index mapping, return the infos to register memory of each group
        def get_target_group(
            local_start: int,
            local_end: int,
            remote_layer_ptr_mapping: dict[str, list[int]],
            remote_layer_block_len_mapping: dict[str, list[int]],
            remote_layer_desc_mapping: dict[str, list[list[tuple[int, int]]]],
            remote_tp_world_size: int,
            layer_region_num_mapping: list[int],
        ):
            num_groups = len(self.kv_group_infos)
            gather_layer_transfers: dict[int, int] = {}
            layer_transfer_infos: list[dict[int, int]] = [
                defaultdict(dict) for _ in range(num_groups)
            ]
            groups_ids = list(range(num_groups))
            local_ptr_infos: list[list[int]] = [[] for _ in range(num_groups)]
            remote_ptr_infos: list[list[int]] = [[] for _ in range(num_groups)]

            local_desc_infos: list[list[list[tuple[int, int]]]] = [
                [] for _ in range(num_groups)
            ]
            remote_desc_infos: list[list[list[tuple[int, int]]]] = [
                [] for _ in range(num_groups)
            ]

            local_block_lens: list[list[int]] = [[] for _ in range(num_groups)]
            remote_block_lens: list[list[int]] = [[] for _ in range(num_groups)]
            layer_desc_infos: list[list[KVCacheSpec]] = [[] for _ in range(num_groups)]
            layer_index_infos: list[list[int]] = [[] for _ in range(num_groups)]

            layer_region_infos: list[list[int]] = [
                [0] * self.num_layers for _ in range(num_groups)
            ]
            group_infos: list[int] = [0 for _ in range(num_groups)]

            layer_count_each_groups: list[set[int]] = [set() for _ in range(num_groups)]
            for group_id, group_info in enumerate(self.kv_group_infos):
                # tuple[int, str, str]: index, layer_name, layer_spec_name
                for index, (_, global_index, layer_name, layer_spec) in enumerate(
                    group_info
                ):
                    if "mtp" in layer_name:
                        continue
                    index_on_pp = global_index - self.pp_layer_start
                    if local_start <= index_on_pp < local_end:
                        if index_on_pp not in layer_index_infos[group_id]:
                            layer_index_infos[group_id].append(index_on_pp)

                        # one transfer on one group
                        layer_count_each_groups[group_id].add(index_on_pp)

                        local_index = self.layer_ptr_mapping[layer_name]
                        remote_index = remote_layer_ptr_mapping[layer_name]
                        local_ptr_infos[group_id].extend(local_index)
                        remote_ptr_infos[group_id].extend(remote_index)
                        assert len(local_index) == len(remote_index)

                        layer_desc_infos[group_id].extend(
                            [layer_spec] * len(local_index)
                        )
                        local_block_len = self.layer_block_len_mapping[layer_name]
                        remote_block_len = remote_layer_block_len_mapping[layer_name]
                        local_desc_info = self.layer_desc_mapping[
                            layer_name
                        ]  # offset, len
                        remote_desc_info = remote_layer_desc_mapping[layer_name]

                        if self._has_mamba:
                            if isinstance(layer_spec, MambaSpec) or not isinstance(
                                layer_spec, MLAAttentionSpec
                            ):
                                # NOTE(gyf): for local tp size == remote tp size, register con + ssm as one memory
                                # 1 for mamba or full-attention, 2 for full-attention with scale,
                                assert (
                                    len(local_desc_info) == 1
                                    and len(remote_desc_info) == 1
                                )
                                if self.tp_world_size == remote_tp_world_size:
                                    local_desc_info[0] = local_desc_info[0][-1:]
                                    remote_desc_info[0] = remote_desc_info[0][-1:]
                                else:
                                    local_desc_info[0] = local_desc_info[0][:-1]
                                    remote_desc_info[0] = remote_desc_info[0][:-1]

                        local_desc_infos[group_id].extend(local_desc_info)
                        remote_desc_infos[group_id].extend(remote_desc_info)
                        local_block_lens[group_id].extend(local_block_len)
                        remote_block_lens[group_id].extend(remote_block_len)

                        num_region_this_layer = sum([len(x) for x in local_desc_info])
                        assert (
                            sum([len(x) for x in remote_desc_info])
                            == num_region_this_layer
                        )

                        layer_region_infos[group_id][index_on_pp] += (
                            num_region_this_layer
                        )
                        layer_region_num_mapping[index_on_pp] = num_region_this_layer
                        group_infos[group_id] += num_region_this_layer

            for group, region_info in enumerate(layer_region_infos):
                x = [
                    0,
                ]
                for i in region_info:
                    if i != 0:
                        x.append(x[-1] + i)
                layer_region_infos[group] = x
            groups_ids = [x for x in groups_ids if len(local_ptr_infos[x]) > 0]
            local_ptr_infos = [x for x in local_ptr_infos if len(x) > 0]
            remote_ptr_infos = [x for x in remote_ptr_infos if len(x) > 0]
            local_desc_infos = [x for x in local_desc_infos if len(x) > 0]
            remote_desc_infos = [x for x in remote_desc_infos if len(x) > 0]
            local_block_lens = [x for x in local_block_lens if len(x) > 0]
            remote_block_lens = [x for x in remote_block_lens if len(x) > 0]
            layer_desc_infos = [x for x in layer_desc_infos if len(x) > 0]
            layer_index_infos = [x for x in layer_index_infos if len(x) > 0]
            layer_region_infos = [x for x in layer_region_infos if len(x) > 1]
            group_infos = [x for x in group_infos if x > 0]

            gather_layer_count = set()
            for layer_count_each_group in layer_count_each_groups:
                gather_layer_count = gather_layer_count | layer_count_each_group

            for layer_index in gather_layer_count:
                count = 0
                for layer_count_each_group in layer_count_each_groups:
                    if layer_index in layer_count_each_group:
                        count += 1
                gather_layer_transfers[layer_index] = count
            layer_transfer_infos = [
                gather_layer_transfers for _ in range(len(groups_ids))
            ]
            return (
                groups_ids,
                local_ptr_infos,
                remote_ptr_infos,
                local_desc_infos,
                remote_desc_infos,
                local_block_lens,
                remote_block_lens,
                layer_desc_infos,
                layer_index_infos,
                group_infos,
                layer_region_infos,
                layer_transfer_infos,
            )

        remote_expire_time = meta.expire_time
        engine_id = meta.engine_id
        assert remote_expire_time >= self._kv_lease_duration, (
            f"the remote engine's: {engine_id}  expire time: {remote_expire_time}, must >= local expire time: {self._kv_lease_duration}, otherwise may get dirty data after remote expired."
        )
        remote_pp_size = meta.pp_size
        remote_tp_size = meta.tp_size
        remote_pp_split = copy.copy(meta.pp_split)
        remote_dp_rank = meta.dp_rank
        remote_pp_split.append(self.num_hidden_layers)
        assert meta.kv_group_infos == self.kv_group_infos_for_check, (
            f"the remote kv group infos: {meta.kv_group_infos} must as same as local kv group infos: {self.kv_group_infos_for_check}"
        )

        tp_num_remotes = max(1, remote_tp_size // self.tp_world_size)
        init_remote_tp_rank = self.transfer_topo.get_target_remote_rank(remote_tp_size)
        init_remote_pp_per_layer_notify: dict[int, tuple[int]] = {
            i: (start, end)
            for i, (start, end) in enumerate(
                zip(remote_pp_split[:-1], remote_pp_split[1:])
            )
        }
        local_pp_split = copy.copy(self.pp_split)
        local_pp_split.append(self.num_hidden_layers)
        init_remote_pp_per_rank_notify: dict[int, int] = _get_pp_notify_infos(
            local_pp_split, remote_pp_split
        )
        decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
        handshake_decoder = msgspec.msgpack.Decoder(NixlHandshakePayload)

        with zmq_ctx(zmq.REQ, envs.VLLM_ILU_NIXL_PROXY_ADDR) as sock:
            # Set receive timeout to 5 seconds to avoid hanging on dead server
            sock.setsockopt(zmq.RCVTIMEO, 5000)  # milliseconds
            remote_agent_layer_record = _get_pp_target_rank_infos(
                self.pp_layer_start, self.pp_layer_end, remote_pp_split
            )

            per_layer_mapping = []
            per_rank_mapping = []
            # each local rank may write to multi remote tp rank
            for i in range(tp_num_remotes):
                remote_tp_rank = init_remote_tp_rank + i
                num_pp_target = len(remote_agent_layer_record)

                layer_region_nums = [None] * self.num_layers
                per_layer_cur_tp_mapping = [
                    [] for _ in range(self.num_layers)
                ]  # each local layer has one remote layer
                per_rank_cur_tp_mapping = [
                    [] for _ in range(self.num_groups * num_pp_target)
                ]  # for per rank, local layers may have multi-remote part and each part may have multi group
                # for num_group == 1, per_rank only have multi-remote
                write = True
                # then, this rank may write to multi remote pp rank
                for pp_index, (
                    remote_pp_rank,
                    _,
                    _,
                    local_start,
                    local_end,
                ) in enumerate(remote_agent_layer_record):
                    global_remote_rank = (
                        remote_pp_rank * remote_tp_size + remote_tp_rank
                    )
                    msg = msgspec.msgpack.encode(
                        (GET_META_MSG, engine_id, global_remote_rank)
                    )
                    metadata_bytes = zmq_comm2proxy(
                        sock, msg, need_recv=True, logger=logger
                    )
                    try:
                        handshake_payload = handshake_decoder.decode(metadata_bytes)
                    except (msgspec.DecodeError, msgspec.ValidationError) as e:
                        raise RuntimeError(
                            f"Failed to decode NixlHandshakePayload. This likely indicates "
                            f"an incompatibility between connector version. Error: {e}"
                        ) from e

                    metadata: NixlAgentMetadata = decoder.decode(
                        handshake_payload.agent_metadata_bytes
                    )

                    # Ensure engine id matches.
                    if metadata.engine_id != engine_id:
                        raise RuntimeError(
                            f"Remote NIXL agent engine ID mismatch. "
                            f"Expected {engine_id},"
                            f"received {metadata.engine_id}."
                        )

                    if metadata.physical_block_size != self.block_size and not (
                        self.use_mla and len(self._all_specs) == 1
                    ):
                        raise RuntimeError(
                            f"for none-mla type model, remote physical_block_size != local physical_block_size is not supported, get"
                            f"remote remote physical_block_size: {metadata.physical_block_size}, local physical_block_size: {self.physical_block_size}"
                        )

                    (
                        groups_ids,
                        local_ptr_infos,
                        remote_ptr_infos,
                        local_desc_infos,
                        remote_desc_infos,
                        local_block_lens,
                        remote_block_lens,
                        layer_desc_infos,
                        layer_index_infos,
                        group_infos,
                        layer_region_infos,
                        layer_transfer_infos,
                    ) = get_target_group(
                        local_start,
                        local_end,
                        metadata.layer_ptr_mapping,
                        metadata.layer_block_len_mapping,
                        metadata.layer_desc_mapping,
                        remote_tp_size,
                        layer_region_nums,
                    )

                    # finally, split those layer to groups.
                    init_per_layer_notify_msg = None
                    init_remote_agent_name = None
                    num_transfer_this_layers = len(layer_index_infos)
                    for (
                        group_id,
                        local_ptr,
                        remote_ptr,
                        local_desc,
                        remote_desc,
                        local_block_len,
                        remote_block_len,
                        layer_desc,
                        layer_index,
                        group_info,
                        layer_regions,
                        layer_transfers,
                    ) in zip(
                        groups_ids,
                        local_ptr_infos,
                        remote_ptr_infos,
                        local_desc_infos,
                        remote_desc_infos,
                        local_block_lens,
                        remote_block_lens,
                        layer_desc_infos,
                        layer_index_infos,
                        group_infos,
                        layer_region_infos,
                        layer_transfer_infos,
                    ):
                        is_mamba = isinstance(layer_desc[0], MambaSpec)

                        (
                            remote_tp_notify_msg,
                            tp_msg_resize,
                            remote_agent_name,
                            write,
                            local_xfer_handle,
                            remote_xfer_handle,
                        ) = self.add_remote_agent(
                            metadata,
                            local_ptr,
                            remote_ptr,
                            local_desc,
                            remote_desc,
                            local_block_len,
                            remote_block_len,
                            layer_desc,
                            remote_tp_rank,
                            remote_tp_size,
                            remote_pp_rank,
                            remote_pp_size,
                            remote_dp_rank,
                            group_id,
                            init_remote_agent_name,
                        )

                        if not write:
                            for mapping_to_layer_index in layer_index:
                                if (
                                    len(
                                        per_layer_cur_tp_mapping[mapping_to_layer_index]
                                    )
                                    == 0
                                ):
                                    per_layer_cur_tp_mapping[
                                        mapping_to_layer_index
                                    ].append(
                                        (
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            False,
                                            0,
                                            is_mamba,
                                        )
                                    )
                            per_rank_cur_tp_mapping[
                                pp_index * self.num_groups + group_id
                            ].append(
                                (None, None, None, None, None, None, False, 0, is_mamba)
                            )
                        else:
                            if init_remote_agent_name is None:
                                assert remote_agent_name is not None
                                init_remote_agent_name = remote_agent_name

                            tp_notify_msg = (
                                f"{self.tp_rank // tp_msg_resize}-"
                                + remote_tp_notify_msg
                            )
                            per_layer_info = init_remote_pp_per_layer_notify[
                                remote_pp_rank
                            ]
                            per_rank_info = init_remote_pp_per_rank_notify[
                                remote_pp_rank
                            ]
                            # pp:tp:group:transfer-nums
                            per_layer_notify_msg = (
                                "{}-"
                                + str(per_layer_info[0])
                                + "-"
                                + str(per_layer_info[1])
                                + ":"
                                + tp_notify_msg
                                + ":"
                                + str(group_id)
                                + "-"
                                + "{}"
                            )
                            if init_per_layer_notify_msg is None:
                                init_per_layer_notify_msg = per_layer_notify_msg

                            per_rank_notify_msg = (
                                f"{self.pp_rank}-"
                                + str(per_rank_info[0])
                                + "-"
                                + str(per_rank_info[1])
                                + ":"
                                + tp_notify_msg
                                + ":"
                                + str(group_id)
                                + "-"
                                + str(num_transfer_this_layers)
                            )

                            is_mamba_group = isinstance(layer_desc[0], MambaSpec)
                            local_num_blocks = (
                                self.num_blocks
                                if not is_mamba_group
                                else self._logical_num_blocks
                            )
                            remote_num_blocks = (
                                metadata.num_blocks
                                if not is_mamba_group
                                else metadata.logical_num_blocks
                            )
                            per_rank_local_region_np = (
                                np.arange(group_info) * local_num_blocks
                            )[:, None]
                            per_rank_remote_region_np = (
                                np.arange(group_info) * remote_num_blocks
                            )[:, None]

                            for index, mapping_to_layer_index in enumerate(layer_index):
                                kv_num_start, kv_num_end = (
                                    layer_regions[index],
                                    layer_regions[index + 1],
                                )

                                per_layer_local_region_np = (
                                    np.arange(kv_num_start, kv_num_end)
                                    * local_num_blocks
                                )[:, None]
                                per_layer_remote_region_np = (
                                    np.arange(kv_num_start, kv_num_end)
                                    * remote_num_blocks
                                )[:, None]
                                per_layer_cur_tp_mapping[mapping_to_layer_index].append(
                                    (
                                        local_xfer_handle,
                                        remote_xfer_handle,
                                        init_remote_agent_name,
                                        str(
                                            per_layer_notify_msg.format(
                                                mapping_to_layer_index
                                                + self.pp_layer_start,
                                                layer_transfers[mapping_to_layer_index],
                                            )
                                        ),
                                        per_layer_local_region_np,
                                        per_layer_remote_region_np,
                                        True,
                                        group_id,
                                        not is_mamba,
                                    )
                                )
                            # per_rank must have some layer to send when some layers(A) use other layers(B)'s kv cache, and only send B's kv cache
                            per_rank_cur_tp_mapping[
                                pp_index * self.num_groups + group_id
                            ].append(
                                (
                                    local_xfer_handle,
                                    remote_xfer_handle,
                                    init_remote_agent_name,
                                    per_rank_notify_msg,
                                    per_rank_local_region_np,
                                    per_rank_remote_region_np,
                                    True,
                                    group_id,
                                    not is_mamba,
                                )
                            )

                # if some layer shared with other, mark this layer only send notify infos.
                if any(len(info) == 0 for info in per_layer_cur_tp_mapping):
                    new_per_layer_cur_tp_mapping = []

                    for index, info in enumerate(per_layer_cur_tp_mapping):
                        if len(info) == 0:
                            # it's safe to use init_per_layer_notify_msg(any group id)
                            new_per_layer_cur_tp_mapping.append(
                                [
                                    (
                                        None,
                                        None,
                                        init_remote_agent_name,
                                        init_per_layer_notify_msg.format(
                                            index + self.pp_layer_start, 1
                                        ),
                                        None,
                                        None,
                                        True,
                                        0,
                                        not is_mamba,
                                    )
                                ]
                            )
                        else:
                            new_per_layer_cur_tp_mapping.append(info)
                    per_layer_cur_tp_mapping = new_per_layer_cur_tp_mapping

                for index, info in enumerate(per_rank_cur_tp_mapping):
                    if len(info) == 0:
                        per_rank_cur_tp_mapping[index] = [
                            (None, None, None, None, None, None, False, 0, is_mamba),
                        ]

                per_layer_mapping.append(per_layer_cur_tp_mapping)
                per_rank_mapping.append(per_rank_cur_tp_mapping)

            # init get per-rank shape[num_tp, num_pp * num_region] pp0_num_region, pp1_num_region
            # init get per-layer shape[num_tp, num_layers] -> [num_layers, num_tp]
            new_per_layer_mapping = [None] * len(per_layer_mapping[0])
            new_per_rank_mapping = [None] * len(per_rank_mapping[0])
            for i in range(len(per_layer_mapping[0])):
                new_per_layer_mapping[i] = [
                    per_layer_mapping[j][i] for j in range(len(per_layer_mapping))
                ]
            for i in range(len(per_rank_mapping[0])):
                new_per_rank_mapping[i] = [
                    per_rank_mapping[j][i] for j in range(len(per_rank_mapping))
                ]

        logger.debug(
            "per_layer_mapping: %s, \nper_rank_mapping: %s",
            new_per_layer_mapping,
            new_per_rank_mapping,
        )

        return new_per_layer_mapping, new_per_rank_mapping

    ############################################################
    # register kv caches
    ############################################################
    def initialize_host_xfer_buffer(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_scales: dict[str, tuple[torch.Tensor]] | None = None,
    ) -> None:
        """
        Initialize transfer buffer in CPU mem for accelerators
        NOT directly supported by NIXL (e.g., tpu)
        """
        new_kv_caches = {}
        tensor_record: dict[int, torch.Tensor] = {}
        for layer_name, cache_or_caches in kv_caches.items():
            data_ptr = cache_or_caches.data_ptr()
            if data_ptr not in tensor_record:
                host_tensor = torch.empty(
                    cache_or_caches.shape, device="cpu", dtype=self.host_dtype
                )
                tensor_record[data_ptr] = host_tensor
            else:
                host_tensor = tensor_record[data_ptr]
            new_kv_caches[layer_name] = host_tensor
        del tensor_record

        if kv_cache_scales is not None:
            logger.warning(
                "for now, host buffer are used for P(kv origin dtype) == D(KV origin dtype),"
                "but P(kv store dtype) != D(KV store dtype), so scales(if have) will be remove"
                "and register with the host tensor"
            )
            kv_cache_scales = None

        self.host_xfer_buffers = new_kv_caches
        return new_kv_caches, kv_cache_scales

    def register_cross_layers_kv_caches(self, kv_cache: torch.Tensor) -> None:
        """Register a cross-layers KV cache tensor with NIXL.

        `use_uniform_kv_cache()` guarantees a single KV cache group whose
        layers all share the same `AttentionSpec`, so any layer name from
        `_layer_specs` yields the correct per-layer spec for `page_size_bytes`.
        """
        first_layer = next(iter(self._layer_specs))
        # Forwarding a real layer name rather than a synthetic key
        self.register_kv_caches({first_layer: kv_cache})

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_scales: dict[str, tuple[torch.Tensor]] | None = None,
    ) -> None:
        """Register the KV Cache data in nixl."""
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
        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config, self.backend_name, self.transfer_topo.cross_layers_blocks
        )

        # NOTE(gyf): hard code for fp8 indexer cache for nv
        if self.use_quant_to_fp8:
            new_kv_caches = {}
            tensor_ptrs = []
            quant_tensor_ptrs = []
            new_device = None
            for layer_name, cache_or_caches in kv_caches.items():
                if ".k_cache" in layer_name:
                    new_shape = list(cache_or_caches.shape)
                    assert len(new_shape) == 3
                    new_shape = new_shape[:-1] + [new_shape[-1] - 4]  # a float scales
                    if self.index_head_dim is None:
                        self.index_head_dim = new_shape[-1]
                    else:
                        assert self.index_head_dim == new_shape[-1]
                    if new_device is None:
                        new_device = cache_or_caches.device
                    new_tensor = torch.empty(
                        new_shape, dtype=self.host_dtype, device=new_device
                    )
                    self.new_allocated_indexer_tensors.append(new_tensor)
                    self.origin_indexer_tensors.append(cache_or_caches)
                    tensor_ptrs.append(new_tensor.data_ptr())
                    quant_tensor_ptrs.append(cache_or_caches.data_ptr())
                    new_kv_caches[layer_name] = new_tensor
                else:
                    new_kv_caches[layer_name] = cache_or_caches
            kv_caches = new_kv_caches
            self.indexer_tensor_ptrs = torch.tensor(
                tensor_ptrs, dtype=torch.long, device=new_device
            )
            self.quant_indexer_tensor_ptrs = torch.tensor(
                quant_tensor_ptrs, dtype=torch.long, device=new_device
            )
            self.device_for_ops = new_device

        if self.use_host_buffer:
            xfer_buffers, kv_cache_scales = self.initialize_host_xfer_buffer(
                kv_caches, kv_cache_scales
            )
        else:
            xfer_buffers = kv_caches
            assert not self.host_xfer_buffers, (
                "host_xfer_buffer should not be initialized when "
                f"kv_buffer_device is {self.kv_buffer_device}"
            )

        logger.info(
            "Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, "
            "use_host_buffer: %s",
            self.use_mla,
            self.kv_buffer_device,
            self.use_host_buffer,
        )

        caches_data = []
        # With hybrid allocator, layers can share a kv cache tensor
        seen_base_addresses = []
        seen_base_addresses_check: dict[int, int] = {}
        tensor_size_bytes = None
        # Per-layer record block_len size, used prt, and (offset, register len).
        self.layer_block_len_mapping = defaultdict(list[int])
        self.layer_ptr_mapping = defaultdict(list[int])
        self.layer_desc_mapping = defaultdict(list[list[tuple[int, int]]])

        if self.use_host_buffer:
            d2h_buffers = [[] for _ in range(self.pp_layer_end - self.pp_layer_start)]
            kv_cache_records = [
                [] for _ in range(self.pp_layer_end - self.pp_layer_start)
            ]
        else:
            d2h_buffers = None
            kv_cache_records = None

        for layer_name, cache_or_caches in xfer_buffers.items():
            if "mtp" in layer_name:
                continue

            layer_spec = self._layer_specs.get(layer_name, None)
            # None means shared with other tensor, skip register this tensor
            if layer_spec is None:
                continue

            # NOTE (NickLucche) Hybrid SSM models assume a layout that is similar to
            # that of FI, with block laid out as in `get_backend_aware_kv_block_len`.
            # However, physical page_size may differ when kernel requires a specific
            # block size. This leads to SSM and FA layers having different num_blocks.
            # `_physical_blocks_per_logical_kv_block` ratio is used to adjust for this.
            if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                # MLA DSv32 Indexer case: UniformTypeKVCacheSpecs merges kv_cache_specs
                layer_spec = layer_spec.kv_cache_specs[layer_name]

            # case1: for full-attention layers, return tensor(2, num_block, ...) if not has_mamba else return list[tensor(2, num_block, ...)]
            # case2: for mamba layers, return cache[0] tensor(...)
            cache_list = self.transfer_topo.get_transfer_cache_regions(
                cache_or_caches, layer_spec
            )
            # `layer_spec.page_size_bytes` only accounts for logical page_size, that is
            # the page_size assuming constant `self._logical_num_blocks`.
            physical_page_size = (
                layer_spec.page_size_bytes
                if isinstance(layer_spec, MambaSpec)
                else layer_spec.page_size_bytes
                // self._physical_blocks_per_logical_kv_block
            )
            # For when registering multiple tensors eg K/V in separate regions.
            physical_page_size = physical_page_size // len(cache_list)
            if self.transfer_topo._cross_layers_blocks:
                # When cross-layers blocks are used, multiply by number of layers
                physical_page_size = physical_page_size * len(
                    self.kv_cache_config.kv_cache_tensors
                )
            num_blocks = (
                self._logical_num_blocks
                if isinstance(layer_spec, MambaSpec)
                else self.num_blocks
            )
            # `page_size` accounts for physical blocks, st KVCache is always
            # [`num_blocks` * `page_size`]
            curr_tensor_size_bytes = num_blocks * physical_page_size
            if tensor_size_bytes is None:
                tensor_size_bytes = curr_tensor_size_bytes

            # TODO (NickLucche) we could eventually unify how we handle FA/FI regions,
            # registering a single tensor for both K/V and splitting logically like FI.
            for cache in cache_list:
                exist = False
                desc_cur_layer: list[tuple[int, int]] = []
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses_check:
                    # if exist, record the index for remote to get the correct data_ptr.
                    exist = True
                    current_index = seen_base_addresses_check[base_addr]
                else:
                    current_index = len(seen_base_addresses_check)
                    seen_base_addresses_check[base_addr] = current_index

                logger.debug(
                    "Registering layer %s with cache shape: %s", layer_name, cache.shape
                )

                if cache.shape[0] != num_blocks:
                    raise AssertionError(
                        "All kv cache tensors must have the same number of "
                        f"blocks; layer={layer_name}, "
                        f"expected_num_blocks={num_blocks}, "
                        f"cache_shape={tuple(cache.shape)}, "
                        f"cache_stride={tuple(cache.stride())}, "
                        f"layer_spec={type(layer_spec).__name__}, "
                        f"backend={self.backend_name}, "
                        "all_backends="
                        f"{[backend.get_name() for backend in self.attn_backends]}, "
                        f"kv_cache_layout={self.kv_cache_layout}, "
                        "blocks_first="
                        f"{self.transfer_topo.is_kv_layout_blocks_first}"
                    )

                if not self.use_mla:
                    # Different kv cache shape is not supported by HeteroTP.
                    # This must also hold true for Mamba-like models.
                    assert tensor_size_bytes == curr_tensor_size_bytes, (
                        "All kv cache tensors must have the same size"
                    )
                # Need to make sure the device ID is non-negative for NIXL,
                # Torch uses -1 to indicate CPU tensors.
                self.device_id = max(cache.get_device(), 0)
                if not exist:
                    seen_base_addresses.append(base_addr)
                    caches_data.append(
                        (base_addr, curr_tensor_size_bytes, self.device_id, "")
                    )

                self.layer_block_len_mapping[layer_name].append(physical_page_size)
                self.layer_ptr_mapping[layer_name].append(current_index)
                if isinstance(layer_spec, MambaSpec):
                    assert self._conv_decomp is not None
                    conv_offsets = self._conv_decomp.local_conv_offsets
                    conv_size, ssm_size = self._mamba_ssm_size

                    for off, sz in conv_offsets:
                        desc_cur_layer.append((off, sz))
                    desc_cur_layer.append((conv_size, ssm_size))
                    desc_cur_layer.append((0, physical_page_size))
                else:
                    if self._has_mamba and not isinstance(layer_spec, MLAAttentionSpec):
                        desc_cur_layer.extend(
                            [
                                (0, physical_page_size // 2),
                                (physical_page_size // 2, physical_page_size // 2),
                                (0, physical_page_size),
                            ]
                        )
                    else:
                        # NOTE(gyf): this kv spec may padded, but we transfor all data.
                        desc_cur_layer.append((0, physical_page_size))
                self.layer_desc_mapping[layer_name].append(desc_cur_layer)
                if exist:
                    continue

            # NOTE(gyf): additional codes for mla host buffer
            if self.use_host_buffer:
                for cache in cache_list:
                    assert isinstance(cache, torch.Tensor)
                    assert isinstance(layer_spec, MLAAttentionSpec)
                    layer_index = int(
                        find_number_on_layer_name(layer_name)[0].replace(".", "")
                    )

                    if ".k_cache" in layer_name:
                        # hard code, mla k_cache on index 0
                        d2h_buffers[layer_index - self.pp_layer_start].insert(
                            0,
                            torch.empty(
                                (self.max_num_batched_tokens, layer_spec.head_size),
                                dtype=self.host_dtype,
                                device="cpu",
                                pin_memory=True,
                            ),
                        )
                        kv_cache_records[layer_index - self.pp_layer_start].insert(
                            0, cache
                        )
                    else:
                        d2h_buffers[layer_index - self.pp_layer_start].append(
                            torch.empty(
                                (self.max_num_batched_tokens, layer_spec.head_size),
                                dtype=self.host_dtype,
                                device="cpu",
                                pin_memory=True,
                            )
                        )
                        kv_cache_records[layer_index - self.pp_layer_start].append(
                            cache
                        )

            # NOTE(gyf): additional codes for scales
            scale_tensors = None
            if kv_cache_scales is not None:
                scale_tensors = kv_cache_scales.get(layer_name, None)
            scale_exist = False
            if scale_tensors is not None:
                assert isinstance(layer_spec, FullAttentionSpec), (
                    "Not support scales for attention layer."
                )
                scale_tensor = scale_tensors[
                    0
                ]  # for scales, only register the first tensor

                base_addr = scale_tensor.data_ptr()
                if base_addr in seen_base_addresses_check:
                    # if exist, record the index for remote to get the correct data_ptr.
                    scale_exist = True
                    current_index = seen_base_addresses_check[base_addr]
                else:
                    current_index = len(seen_base_addresses_check)
                    seen_base_addresses_check[base_addr] = current_index

                curr_tensor_size_bytes = (
                    scale_tensor.numel() * scale_tensor.element_size()
                )

                # num_blocks, 2, block_size             for mla int8 scale
                # num_blocks, num_heads, block_size     for non-mla int8 scale
                attention_num_block = scale_tensor.shape[0]
                assert attention_num_block == num_blocks

                if not scale_exist:
                    seen_base_addresses.append(base_addr)
                    caches_data.append(
                        (base_addr, curr_tensor_size_bytes, self.device_id, "")
                    )

                self.layer_block_len_mapping[layer_name].append(
                    curr_tensor_size_bytes // self.num_blocks
                )
                self.layer_ptr_mapping[layer_name].append(current_index)
                # For now, scale only has one desc per ptr.
                self.layer_desc_mapping[layer_name].append(
                    [(0, curr_tensor_size_bytes // self.num_blocks)]
                )

        if self.use_host_buffer:
            self.d2h_manager = D2HManger(d2h_buffers, kv_cache_records)

        descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
        logger.debug("Registering descs: %s", seen_base_addresses)
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        logger.debug("Done registering descs")
        self._registered_descs.append(descs)

        # Register local/src descr for NIXL xfer.
        self.seen_base_addresses = seen_base_addresses

        if self._has_mamba:
            logger.info(
                "Hybrid SSM registration: num_blocks=%s, "
                "logical_num_blocks=%s, ratio=%s, "
                "mamba_ssm_size=%s",
                self.num_blocks,
                self._logical_num_blocks,
                self._physical_blocks_per_logical_kv_block,
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
            physical_blocks_per_logical_kv_block=(
                self._physical_blocks_per_logical_kv_block
            ),
            # NOTE(gyf): extra infos pass to remote.
            physical_block_size=self.block_size,
            logical_num_blocks=self._logical_num_blocks,
            layer_ptr_mapping=self.layer_ptr_mapping,
            layer_block_len_mapping=self.layer_block_len_mapping,
            layer_desc_mapping=self.layer_desc_mapping,
            global_rank=self.global_rank,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
        )
        # Wrap metadata in payload with hash for defensive decoding
        assert self.compat_hash is not None
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )

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

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        for req_id, meta in metadata.reqs_to_recv.items():
            # the reqs in reqs_to_recv do not contains abort or full cache local.
            self._reqs_to_recv[req_id] = meta
            # the adding point of recv time is later than send, so when send done, recv must done, send expires, recv must expires.
            self._reqs_ready_to_recv_time_record[req_id] = (
                time.perf_counter() + self._kv_lease_duration
            )
            if req_id in self.pre_recvs:
                self.pre_recvs_to_this_step.add(req_id)
                self.pre_recvs.remove(req_id)

        self._reqs_to_send_layerwise = metadata.reqs_to_send_layerwise
        self._reqs_to_send_per_rank = metadata.reqs_to_send
        self._reqs_to_send_record.update(metadata.reqs_to_send)
        self._reqs_to_send_record.update(metadata.reqs_to_send_layerwise)

        # abort reqs for D only
        if metadata.reqs_to_abort:
            self._reqs_to_abort_recv.update(metadata.reqs_to_abort)

        for req_id, expiration_time in metadata.reqs_to_send_time_record.items():
            self._reqs_ready_to_send_time_record[req_id] = expiration_time

    def try_remap_block_ids(self, req_meta: ReqMeta):
        engine_id = req_meta.engine_id

        block_size_ratio = self._block_size_ratio[engine_id]
        if block_size_ratio == 1:
            return

        token_lens = req_meta.prompt_token_len
        kernel_block_arange = np.arange(block_size_ratio)
        block_size = self.block_size // kernel_block_arange
        req_meta.local_block_ids_np = [
            (blocks.reshape(-1, 1) * block_size_ratio + kernel_block_arange).reshape(
                -1
            )[: cdiv(token_lens, block_size)]
            if self.remap_to_kernel_size[i] > 0
            else blocks
            for i, blocks in enumerate(req_meta.local_block_ids_np)
        ]

    # NOTE(gyf): add later
    def try_sending_kv_cache_chunk(self, req_id: str):
        pass

    def try_sending_kv_cache(self, req_id: str):
        for layer_index, meta in self._reqs_ready_to_send[req_id]:
            self._write_blocks(
                meta.local_block_ids_np,
                self._reqs_to_blocks[req_id][0],
                meta.engine_id,
                req_id,
                layer_index,
            )

        if layer_index == -1 or layer_index == self.pp_layer_end_closed:
            self._reqs_ready_to_send.pop(req_id)
            self._reqs_to_blocks.pop(req_id)
        else:
            self._reqs_ready_to_send[req_id].clear()

    def _write_blocks(
        self,
        local_block_ids: list[
            np.ndarray
        ],  # local and remote are process for mapping or re-calculate
        remote_block_ids: list[np.ndarray],
        dst_engine_id: str,
        request_id: str,
        layer_index: int,
    ):
        # list[array] of shape[1,N], transfer when N > 0 else not
        transfer = remote_block_ids[0].shape[1] > 0
        if not transfer:
            # decode get a full cache or abort
            # do not need recv msg from p
            if request_id not in self._sending_transfers:
                self._sending_transfers[request_id] = TransferWrapper(
                    True, [[None, True, None, dst_engine_id]]
                )

            return

        only_send_msg = False
        # local abort before schedule, but get remote block info, do not send kv, send notify info
        if local_block_ids[0].shape[1] == 0:
            only_send_msg = True

        # true for per-group send, all layer's kv will be send when use per-group send
        ready_for_check = True
        if layer_index != -1:
            ready_for_check = (
                layer_index == self.pp_layer_end_closed
            )  # last layer on this pp rank, get finished can chenk this req send status
            remote_infos = [
                self._remote_agents_per_layer[dst_engine_id][
                    layer_index - self.pp_layer_start
                ]
            ]
        else:
            remote_infos = self._remote_agents_per_rank[dst_engine_id]

        new_handles = []
        start_task_success = []
        # each layer/group may write kv to multi remote engine(PP remote)
        for remote_info_one_layer_or_one_group in remote_infos:
            # Then (TP remote)
            local_descs_ids: list[np.ndarray] = [None] * self.num_groups
            remote_descs_ids: list[np.ndarray] = [None] * self.num_groups
            for remote_info_tps in remote_info_one_layer_or_one_group:
                # layer may has multi group:
                # group only has one group
                for remote_info_groups in remote_info_tps:
                    (
                        local_xfer_side_handle,
                        remote_xfer_side_handle,
                        remote_agent_name,
                        notify_msg,
                        locol_region_np,
                        remote_region_np,
                        write,
                        group_id,
                        not_mamba,
                    ) = remote_info_groups

                    if not write:
                        new_handles.append([None, True, None, dst_engine_id])
                        start_task_success.append(True)
                        continue

                    if local_xfer_side_handle is None:
                        only_send_msg = True

                    notif_id = f"{request_id}:{notify_msg}".encode()

                    if only_send_msg:
                        self.nixl_wrapper.send_notif(remote_agent_name, notif_id)
                        new_handles.append(
                            [None, True, time.perf_counter(), dst_engine_id]
                        )
                        start_task_success.append(True)
                        continue

                    if local_descs_ids[group_id] is None:
                        remote_num_block = remote_block_ids[group_id].shape[1]
                        if not self._has_mamba or not_mamba:
                            local_descs_id = (
                                locol_region_np
                                + local_block_ids[group_id][:, -remote_num_block:]
                            ).flatten()
                            remote_descs_id = (
                                remote_region_np + remote_block_ids[group_id]
                            ).flatten()
                        else:
                            num_blocks = min(
                                local_block_ids[group_id].shape[1], remote_num_block
                            )
                            local_descs_id = (
                                locol_region_np
                                + local_block_ids[group_id][:, :num_blocks]
                            ).flatten()
                            remote_descs_id = (
                                remote_region_np
                                + remote_block_ids[group_id][:, :num_blocks]
                            ).flatten()
                        local_descs_ids[group_id] = local_descs_id
                        remote_descs_ids[group_id] = remote_descs_id

                        assert len(local_descs_id) == len(remote_descs_id)
                    else:
                        local_descs_id = local_descs_ids[group_id]
                        remote_descs_id = remote_descs_ids[group_id]

                    # Prepare transfer with Nixl.
                    handle = None
                    try:
                        handle = self.nixl_wrapper.make_prepped_xfer(
                            "WRITE",
                            local_xfer_side_handle,
                            local_descs_id,
                            remote_xfer_side_handle,
                            remote_descs_id,
                            notif_msg=notif_id,
                        )
                        # Begin async xfer.
                        self.nixl_wrapper.transfer(handle)

                        # Use handle to check completion in future step().
                        new_handles.append(
                            [handle, False, time.perf_counter(), dst_engine_id]
                        )
                        start_task_success.append(True)

                        logger.debug("transfer with notify_id: %s", notif_id)
                    except Exception as e:
                        # mark all (logical) blocks for this request as invalid
                        self._log_failure(
                            failure_type="transfer_setup_failed",
                            req_id=request_id,
                            msg="Marking blocks as invalid",
                            error=e,
                        )
                        self.xfer_stats.record_failed_transfer()
                        if handle is not None:
                            self.nixl_wrapper.release_xfer_handle(handle)
                            start_task_success.append(False)

        if not all(start_task_success):
            new_handles = [
                handle
                for handle, success in zip(new_handles, start_task_success)
                if success
            ]

        if request_id not in self._sending_transfers:
            headles = TransferWrapper(False, [])
            self._sending_transfers[request_id] = headles
        else:
            headles = self._sending_transfers[request_id]
        headles.transfer_list.extend(new_handles)
        headles.ready_for_check = ready_for_check

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

        # save kv layer will be called after start_load_kv
        connector_metadata: NixlConnectorMetadata = None
        if not (connector_metadata := self._connector_metadata):
            return

        layer_index = self.layer_index
        if reqs := self._reqs_to_send_layerwise:
            self._async_loop.call_soon_threadsafe(
                self._reqs_to_send.put_nowait,
                (reqs, layer_index, torch.cuda.current_stream().record_event()),
            )

        self.layer_index += 1
        self.un_send_layer_nums -= 1
        if not self.un_send_layer_nums:
            if reqs := self._reqs_to_send_per_rank:
                self._async_loop.call_soon_threadsafe(
                    self._reqs_to_send.put_nowait,
                    (reqs, -1, torch.cuda.current_stream().record_event()),
                )

    def save_kv_layer_host(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        if self.is_consumer:
            return

        # save kv layer is call after start_load_kv
        connector_metadata: NixlConnectorMetadata = None
        if not (connector_metadata := self._connector_metadata):
            return

        # bnd means shape: [B, Head_nums, head_dims]
        layer_index = self.layer_index

        kv_cache_bnd = kwargs.pop("kv_cache_bnd")
        skip_this_for_send_kv = kwargs.pop("skip_this", False)
        slot_mapping_cpu = None

        if layer_index == self.pp_layer_start:
            slot_mapping_cpu = getattr(attn_metadata, "slot_mapping_cpu", None)

            if slot_mapping_cpu is None:
                logger.error(
                    "attn_metadata type: %s must provide slot_mapping_cpu when use host buffer",
                    attn_metadata.__class__.__name__,
                )
                raise NotImplementedError

        if reqs := self._reqs_to_send_per_rank:
            if not skip_this_for_send_kv:
                self.layer_index += 1
                self.un_send_layer_nums -= 1
            if not self.un_send_layer_nums:
                self._async_loop.call_soon_threadsafe(
                    self._reqs_to_send_d2h.put_nowait,
                    (layer_index, kv_cache_bnd, slot_mapping_cpu, reqs),
                )
            else:
                self._async_loop.call_soon_threadsafe(
                    self._reqs_to_send_d2h.put_nowait,
                    (layer_index, kv_cache_bnd, slot_mapping_cpu, None),
                )

    ############################################################
    # Get results for sending and recving
    ############################################################
    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which remote write done.
        """

        notified_req_ids: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                req_id, remote_pp_info, remote_tp_info, remote_pp_info_2 = notif.decode(
                    "utf-8"
                ).rsplit(":", 3)

                remote_pp_info = remote_pp_info.split("-", 2)
                remote_pp_rank_or_layer_index, remote_pp_start, remote_pp_end = (
                    int(remote_pp_info[0]),
                    int(remote_pp_info[1]),
                    int(remote_pp_info[2]),
                )

                remote_pp_info = remote_pp_info_2.split("-", 1)
                rank_or_layer_transfer_index, rank_or_layer_transfer_nums = (
                    int(remote_pp_info[0]),
                    int(remote_pp_info[1]),
                )

                remote_tp_info = remote_tp_info.split("-", 2)
                remote_tp_rank, remote_tp_start, remote_tp_end = (
                    int(remote_tp_info[0]),
                    int(remote_tp_info[1]),
                    int(remote_tp_info[2]),
                )
                if (
                    req_not_on_this_step := req_id not in self._reqs_to_recv
                ) and not self.use_async_scheduling:
                    logger.error(
                        "Potentially invalid KV blocks for "
                        "unrecognized request %s were retrieved by "
                        "a decode worker. They may have expired.",
                        req_id,
                    )
                    continue

                if req_id not in self.consumer_notification_counts_by_req:
                    self.consumer_notification_counts_by_req[req_id] = (
                        self.notification_class(
                            remote_tp_start,
                            remote_tp_end,
                            remote_pp_start,
                            remote_pp_end,
                            self.num_groups,
                        )
                    )
                transfer_helper = self.consumer_notification_counts_by_req[req_id]
                transfer_helper.update(
                    remote_tp_rank,
                    remote_pp_rank_or_layer_index,
                    rank_or_layer_transfer_index,
                    rank_or_layer_transfer_nums,
                    logger,
                )

                # for reqs that aborted, only pop them when they are done or expired
                if transfer_helper.is_done():
                    if not req_not_on_this_step:
                        notified_req_ids.add(req_id)
                    else:
                        self.pre_recvs.add(req_id)
                if req_id in self._reqs_disconnected_need_check:
                    raise RuntimeError(
                        "req: %s still get remote info after remote disconnected 300s. ",
                        req_id,
                    )

        return notified_req_ids

    async def fetch_finished_status(self) -> tuple[set[int], set[str], set[str]]:
        finished_send_reqs = self.finished_send_reqs
        self.finished_send_reqs = set()

        disconnected_reqs = set()
        disconnected_engines = set()
        disconnect_engine_checked = set()

        if self._reqs_check_done_after_disconnected:
            disconnected_reqs = self._reqs_check_done_after_disconnected
            self._reqs_check_done_after_disconnected = set()

        if self._engines_check_done_after_disconnected:
            disconnected_engines = self._engines_check_done_after_disconnected
            self._engines_check_done_after_disconnected = set()

        if self.disconnected_engine:
            disconnect_engine_checked = self.disconnected_engine
            self.disconnected_engine = {}

        return (
            finished_send_reqs,
            disconnected_reqs,
            disconnected_engines,
            disconnect_engine_checked,
        )

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

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        # query status from loop
        status = asyncio.run_coroutine_threadsafe(
            self.fetch_finished_status(), self._async_loop
        )
        # recv
        done_recving = self._get_new_notifs()
        done_recving.update(self.pre_recvs_to_this_step)
        self.pre_recvs_to_this_step = set()

        block_ids_to_permute = []
        block_ids_to_quant = []
        for req_id in done_recving:
            # clean up metadata for completed requests
            del self.consumer_notification_counts_by_req[req_id]
            self._reqs_ready_to_recv_time_record.pop(req_id)
            meta = self._reqs_to_recv.pop(req_id)
            assert meta is not None, f"{req_id} not found in recving_metadata list"

            if self.enable_permute_local_kv and meta.need_permute_after_recv:
                block_ids_to_permute += meta.local_block_ids[0]

            if self.use_quant_to_fp8:
                block_ids_to_quant += meta.local_block_ids[0]

        # when remote is HND and local is NHD, process a permute op
        if len(block_ids_to_permute) > 0:
            self.permute_device_kv(block_ids_to_permute)

        if len(block_ids_to_quant) > 0:
            self.quant_device_kv(block_ids_to_quant)

        now = time.perf_counter()
        expire_reqs = []
        for req_id, expires in self._reqs_ready_to_recv_time_record.items():
            if now > expires:
                # for now, when recv timeout, remote will exit, so we can make sure we will not get dirty data, this request can be done
                expire_reqs.append(req_id)
                done_recving.add(req_id)
                self.consumer_notification_counts_by_req.pop(req_id, None)
                meta = self._reqs_to_recv.pop(req_id)
                if req_id not in self._reqs_to_abort_recv:
                    # NOTE(gyf): check this work?
                    for block_id in meta.local_block_ids[:1]:
                        self._invalid_block_ids.update(block_id)
                else:
                    self._reqs_to_abort_recv.pop(req_id)
        for req_id in expire_reqs:
            self._reqs_ready_to_recv_time_record.pop(req_id)

        # send, reqs disconnected done, engines disconnected done, first disconnect engine msg
        (
            done_sending,
            disconnected_reqs,
            disconnected_engines,
            disconnect_engine_first_check,
        ) = status.result()
        # remove disconnected_reqs
        self._reqs_disconnected_need_check -= disconnected_reqs
        for req_id in done_sending:
            self._reqs_ready_to_send_time_record.pop(req_id)

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()
        while self._reqs_ready_to_send_time_record:
            req_id, expires = next(iter(self._reqs_ready_to_send_time_record.items()))
            # Sorted dict, oldest requests are put first so we can exit early.
            if now < expires:
                break
            raise RuntimeError(
                f"request : {req_id} send kv expired, expired time is set to: {self._kv_lease_duration}, bug happened"
            )

        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug(
                "Rank tp: %s, pp: %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                self.pp_rank,
                len(done_sending),
                len(done_recving),
            )

        check_req_nums = 0
        if disconnect_engine_first_check:
            reqs_tp_remove = []
            now = time.perf_counter()
            info_to_update: dict[str, list[list[Any]]] = {}
            for engine_id, added_time in disconnect_engine_first_check.items():
                if engine_id not in info_to_update:
                    info_to_update[engine_id] = []
                if self.is_consumer:
                    for req_id, meta in self._reqs_to_recv.items():
                        if meta.engine_id == engine_id:
                            # 30 secs for remote exit..
                            expire_time = min(
                                added_time + 30.0,
                                self._reqs_ready_to_recv_time_record[req_id],
                            )
                            self._reqs_ready_to_recv_time_record[req_id] = expire_time

                            if now > expire_time:
                                done_recving.add(req_id)
                                reqs_tp_remove.append(req_id)

                                self._reqs_ready_to_recv_time_record.pop(req_id)
                                self._reqs_disconnected_need_check.add(req_id)
                                self.consumer_notification_counts_by_req.pop(
                                    req_id, None
                                )
                                info_to_update[engine_id].append([req_id, 0])
                                check_req_nums += 1

                                meta = self._reqs_to_recv[req_id]
                                if req_id not in self._reqs_to_abort_recv:
                                    for block_id in meta.local_block_ids[:1]:
                                        self._invalid_block_ids.update(block_id)
                                else:
                                    self._reqs_to_abort_recv.pop(req_id)

                    for req_id in reqs_tp_remove:
                        self._reqs_to_recv.pop(req_id)
                else:
                    # do not transfer, but remote disconnected.
                    reqs_do_not_start_transfer = []
                    for req_id, meta in self._reqs_to_send_record.items():
                        if meta.engine_id == engine_id and not meta.start_transfer:
                            reqs_do_not_start_transfer.append(req_id)
                            info_to_update[engine_id].append([req_id, 0])
                            check_req_nums += 1

                    for req_id in reqs_do_not_start_transfer:
                        self._reqs_ready_to_send_time_record.pop(req_id)
                        self._reqs_to_send_record.pop(req_id)

        if check_req_nums:
            self._async_loop.call_soon_threadsafe(
                self._reqs_to_checked.put_nowait, info_to_update
            )

        for engine_id in disconnected_engines:
            # release related resources
            with self.net_info_lock:
                for dst_xfer_side_handle in self.dst_xfer_side_handles.pop(
                    engine_id, []
                ):
                    self.nixl_wrapper.release_dlist_handle(dst_xfer_side_handle)

                for agent_name in self._remote_agents.pop(engine_id, []):
                    self.nixl_wrapper.remove_remote_agent(agent_name)

        return done_sending, done_recving

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
        self.shutdown()

    def shutdown(self):
        """Shutdown the connector worker."""
        if dict_lock := getattr(self, "dict_lock", None):
            with dict_lock:
                for handles in self._sending_transfers.values():
                    for handle, _, _, _ in handles.transfer_list:
                        if handle is not None:
                            self.nixl_wrapper.release_xfer_handle(handle)
            self._sending_transfers.clear()
        else:
            # error on init
            return

        # release all local handle
        for (
            src_xfer_side_handle_by_block
        ) in self.src_xfer_side_handles.values():  # block order
            for (
                src_xfer_side_handle
            ) in src_xfer_side_handle_by_block.values():  # multi-remote
                self.nixl_wrapper.release_dlist_handle(src_xfer_side_handle)
        self.src_xfer_side_handles.clear()

        # release all remote handle
        with self.net_info_lock:
            for dst_xfer_side_handles in self.dst_xfer_side_handles.values():
                for dst_xfer_side_handle in dst_xfer_side_handles:
                    self.nixl_wrapper.release_dlist_handle(dst_xfer_side_handle)
            self.dst_xfer_side_handles.clear()

            # remove all remote agent
            for remote_agents in self._remote_agents.values():
                for agent_name in remote_agents:
                    self.nixl_wrapper.remove_remote_agent(agent_name)
            self._remote_agents.clear()

        for desc in self._registered_descs:
            self.nixl_wrapper.deregister_memory(desc)
        self._registered_descs.clear()

    ############################################################
    # async loop for recv info from scheduler, run kv cache sending...
    ############################################################
    def connecter_loop(self):
        asyncio.set_event_loop(self._async_loop)
        torch.cuda.set_device(torch.distributed.get_rank())

        async def tasks():
            if self.use_host_buffer:
                await asyncio.wait(
                    [
                        asyncio.create_task(self._scheduler_listener()),
                        asyncio.create_task(self._run_write_kv_cache_host()),
                        asyncio.create_task(self._run_d2h()),
                        asyncio.create_task(self._run_get_finished()),
                        asyncio.create_task(self._clean_resource()),
                    ],
                    return_when=asyncio.ALL_COMPLETED,
                )
            else:
                await asyncio.wait(
                    [
                        asyncio.create_task(self._scheduler_listener()),
                        asyncio.create_task(self._run_write_kv_cache()),
                        asyncio.create_task(self._run_get_finished()),
                        asyncio.create_task(self._clean_resource()),
                    ],
                    return_when=asyncio.ALL_COMPLETED,
                )

        try:
            self._async_loop.run_until_complete(tasks())
        except:
            logger.fatal("An exception occurred in the connecter background thread")
            traceback.print_exc()

    async def _scheduler_listener(self) -> None:
        scheduler_addr = envs.VLLM_ILU_NIXL_SCHEDULER_ADDR
        with make_zmq_socket(
            path=scheduler_addr, ctx=self.ctx, socket_type=zmq.XSUB, linger=0
        ) as recv_from_scheduler:
            # subscribe to scheduler
            await recv_from_scheduler.send(b"\x01")
            if await recv_from_scheduler.recv() != b"READY":
                raise "NIXL worker listener failed to connect to scheduler"
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
                        block_info: tuple[str, list[list[int]], int] = remote_infos
                        req_id, block_ids = block_info[0], block_info[1]
                        remote_block_ids = [
                            np.array(block_id, dtype=np.int32)[None, :]
                            for block_id in block_ids
                        ]
                        self._reqs_to_blocks[req_id] = [remote_block_ids, True]

                        if self.use_host_buffer:
                            await self._reqs_to_send_host.put((req_id, None))
                        else:
                            await self._reqs_to_send.put((req_id, None, None))
                    elif op_type == "disconnect":
                        remote_engine_id = remote_infos
                        self.disconnected_engine[remote_engine_id] = time.perf_counter()
                    elif op_type == "connect":
                        node_meta = NodeMeta(**remote_infos)
                        per_layer_infos, per_rank_info = self._nixl_handshake(node_meta)
                        self._remote_agents_per_layer[node_meta.engine_id] = (
                            per_layer_infos
                        )
                        self._remote_agents_per_rank[node_meta.engine_id] = (
                            per_rank_info
                        )
                        logger.info(f"connect to remote engine: {node_meta.engine_id}")
                    else:
                        logger.info(
                            f"get a unsupported op: {op_type}, supported ops: block_ids, connect, disconnect, check_helper"
                        )
                except Exception as e:
                    logger.info("error on fun _scheduler_listener: %s", e)
                    traceback.print_exc()

    async def _clean_resource(self) -> None:
        # NOTE(gyf):
        # one bug:
        # kv_consumer still get info from remote after remote disconnected 300s
        # one memory leak:
        # kv_provider forward still not done after remote disconnected 300s
        all_disconnect_to_checks: dict[str, list[list[str, int]]] = {}
        try:
            while True:
                engine_req_to_checks = await self._reqs_to_checked.get()
                all_disconnect_to_checks.update(engine_req_to_checks)

                await asyncio.sleep(10.0)

                engine_to_rm = []
                for engine_id, reqs in all_disconnect_to_checks.items():
                    reqs_to_rm = []
                    for req in reqs:
                        req_id = req[0]
                        req[1] += 1
                        if req[1] == 5:
                            reqs_to_rm.append(req)
                            # afger 300s, the forward for this reqs should done, delete it.
                            if req_id in self._reqs_ready_to_send:
                                self._reqs_ready_to_send.pop(req_id)
                                continue
                            self._reqs_check_done_after_disconnected.add(req[0])

                    for req in reqs_to_rm:
                        reqs.remove(req)

                    if len(reqs) == 0:
                        engine_to_rm.append(engine_id)

                for engine_id in engine_to_rm:
                    all_disconnect_to_checks.pop(engine_id)
                    logger.info(
                        "remote engine id : %s removed during 300s afer get disconnected info, if any msgs come from that remote, this process will crash."
                        "the upper developer should ensure this",
                        engine_id,
                    )
                    self._engines_check_done_after_disconnected.add(engine_id)
        except Exception as e:
            logger.info("error when _clean_resource: %s, ", e)

    async def _run_get_finished(self) -> None:
        while True:
            done_req_ids: set[str] = set()
            check_list = list(self._sending_transfers.items())

            for req_id, handle_wrapper in check_list:
                if not handle_wrapper.ready_for_check:
                    continue
                in_progress = False
                handles = handle_wrapper.transfer_list
                for index, (handle, done, _xfer_stime, remote_engine_id) in enumerate(
                    handles
                ):
                    if done:
                        continue
                    xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                    if xfer_state == "DONE":
                        # Get telemetry from NIXL
                        res = self.nixl_wrapper.get_xfer_telemetry(handle)
                        self.xfer_stats.record_transfer(res)
                        handles[index][1] = True
                    elif xfer_state == "PROC":
                        in_progress = True
                        continue
                    else:
                        handles[index][1] = True
                        # transfer failed:
                        # for P, process is done
                        # for D, blocks marked as invalid after expired
                        logger.error(
                            "NIXL transfer failed for request %s with state %s. "
                            "Marking blocks as invalid.",
                            req_id,
                            xfer_state,
                        )
                        self.xfer_stats.record_failed_transfer()

                if not in_progress:
                    for handle, done, _, _ in handles:
                        if handle is not None:
                            self.nixl_wrapper.release_xfer_handle(handle)
                    done_req_ids.add(req_id)
                    self.finished_send_reqs.add(req_id)

            for req_id in done_req_ids:
                del self._sending_transfers[req_id]

            if self.stop_event.is_set():
                break

            # query status per 5ms
            await asyncio.sleep(0.005)

    async def _run_write_kv_cache(self):
        while True:
            req_meta, layer_index, event = await self._reqs_to_send.get()
            if event is None:
                req_id = req_meta
                if self._reqs_ready_to_send[req_id]:
                    remote_infos = self._reqs_to_blocks[req_id]
                    assert remote_infos[1]
                    remote_infos[1] = False

                    meta = self._reqs_ready_to_send[req_id][0][1]
                    meta.start_transfer = True
                    self.try_remap_block_ids(meta)
                    self.try_sending_kv_cache(req_id)
                else:
                    logger.info(
                        "please check the req_id or env: VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1"
                    )
            else:
                while not event.query():
                    await asyncio.sleep(0.0005)

                for req_id, meta in req_meta.items():
                    self._reqs_ready_to_send[req_id].append([layer_index, meta])
                    if remote_infos := self._reqs_to_blocks[req_id]:
                        if remote_infos[1]:
                            remote_infos[1] = False

                            meta.start_transfer = True
                            self.try_remap_block_ids(meta)
                        self.try_sending_kv_cache(req_id)

    async def _run_d2h(self):
        while True:
            (
                layer_index,
                kv_cache_bnd,
                slot_mapping_cpu,
                req_meta,
            ) = await self._reqs_to_send_d2h.get()
            index = layer_index - self.pp_layer_start
            if index == 0:
                # wait last step last pp layer h2h done
                while not self.d2h_manager.check_prestep_status():
                    # cpu buffer h2h copy is very quiky based on previous experience, so come to next step, the
                    # previous buffer h2h work should be done, but still need check here.
                    await asyncio.sleep(0.001)
                self.d2h_manager.bind_slotmapping(slot_mapping_cpu)

            event = self.d2h_manager.transfer_d2h_once(index, kv_cache_bnd)
            if event is not None:
                while not event.query():
                    await asyncio.sleep(0.005)
                self.d2h_manager.transfer_h2h_once(index)

            if index == self.num_layers_end_index and event is not None:
                self._reqs_to_send_host.put_nowait((req_meta, -1))

    async def _run_write_kv_cache_host(self):
        while True:
            req_meta, layer_index = await self._reqs_to_send_host.get()
            index = self.num_layers_end_index
            # wait h2h done
            while not self.d2h_manager.check_status(index):
                await asyncio.sleep(0.001)

            self.d2h_manager.reset()
            if layer_index is None:
                req_id = req_meta
                if self._reqs_ready_to_send[req_id]:
                    remote_infos = self._reqs_to_blocks[req_id]
                    assert remote_infos[1]
                    remote_infos[1] = False

                    meta = self._reqs_ready_to_send[req_id][0][1]
                    meta.start_transfer = True
                    self.try_remap_block_ids(meta)
                    self.try_sending_kv_cache(req_id)
                else:
                    logger.info(
                        "please check the req_id or env: VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1"
                    )
            else:
                for req_id, meta in req_meta.items():
                    self._reqs_ready_to_send[req_id].append([layer_index, meta])
                    if remote_infos := self._reqs_to_blocks[req_id]:
                        if remote_infos[1]:
                            remote_infos[1] = False

                            meta.start_transfer = True
                            self.try_remap_block_ids(meta)
                        self.try_sending_kv_cache(req_id)
