# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side PP extensions for the Ilu NIXL connector."""

import time
from collections import defaultdict
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import msgspec
import numpy as np
import torch
import zmq
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineTransferInfo,
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NixlHandshakePayload,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    _is_attention_spec,
    _is_ssm_spec,
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from .fp8_layout import use_fp8_ds_mla_wire_staging
from .metadata import IluNixlAgentMetadata, IluNixlTransferRegion
from .pp_utils import infer_pp_split

logger = init_logger(__name__)

_SEMANTIC_COMPAT_BACKEND = "ILU_NIXL_SEMANTIC_REGIONS_V1"


@dataclass(frozen=True)
class _HmaTransferRoute:
    local_group_idx: int
    remote_group_idx: int
    is_ssm: bool
    local_desc_region_ids: np.ndarray
    remote_desc_region_ids: np.ndarray
    raw_local_region_ids: tuple[int, ...]
    raw_remote_region_ids: tuple[int, ...]


class _BlocksFirstTopologyBackend:
    """Topology-only view for Iluvatar's legacy K/V-first FA cache."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)


class IluNixlWorkerMixin:
    """Mixin that makes the official NIXL worker homogeneous-PP aware."""

    def __init__(self, *args, **kwargs):
        vllm_config = args[0] if args else kwargs.get("vllm_config")
        kv_cache_config = args[2] if len(args) > 2 else kwargs.get("kv_cache_config")
        parallel_config = getattr(vllm_config, "parallel_config", None)
        original_pp_size = getattr(parallel_config, "pipeline_parallel_size", 1)
        is_pp = original_pp_size > 1
        is_consumer = (
            vllm_config is not None
            and vllm_config.kv_transfer_config is not None
            and vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        requires_hma = (
            vllm_config is not None
            and kv_cache_config is not None
            and not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(group.kv_cache_spec, FullAttentionSpec)
                for group in kv_cache_config.kv_cache_groups
            )
        )

        # Upstream rejects both consumer PP and PP with HMA during worker
        # construction. IluNixl implements PP-aware notification accounting and
        # region metadata for those cases, so hide PP only while constructing
        # the upstream worker, then restore the real topology.
        mask_pp_during_init = is_pp and (is_consumer or requires_hma)

        # Upstream KV-cache registration can handle uniform PP layouts. Only
        # HMA needs IluNixl's semantic PP/HMA registration path below.
        self._mask_pp_during_register = is_pp and requires_hma
        if mask_pp_during_init:
            try:
                parallel_config.pipeline_parallel_size = 1
            except Exception:
                object.__setattr__(parallel_config, "pipeline_parallel_size", 1)
        try:
            super().__init__(*args, **kwargs)
        finally:
            if mask_pp_during_init:
                try:
                    parallel_config.pipeline_parallel_size = original_pp_size
                except Exception:
                    object.__setattr__(
                        parallel_config, "pipeline_parallel_size", original_pp_size
                    )

        if mask_pp_during_init:
            self.pp_size = original_pp_size
        self.fp8_wire_staging_enabled = use_fp8_ds_mla_wire_staging(vllm_config)
        self.fp8_wire_staging = None
        self.compute_device_kv_caches: dict[str, torch.Tensor] = {}
        self._virtual_packed_regions = False
        self._semantic_region_engines: set[str] = set()
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_world_size = get_pp_group().world_size
        self.pp_split = envs.VLLM_PP_LAYER_PARTITION or infer_pp_split(self.vllm_config)
        assert len(self.pp_split.split(",")) == self.pp_world_size, (
            "PP split count must equal pp_size"
        )
        self.layer_region_ids: dict[str, list[int]] = {}
        self.region_ids_by_group: list[np.ndarray] = []
        self.transfer_regions: list[IluNixlTransferRegion] = []
        self.local_region_ids_by_group: dict[str, dict[int, list[np.ndarray]]] = {}
        self.remote_region_ids_by_group: dict[str, dict[int, list[np.ndarray]]] = {}
        self._mamba_hma_routes: dict[str, dict[int, list[_HmaTransferRoute]]] = {}
        self._remote_fa_num_regions: dict[str, dict[int, int]] = {}
        self._selected_ssm_remote_region_ids: dict[str, dict[int, list[int]]] = {}
        self._remote_pp_size_by_engine: dict[str, int] = {}
        self._remote_pp_split_by_engine: dict[str, str] = {}
        self._pp_mamba_local_xfer_handle: int | None = None

        if self.fp8_wire_staging_enabled and self.enforce_compat_hash:
            raise ValueError(
                "IluNixl FP8 DS MLA wire staging requires "
                "kv_connector_extra_config.enforce_handshake_compat=false "
                "until both endpoints hash the virtual wire layout"
            )
        if self.fp8_wire_staging_enabled and self.use_host_buffer:
            raise ValueError(
                "IluNixl FP8 DS MLA wire staging is incompatible with use_host_buffer"
            )
        if (
            self.fp8_wire_staging_enabled
            and self.vllm_config.speculative_config is not None
        ):
            raise ValueError(
                "IluNixl FP8 DS MLA wire staging does not support speculative "
                "prefill-side execution"
            )

    @staticmethod
    def _pp_layer_range(
        pp_split: str,
        pp_rank: int,
        layer_region_ids: dict[str, list[int]] | None = None,
    ) -> tuple[int, int]:
        split = [int(x) for x in pp_split.split(",")]
        start = sum(split[:pp_rank])
        end = start + split[pp_rank]
        if pp_rank == len(split) - 1 and layer_region_ids:
            for layer_name in layer_region_ids:
                with suppress(AssertionError, ValueError):
                    end = max(end, extract_layer_index(layer_name) + 1)
        return start, end

    def _compute_compatibility_hash(
        self, _cross_layers_blocks: bool | None = None
    ) -> str:
        # The community hash uses the first backend in this worker's local PP
        # slice. Multi-backend models can therefore hash differently across
        # heterogeneous PP partitions even when overlapping KV regions are
        # byte-compatible. IluNixl validates the selected semantic regions and
        # their block lengths during add_remote_agent, so use a stable connector
        # capability here instead of a stage-local backend name.
        return compute_nixl_compatibility_hash(
            self.vllm_config,
            _SEMANTIC_COMPAT_BACKEND,
            transfer_mode=getattr(self, "_TRANSFER_MODE", "pull"),
        )

    def _cross_layers_blocks_enabled(self) -> bool:
        transfer_config = getattr(self, "kv_transfer_config", None)
        if transfer_config is None:
            return False
        value = transfer_config.get_from_extra_config(
            "enable_cross_layers_blocks", False
        )
        return str(value).lower() == "true"

    def _should_split_virtual_region(self) -> bool:
        """Keep compatibility with pre-0.29 topology metadata.

        vLLM 0.31 emits Mamba sub-descriptors itself and removed the virtual
        split flag. Only an older topology that explicitly advertises the
        split may request it, and never for Mamba.
        """
        return bool(
            self.transfer_topo is not None
            and getattr(self.transfer_topo, "virtually_split_kv_in_blocks", False)
            and not self._has_mamba
        )

    def _new_transfer_topology(
        self,
        *,
        attn_backends=None,
        tensor_shape=None,
        is_mla: bool | None = None,
        is_mamba: bool | None = None,
    ) -> TransferTopology:
        topology = TransferTopology(
            tp_rank=getattr(self, "transfer_tp_rank", self.tp_rank),
            tp_size=getattr(self, "transfer_tp_size", self.world_size),
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla if is_mla is None else is_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=attn_backends or self.attn_backends,
            dcp_size=getattr(self, "dcp_size", 1),
            tensor_shape=tensor_shape,
            is_mamba=self._has_mamba if is_mamba is None else is_mamba,
        )
        self.compat_hash = self._compute_compatibility_hash(
            self._cross_layers_blocks_enabled()
        )
        return topology

    @classmethod
    def _target_pp_ranks(
        cls,
        local_pp_split: str,
        local_pp_rank: int,
        remote_pp_split: str,
    ) -> list[int]:
        local_start, local_end = cls._pp_layer_range(local_pp_split, local_pp_rank)
        remote_split = [int(x) for x in remote_pp_split.split(",")]
        target_pp_ranks: list[int] = []
        remote_start = 0
        for remote_pp_rank, num_layers in enumerate(remote_split):
            remote_end = remote_start + num_layers
            if max(local_start, remote_start) < min(local_end, remote_end):
                target_pp_ranks.append(remote_pp_rank)
            remote_start = remote_end
        return target_pp_ranks

    @classmethod
    def _overlap_pp_count(
        cls,
        producer_pp_rank: int,
        producer_pp_split: str,
        consumer_pp_split: str,
    ) -> int:
        return len(
            cls._target_pp_ranks(
                producer_pp_split,
                producer_pp_rank,
                consumer_pp_split,
            )
        )

    @staticmethod
    def _cache_kind_from_layer_name(layer_name: str) -> str:
        if ".swa_cache" in layer_name:
            return "swa_cache"
        if ".indexer.k_cache" in layer_name:
            return "indexer_k_cache"
        if ".indexer.tail_cache" in layer_name:
            return "kpool_tail_cache"
        if ".indexer.compressor" in layer_name:
            return "indexer_compressor"
        if ".compressor.state_cache" in layer_name:
            return "compressor_state_cache"
        if ".ilu_deepseek_v4_multi_head_latent_attention" in layer_name:
            return "main_mla"
        if layer_name.endswith(".mla_attn") or layer_name.endswith(".attn"):
            return "main_mla"
        if layer_name.endswith(".linear_attn") or layer_name.endswith(".self_attn"):
            return "mamba_cache"
        return "unknown"

    @classmethod
    def _transfer_identity(cls, layer_name: str, group_idx: int) -> str:
        try:
            layer_idx = extract_layer_index(layer_name)
        except (AssertionError, ValueError):
            layer_idx = -1
        kind = cls._cache_kind_from_layer_name(layer_name)
        return f"{layer_idx}:{kind}:group{group_idx}"

    @staticmethod
    def _regions_by_id(
        transfer_regions: list[IluNixlTransferRegion] | None,
    ) -> dict[int, IluNixlTransferRegion]:
        return {region.region_id: region for region in transfer_regions or []}

    @classmethod
    def _region_ids_for_layer_range(
        cls,
        layer_region_ids: dict[str, list[int]],
        layer_start: int,
        layer_end: int,
    ) -> list[int]:
        selected: list[int] = []
        layers: list[tuple[int, str, list[int]]] = []
        for layer_name, region_ids in layer_region_ids.items():
            try:
                layer_idx = extract_layer_index(layer_name)
            except (AssertionError, ValueError):
                continue
            if layer_start <= layer_idx < layer_end:
                layers.append((layer_idx, layer_name, region_ids))
        for _, _, region_ids in sorted(layers):
            for region_id in region_ids:
                if region_id not in selected:
                    selected.append(region_id)
        return selected

    def _actual_to_virtual_region_ids(self) -> dict[int, list[int]]:
        mapping: dict[int, list[int]] = {}
        next_region_id = 0
        for actual_region_id in range(len(self.block_len_per_layer)):
            mapping[actual_region_id] = [next_region_id]
            next_region_id += 1
            if self._should_split_virtual_region() and not self._is_region_replicated(
                actual_region_id
            ):
                mapping[actual_region_id].append(next_region_id)
                next_region_id += 1
        return mapping

    def _build_layer_region_ids_from_kv_caches(
        self, kv_caches: dict[str, Any]
    ) -> dict[str, list[int]]:
        assert self.transfer_topo is not None
        xfer_buffers = self.host_xfer_buffers if self.use_host_buffer else kv_caches
        base_addrs = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        base_addr_to_actual_region = {
            base_addr: region_id for region_id, base_addr in enumerate(base_addrs)
        }
        actual_to_virtual = self._actual_to_virtual_region_ids()

        layer_region_ids: dict[str, list[int]] = defaultdict(list)
        for layer_name, cache in xfer_buffers.items():
            # Layers with no spec of their own share another layer's KV cache;
            # upstream `register_kv_caches` skips them, so no region is emitted.
            if self._layer_specs.get(layer_name) is None:
                continue
            actual_region_id = base_addr_to_actual_region.get(cache.data_ptr())
            if actual_region_id is None:
                continue
            for region_id in actual_to_virtual[actual_region_id]:
                if region_id not in layer_region_ids[layer_name]:
                    layer_region_ids[layer_name].append(region_id)
        self._validate_layer_region_ids(
            layer_region_ids, xfer_buffers, actual_to_virtual
        )
        return dict(layer_region_ids)

    def _validate_layer_region_ids(
        self,
        layer_region_ids: dict[str, list[int]],
        xfer_buffers: dict[str, Any],
        actual_to_virtual: dict[int, list[int]],
    ) -> None:
        missing = [
            layer_name
            for layer_name in xfer_buffers
            if self._layer_specs.get(layer_name) is not None
            and not layer_region_ids.get(layer_name)
        ]
        if missing:
            raise RuntimeError(
                "IluNixl failed to map layers onto registered NIXL regions. "
                "Those layers would never be transferred. "
                f"layers_without_region={missing}, "
                f"num_regions={self.num_regions}"
            )
        num_virtual_regions = sum(len(ids) for ids in actual_to_virtual.values())
        covered = {region_id for ids in layer_region_ids.values() for region_id in ids}
        if len(covered) != num_virtual_regions:
            logger.warning(
                "IluNixl region mapping does not cover every registered NIXL "
                "region: mapped=%d, expected=%d. Uncovered regions are excluded "
                "from PP/HMA transfer metadata.",
                len(covered),
                num_virtual_regions,
            )

    def _build_region_ids_by_group(
        self, layer_region_ids: dict[str, list[int]]
    ) -> list[np.ndarray]:
        region_ids_by_group: list[np.ndarray] = []
        for group in self.kv_cache_config.kv_cache_groups:
            group_region_ids: list[int] = []
            for layer_name in group.layer_names:
                group_region_ids.extend(layer_region_ids.get(layer_name, []))
            region_ids_by_group.append(
                np.asarray(list(dict.fromkeys(group_region_ids)), dtype=np.int64)
            )
        if not region_ids_by_group:
            region_ids_by_group = [np.arange(self.num_regions, dtype=np.int64)]
        return region_ids_by_group

    @staticmethod
    def _filter_group_region_ids(
        group_region_ids: list[list[int]] | list[np.ndarray],
        selected_region_ids: list[int],
        *,
        compact: bool,
    ) -> list[np.ndarray]:
        selected = set(selected_region_ids)
        compact_map = {
            region_id: compact_id
            for compact_id, region_id in enumerate(selected_region_ids)
        }
        result: list[np.ndarray] = []
        for group in group_region_ids:
            if compact:
                ids = [compact_map[int(r)] for r in group if int(r) in compact_map]
            else:
                ids = [int(r) for r in group if int(r) in selected]
            result.append(np.asarray(ids, dtype=np.int64))
        return result

    def _block_len_for_virtual_region(self, region_id: int) -> int:
        next_region_id = 0
        for actual_region_id, block_len in enumerate(self.block_len_per_layer):
            if region_id == next_region_id:
                if (
                    self._should_split_virtual_region()
                    and not self._is_region_replicated(actual_region_id)
                ):
                    return block_len // 2
                return block_len
            next_region_id += 1
            if self._should_split_virtual_region() and not self._is_region_replicated(
                actual_region_id
            ):
                if region_id == next_region_id:
                    return block_len // 2
                next_region_id += 1
        return self.block_len_per_layer[region_id]

    def _build_hma_transfer_regions(
        self,
        layer_region_ids: dict[str, list[int]],
        region_ids_by_group: list[np.ndarray],
    ) -> list[IluNixlTransferRegion]:
        region_to_layers: dict[int, list[str]] = defaultdict(list)
        for layer_name, region_ids in layer_region_ids.items():
            for region_id in region_ids:
                region_to_layers[int(region_id)].append(layer_name)

        region_to_groups: dict[int, list[int]] = defaultdict(list)
        for group_idx, group_region_ids in enumerate(region_ids_by_group):
            for region_id in group_region_ids:
                region_to_groups[int(region_id)].append(group_idx)

        layer_to_groups = self._layer_to_groups(self.kv_cache_config)

        transfer_regions: list[IluNixlTransferRegion] = []
        for region_id in sorted(region_to_layers):
            shared_by = sorted(set(region_to_layers[region_id]))
            layer_indices: list[int] = []
            cache_kinds: set[str] = set()
            identities: set[str] = set()
            for layer_name in shared_by:
                with suppress(AssertionError, ValueError):
                    layer_indices.append(extract_layer_index(layer_name))
                cache_kinds.add(self._cache_kind_from_layer_name(layer_name))
                for group_idx in layer_to_groups.get(layer_name, []):
                    identities.add(self._transfer_identity(layer_name, group_idx))
            transfer_regions.append(
                IluNixlTransferRegion(
                    region_id=region_id,
                    block_len=self._block_len_for_virtual_region(region_id),
                    shared_by=shared_by,
                    layer_indices=sorted(set(layer_indices)),
                    group_ids=sorted(set(region_to_groups.get(region_id, []))),
                    cache_kinds=sorted(cache_kinds),
                    identities=sorted(identities),
                    physical_backing_id=region_id,
                    byte_offset=0,
                    block_stride=self._block_len_for_virtual_region(region_id),
                )
            )
        return transfer_regions

    @staticmethod
    def _layer_specs_for_config(kv_cache_config) -> dict[str, Any]:
        specs: dict[str, Any] = {}
        for group in kv_cache_config.kv_cache_groups:
            group_spec = group.kv_cache_spec
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                specs.update(group_spec.kv_cache_specs)
            else:
                specs.update(dict.fromkeys(group.layer_names, group_spec))
        return specs

    @staticmethod
    def _layer_to_groups(kv_cache_config) -> dict[str, list[int]]:
        result: dict[str, list[int]] = defaultdict(list)
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in group.layer_names:
                result[layer_name].append(group_idx)
        return dict(result)

    @staticmethod
    def _is_packed_layout(kv_cache_config, kv_caches: dict[str, Any]) -> bool:
        tensors = getattr(kv_cache_config, "kv_cache_tensors", ())
        if not tensors or not kv_caches:
            return False
        if not any(getattr(tensor, "block_stride", 0) > 0 for tensor in tensors):
            return False
        storage_ptrs = {
            cache.untyped_storage().data_ptr() for cache in kv_caches.values()
        }
        return len(storage_ptrs) == 1

    @staticmethod
    def _virtual_slot_desc(
        backing_base: int,
        block_id: int,
        block_stride: int,
        slot_offset: int,
        page_size: int,
        device_id: int,
    ) -> tuple[int, int, int]:
        if block_stride <= 0 or page_size <= 0 or page_size > block_stride:
            raise ValueError(
                "Packed virtual slot stride and page size must be positive "
                "and the page must fit in one block"
            )
        if slot_offset < 0:
            raise ValueError(
                f"Packed virtual slot offset must be non-negative: offset={slot_offset}"
            )
        # Block-outer: slot_offset is an intra-block window offset, so the page
        # must fit in [slot_offset, block_stride).
        # Layer-outer (LBHNC): slot_offset is the absolute layer base and may
        # be >> block_stride (and need not be stride-aligned when mixed pages
        # share a pool). Addr = backing + slot_offset + block_id * stride.
        if slot_offset < block_stride and slot_offset + page_size > block_stride:
            raise ValueError(
                "Packed virtual slot lies outside its block stride: "
                f"offset={slot_offset}, page_size={page_size}, "
                f"block_stride={block_stride}"
            )
        return (
            backing_base + block_id * block_stride + slot_offset,
            page_size,
            device_id,
        )

    def _build_packed_virtual_layout(
        self,
        kv_caches: dict[str, torch.Tensor],
        layout_config,
        block_id_config,
    ) -> tuple[
        dict[str, list[int]],
        list[np.ndarray],
        list[IluNixlTransferRegion],
        list[bool],
    ]:
        specs = self._layer_specs_for_config(layout_config)
        layer_to_groups = self._layer_to_groups(block_id_config)

        first_cache = next(iter(kv_caches.values()))
        backing_ptr = first_cache.untyped_storage().data_ptr()
        backing_nbytes = first_cache.untyped_storage().nbytes()
        layer_region_ids: dict[str, list[int]] = defaultdict(list)
        transfer_regions: list[IluNixlTransferRegion] = []
        region_is_mla: list[bool] = []

        for tensor_config in layout_config.kv_cache_tensors:
            shared_by = [
                layer_name
                for layer_name in tensor_config.layers
                if layer_name in kv_caches
            ]
            if not shared_by:
                continue
            tensor_stride = int(tensor_config.block_stride)
            if tensor_stride <= 0:
                raise ValueError(
                    "Packed IluNixl layout contains a non-packed tensor slot"
                )
            if int(tensor_config.size) != backing_nbytes:
                raise ValueError(
                    "Packed IluNixl slot size does not cover the registered "
                    f"backing: {tensor_config.size} != {backing_nbytes}"
                )

            # Block-outermost packing places layer l at
            # offset + l * layer_stride. tensor.offset is only layer 0.
            # layer_stride 0 keeps overlaid slots on that same base.
            layer_positions = {
                layer_name: index
                for index, layer_name in enumerate(tensor_config.layers)
            }
            layer_stride = int(getattr(tensor_config, "layer_stride", 0))
            tensor_offset = int(tensor_config.offset)
            num_blocks = int(layout_config.num_blocks)

            def layer_byte_offset(
                layer_name: str,
                *,
                _tensor_offset: int = tensor_offset,
                _layer_positions: dict[str, int] = layer_positions,
                _layer_stride: int = layer_stride,
            ) -> int:
                return _tensor_offset + _layer_positions[layer_name] * _layer_stride

            missing_specs = [
                layer_name for layer_name in shared_by if layer_name not in specs
            ]
            if missing_specs:
                raise ValueError(
                    "Packed IluNixl layout has no KV cache spec for layers: "
                    + ", ".join(sorted(missing_specs))
                )

            # Main K/V and the indexer page are different sizes (e.g. 262144 vs
            # 32768 on MiniMax). Each slot keeps its own block stride. Under
            # layer-outer (LBHNC) the shared pool is N * sum(pages), so it need
            # not be a multiple of one tensor's block_stride * N; only each
            # layer's last page must fit. Block-outer / overlaid slots still
            # require the stride to tile the backing.
            block_span = tensor_stride * num_blocks
            is_block_outer = layer_stride == 0 or layer_stride < tensor_stride
            if is_block_outer and (block_span <= 0 or backing_nbytes % block_span != 0):
                raise ValueError(
                    "Packed IluNixl backing size is not a whole number of "
                    "num_blocks * block_stride: "
                    f"{backing_nbytes} != N * {num_blocks} * {tensor_stride}"
                )
            if num_blocks <= 0 or block_span <= 0:
                raise ValueError(
                    "Packed IluNixl layout has non-positive num_blocks or "
                    f"block_stride: N={num_blocks}, stride={tensor_stride}"
                )
            for layer_name in shared_by:
                page_size = int(specs[layer_name].page_size_bytes)
                last_end = (
                    layer_byte_offset(layer_name)
                    + (num_blocks - 1) * tensor_stride
                    + page_size
                )
                if last_end > backing_nbytes:
                    raise ValueError(
                        "Packed IluNixl layer page exceeds registered backing: "
                        f"{layer_name} end={last_end} > backing={backing_nbytes}"
                    )

            for layer_name in shared_by:
                cache = kv_caches[layer_name]
                if cache.untyped_storage().data_ptr() != backing_ptr:
                    raise ValueError(
                        f"Packed cache layer {layer_name} uses another backing"
                    )
                slot_offset = layer_byte_offset(layer_name)
                actual_offset = cache.data_ptr() - backing_ptr
                actual_stride = cache.stride(0) * cache.element_size()
                if actual_offset != slot_offset or actual_stride != tensor_stride:
                    raise ValueError(
                        "Packed cache view disagrees with KV planning metadata "
                        f"for {layer_name}: offset={actual_offset}/{slot_offset}, "
                        f"stride={actual_stride}/{tensor_stride}"
                    )

            semantic_slots: dict[tuple[int, bool, str, int], list[str]] = defaultdict(
                list
            )
            for layer_name in shared_by:
                spec = specs[layer_name]
                is_mla_region = isinstance(
                    spec, (MLAAttentionSpec, SlidingWindowMLASpec)
                )
                cache_kind = self._cache_kind_from_layer_name(layer_name)
                semantic_slots[
                    (
                        int(spec.page_size_bytes),
                        is_mla_region,
                        cache_kind,
                        layer_byte_offset(layer_name),
                    )
                ].append(layer_name)

            for (
                page_size,
                is_mla_region,
                cache_kind,
                slot_offset,
            ), region_layers in sorted(semantic_slots.items()):
                self._virtual_slot_desc(
                    backing_ptr,
                    0,
                    tensor_stride,
                    slot_offset,
                    page_size,
                    max(first_cache.get_device(), 0),
                )

                region_id = len(transfer_regions)
                group_ids = sorted(
                    {
                        group_idx
                        for layer_name in region_layers
                        for group_idx in layer_to_groups.get(layer_name, [])
                    }
                )
                if not group_ids:
                    raise ValueError(
                        "Packed virtual slot has no block-ID owner group: "
                        + ", ".join(region_layers)
                    )
                identities = sorted(
                    {
                        self._transfer_identity(layer_name, group_idx)
                        for layer_name in region_layers
                        for group_idx in layer_to_groups.get(layer_name, [])
                    }
                )
                layer_indices: list[int] = []
                for layer_name in region_layers:
                    with suppress(AssertionError, ValueError):
                        layer_indices.append(extract_layer_index(layer_name))
                    layer_region_ids[layer_name].append(region_id)
                transfer_regions.append(
                    IluNixlTransferRegion(
                        region_id=region_id,
                        block_len=page_size,
                        shared_by=sorted(region_layers),
                        layer_indices=sorted(set(layer_indices)),
                        group_ids=group_ids,
                        cache_kinds=[cache_kind],
                        identities=identities,
                        physical_backing_id=0,
                        byte_offset=slot_offset,
                        block_stride=tensor_stride,
                    )
                )
                region_is_mla.append(is_mla_region)

        if not transfer_regions:
            raise ValueError("Packed IluNixl layout has no local semantic slots")

        region_ids_by_group = self._build_region_ids_by_group(dict(layer_region_ids))
        return (
            dict(layer_region_ids),
            region_ids_by_group,
            transfer_regions,
            region_is_mla,
        )

    def _set_handshake_metadata(
        self,
        *,
        kv_cache_layout: str | None = None,
        memory_type: str | None = None,
    ) -> None:
        agent_metadata = IluNixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=(
                self.kv_caches_base_addr[self.engine_id][self.tp_rank]
            ),
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            block_strides=self.block_stride_per_layer,
            kv_cache_layout=kv_cache_layout or self.kv_cache_layout,
            block_size=self.block_size,
            ssm_sizes=self._mamba_ssm_size,
            attn_backend_name=self.backend_name,
            physical_blocks_per_logical_kv_block=(
                self._physical_blocks_per_logical_kv_block
            ),
            region_num_blocks=getattr(self, "region_num_blocks", None),
            region_group_ids=getattr(self, "region_group_ids", None),
            region_names=getattr(self, "region_names", None),
            region_mem_types=getattr(self, "region_mem_types", None),
            dcp_size=getattr(self, "dcp_size", 1),
            pcp_size=getattr(self, "pcp_size", 1),
            pp_split=self.pp_split,
            pp_rank=self.pp_rank,
            tp_rank=self.tp_rank,
            layer_region_ids=self.layer_region_ids,
            region_ids_by_group=[
                region_ids.tolist() for region_ids in self.region_ids_by_group
            ],
            transfer_regions=self.transfer_regions,
            memory_type=memory_type or self.nixl_memory_type,
        )
        assert self.compat_hash is not None
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=msgspec.msgpack.encode(agent_metadata),
        )

    def _finish_memory_registration(
        self,
        kv_caches: dict[str, torch.Tensor],
        caches_data: list[tuple[int, int, int, str]],
        base_addresses: list[int],
        *,
        memory_type: str | None = None,
    ) -> None:
        memory_type = memory_type or self.nixl_memory_type
        # Packed and K/V-first registration skip the community per-region
        # loop. register_local_xfer_handler still reads region_mem_types[0].
        if not self.region_mem_types:
            self.region_mem_types = [memory_type] * self.num_regions
        self.dst_region_mem_types[self.engine_id] = self.region_mem_types
        descs = self.nixl_wrapper.get_reg_descs(caches_data, memory_type)
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        self._registered_descs.append(descs)
        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = base_addresses
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        self.src_xfer_handles_by_block_size[self.block_size], self.src_blocks_data = (
            self.register_local_xfer_handler(self.block_size)
        )
        self.device_kv_caches = kv_caches
        self._set_handshake_metadata(memory_type=memory_type)

    def _uses_kv_first_fa_layout(self, kv_caches: dict[str, torch.Tensor]) -> bool:
        if self.use_mla or self._has_mamba or not kv_caches:
            return False
        registered = [
            cache
            for layer_name, cache in kv_caches.items()
            if self._layer_specs.get(layer_name) is not None
        ]
        return bool(registered) and all(
            cache.ndim == 5
            and cache.shape[0] in (2, 3)
            and cache.shape[1] == self.num_blocks
            for cache in registered
        )

    def _register_kv_first_fa_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register legacy ``[K/V, blocks, heads, block, dim]`` caches."""
        if self.use_host_buffer:
            raise ValueError(
                "Iluvatar K/V-first KV caches do not support a host transfer buffer"
            )
        if self._physical_blocks_per_logical_kv_block != 1:
            raise ValueError(
                "Iluvatar K/V-first KV caches require matching logical and "
                "physical kernel block sizes"
            )

        first_cache = next(
            cache
            for layer_name, cache in kv_caches.items()
            if self._layer_specs.get(layer_name) is not None
        )
        self.transfer_topo = self._new_transfer_topology(
            attn_backends=[_BlocksFirstTopologyBackend],
            tensor_shape=first_cache.shape[1:],
            is_mamba=False,
            is_mla=False,
        )
        if self._cross_layers_blocks_enabled():
            raise ValueError(
                "Iluvatar K/V-first registration does not support cross-layer caches"
            )

        layer_to_groups = self._layer_to_groups(self.kv_cache_config)

        caches_data: list[tuple[int, int, int, str]] = []
        base_addresses: list[int] = []
        layer_region_ids: dict[str, list[int]] = defaultdict(list)
        transfer_regions: list[IluNixlTransferRegion] = []
        self._region_is_mla = []

        for layer_name, cache in kv_caches.items():
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                continue
            if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                layer_spec = layer_spec.kv_cache_specs[layer_name]

            component_count = int(cache.shape[0])
            if layer_spec.page_size_bytes % component_count:
                raise ValueError(
                    f"KV page for {layer_name} cannot be split into "
                    f"{component_count} K/V-first components"
                )
            expected_block_len = layer_spec.page_size_bytes // component_count
            for component_idx in range(component_count):
                component = cache[component_idx]
                block_len = component.stride(0) * component.element_size()
                if block_len != expected_block_len:
                    raise ValueError(
                        "K/V-first cache stride disagrees with KV spec for "
                        f"{layer_name} component {component_idx}: "
                        f"{block_len} != {expected_block_len}"
                    )
                base_addr = component.data_ptr()
                device_id = max(component.get_device(), 0)
                region_id = len(base_addresses)
                region_size = self.num_blocks * block_len
                caches_data.append((base_addr, region_size, device_id, ""))
                base_addresses.append(base_addr)
                layer_region_ids[layer_name].append(region_id)
                group_ids = layer_to_groups.get(layer_name, [])
                cache_kind = f"full_attn_component_{component_idx}"
                try:
                    layer_idx = extract_layer_index(layer_name)
                except (AssertionError, ValueError):
                    layer_idx = -1
                transfer_regions.append(
                    IluNixlTransferRegion(
                        region_id=region_id,
                        block_len=block_len,
                        shared_by=[layer_name],
                        layer_indices=[layer_idx] if layer_idx >= 0 else [],
                        group_ids=group_ids,
                        cache_kinds=[cache_kind],
                        identities=[
                            f"{layer_idx}:{cache_kind}:group{group_idx}"
                            for group_idx in group_ids
                        ],
                        physical_backing_id=region_id,
                        byte_offset=0,
                        block_stride=block_len,
                    )
                )
                self._region_is_mla.append(False)
                self.device_id = device_id

        self.layer_region_ids = dict(layer_region_ids)
        self.region_ids_by_group = self._build_region_ids_by_group(
            self.layer_region_ids
        )
        self.transfer_regions = transfer_regions
        self.block_len_per_layer = [
            region.block_len for region in self.transfer_regions
        ]
        self.block_stride_per_layer = [
            region.block_stride for region in self.transfer_regions
        ]
        self.num_regions = len(self.transfer_regions)
        self.num_descs = self.num_regions * self.num_blocks
        self._finish_memory_registration(kv_caches, caches_data, base_addresses)

    def _register_virtual_packed_kv_cache(
        self,
        kv_caches: dict[str, torch.Tensor],
        layout_config,
        block_id_config,
    ) -> None:
        if self.use_host_buffer:
            raise ValueError(
                "Packed virtual IluNixl regions cannot use a host transfer buffer"
            )
        if self._physical_blocks_per_logical_kv_block != 1:
            raise ValueError(
                "Packed virtual IluNixl regions require logical and physical "
                "kernel block sizes to match"
            )
        if layout_config.num_blocks != self.num_blocks:
            raise ValueError(
                "Packed layout and connector block counts differ: "
                f"{layout_config.num_blocks} != {self.num_blocks}"
            )

        (
            self.layer_region_ids,
            self.region_ids_by_group,
            self.transfer_regions,
            self._region_is_mla,
        ) = self._build_packed_virtual_layout(kv_caches, layout_config, block_id_config)
        self.block_len_per_layer = [
            region.block_len for region in self.transfer_regions
        ]
        self.block_stride_per_layer = [
            region.block_stride for region in self.transfer_regions
        ]
        self.num_regions = len(self.transfer_regions)
        self.num_descs = self.num_regions * self.num_blocks
        self._virtual_packed_regions = True

        self.transfer_topo = self._new_transfer_topology()
        if self._cross_layers_blocks_enabled():
            raise ValueError(
                "Packed virtual IluNixl regions do not support uniform "
                "cross-layer KV cache allocation"
            )

        storage = next(iter(kv_caches.values())).untyped_storage()
        base_addr = storage.data_ptr()
        if storage.device.type == "cpu":
            backing_memory_type = "DRAM"
            device_id = 0
        else:
            backing_memory_type = self.nixl_memory_type
            device_id = storage.device.index
            if device_id is None:
                raise ValueError(
                    "Packed virtual IluNixl device backing has no device index"
                )
        self.nixl_memory_type = backing_memory_type
        self.device_id = device_id
        physical_data = [(base_addr, storage.nbytes(), device_id, "")]
        self._finish_memory_registration(
            kv_caches,
            physical_data,
            [base_addr],
            memory_type=backing_memory_type,
        )

    def register_cross_layers_kv_caches(self, kv_cache: torch.Tensor) -> None:
        if self.fp8_wire_staging_enabled:
            raise ValueError(
                "IluNixl FP8 DS MLA wire staging does not support uniform "
                "cross-layer KV cache allocation"
            )
        super().register_cross_layers_kv_caches(kv_cache)

    def register_kv_caches(self, kv_caches: dict[str, Any]):
        if self.fp8_wire_staging_enabled:
            from .fp8_staging import FP8DSMLAWireStaging

            self.compute_device_kv_caches = kv_caches
            staging = FP8DSMLAWireStaging(
                self.vllm_config,
                self.kv_cache_config,
                kv_caches,
            )
            self.fp8_wire_staging = staging
            self._register_virtual_packed_kv_cache(
                staging.wire_caches,
                staging.wire_config,
                self.kv_cache_config,
            )
            return

        # Kernel block size can be smaller than the scheduler block (hybrid
        # models: 256 vs 64). Packed virtual descriptors are one slot per
        # logical block, so that case uses the community registration, which
        # splits each logical page into kernel pages.
        if (
            self._is_packed_layout(self.kv_cache_config, kv_caches)
            and self._physical_blocks_per_logical_kv_block == 1
        ):
            self._register_virtual_packed_kv_cache(
                kv_caches,
                self.kv_cache_config,
                self.kv_cache_config,
            )
            return

        if self._uses_kv_first_fa_layout(kv_caches):
            self._register_kv_first_fa_caches(kv_caches)
            return

        if self._mask_pp_during_register:
            original_pp_size = self.pp_size
            # vLLM 0.24's generic NIXL register path assumes uniform regions per
            # PP layer slice. HMA violates that assumption, and IluNixl builds
            # semantic PP/HMA region metadata below instead.
            self.pp_size = 1
            try:
                super().register_kv_caches(kv_caches)
            finally:
                self.pp_size = original_pp_size
        else:
            super().register_kv_caches(kv_caches)

        self.layer_region_ids = self._build_layer_region_ids_from_kv_caches(kv_caches)
        self.region_ids_by_group = self._build_region_ids_by_group(
            self.layer_region_ids
        )
        self.transfer_regions = self._build_hma_transfer_regions(
            self.layer_region_ids, self.region_ids_by_group
        )
        assert self.transfer_topo is not None
        self.compat_hash = self._compute_compatibility_hash(
            self._cross_layers_blocks_enabled()
        )
        self._set_handshake_metadata(
            kv_cache_layout=(
                self.kv_cache_layout
                if not self.use_host_buffer
                else self.host_buffer_kv_cache_layout
            )
        )

    @classmethod
    def _align_hma_transfer_regions(
        cls,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
        local_transfer_regions: list[IluNixlTransferRegion],
        remote_transfer_regions: list[IluNixlTransferRegion],
        num_groups: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]:
        local_regions = cls._regions_by_id(local_transfer_regions)
        remote_regions = cls._regions_by_id(remote_transfer_regions)

        selected_local = [
            local_regions[region_id]
            for region_id in selected_local_region_ids
            if region_id in local_regions
        ]
        selected_remote = [
            remote_regions[region_id]
            for region_id in selected_remote_region_ids
            if region_id in remote_regions
        ]

        local_by_identity: dict[str, IluNixlTransferRegion] = {}
        for region in selected_local:
            for identity in region.identities:
                local_by_identity[identity] = region

        remote_by_identity: dict[str, IluNixlTransferRegion] = {}
        for region in selected_remote:
            for identity in region.identities:
                remote_by_identity[identity] = region

        def identity_sort_key(identity: str) -> tuple[int, str, int]:
            layer_idx, kind, group = identity.split(":", 2)
            return (int(layer_idx), kind, int(group.removeprefix("group")))

        common_identities = sorted(
            set(local_by_identity).intersection(remote_by_identity),
            key=identity_sort_key,
        )
        if not common_identities:
            raise RuntimeError(
                "Heterogeneous PP HMA semantic alignment found no shared "
                "cache identities between local and remote selected regions. "
                f"local_regions={selected_local_region_ids}, "
                f"remote_regions={selected_remote_region_ids}"
            )

        group_pairs: list[list[tuple[int, int, str]]] = [[] for _ in range(num_groups)]
        for identity in common_identities:
            local_region = local_by_identity[identity]
            remote_region = remote_by_identity[identity]
            group_idx = int(identity.rsplit("group", 1)[1])
            if group_idx >= num_groups:
                continue
            group_pairs[group_idx].append(
                (local_region.region_id, remote_region.region_id, identity)
            )

        conflicts: list[dict[str, Any]] = []
        local_group_region_ids: list[np.ndarray] = []
        remote_group_region_ids: list[np.ndarray] = []
        selected_local_ids: list[int] = []
        selected_remote_ids: list[int] = []

        for group_idx, pairs in enumerate(group_pairs):
            local_to_remote: dict[int, int] = {}
            remote_to_local: dict[int, int] = {}
            local_group: list[int] = []
            remote_group: list[int] = []
            for local_region_id, remote_region_id, identity in pairs:
                prior_remote = local_to_remote.get(local_region_id)
                prior_local = remote_to_local.get(remote_region_id)
                if prior_remote is not None and prior_remote != remote_region_id:
                    conflicts.append(
                        {
                            "group": group_idx,
                            "local_region": local_region_id,
                            "remote_regions": sorted({prior_remote, remote_region_id}),
                            "identity": identity,
                        }
                    )
                    continue
                if prior_local is not None and prior_local != local_region_id:
                    conflicts.append(
                        {
                            "group": group_idx,
                            "remote_region": remote_region_id,
                            "local_regions": sorted({prior_local, local_region_id}),
                            "identity": identity,
                        }
                    )
                    continue
                local_to_remote[local_region_id] = remote_region_id
                remote_to_local[remote_region_id] = local_region_id
                if local_region_id not in local_group:
                    local_group.append(local_region_id)
                if remote_region_id not in remote_group:
                    remote_group.append(remote_region_id)
                if local_region_id not in selected_local_ids:
                    selected_local_ids.append(local_region_id)
                if remote_region_id not in selected_remote_ids:
                    selected_remote_ids.append(remote_region_id)

            local_group_region_ids.append(np.asarray(local_group, dtype=np.int64))
            remote_group_region_ids.append(np.asarray(remote_group, dtype=np.int64))

        if conflicts:
            raise RuntimeError(
                "Heterogeneous PP HMA needs slice/scatter transfer inside a "
                "single KV group because one semantic owner maps to multiple "
                f"physical regions. conflicts={conflicts[:8]}"
            )

        compact_remote_map = {
            region_id: compact_id
            for compact_id, region_id in enumerate(selected_remote_ids)
        }
        remote_group_region_ids = [
            np.asarray(
                [compact_remote_map[int(region_id)] for region_id in group],
                dtype=np.int64,
            )
            for group in remote_group_region_ids
        ]
        return (
            local_group_region_ids,
            remote_group_region_ids,
            selected_local_ids,
            selected_remote_ids,
        )

    @staticmethod
    def _parse_transfer_identity(identity: str) -> tuple[int, str, int]:
        layer_idx, kind, group = identity.split(":", 2)
        return (int(layer_idx), kind, int(group.removeprefix("group")))

    def _build_mamba_hma_transfer_routes(
        self,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
        local_transfer_regions: list[IluNixlTransferRegion],
        remote_transfer_regions: list[IluNixlTransferRegion],
        overlap_start: int | None = None,
        overlap_end: int | None = None,
    ) -> tuple[
        list[_HmaTransferRoute],
        list[int],
        list[int],
        list[int],
    ]:
        local_regions = self._regions_by_id(local_transfer_regions)
        remote_regions = self._regions_by_id(remote_transfer_regions)

        selected_local = [
            local_regions[region_id]
            for region_id in selected_local_region_ids
            if region_id in local_regions
        ]
        selected_remote = [
            remote_regions[region_id]
            for region_id in selected_remote_region_ids
            if region_id in remote_regions
        ]

        def in_overlap(layer_idx: int) -> bool:
            if overlap_start is None or overlap_end is None or layer_idx < 0:
                return True
            return overlap_start <= layer_idx < overlap_end

        local_by_key: dict[tuple[int, str], tuple[int, int]] = {}
        for region in selected_local:
            for identity in region.identities:
                layer_idx, kind, group_idx = self._parse_transfer_identity(identity)
                if in_overlap(layer_idx):
                    local_by_key[(layer_idx, kind)] = (region.region_id, group_idx)

        remote_by_key: dict[tuple[int, str], tuple[int, int]] = {}
        for region in selected_remote:
            for identity in region.identities:
                layer_idx, kind, group_idx = self._parse_transfer_identity(identity)
                if in_overlap(layer_idx):
                    remote_by_key[(layer_idx, kind)] = (region.region_id, group_idx)

        common_keys = sorted(
            set(local_by_key).intersection(remote_by_key),
            key=lambda key: (
                key[0],
                local_by_key[key][0],
                remote_by_key[key][0],
                key[1],
            ),
        )
        if not common_keys:
            raise RuntimeError(
                "Heterogeneous PP HMA semantic alignment found no shared "
                "cache identities between local and remote selected regions. "
                f"local_regions={selected_local_region_ids}, "
                f"remote_regions={selected_remote_region_ids}"
            )

        num_local_groups = len(
            getattr(self, "_group_spec_types", None)
            or getattr(self, "region_ids_by_group", ())
        )
        route_buckets: dict[tuple[int, int], list[tuple[int, int, tuple[int, str]]]] = (
            defaultdict(list)
        )
        for key in common_keys:
            local_region_id, local_group_idx = local_by_key[key]
            remote_region_id, remote_group_idx = remote_by_key[key]
            if num_local_groups and local_group_idx >= num_local_groups:
                continue
            route_buckets[(local_group_idx, remote_group_idx)].append(
                (local_region_id, remote_region_id, key)
            )

        conflicts: list[dict[str, Any]] = []
        bucket_regions: list[tuple[int, int, bool, list[int], list[int]]] = []
        fa_region_pairs: list[tuple[int, int]] = []
        fa_local_to_remote: dict[int, int] = {}
        fa_remote_to_local: dict[int, int] = {}
        selected_ssm_remote_ids: list[int] = []

        group_spec_types = getattr(self, "_group_spec_types", None)
        for (local_group_idx, remote_group_idx), pairs in sorted(route_buckets.items()):
            if group_spec_types is not None and local_group_idx < len(group_spec_types):
                is_ssm = _is_ssm_spec(group_spec_types[local_group_idx])
            else:
                is_ssm = any(key[1] == "mamba_cache" for _, _, key in pairs)

            local_to_remote: dict[int, int] = {}
            remote_to_local: dict[int, int] = {}
            raw_local_group: list[int] = []
            raw_remote_group: list[int] = []

            for local_region_id, remote_region_id, key in pairs:
                identity_str = (
                    f"{key[0]}:{key[1]}:group{local_group_idx}->group{remote_group_idx}"
                )
                prior_remote = local_to_remote.get(local_region_id)
                prior_local = remote_to_local.get(remote_region_id)
                if prior_remote is not None and prior_remote != remote_region_id:
                    conflicts.append(
                        {
                            "group": (local_group_idx, remote_group_idx),
                            "local_region": local_region_id,
                            "remote_regions": sorted({prior_remote, remote_region_id}),
                            "identity": identity_str,
                        }
                    )
                    continue
                if prior_local is not None and prior_local != local_region_id:
                    conflicts.append(
                        {
                            "group": (local_group_idx, remote_group_idx),
                            "remote_region": remote_region_id,
                            "local_regions": sorted({prior_local, local_region_id}),
                            "identity": identity_str,
                        }
                    )
                    continue
                local_to_remote[local_region_id] = remote_region_id
                remote_to_local[remote_region_id] = local_region_id
                if local_region_id not in raw_local_group:
                    raw_local_group.append(local_region_id)
                if remote_region_id not in raw_remote_group:
                    raw_remote_group.append(remote_region_id)

                if is_ssm:
                    if remote_region_id not in selected_ssm_remote_ids:
                        selected_ssm_remote_ids.append(remote_region_id)
                else:
                    prior_fa_remote = fa_local_to_remote.get(local_region_id)
                    prior_fa_local = fa_remote_to_local.get(remote_region_id)
                    if (
                        prior_fa_remote is not None
                        and prior_fa_remote != remote_region_id
                    ) or (
                        prior_fa_local is not None and prior_fa_local != local_region_id
                    ):
                        conflicts.append(
                            {
                                "group": (local_group_idx, remote_group_idx),
                                "local_region": local_region_id,
                                "remote_region": remote_region_id,
                                "identity": identity_str,
                            }
                        )
                        continue
                    fa_local_to_remote[local_region_id] = remote_region_id
                    fa_remote_to_local[remote_region_id] = local_region_id
                    pair = (local_region_id, remote_region_id)
                    if pair not in fa_region_pairs:
                        fa_region_pairs.append(pair)

            bucket_regions.append(
                (
                    local_group_idx,
                    remote_group_idx,
                    is_ssm,
                    raw_local_group,
                    raw_remote_group,
                )
            )

        if conflicts:
            raise RuntimeError(
                "Heterogeneous PP HMA needs slice/scatter transfer inside a "
                "single KV group because one semantic owner maps to multiple "
                f"physical regions. conflicts={conflicts[:8]}"
            )

        selected_fa_local_ids = [pair[0] for pair in fa_region_pairs]
        selected_fa_remote_ids = [pair[1] for pair in fa_region_pairs]
        compact_fa_remote_map = {
            reg_id: compact_id
            for compact_id, reg_id in enumerate(selected_fa_remote_ids)
        }

        ssm_region_indices = getattr(self, "_ssm_region_indices", None)
        num_local_regions = getattr(self, "num_regions", len(local_transfer_regions))
        local_ssm_regions = (
            list(ssm_region_indices)
            if ssm_region_indices
            else list(range(num_local_regions))
        )
        local_ssm_slot_map = {
            int(reg_id): slot_idx for slot_idx, reg_id in enumerate(local_ssm_regions)
        }
        remote_ssm_slot_map = {
            int(reg_id): slot_idx
            for slot_idx, reg_id in enumerate(selected_ssm_remote_ids)
        }

        routes: list[_HmaTransferRoute] = []
        for (
            local_group_idx,
            remote_group_idx,
            is_ssm,
            raw_local_group,
            raw_remote_group,
        ) in bucket_regions:
            if is_ssm:
                for reg_id in raw_local_group:
                    if reg_id not in local_ssm_slot_map:
                        raise RuntimeError(
                            f"Local SSM region {reg_id} is not registered in "
                            f"_ssm_region_indices={local_ssm_regions}"
                        )
                local_desc_ids = np.asarray(
                    [local_ssm_slot_map[reg_id] for reg_id in raw_local_group],
                    dtype=np.int64,
                )
                remote_desc_ids = np.asarray(
                    [remote_ssm_slot_map[reg_id] for reg_id in raw_remote_group],
                    dtype=np.int64,
                )
            else:
                local_desc_ids = np.asarray(raw_local_group, dtype=np.int64)
                remote_desc_ids = np.asarray(
                    [compact_fa_remote_map[reg_id] for reg_id in raw_remote_group],
                    dtype=np.int64,
                )

            routes.append(
                _HmaTransferRoute(
                    local_group_idx=local_group_idx,
                    remote_group_idx=remote_group_idx,
                    is_ssm=is_ssm,
                    local_desc_region_ids=local_desc_ids,
                    remote_desc_region_ids=remote_desc_ids,
                    raw_local_region_ids=tuple(raw_local_group),
                    raw_remote_region_ids=tuple(raw_remote_group),
                )
            )

        return (
            routes,
            selected_fa_local_ids,
            selected_fa_remote_ids,
            selected_ssm_remote_ids,
        )

    def _gqa_expected_remote_block_len(
        self,
        local_len: int,
        remote_tp_size: int,
        block_size_ratio: int,
    ) -> int:
        """Expected remote FA region bytes for GQA/MHA under heterogeneous TP.

        Raw ``tp_ratio`` is wrong when ``TP > num_kv_heads``: extra ranks
        replicate heads, so per-rank bytes track physical heads, not world size.
        Example: heads=4, Decode TP8, Prefill TP1 → ratio 4, not 8.
        """
        assert self.transfer_topo is not None
        total_heads = self.transfer_topo.total_num_kv_heads
        local_heads = max(1, total_heads // self.transfer_topo.tp_size)
        remote_heads = max(1, total_heads // remote_tp_size)
        assert (local_len * remote_heads) % local_heads == 0, (
            "GQA region length not divisible by physical head ratio: "
            f"local={local_len}, local_heads={local_heads}, "
            f"remote_heads={remote_heads}, total_heads={total_heads}, "
            f"local_tp={self.transfer_topo.tp_size}, remote_tp={remote_tp_size}"
        )
        if remote_heads < local_heads:
            assert block_size_ratio == 1, (
                "Different local/remote block sizes are not supported "
                "when remote has fewer KV heads per rank than local."
            )
        return (local_len * remote_heads) // local_heads // block_size_ratio

    def _validate_pp_selected_region_lengths(
        self,
        nixl_agent_meta: IluNixlAgentMetadata,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
        remote_tp_size: int,
    ) -> None:
        assert self.transfer_topo is not None
        local_layout = (
            self.host_buffer_kv_cache_layout
            if self.use_host_buffer
            else self.kv_cache_layout
        )
        if not self.use_mla and nixl_agent_meta.kv_cache_layout != local_layout:
            raise RuntimeError(
                "Heterogeneous PP expects the same non-MLA KV cache layout: "
                f"local={local_layout}, remote={nixl_agent_meta.kv_cache_layout}"
            )
        assert len(selected_local_region_ids) == len(selected_remote_region_ids), (
            "Heterogeneous PP selected region count mismatch: "
            f"local={len(selected_local_region_ids)}, "
            f"remote={len(selected_remote_region_ids)}"
        )
        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )
        # MLA (and region-level full replicas) keep equal per-rank bytes.
        # GQA must use physical-head ratio — not tp_ratio — when TP>kv_heads.
        model_replicated = self.use_mla
        for pair_idx, (local_region_id, remote_region_id) in enumerate(
            zip(selected_local_region_ids, selected_remote_region_ids)
        ):
            local_len = self.block_len_per_layer[local_region_id]
            remote_len = nixl_agent_meta.block_lens[remote_region_id]
            replicated = model_replicated or self._is_region_replicated(local_region_id)
            if replicated:
                expected_remote_len = local_len // block_size_ratio
            else:
                expected_remote_len = self._gqa_expected_remote_block_len(
                    local_len, remote_tp_size, block_size_ratio
                )
            assert remote_len == expected_remote_len, (
                "Heterogeneous PP selected region size mismatch at "
                f"pair={pair_idx}: local_region={local_region_id}, "
                f"remote_region={remote_region_id}, local={local_len}, "
                f"remote={remote_len}, expected_remote={expected_remote_len}, "
                f"tp_ratio={tp_ratio}, block_size_ratio={block_size_ratio}, "
                f"local_heads={max(1, self.transfer_topo.total_num_kv_heads // self.transfer_topo.tp_size)}, "
                f"remote_heads={max(1, self.transfer_topo.total_num_kv_heads // remote_tp_size)}"
            )
        assert self.dst_num_blocks[nixl_agent_meta.engine_id] == (
            nixl_agent_meta.num_blocks
        )

    def _validate_pp_hma_region_lengths(
        self,
        nixl_agent_meta: IluNixlAgentMetadata,
        local_region_ids_by_group: list[np.ndarray],
        remote_region_ids_by_group: list[np.ndarray],
        selected_remote_region_ids: list[int],
        remote_tp_size: int,
        routes: list[_HmaTransferRoute] | None = None,
    ) -> None:
        assert self.transfer_topo is not None
        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )
        assert block_size_ratio == 1, (
            "HMA does not support different remote block size yet"
        )
        hybrid_mla_mamba = bool(
            getattr(self, "_has_mamba", False) and getattr(self, "use_mla", False)
        )
        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)

        if routes is not None:
            for route in routes:
                if len(route.raw_local_region_ids) != len(route.raw_remote_region_ids):
                    raise AssertionError(
                        "KV cache selected region count must match within each "
                        "HMA route after semantic alignment: "
                        f"group={route.local_group_idx}->{route.remote_group_idx}, "
                        f"local={len(route.raw_local_region_ids)}, "
                        f"remote={len(route.raw_remote_region_ids)}"
                    )
                for pair_idx, (local_region_id, remote_region_id) in enumerate(
                    zip(route.raw_local_region_ids, route.raw_remote_region_ids)
                ):
                    local_block_len = self.block_len_per_layer[int(local_region_id)]
                    remote_block_len = nixl_agent_meta.block_lens[int(remote_region_id)]
                    replicated = hybrid_mla_mamba or (
                        not route.is_ssm
                        and self._is_region_replicated(int(local_region_id))
                    )
                    expected_remote_len = local_block_len // block_size_ratio
                    if not replicated:
                        if route.is_ssm:
                            # SSM follows TP world-size sharding, not KV heads.
                            expected_remote_len = (
                                local_block_len * tp_ratio
                                if tp_ratio >= 1
                                else local_block_len // -tp_ratio
                            )
                        else:
                            expected_remote_len = self._gqa_expected_remote_block_len(
                                local_block_len, remote_tp_size, block_size_ratio
                            )
                    assert remote_block_len == expected_remote_len, (
                        "KV cache selected region size must match between P and D "
                        "when using HMA with heterogeneous PP at "
                        f"group={route.local_group_idx}->{route.remote_group_idx}, "
                        f"pair={pair_idx}: "
                        f"local_region={int(local_region_id)} "
                        f"local={local_block_len}, "
                        f"remote_region={int(remote_region_id)} "
                        f"remote={remote_block_len}, "
                        f"expected_remote={expected_remote_len}"
                    )
            assert self.dst_num_blocks[nixl_agent_meta.engine_id] == (
                nixl_agent_meta.num_blocks
            )
            return

        assert len(local_region_ids_by_group) == len(remote_region_ids_by_group), (
            "KV cache HMA group count must match after semantic alignment: "
            f"local={len(local_region_ids_by_group)}, "
            f"remote={len(remote_region_ids_by_group)}"
        )
        for group_idx, (local_group, remote_group) in enumerate(
            zip(local_region_ids_by_group, remote_region_ids_by_group)
        ):
            if len(local_group) != len(remote_group):
                raise AssertionError(
                    "KV cache selected region count must match within each "
                    "HMA group after semantic alignment: "
                    f"group={group_idx}, local={len(local_group)}, "
                    f"remote={len(remote_group)}"
                )
            for pair_idx, (local_region_id, remote_compact_id) in enumerate(
                zip(local_group, remote_group)
            ):
                remote_region_id = selected_remote_region_ids[int(remote_compact_id)]
                local_block_len = self.block_len_per_layer[int(local_region_id)]
                remote_block_len = nixl_agent_meta.block_lens[int(remote_region_id)]
                is_ssm = _is_ssm_spec(self._group_spec_types[group_idx])
                replicated = hybrid_mla_mamba or (
                    not is_ssm and self._is_region_replicated(int(local_region_id))
                )
                expected_remote_len = local_block_len // block_size_ratio
                if not replicated:
                    if is_ssm:
                        expected_remote_len = (
                            local_block_len * tp_ratio
                            if tp_ratio >= 1
                            else local_block_len // -tp_ratio
                        )
                    else:
                        expected_remote_len = self._gqa_expected_remote_block_len(
                            local_block_len, remote_tp_size, block_size_ratio
                        )
                assert remote_block_len == expected_remote_len, (
                    "KV cache selected region size must match between P and D "
                    "when using HMA with heterogeneous PP at "
                    f"group={group_idx}, pair={pair_idx}: "
                    f"local_region={int(local_region_id)} "
                    f"local={local_block_len}, "
                    f"remote_region={remote_region_id} "
                    f"remote={remote_block_len}, "
                    f"expected_remote={expected_remote_len}"
                )
        assert self.dst_num_blocks[nixl_agent_meta.engine_id] == (
            nixl_agent_meta.num_blocks
        )

    def _select_pp_regions(
        self,
        nixl_agent_meta: IluNixlAgentMetadata,
        remote_pp_rank: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]:
        local_start, local_end = self._pp_layer_range(
            self.pp_split, self.pp_rank, self.layer_region_ids
        )
        remote_start, remote_end = self._pp_layer_range(
            nixl_agent_meta.pp_split,
            remote_pp_rank,
            nixl_agent_meta.layer_region_ids,
        )
        overlap_start = max(local_start, remote_start)
        overlap_end = min(local_end, remote_end)
        if overlap_start >= overlap_end:
            raise RuntimeError(
                "Heterogeneous PP handshake selected a remote PP stage "
                "with no overlapping layers: "
                f"local=[{local_start}, {local_end}), "
                f"remote=[{remote_start}, {remote_end})."
            )

        selected_local_region_ids = self._region_ids_for_layer_range(
            self.layer_region_ids, overlap_start, overlap_end
        )
        assert nixl_agent_meta.layer_region_ids is not None
        selected_remote_region_ids = self._region_ids_for_layer_range(
            nixl_agent_meta.layer_region_ids, overlap_start, overlap_end
        )

        if self._is_hma_required:
            assert nixl_agent_meta.transfer_regions is not None
            return self._align_hma_transfer_regions(
                selected_local_region_ids,
                selected_remote_region_ids,
                self.transfer_regions,
                nixl_agent_meta.transfer_regions,
                len(self.region_ids_by_group),
            )

        assert nixl_agent_meta.region_ids_by_group is not None
        return (
            self._filter_group_region_ids(
                self.region_ids_by_group,
                selected_local_region_ids,
                compact=False,
            ),
            self._filter_group_region_ids(
                nixl_agent_meta.region_ids_by_group,
                selected_remote_region_ids,
                compact=True,
            ),
            selected_local_region_ids,
            selected_remote_region_ids,
        )

    @staticmethod
    def _metadata_uses_virtual_packed_regions(
        nixl_agent_meta: IluNixlAgentMetadata,
    ) -> bool:
        return bool(
            nixl_agent_meta.transfer_regions
            and any(
                region.block_stride > 0
                and (
                    region.byte_offset != 0
                    or region.block_stride != region.block_len
                    or region.physical_backing_id != region.region_id
                )
                for region in nixl_agent_meta.transfer_regions
            )
        )

    def _build_fa_local(
        self,
        base_addresses: list[int],
        block_size_ratio: int,
    ) -> np.ndarray:
        if not self._virtual_packed_regions:
            return super()._build_fa_local(base_addresses, block_size_ratio)
        if block_size_ratio != 1:
            raise ValueError(
                "Packed virtual IluNixl descriptors do not support a remote "
                "logical block-size mismatch"
            )
        regions = self._regions_by_id(self.transfer_regions)
        blocks_data: list[tuple[int, int, int]] = []
        for region_id in range(self.num_regions):
            region = regions[region_id]
            if region.physical_backing_id >= len(base_addresses):
                raise ValueError(
                    "Packed virtual region references an unregistered physical "
                    f"backing: region={region_id}, "
                    f"backing={region.physical_backing_id}"
                )
            backing_base = base_addresses[region.physical_backing_id]
            for block_id in range(self.num_blocks):
                blocks_data.append(
                    self._virtual_slot_desc(
                        backing_base,
                        block_id,
                        region.block_stride,
                        region.byte_offset,
                        region.block_len,
                        self.device_id,
                    )
                )
        return np.asarray(blocks_data, dtype=np.uint64)

    def _remote_transfer_regions(
        self,
        engine_id: str,
        nixl_agent_meta: IluNixlAgentMetadata,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
    ) -> list[tuple[int, int, int]]:
        """Return (remote_region_id, transfer_len, rank_offset)."""
        if not selected_local_region_ids:
            return []
        assert self.transfer_topo is not None
        fa_group_idx = next(
            i for i, t in enumerate(self._group_spec_types) if _is_attention_spec(t)
        )
        plan = self.tp_mappings[engine_id]
        split_reads = len(plan.source_ranks_per_group[fa_group_idx])
        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )

        regions: list[tuple[int, int, int]] = []
        for local_region_id, remote_region_id in zip(
            selected_local_region_ids, selected_remote_region_ids
        ):
            replicated = self._is_region_replicated(local_region_id)
            # Use the stride registered for this transfer region.  The IluNixl
            # worker mixin does not implement get_backend_aware_kv_block_len.
            local_block_len = self.block_len_per_layer[local_region_id]
            remote_kv_block_len = local_block_len // block_size_ratio
            if block_size_ratio > 1:
                local_block_len = remote_kv_block_len

            num_reads = 1 if replicated else split_reads
            rank_offset = (
                0 if replicated else plan.rank_offset_factor * remote_kv_block_len
            )
            regions.append(
                (remote_region_id, local_block_len // num_reads, rank_offset)
            )
        return regions

    def _build_pp_mamba_remote(
        self,
        nixl_agent_meta: IluNixlAgentMetadata,
        selected_remote_region_ids: list[int],
        remote_tp_size: int,
    ) -> np.ndarray:
        if not selected_remote_region_ids:
            return np.empty((0, 3), dtype=np.uint64)
        assert self.transfer_topo is not None and self._conv_decomp is not None
        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        local_offset = self.tp_rank % max(tp_ratio, 1)
        conv_offsets = self._conv_decomp.remote_conv_offsets(local_offset, tp_ratio)
        ssm_read_size = (
            self._mamba_ssm_size[1] if tp_ratio >= 1 else nixl_agent_meta.ssm_sizes[1]
        )
        remote_ratio = nixl_agent_meta.physical_blocks_per_logical_kv_block
        num_blocks = nixl_agent_meta.num_blocks // remote_ratio
        block_arange = np.arange(num_blocks, dtype=np.uint64)
        block_strides = nixl_agent_meta.block_strides or nixl_agent_meta.block_lens
        parts: list[np.ndarray] = []
        for region_id in selected_remote_region_ids:
            page_stride = block_strides[region_id] * remote_ratio
            block_addrs = (
                nixl_agent_meta.kv_caches_base_addr[region_id]
                + block_arange * page_stride
            )
            for offset, size in conv_offsets:
                parts.append(
                    self._stack_descs(
                        block_addrs + offset, size, nixl_agent_meta.device_id
                    )
                )
            ssm_addrs = (
                block_addrs
                + nixl_agent_meta.ssm_sizes[0]
                + local_offset * ssm_read_size
            )
            parts.append(
                self._stack_descs(ssm_addrs, ssm_read_size, nixl_agent_meta.device_id)
            )
        return np.concatenate(parts) if parts else np.empty((0, 3), dtype=np.uint64)

    def _ensure_pp_mamba_local_xfer_handle(self) -> int:
        if self._pp_mamba_local_xfer_handle is not None:
            return self._pp_mamba_local_xfer_handle
        if (
            hasattr(self, "src_xfer_handles_by_block_size")
            and getattr(self, "block_size", None) in self.src_xfer_handles_by_block_size
        ):
            self._pp_mamba_local_xfer_handle = self.src_xfer_handles_by_block_size[
                self.block_size
            ]
            return self._pp_mamba_local_xfer_handle

        base_addresses = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        blocks_data: list[tuple[int, int, int]] = []
        block_strides = getattr(
            self, "block_stride_per_layer", self.block_len_per_layer
        )
        for region_id, base_addr in enumerate(base_addresses):
            page_size = self.block_len_per_layer[region_id]
            page_stride = block_strides[region_id]
            for block_id in range(self.num_blocks):
                blocks_data.append(
                    (
                        base_addr + block_id * page_stride,
                        page_size,
                        self.device_id,
                    )
                )

        fa_blocks = (
            np.asarray(blocks_data, dtype=np.uint64)
            if blocks_data
            else np.empty((0, 3), dtype=np.uint64)
        )
        blocks = np.concatenate(
            [
                fa_blocks,
                self._build_mamba_local(base_addresses),
            ]
        )
        descs = self.nixl_wrapper.get_xfer_descs(blocks, self.nixl_memory_type)
        self._pp_mamba_local_xfer_handle = self.nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", descs
        )
        return self._pp_mamba_local_xfer_handle

    def _register_pp_remote_xfer_handle(
        self,
        engine_id: str,
        remote_key: int,
        remote_agent_name: str,
        nixl_agent_meta: IluNixlAgentMetadata,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
        selected_ssm_remote_region_ids: list[int] | None = None,
    ) -> None:
        blocks_data: list[tuple[int, int, int]] = []
        remote_regions = self._regions_by_id(nixl_agent_meta.transfer_regions)
        virtual_packed = self._metadata_uses_virtual_packed_regions(nixl_agent_meta)
        block_strides = nixl_agent_meta.block_strides or nixl_agent_meta.block_lens
        for region_id, transfer_len, rank_offset in self._remote_transfer_regions(
            engine_id,
            nixl_agent_meta,
            selected_local_region_ids,
            selected_remote_region_ids,
        ):
            if virtual_packed:
                region = remote_regions.get(region_id)
                if region is None:
                    raise RuntimeError(
                        "Packed remote descriptor is missing transfer-region "
                        f"metadata for region {region_id}"
                    )
                if transfer_len + rank_offset > region.block_len:
                    raise RuntimeError(
                        "Packed remote TP slice exceeds its semantic slot for "
                        f"region {region_id}: offset={rank_offset}, "
                        f"length={transfer_len}, page_size={region.block_len}"
                    )
                try:
                    physical_base = nixl_agent_meta.kv_caches_base_addr[
                        region.physical_backing_id
                    ]
                except IndexError as exc:
                    raise RuntimeError(
                        "Packed remote region references an unavailable physical "
                        f"backing: region={region_id}, "
                        f"backing={region.physical_backing_id}"
                    ) from exc
                slot_base = physical_base + region.byte_offset
                page_stride = region.block_stride
            else:
                slot_base = nixl_agent_meta.kv_caches_base_addr[region_id]
                page_stride = block_strides[region_id]
            for block_id in range(nixl_agent_meta.num_blocks):
                blocks_data.append(
                    (
                        slot_base + block_id * page_stride + rank_offset,
                        transfer_len,
                        nixl_agent_meta.device_id,
                    )
                )

        if not hasattr(self, "_remote_fa_num_regions"):
            self._remote_fa_num_regions = {}
        self._remote_fa_num_regions.setdefault(engine_id, {})[remote_key] = len(
            selected_remote_region_ids
        )

        if getattr(self, "_has_mamba", False):
            ssm_remote_ids = selected_ssm_remote_region_ids
            if ssm_remote_ids is None:
                ssm_remote_ids = (
                    getattr(self, "_selected_ssm_remote_region_ids", {})
                    .get(engine_id, {})
                    .get(remote_key)
                )
            if ssm_remote_ids is None:
                ssm_remote_ids = selected_remote_region_ids
            fa_blocks = (
                np.asarray(blocks_data, dtype=np.uint64)
                if blocks_data
                else np.empty((0, 3), dtype=np.uint64)
            )
            blocks_data = np.concatenate(
                [
                    fa_blocks,
                    self._build_pp_mamba_remote(
                        nixl_agent_meta,
                        ssm_remote_ids,
                        self.transfer_topo.get_engine_info(engine_id).remote_tp_size,
                    ),
                ]
            )
        descs = self.nixl_wrapper.get_xfer_descs(
            blocks_data, nixl_agent_meta.memory_type
        )
        self.dst_xfer_side_handles[engine_id][remote_key] = (
            self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        )

    def _needs_split_local_xfer_handles(self, tp_ratio: int, plan) -> bool:
        return tp_ratio < 0 and (
            not getattr(self, "use_mla", False)
            or len(getattr(plan, "all_source_ranks", ())) > 1
        )

    def _ensure_pp_local_split_handles(
        self, engine_id: str, remote_tp_size: int
    ) -> None:
        assert self.transfer_topo is not None
        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        plan = self.tp_mappings[engine_id]
        if (
            not self._needs_split_local_xfer_handles(tp_ratio, plan)
            or tp_ratio in self.src_xfer_handles_by_tp_ratio
        ):
            return

        self.src_xfer_handles_by_tp_ratio[tp_ratio] = []
        for handle_data in self._build_local_splits_from_plan(
            plan,
            self.src_blocks_data,
            self.num_descs,
        ):
            local_descs = self.nixl_wrapper.get_xfer_descs(
                handle_data, self.nixl_memory_type
            )
            self.src_xfer_handles_by_tp_ratio[tp_ratio].append(
                self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", local_descs)
            )

    def add_remote_agent(
        self,
        nixl_agent_meta: IluNixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
        remote_pp_rank: int = 0,
        remote_pp_size: int = 1,
    ) -> str:
        remote_virtual_packed = self._metadata_uses_virtual_packed_regions(
            nixl_agent_meta
        )
        if (
            remote_pp_size == self.pp_world_size
            and nixl_agent_meta.pp_split in ("", self.pp_split)
            and not self._virtual_packed_regions
            and not remote_virtual_packed
        ):
            return super().add_remote_agent(
                nixl_agent_meta, remote_tp_rank, remote_tp_size
            )

        if (
            nixl_agent_meta.layer_region_ids is None
            or nixl_agent_meta.region_ids_by_group is None
        ):
            raise RuntimeError(
                "Heterogeneous PP requires layer_region_ids and "
                "region_ids_by_group in the "
                "IluNixl handshake metadata."
            )
        if self._is_hma_required and nixl_agent_meta.transfer_regions is None:
            raise RuntimeError(
                "Heterogeneous PP with HMA requires transfer_regions in the "
                "IluNixl handshake metadata."
            )
        engine_id = nixl_agent_meta.engine_id
        is_hetero_pp_topology = remote_pp_size != self.pp_world_size or (
            bool(nixl_agent_meta.pp_split) and nixl_agent_meta.pp_split != self.pp_split
        )
        remote_key = (
            remote_pp_rank * remote_tp_size + remote_tp_rank
            if is_hetero_pp_topology
            else remote_tp_rank
        )
        if remote_key in self._remote_agents.get(engine_id, {}):
            return self._remote_agents[engine_id][remote_key]

        assert self.transfer_topo is not None
        physical_blocks_per_logical = (
            nixl_agent_meta.physical_blocks_per_logical_kv_block
        )
        transfer_info = EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=nixl_agent_meta.block_size,
            remote_block_len=nixl_agent_meta.block_lens[0],
            remote_physical_blocks_per_logical=physical_blocks_per_logical,
        )
        self.transfer_topo.register_remote_engine(engine_id, transfer_info)
        self.tp_mappings[engine_id] = compute_tp_mapping(
            transfer_topology=self.transfer_topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
        )
        self._remote_pp_size_by_engine[engine_id] = remote_pp_size
        self._remote_pp_split_by_engine[engine_id] = nixl_agent_meta.pp_split
        self._semantic_region_engines.add(engine_id)

        routes: list[_HmaTransferRoute] | None = None
        selected_ssm_remote_ids: list[int] | None = None
        if self._is_hma_required and getattr(self, "_has_mamba", False):
            local_start, local_end = self._pp_layer_range(
                self.pp_split, self.pp_rank, self.layer_region_ids
            )
            remote_start, remote_end = self._pp_layer_range(
                nixl_agent_meta.pp_split,
                remote_pp_rank,
                nixl_agent_meta.layer_region_ids,
            )
            overlap_start = max(local_start, remote_start)
            overlap_end = min(local_end, remote_end)
            if overlap_start >= overlap_end:
                raise RuntimeError(
                    "Heterogeneous PP handshake selected a remote PP stage "
                    "with no overlapping layers: "
                    f"local=[{local_start}, {local_end}), "
                    f"remote=[{remote_start}, {remote_end})."
                )
            raw_selected_local = self._region_ids_for_layer_range(
                self.layer_region_ids, overlap_start, overlap_end
            )
            assert nixl_agent_meta.layer_region_ids is not None
            raw_selected_remote = self._region_ids_for_layer_range(
                nixl_agent_meta.layer_region_ids, overlap_start, overlap_end
            )
            assert nixl_agent_meta.transfer_regions is not None
            (
                routes,
                selected_local_region_ids,
                selected_remote_region_ids,
                selected_ssm_remote_ids,
            ) = self._build_mamba_hma_transfer_routes(
                raw_selected_local,
                raw_selected_remote,
                self.transfer_regions,
                nixl_agent_meta.transfer_regions,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
            )
            num_groups = len(self.region_ids_by_group)
            local_by_group: list[list[int]] = [[] for _ in range(num_groups)]
            remote_by_group: list[list[int]] = [[] for _ in range(num_groups)]
            for route in routes:
                if route.local_group_idx < num_groups:
                    for rid in route.local_desc_region_ids.tolist():
                        if rid not in local_by_group[route.local_group_idx]:
                            local_by_group[route.local_group_idx].append(rid)
                    for rid in route.remote_desc_region_ids.tolist():
                        if rid not in remote_by_group[route.local_group_idx]:
                            remote_by_group[route.local_group_idx].append(rid)
            local_region_ids_by_group = [
                np.asarray(g, dtype=np.int64) for g in local_by_group
            ]
            remote_region_ids_by_group = [
                np.asarray(g, dtype=np.int64) for g in remote_by_group
            ]
            if not hasattr(self, "_mamba_hma_routes"):
                self._mamba_hma_routes = {}
            if not hasattr(self, "_remote_fa_num_regions"):
                self._remote_fa_num_regions = {}
            if not hasattr(self, "_selected_ssm_remote_region_ids"):
                self._selected_ssm_remote_region_ids = {}
            self._mamba_hma_routes.setdefault(engine_id, {})[remote_key] = routes
            self._remote_fa_num_regions.setdefault(engine_id, {})[remote_key] = len(
                selected_remote_region_ids
            )
            self._selected_ssm_remote_region_ids.setdefault(engine_id, {})[
                remote_key
            ] = selected_ssm_remote_ids
        else:
            (
                local_region_ids_by_group,
                remote_region_ids_by_group,
                selected_local_region_ids,
                selected_remote_region_ids,
            ) = self._select_pp_regions(nixl_agent_meta, remote_pp_rank)

        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        if self._is_hma_required:
            self._validate_pp_hma_region_lengths(
                nixl_agent_meta,
                local_region_ids_by_group,
                remote_region_ids_by_group,
                selected_remote_region_ids,
                remote_tp_size,
                routes=routes,
            )
        else:
            self._validate_pp_selected_region_lengths(
                nixl_agent_meta,
                selected_local_region_ids,
                selected_remote_region_ids,
                remote_tp_size,
            )
        self.local_region_ids_by_group.setdefault(engine_id, {})[remote_key] = (
            local_region_ids_by_group
        )
        self.remote_region_ids_by_group.setdefault(engine_id, {})[remote_key] = (
            remote_region_ids_by_group
        )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )
        if remote_virtual_packed:
            self.kv_caches_base_addr[engine_id][remote_key] = list(
                nixl_agent_meta.kv_caches_base_addr
            )
        else:
            self.kv_caches_base_addr[engine_id][remote_key] = [
                nixl_agent_meta.kv_caches_base_addr[i]
                for i in selected_remote_region_ids
            ]

        self._register_pp_remote_xfer_handle(
            engine_id,
            remote_key,
            remote_agent_name,
            nixl_agent_meta,
            selected_local_region_ids,
            selected_remote_region_ids,
            selected_ssm_remote_region_ids=selected_ssm_remote_ids,
        )
        self._ensure_pp_local_split_handles(engine_id, remote_tp_size)
        return remote_agent_name

    def _make_pp_aware_notif_id(
        self,
        remote_request_id: str,
        dst_engine_id: str,
        remote_key: int,
    ) -> bytes:
        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        remote_pp_rank = remote_key // remote_info.remote_tp_size
        remote_pp_split = self._remote_pp_split_by_engine[dst_engine_id]
        pp_consumers = self._overlap_pp_count(
            remote_pp_rank,
            remote_pp_split,
            self.pp_split,
        )
        return f"{remote_request_id}:{self.world_size}:{pp_consumers}".encode()

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        remote_pp_size: int,
        remote_pp_split: str,
        expected_engine_id: str,
    ) -> dict[int, str]:
        if not self.use_host_buffer:
            current_platform.set_device(self.device_id)

        assert self.transfer_topo is not None
        p_remote_ranks = self.transfer_topo.handshake_target_ranks(remote_tp_size)
        if remote_pp_split:
            remote_pp_ranks = self._target_pp_ranks(
                self.pp_split, self.pp_rank, remote_pp_split
            )
        elif remote_pp_size == self.pp_world_size:
            remote_pp_ranks = [self.pp_rank]
        else:
            raise RuntimeError(
                "IluNixl heterogeneous PP handshake requires remote "
                "pp_split in kv_transfer_params."
            )
        remote_rank_to_agent_name = {}
        path = make_zmq_path("tcp", host, port)

        with zmq_ctx(zmq.REQ, path) as sock:
            for remote_pp_rank in remote_pp_ranks:
                for remote_rank in p_remote_ranks:
                    target_rank = remote_pp_rank * remote_tp_size + remote_rank
                    logger.debug(
                        "Querying metadata on path: %s at remote pp rank %s, "
                        "tp rank %s (global rank %s)",
                        path,
                        remote_pp_rank,
                        remote_rank,
                        target_rank,
                    )

                    start_time = time.perf_counter()
                    msg = msgspec.msgpack.encode((GET_META_MSG, target_rank))
                    sock.setsockopt(zmq.RCVTIMEO, 5000)
                    sock.send(msg)
                    handshake_bytes = sock.recv()

                    handshake_decoder = msgspec.msgpack.Decoder(NixlHandshakePayload)
                    try:
                        handshake_payload = handshake_decoder.decode(handshake_bytes)
                    except (msgspec.DecodeError, msgspec.ValidationError) as e:
                        raise RuntimeError(
                            "Failed to decode NixlHandshakePayload. This likely "
                            "indicates an incompatible connector version. "
                            f"Error: {e}"
                        ) from e

                    got_metadata_time = time.perf_counter()
                    logger.debug(
                        "NIXL handshake: get metadata took: %s",
                        got_metadata_time - start_time,
                    )

                    assert self.compat_hash is not None
                    if (
                        self.enforce_compat_hash
                        and handshake_payload.compatibility_hash != self.compat_hash
                    ):
                        raise RuntimeError(
                            "NIXL compatibility hash mismatch. "
                            f"Local: {self.compat_hash}, "
                            f"Remote: {handshake_payload.compatibility_hash}."
                        )

                    metadata_decoder = msgspec.msgpack.Decoder(IluNixlAgentMetadata)
                    try:
                        metadata = metadata_decoder.decode(
                            handshake_payload.agent_metadata_bytes
                        )
                    except (msgspec.DecodeError, msgspec.ValidationError) as e:
                        raise RuntimeError(
                            f"Failed to decode IluNixlAgentMetadata. Error: {e}"
                        ) from e

                    if metadata.engine_id != expected_engine_id:
                        raise RuntimeError(
                            "Remote NIXL agent engine ID mismatch. "
                            f"Expected {expected_engine_id}, "
                            f"received {metadata.engine_id}."
                        )

                    remote_agent_name = self.add_remote_agent(
                        metadata,
                        remote_rank,
                        remote_tp_size,
                        remote_pp_rank,
                        remote_pp_size,
                    )
                    logger.debug(
                        "NIXL handshake: add agent took: %s",
                        time.perf_counter() - got_metadata_time,
                    )
                    remote_key = (
                        target_rank
                        if (
                            remote_pp_size != self.pp_world_size
                            or (
                                bool(remote_pp_split)
                                and remote_pp_split != self.pp_split
                            )
                        )
                        else remote_rank
                    )
                    remote_rank_to_agent_name[remote_key] = remote_agent_name

        return remote_rank_to_agent_name

    def _ensure_handshake(
        self,
        engine_id,
        host: str,
        port: int,
        tp_size: int,
        pp_size: int = 1,
        pp_split: str = "",
    ) -> Future[dict[int, str]] | None:
        self._evict_stale_engines()
        with self._handshake_lock:
            if engine_id in self._remote_agents:
                return None
            fut = self._handshake_futures.get(engine_id)
            if fut is not None:
                return fut
            fut = self._handshake_initiation_executor.submit(
                self._nixl_handshake,
                host,
                port,
                tp_size,
                pp_size,
                pp_split,
                engine_id,
            )
            self._handshake_futures[engine_id] = fut

            def done_callback(f: Future[dict[int, str]], eid=engine_id):
                with self._handshake_lock:
                    del self._handshake_futures[eid]
                    try:
                        self._remote_agents[eid] = f.result()
                        self._engine_last_active[eid] = time.perf_counter()
                        self._track_remote_engine_replacement(eid, host, port)
                    except Exception as e:
                        self._log_failure(
                            failure_type="handshake_setup_failed",
                            req_id=None,
                            error=e,
                            remote_engine_id=eid,
                        )

            fut.add_done_callback(done_callback)
            return fut

    def _background_nixl_handshake(self, req_id: str, remote_engine_id, meta):
        assert meta.remote is not None
        fut = self._ensure_handshake(
            remote_engine_id,
            meta.remote.host,
            meta.remote.port,
            meta.tp_size,
            getattr(meta, "remote_pp_size", 1),
            getattr(meta, "remote_pp_split", ""),
        )
        if fut is None:
            self._ready_requests.put((req_id, meta))
            return

        def request_ready(f: Future[Any], entry=(req_id, meta)):
            try:
                f.result()
                self._ready_requests.put(entry)
            except Exception as e:
                self._log_failure(
                    failure_type="handshake_failed",
                    req_id=req_id,
                    error=e,
                    meta=meta,
                )
                self._handle_failed_transfer(req_id, None)

        fut.add_done_callback(request_ready)

    def _is_heterogeneous_pp_engine(self, engine_id: str) -> bool:
        return (
            self._remote_pp_size_by_engine.get(engine_id, self.pp_world_size)
            != self.pp_world_size
            or engine_id in self._semantic_region_engines
        )

    def _compute_desc_ids_for_regions(
        self,
        block_ids: list[int],
        region_ids: np.ndarray,
        num_blocks: int,
        block_size_ratio: float | None = None,
    ) -> np.ndarray:
        if block_size_ratio is not None:
            num_blocks = int(num_blocks * block_size_ratio)
        if len(block_ids) == 0 or len(region_ids) == 0:
            return np.asarray([], dtype=np.int64)
        return (
            region_ids[:, None] * num_blocks
            + np.asarray(block_ids, dtype=np.int64)[None, :]
        ).flatten()

    def _compute_mamba_desc_ids_for_regions(
        self,
        block_ids: list[int],
        region_ids: np.ndarray,
        num_physical_blocks: int,
        physical_blocks_per_logical: int,
        num_fa_regions: int,
        num_fa_components: int = 1,
    ) -> np.ndarray:
        if len(block_ids) == 0 or len(region_ids) == 0:
            return np.asarray([], dtype=np.int64)
        assert self._conv_decomp is not None
        num_logical_blocks = num_physical_blocks // physical_blocks_per_logical
        regions_per_layer = len(self._conv_decomp.local_conv_offsets) + 1
        return (
            num_fa_regions * num_fa_components * num_physical_blocks
            + (
                region_ids[:, None, None] * regions_per_layer
                + np.arange(regions_per_layer, dtype=np.int64)[None, :, None]
            )
            * num_logical_blocks
            + np.asarray(block_ids, dtype=np.int64)[None, None, :]
        ).flatten()

    @staticmethod
    def _compute_fa_component_desc_ids(
        block_ids: list[int],
        region_ids: np.ndarray,
        num_blocks: int,
    ) -> np.ndarray:
        if len(block_ids) == 0 or len(region_ids) == 0:
            return np.asarray([], dtype=np.int64)
        return (
            (
                region_ids[:, None, None] * 2
                + np.arange(2, dtype=np.int64)[None, :, None]
            )
            * num_blocks
            + np.asarray(block_ids, dtype=np.int64)[None, None, :]
        ).flatten()

    def _logical_to_kernel_block_ids(self, block_ids, ratio: int):
        if ratio == 1:
            return block_ids
        transfer_groups = getattr(
            getattr(self, "kv_cache_config", None),
            "transfer_groups",
            getattr(getattr(self, "kv_cache_config", None), "kv_cache_groups", ()),
        )
        if not transfer_groups or len(block_ids) <= len(transfer_groups):
            return super()._logical_to_kernel_block_ids(block_ids, ratio)
        prefix = super()._logical_to_kernel_block_ids(
            block_ids[: len(transfer_groups)], ratio
        )
        # Extra groups from heterogeneous PP Mamba group splitting are SSM
        # groups whose block IDs are single state slots (never ratio-expanded).
        return list(prefix) + [
            list(group) for group in block_ids[len(transfer_groups) :]
        ]

    def _read_blocks_for_req(self, req_id: str, meta):
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        if not self._is_heterogeneous_pp_engine(engine_id):
            return super()._read_blocks_for_req(req_id, meta)

        self._engine_last_active[engine_id] = time.perf_counter()
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids

        # abort 清理伪请求 / 本地 prefix cache 全命中时会拿到空的 local_block_ids（或空的 remote_block_ids）。
        # 此时 decode 侧没有任何本地块需要写入，只需向远端各 PP/TP rank 发送释放通知。
        # 若不跳过下面的 route 对齐检查，会触发 route.local_group_idx >= len([]) 即 0 >= 0 恒真而 raise。
        if not any(local_block_ids) or not any(remote_block_ids):
            for remote_key in sorted(self._remote_agents.get(engine_id, ())):
                notif_id = self._make_pp_aware_notif_id(
                    meta.remote.request_id, engine_id, remote_key
                )
                self.nixl_wrapper.send_notif(
                    self._remote_agents[engine_id][remote_key],
                    notif_msg=notif_id,
                )
            if getattr(meta, "awaiting_kvs", False):
                self._recving_transfers.setdefault(req_id, [])
            return

        engine_routes = getattr(self, "_mamba_hma_routes", {}).get(engine_id)
        transfer_group_ids = list(range(len(local_block_ids)))
        if engine_routes:
            for routes in engine_routes.values():
                for route in routes:
                    if route.local_group_idx >= len(
                        local_block_ids
                    ) or route.remote_group_idx >= len(remote_block_ids):
                        raise RuntimeError(
                            "Cannot align producer and consumer KV cache groups "
                            "for Mamba HMA route: "
                            f"local_group={route.local_group_idx}/"
                            f"{len(local_block_ids)}, "
                            f"remote_group={route.remote_group_idx}/"
                            f"{len(remote_block_ids)}"
                        )
        elif len(remote_block_ids) < len(local_block_ids):
            # A decode-only speculative model owns an additional EAGLE KV group.
            # That group has no producer-side blocks and must stay local to decode.
            # Keep the common target-model groups positionally aligned for transfer.
            transfer_groups = getattr(
                self.kv_cache_config,
                "transfer_groups",
                self.kv_cache_config.kv_cache_groups,
            )
            transfer_group_ids = [
                group_id
                for group_id, group in enumerate(transfer_groups)
                if not group.is_eagle_group
            ]
            if len(transfer_group_ids) != len(remote_block_ids):
                raise RuntimeError(
                    "Cannot align producer and consumer KV cache groups: "
                    f"producer={len(remote_block_ids)}, "
                    f"consumer={len(local_block_ids)}, "
                    f"transferable_consumer={len(transfer_group_ids)}"
                )

        remote_keys = sorted(self._remote_agents[engine_id])
        for remote_key in remote_keys:
            remote_tp_rank = remote_key % remote_info.remote_tp_size
            if engine_routes:
                read_spec = ReadSpec(
                    remote_rank=remote_key,
                    local_block_ids=[
                        list(local_block_ids[local_group_id])
                        if local_group_id < len(plan.source_ranks_per_group)
                        and remote_tp_rank
                        in plan.source_ranks_per_group[local_group_id]
                        else []
                        for local_group_id in range(len(local_block_ids))
                    ],
                    remote_block_ids=[
                        list(group_blocks) for group_blocks in remote_block_ids
                    ],
                )
            else:
                read_spec = ReadSpec(
                    remote_rank=remote_key,
                    local_block_ids=[
                        list(local_block_ids[local_group_id])
                        if remote_tp_rank in plan.source_ranks_per_group[local_group_id]
                        else []
                        for local_group_id in transfer_group_ids
                    ],
                    remote_block_ids=[
                        list(remote_block_ids[remote_group_id])
                        if remote_tp_rank in plan.source_ranks_per_group[local_group_id]
                        else []
                        for remote_group_id, local_group_id in enumerate(
                            transfer_group_ids
                        )
                    ],
                )
            if not any(read_spec.local_block_ids):
                if self.use_mla:
                    notif_id = self._make_pp_aware_notif_id(
                        meta.remote.request_id, engine_id, remote_key
                    )
                    self.nixl_wrapper.send_notif(
                        self._remote_agents[engine_id][remote_key],
                        notif_msg=notif_id,
                    )
                continue

            if self._needs_split_local_xfer_handles(tp_ratio, plan):
                split_idx = plan.all_source_ranks.index(remote_tp_rank)
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[tp_ratio][
                    split_idx
                ]
            else:
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_info.remote_block_size
                ]
            remote_xfer_side_handle = self.dst_xfer_side_handles[engine_id][remote_key]
            self._read_blocks(
                read_spec=read_spec,
                request_id=req_id,
                dst_engine_id=engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
                # >>> ILU MODIFICATION >>>
                # Forward community v0.31.0 ``awaiting_kvs`` so parked requests
                # stay tracked. local_dram_handle defaults to None: the ILU
                # HMA path does not stage to DRAM.
                # <<< ILU MODIFICATION <<<
                awaiting_kvs=getattr(meta, "awaiting_kvs", False),
            )

    def _read_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
        expected_consumers: int = 1,
        local_dram_handle: int | None = None,
        awaiting_kvs: bool = False,
    ):
        if not self._is_heterogeneous_pp_engine(dst_engine_id):
            return super()._read_blocks(
                read_spec,
                dst_engine_id,
                request_id,
                remote_request_id,
                local_xfer_side_handle,
                local_dram_handle,
                remote_xfer_side_handle,
                expected_consumers=expected_consumers,
                awaiting_kvs=awaiting_kvs,
            )

        assert self.transfer_topo is not None
        remote_key = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids
        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )

        notif_id = self._make_pp_aware_notif_id(
            remote_request_id, dst_engine_id, remote_key
        )
        if len(local_block_ids) == 0 or not any(local_block_ids):
            agent_name = self._remote_agents[dst_engine_id][remote_key]
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            if awaiting_kvs:
                self._recving_transfers.setdefault(request_id, [])
            return

        routes = (
            getattr(self, "_mamba_hma_routes", {})
            .get(dst_engine_id, {})
            .get(remote_key)
        )
        local_descs_list: list[np.ndarray] = []
        remote_descs_list: list[np.ndarray] = []

        if routes is not None:
            num_local_groups = len(local_block_ids)
            num_remote_fa_regions = (
                getattr(self, "_remote_fa_num_regions", {})
                .get(dst_engine_id, {})
                .get(remote_key)
            )
            if num_remote_fa_regions is None:
                num_remote_fa_regions = max(
                    (
                        int(r.remote_desc_region_ids.max()) + 1
                        for r in routes
                        if not r.is_ssm and len(r.remote_desc_region_ids)
                    ),
                    default=0,
                )
            for route in routes:
                group_local_blocks = list(local_block_ids[route.local_group_idx])
                if not group_local_blocks:
                    continue
                group_remote_blocks = list(remote_block_ids[route.remote_group_idx])
                padded_local = [[] for _ in range(num_local_groups)]
                padded_remote = [[] for _ in range(num_local_groups)]
                padded_local[route.local_group_idx] = group_local_blocks
                padded_remote[route.local_group_idx] = group_remote_blocks
                aligned_local, aligned_remote = self._apply_prefix_caching(
                    decode_block_ids=padded_local,
                    prefill_block_ids=padded_remote,
                    decode_physical_per_logical=(
                        self._physical_blocks_per_logical_kv_block
                    ),
                    prefill_physical_per_logical=(
                        remote_info.remote_physical_blocks_per_logical
                    ),
                )
                group_local_blocks = aligned_local[route.local_group_idx]
                group_remote_blocks = aligned_remote[route.local_group_idx]
                if not group_local_blocks:
                    continue
                if route.is_ssm:
                    local_descs_list.append(
                        self._compute_mamba_desc_ids_for_regions(
                            group_local_blocks,
                            route.local_desc_region_ids,
                            self.dst_num_blocks[self.engine_id],
                            self._physical_blocks_per_logical_kv_block,
                            self.num_regions,
                            1,
                        )
                    )
                    remote_descs_list.append(
                        self._compute_mamba_desc_ids_for_regions(
                            group_remote_blocks,
                            route.remote_desc_region_ids,
                            self.dst_num_blocks[dst_engine_id],
                            remote_info.remote_physical_blocks_per_logical,
                            num_remote_fa_regions,
                            1,
                        )
                    )
                else:
                    local_descs_list.append(
                        self._compute_desc_ids_for_regions(
                            group_local_blocks,
                            route.local_desc_region_ids,
                            self.dst_num_blocks[self.engine_id],
                            block_size_ratio,
                        )
                    )
                    remote_descs_list.append(
                        self._compute_desc_ids_for_regions(
                            group_remote_blocks,
                            route.remote_desc_region_ids,
                            self.dst_num_blocks[dst_engine_id],
                            None,
                        )
                    )
        else:
            # Local blocks are the decode side; remote blocks are the prefill side.
            # The two physical-per-logical ratios are independent under hetero PD.
            local_block_ids, remote_block_ids = self._apply_prefix_caching(
                decode_block_ids=local_block_ids,
                prefill_block_ids=remote_block_ids,
                decode_physical_per_logical=(
                    self._physical_blocks_per_logical_kv_block
                ),
                prefill_physical_per_logical=(
                    remote_info.remote_physical_blocks_per_logical
                ),
            )

            local_region_ids_by_group = self.local_region_ids_by_group[dst_engine_id][
                remote_key
            ]
            remote_region_ids_by_group = self.remote_region_ids_by_group[dst_engine_id][
                remote_key
            ]

            for group_idx, (group_local_blocks, group_remote_blocks) in enumerate(
                zip(local_block_ids, remote_block_ids)
            ):
                if not group_local_blocks:
                    continue
                if _is_ssm_spec(self._group_spec_types[group_idx]):
                    remote_num_regions = (
                        getattr(self, "_remote_fa_num_regions", {})
                        .get(dst_engine_id, {})
                        .get(remote_key)
                    )
                    if remote_num_regions is None:
                        remote_num_regions = max(
                            (
                                int(ids.max()) + 1
                                for ids in remote_region_ids_by_group
                                if len(ids)
                            ),
                            default=0,
                        )
                    local_descs_list.append(
                        self._compute_mamba_desc_ids_for_regions(
                            group_local_blocks,
                            local_region_ids_by_group[group_idx],
                            self.dst_num_blocks[self.engine_id],
                            self._physical_blocks_per_logical_kv_block,
                            self.num_regions,
                            1,
                        )
                    )
                    remote_descs_list.append(
                        self._compute_mamba_desc_ids_for_regions(
                            group_remote_blocks,
                            remote_region_ids_by_group[group_idx],
                            self.dst_num_blocks[dst_engine_id],
                            remote_info.remote_physical_blocks_per_logical,
                            remote_num_regions,
                            1,
                        )
                    )
                    continue
                local_descs_list.append(
                    self._compute_desc_ids_for_regions(
                        group_local_blocks,
                        local_region_ids_by_group[group_idx],
                        self.dst_num_blocks[self.engine_id],
                        block_size_ratio,
                    )
                )
                remote_descs_list.append(
                    self._compute_desc_ids_for_regions(
                        group_remote_blocks,
                        remote_region_ids_by_group[group_idx],
                        self.dst_num_blocks[dst_engine_id],
                        None,
                    )
                )

        if not local_descs_list:
            agent_name = self._remote_agents[dst_engine_id][remote_key]
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            if awaiting_kvs:
                self._recving_transfers.setdefault(request_id, [])
            return

        local_block_descs_ids = np.concatenate(local_descs_list)
        remote_block_descs_ids = np.concatenate(remote_descs_list)
        assert len(local_block_descs_ids) == len(remote_block_descs_ids), (
            "Heterogeneous PP HMA descriptor count mismatch: "
            f"local={len(local_block_descs_ids)}, "
            f"remote={len(remote_block_descs_ids)}"
        )

        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )
            self.nixl_wrapper.transfer(handle)
            self._recving_transfers[request_id].append(handle)
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Marking blocks as invalid",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_key,
            )
            self._handle_failed_transfer(request_id, handle)

    def _send_heartbeats(self, metadata) -> None:
        for engine_id, hb_info in metadata.heartbeat_by_engine.items():
            if (
                self._ensure_handshake(
                    engine_id,
                    hb_info.host,
                    hb_info.port,
                    hb_info.tp_size,
                    getattr(hb_info, "pp_size", 1),
                    getattr(hb_info, "pp_split", ""),
                )
                is not None
            ):
                continue

            hb_msg = ("HB:" + ",".join(hb_info.req_ids)).encode()
            for agent_name in self._remote_agents[engine_id].values():
                try:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=hb_msg)
                except Exception:
                    logger.debug(
                        "Failed to send heartbeat to engine %s",
                        engine_id,
                        exc_info=True,
                    )

    def export_fp8_wire_staging(self, metadata) -> None:
        if not self.fp8_wire_staging_enabled:
            return
        if self.fp8_wire_staging is None:
            raise RuntimeError("IluNixl FP8 DS MLA wire staging is not initialized")
        self.fp8_wire_staging.export_requests(
            [
                (req_meta.local_block_ids, req_meta.num_tokens)
                for req_meta in metadata.reqs_to_save.values()
            ]
        )
