# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side KV memory registration, virtual-packed layouts, and compat hashing."""

from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from functools import cached_property
from typing import Any

import msgspec
import numpy as np
import torch
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    _is_ssm_spec,
)
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import KpoolTailSpec, UniformTypeKVCacheSpecs

from ..ilu_nixl_common.memory_registration import MemoryRegistrationPlanner
from ..ilu_nixl_common.semantic_pairing import (
    cache_kind_from_layer_name,
    transfer_identity,
)
from ..ilu_nixl_common.virtual_packed_reg import (
    append_paired_scale_regions,
    build_packed_virtual_layout,
    build_region_ids_by_group,
    exclusive_cumsum,
    is_virtual_packed_layout,
    layer_specs_for_config,
    layer_to_groups,
    virtual_slot_desc,
)
from ..skip_layers import get_skip_layers
from ..ilu_nixl_common.dsv41 import (
    resolve_dsv41_layout,
    resolve_indexer_kv_layout_name,
    resolve_main_kv_format_name,
)
from .metadata import (
    IluNixlAgentMetadata,
    IluNixlHandshakePayload,
    IluNixlTransferRegion,
)

logger = init_logger(__name__)

_SEMANTIC_COMPAT_BACKEND = "ILU_NIXL_SEMANTIC_REGIONS_V1"


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


class IluNixlRegistrationMixin:
    """Registration & handshake-metadata construction mixin for IluNixl workers."""

    def _sync_block_size_with_kernel(self) -> None:
        self._scheduler_block_size = self.block_size
        if resolve_dsv41_layout(self.vllm_config) is not None:
            self._logical_num_blocks = self.num_blocks
            self._physical_blocks_per_logical_kv_block = 1
            return
        super()._sync_block_size_with_kernel()

    def _cross_layers_blocks_enabled(self) -> bool:
        transfer_config = getattr(self, "kv_transfer_config", None)
        if transfer_config is None:
            return False
        value = transfer_config.get_from_extra_config(
            "enable_cross_layers_blocks", False
        )
        return str(value).lower() == "true"

    def _should_split_virtual_region(self) -> bool:
        return bool(
            self.transfer_topo is not None
            and getattr(
                self.transfer_topo, "virtually_split_kv_in_blocks", False
            )
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
        topo_kwargs: dict[str, Any] = dict(
            tp_rank=getattr(self, "transfer_tp_rank", self.tp_rank),
            tp_size=getattr(self, "transfer_tp_size", self.world_size),
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla if is_mla is None else is_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=attn_backends or self.attn_backends,
            tensor_shape=tensor_shape,
            is_mamba=self._has_mamba if is_mamba is None else is_mamba,
        )
        try:
            topology = TransferTopology(
                **topo_kwargs,
                dcp_size=getattr(self, "dcp_size", 1),
            )
        except TypeError:
            topology = TransferTopology(**topo_kwargs)
        self.compat_hash = self._compute_compatibility_hash(
            self._cross_layers_blocks_enabled()
        )
        return topology

    @staticmethod
    def _exclusive_cumsum(counts: list[int]) -> np.ndarray:
        return exclusive_cumsum(counts)

    @staticmethod
    def _layer_specs_for_config(kv_cache_config) -> dict[str, Any]:
        return layer_specs_for_config(kv_cache_config)

    @staticmethod
    def _layer_to_groups(kv_cache_config) -> dict[str, list[int]]:
        return layer_to_groups(kv_cache_config)

    @staticmethod
    def _is_packed_layout(kv_cache_config, kv_caches: dict[str, Any]) -> bool:
        return is_virtual_packed_layout(kv_cache_config, kv_caches)

    @staticmethod
    def _virtual_slot_desc(
        backing_base: int,
        block_id: int,
        block_stride: int,
        slot_offset: int,
        page_size: int,
        device_id: int,
    ) -> tuple[int, int, int]:
        return virtual_slot_desc(
            backing_base,
            block_id,
            block_stride,
            slot_offset,
            page_size,
            device_id,
        )

    def _build_region_ids_by_group(
        self, layer_region_ids: dict[str, list[int]]
    ) -> list[np.ndarray]:
        return build_region_ids_by_group(
            self.kv_cache_config,
            layer_region_ids,
            getattr(self, "num_regions", 0),
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
        return build_packed_virtual_layout(
            kv_caches,
            layout_config,
            block_id_config,
            is_dsv41=getattr(self, "dsv41_layout", None) is not None,
            region_cls=IluNixlTransferRegion,
        )

    def _append_paired_scale_regions(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_scales: dict[str, torch.Tensor] | None,
        layout_config,
        block_id_config,
        physical_data: list[tuple[int, int, int, str]],
        base_addresses: list[int],
    ) -> None:
        self.region_ids_by_group = append_paired_scale_regions(
            vllm_config=self.vllm_config,
            device_id=self.device_id,
            kv_caches=kv_caches,
            kv_cache_scales=kv_cache_scales,
            layout_config=layout_config,
            block_id_config=block_id_config,
            physical_data=physical_data,
            base_addresses=base_addresses,
            layer_region_ids=self.layer_region_ids,
            transfer_regions=self.transfer_regions,
            region_is_mla=self._region_is_mla,
            region_cls=IluNixlTransferRegion,
        )

    def _actual_to_virtual_region_ids(self) -> dict[int, list[int]]:
        mapping: dict[int, list[int]] = {}
        next_region_id = 0
        for actual_region_id in range(len(self.block_len_per_layer)):
            mapping[actual_region_id] = [next_region_id]
            next_region_id += 1
            if (
                self._should_split_virtual_region()
                and not self._is_region_replicated(actual_region_id)
            ):
                mapping[actual_region_id].append(next_region_id)
                next_region_id += 1
        return mapping

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
            if (
                self._should_split_virtual_region()
                and not self._is_region_replicated(actual_region_id)
            ):
                if region_id == next_region_id:
                    return block_len // 2
                next_region_id += 1
        return self.block_len_per_layer[region_id]

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

        l_to_groups = self._layer_to_groups(self.kv_cache_config)
        layer_specs = self._layer_specs_for_config(self.kv_cache_config)
        region_num_blocks_list = getattr(self, "region_num_blocks", None)
        default_num_blocks = int(getattr(self, "num_blocks", 0))
        kv_caches = getattr(self, "device_kv_caches", {}) or {}
        logical_num_blocks = int(
            getattr(self, "_logical_num_blocks", default_num_blocks)
            or default_num_blocks
        )

        transfer_regions: list[IluNixlTransferRegion] = []
        for region_id in sorted(region_to_layers):
            shared_by = sorted(set(region_to_layers[region_id]))
            layer_indices: list[int] = []
            cache_kinds: set[str] = set()
            identities: set[str] = set()
            identity_groups: dict[str, list[int]] = defaultdict(list)
            for layer_name in shared_by:
                with suppress(AssertionError, ValueError):
                    layer_indices.append(extract_layer_index(layer_name))
                cache_kinds.add(cache_kind_from_layer_name(layer_name))
                ident = transfer_identity(layer_name)
                identities.add(ident)
                for group_idx in l_to_groups.get(layer_name, []):
                    if group_idx not in identity_groups[ident]:
                        identity_groups[ident].append(group_idx)

            sample_spec = next(
                (layer_specs[name] for name in shared_by if name in layer_specs),
                None,
            )
            reg_num_blocks = (
                int(region_num_blocks_list[region_id])
                if region_num_blocks_list is not None
                and region_id < len(region_num_blocks_list)
                else default_num_blocks
            )
            phys_per_logical = int(
                getattr(self, "_physical_blocks_per_logical_kv_block", 1) or 1
            )
            if phys_per_logical != 1:
                # Non-packed registration already split each logical block into
                # physical kernel pages of ``self.block_size`` tokens.
                kernel_blocks = 1
                block_tokens = int(getattr(self, "block_size", 0) or 0)
            else:
                block_tokens = max(
                    0, int(getattr(sample_spec, "block_size", 0) or 0)
                )
                sample_cache = next(
                    (kv_caches[name] for name in shared_by if name in kv_caches),
                    None,
                )
                kernel_blocks = 1
                if (
                    sample_cache is not None
                    and sample_spec is not None
                    and logical_num_blocks > 0
                ):
                    with suppress(ValueError):
                        kernel_blocks = (
                            MemoryRegistrationPlanner.kernel_blocks_per_block(
                                layer_name=shared_by[0],
                                tensor=sample_cache,
                                spec=sample_spec,
                                num_blocks=logical_num_blocks,
                            )
                        )
            block_len = self._block_len_for_virtual_region(region_id)
            transfer_regions.append(
                IluNixlTransferRegion(
                    region_id=region_id,
                    block_len=block_len,
                    shared_by=shared_by,
                    layer_indices=sorted(set(layer_indices)),
                    group_ids=sorted(set(region_to_groups.get(region_id, []))),
                    cache_kinds=sorted(cache_kinds),
                    identities=sorted(identities),
                    physical_backing_id=region_id,
                    byte_offset=0,
                    block_stride=block_len,
                    kernel_blocks=kernel_blocks,
                    block_tokens=block_tokens,
                    num_blocks=reg_num_blocks,
                    identity_groups={
                        k: sorted(v) for k, v in identity_groups.items()
                    },
                )
            )
        return transfer_regions

    def _dump_registered_regions(self) -> None:
        tp_size = getattr(self, "world_size", 1)
        pp_size = getattr(self, "pp_world_size", 1)
        lines = [
            f"[ILU-NIXL] engine={self.engine_id} "
            f"pp={getattr(self, 'pp_rank', 0)}/{pp_size} "
            f"tp={getattr(self, 'tp_rank', 0)}/{tp_size} "
            f"regions={len(self.transfer_regions)}"
        ]
        for region in self.transfer_regions:
            label = ",".join(region.shared_by[:2])
            if len(region.shared_by) > 2:
                label += f",+{len(region.shared_by) - 2}"
            groups_str = ",".join(str(g) for g in region.group_ids)
            lines.append(
                f"  r{region.region_id:<2d} {label:<36s} "
                f"kernel_blocks={region.kernel_blocks} "
                f"block_tokens={region.block_tokens} "
                f"num_blocks={region.num_blocks or self.num_blocks} "
                f"len={region.block_len} stride={region.block_stride} "
                f"group={groups_str}"
            )
        skipped = getattr(self, "_skipped_layers", ())
        if skipped:
            lines.append("  skipped: " + ", ".join(skipped))
        logger.info("\n".join(lines))

    def _set_handshake_metadata(
        self,
        *,
        kv_cache_layout: str | None = None,
        memory_type: str | None = None,
    ) -> None:
        dsv41 = getattr(self, "dsv41_layout", None)
        # Wire staging publishes D-facing FP8 pages; advertise wire tags, not
        # the producer's compressed compute codecs (i8g64 / split indexer).
        if getattr(self, "fp8_wire_staging_enabled", False):
            main_kv_fmt = "none"
            idx_kv_layout = "legacy"
        else:
            main_kv_fmt = (
                dsv41.main_kv_format
                if dsv41 is not None
                else resolve_main_kv_format_name()
            )
            idx_kv_layout = (
                dsv41.indexer_kv_layout
                if dsv41 is not None
                else resolve_indexer_kv_layout_name()
            )
        agent_metadata = IluNixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=(
                self.kv_caches_base_addr[self.engine_id][self.tp_rank]
            ),
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            block_strides=getattr(
                self, "block_stride_per_layer", self.block_len_per_layer
            ),
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
            virtual_packed=getattr(self, "_virtual_packed_regions", False),
            kv_layer_end=dsv41.kv_layer_end if dsv41 is not None else None,
            swa_bounded_replay=(
                dsv41.swa_bounded_replay if dsv41 is not None else None
            ),
            skip_layers=list(getattr(self, "_skipped_layers", ())) or None,
            kv_source_layers=(
                list(dsv41.kv_source_layers) if dsv41 is not None else None
            ),
            index_source_layers=(
                list(dsv41.index_source_layers) if dsv41 is not None else None
            ),
            compress_ratios=(
                list(dsv41.compress_ratios) if dsv41 is not None else None
            ),
            main_kv_format=main_kv_fmt,
            indexer_kv_layout=idx_kv_layout,
            supports_remote_block_starts=True,
            supports_structured_notif=True,
        )
        assert self.compat_hash is not None
        self.xfer_handshake_metadata = IluNixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=msgspec.msgpack.encode(agent_metadata),
            main_kv_format=main_kv_fmt,
            indexer_kv_layout=idx_kv_layout,
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

    def _uses_kv_first_fa_layout(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> bool:
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

    def _register_kv_first_fa_caches(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
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

        l_to_groups = self._layer_to_groups(self.kv_cache_config)

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
                group_ids = l_to_groups.get(layer_name, [])
                cache_kind = f"full_attn_component_{component_idx}"
                try:
                    layer_idx = extract_layer_index(layer_name)
                except (AssertionError, ValueError):
                    layer_idx = -1
                ident = f"{layer_idx}:{cache_kind}"
                transfer_regions.append(
                    IluNixlTransferRegion(
                        region_id=region_id,
                        block_len=block_len,
                        shared_by=[layer_name],
                        layer_indices=[layer_idx] if layer_idx >= 0 else [],
                        group_ids=group_ids,
                        cache_kinds=[cache_kind],
                        identities=[ident],
                        physical_backing_id=region_id,
                        byte_offset=0,
                        block_stride=block_len,
                        kernel_blocks=1,
                        block_tokens=int(getattr(layer_spec, "block_size", 0) or 0),
                        num_blocks=self.num_blocks,
                        identity_groups={ident: sorted(group_ids)},
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
        self.region_num_blocks = [
            region.num_blocks or self.num_blocks
            for region in self.transfer_regions
        ]
        self.num_regions = len(self.transfer_regions)
        self.num_descs = sum(self.region_num_blocks)
        self._local_desc_cumsum = self._exclusive_cumsum(self.region_num_blocks)
        self._finish_memory_registration(kv_caches, caches_data, base_addresses)
        self._dump_registered_regions()

    def _register_empty_rank(self, original_kv_caches: dict[str, Any]) -> None:
        logical_num_blocks = int(
            getattr(self, "_logical_num_blocks", self.num_blocks)
        )
        if self._physical_blocks_per_logical_kv_block != 1:
            self.num_blocks = logical_num_blocks
            self.block_size = int(
                getattr(self, "_scheduler_block_size", self.block_size)
            )
            self._physical_blocks_per_logical_kv_block = 1
        self._virtual_packed_regions = True
        transfer_cfg = getattr(self, "kv_transfer_config", None)
        self._empty_producer_rank = (
            getattr(transfer_cfg, "kv_role", None) == "kv_producer"
        )
        if original_kv_caches:
            sample = next(iter(original_kv_caches.values()))
            self.device_id = (
                0 if sample.device.type == "cpu" else max(sample.get_device(), 0)
            )
        self.layer_region_ids = {}
        self.transfer_regions = []
        self._region_is_mla = []
        self.region_ids_by_group = self._build_region_ids_by_group({})
        self.block_len_per_layer = []
        self.block_stride_per_layer = []
        self.region_num_blocks = []
        self.region_mem_types = []
        self.num_regions = 0
        self.num_descs = 0
        self._local_desc_cumsum = np.zeros(0, dtype=np.int64)
        self.transfer_topo = self._new_transfer_topology()
        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = []
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        self.dst_region_mem_types[self.engine_id] = []
        self.device_kv_caches = {}
        self._set_handshake_metadata(memory_type=self.nixl_memory_type)
        self._dump_registered_regions()

    def _register_virtual_packed_kv_cache(
        self,
        kv_caches: dict[str, torch.Tensor],
        layout_config,
        block_id_config,
        *,
        kv_cache_scales: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if self.use_host_buffer:
            raise ValueError(
                "Packed virtual IluNixl regions cannot use a host transfer buffer"
            )
        logical_num_blocks = int(
            getattr(self, "_logical_num_blocks", self.num_blocks)
        )
        if getattr(self, "_physical_blocks_per_logical_kv_block", 1) != 1:
            self.num_blocks = logical_num_blocks
            self.block_size = int(
                getattr(self, "_scheduler_block_size", self.block_size)
            )
            self._physical_blocks_per_logical_kv_block = 1

        if layout_config.num_blocks != self.num_blocks:
            raise ValueError(
                "Packed layout and connector block counts differ: "
                f"{layout_config.num_blocks} != {self.num_blocks}"
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

        (
            self.layer_region_ids,
            self.region_ids_by_group,
            self.transfer_regions,
            self._region_is_mla,
        ) = self._build_packed_virtual_layout(kv_caches, layout_config, block_id_config)

        physical_data = [(base_addr, storage.nbytes(), device_id, "")]
        base_addresses = [base_addr]
        self._append_paired_scale_regions(
            kv_caches,
            kv_cache_scales,
            layout_config,
            block_id_config,
            physical_data,
            base_addresses,
        )

        self.block_len_per_layer = [
            region.block_len for region in self.transfer_regions
        ]
        self.block_stride_per_layer = [
            region.block_stride for region in self.transfer_regions
        ]
        self.region_num_blocks = [
            region.num_blocks or self.num_blocks
            for region in self.transfer_regions
        ]
        self.num_regions = len(self.transfer_regions)
        self.num_descs = sum(self.region_num_blocks)
        self._local_desc_cumsum = self._exclusive_cumsum(self.region_num_blocks)
        self._virtual_packed_regions = True

        self.transfer_topo = self._new_transfer_topology()
        if self._cross_layers_blocks_enabled():
            raise ValueError(
                "Packed virtual IluNixl regions do not support uniform "
                "cross-layer KV cache allocation"
            )

        self._finish_memory_registration(
            kv_caches,
            physical_data,
            base_addresses,
            memory_type=backing_memory_type,
        )
        self._dump_registered_regions()

    def register_cross_layers_kv_caches(self, kv_cache: torch.Tensor) -> None:
        if self.fp8_wire_staging_enabled:
            raise ValueError(
                "IluNixl FP8 DS MLA wire staging does not support uniform "
                "cross-layer KV cache allocation"
            )
        super().register_cross_layers_kv_caches(kv_cache)

    def register_kv_caches(
        self,
        kv_caches: dict[str, Any],
        kv_cache_scales: dict[str, torch.Tensor] | None = None,
    ):
        if self.fp8_wire_staging_enabled:
            from .fp8_layout import get_fp8_wire_block_size
            from .fp8_staging import FP8DSMLAWireStaging

            wire_bs = get_fp8_wire_block_size(self.vllm_config)
            skip_layers = get_skip_layers()
            if skip_layers:
                active_compute_caches = {
                    layer_name: cache
                    for layer_name, cache in kv_caches.items()
                    if layer_name not in skip_layers and "mtp" not in layer_name
                }
                self._skipped_layers = sorted(
                    layer_name
                    for layer_name in kv_caches
                    if layer_name not in active_compute_caches
                )
            else:
                active_compute_caches = kv_caches
            if not active_compute_caches:
                if wire_bs is not None:
                    self.block_size = wire_bs
                self.fp8_wire_staging = None
                self._register_empty_rank(kv_caches)
                return

            self.compute_device_kv_caches = kv_caches
            staging = FP8DSMLAWireStaging(
                self.vllm_config,
                self.kv_cache_config,
                active_compute_caches,
                kv_cache_scales=kv_cache_scales,
            )
            self.fp8_wire_staging = staging
            if wire_bs is not None:
                self.block_size = wire_bs
            self._register_virtual_packed_kv_cache(
                staging.wire_caches,
                staging.wire_config,
                self.kv_cache_config,
            )
            return

        skip_layers = get_skip_layers()
        active_kv_caches = {
            layer_name: cache
            for layer_name, cache in kv_caches.items()
            if layer_name not in skip_layers and "mtp" not in layer_name
        }
        self._skipped_layers = sorted(
            layer_name for layer_name in kv_caches if layer_name not in active_kv_caches
        )
        if not active_kv_caches:
            self._register_empty_rank(kv_caches)
            return

        # Packed virtual descriptors emit one slot per logical (scheduler)
        # block. Hybrid Mamba+Attention models whose kernel page size is smaller
        # than the mamba-aligned scheduler block (e.g. GLM-5.3 2304/4480 -> 64)
        # must use upstream non-packed kernel-page registration below so both P
        # and D advertise the uniform 64-token kernel page size. DeepSeek-V4.1
        # intentionally registers scheduler blocks directly with per-region
        # kernel_blocks metadata.
        if self._is_packed_layout(self.kv_cache_config, active_kv_caches) and (
            getattr(self, "_physical_blocks_per_logical_kv_block", 1) == 1
            or getattr(self, "dsv41_layout", None) is not None
        ):
            self._register_virtual_packed_kv_cache(
                active_kv_caches,
                self.kv_cache_config,
                self.kv_cache_config,
                kv_cache_scales=kv_cache_scales,
            )
            return

        if self._uses_kv_first_fa_layout(active_kv_caches):
            self._register_kv_first_fa_caches(active_kv_caches)
            return

        if getattr(self, "_mask_pp_during_register", False):
            original_pp_size = self.pp_size
            self.pp_size = 1
            try:
                super().register_kv_caches(active_kv_caches)
            finally:
                self.pp_size = original_pp_size
        else:
            super().register_kv_caches(active_kv_caches)

        self.layer_region_ids = self._build_layer_region_ids_from_kv_caches(
            active_kv_caches
        )
        self.region_ids_by_group = self._build_region_ids_by_group(
            self.layer_region_ids
        )
        self.transfer_regions = self._build_hma_transfer_regions(
            self.layer_region_ids, self.region_ids_by_group
        )
        self._local_desc_cumsum = self._exclusive_cumsum(
            [
                region.num_blocks or self.num_blocks
                for region in self.transfer_regions
            ]
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
        self._dump_registered_regions()

    @cached_property
    def _attention_kv_caches(self) -> list[torch.Tensor]:
        """Attention KV caches aligned to ``self.num_blocks`` for stale-page zeroing."""
        assert self.device_kv_caches, (
            "_attention_kv_caches accessed before register_kv_caches"
        )
        transfer_groups = getattr(
            self.kv_cache_config,
            "transfer_groups",
            self.kv_cache_config.kv_cache_groups,
        )
        mamba_layers = {
            name
            for g, group in enumerate(transfer_groups)
            if g < len(self._group_spec_types)
            and _is_ssm_spec(self._group_spec_types[g])
            for name in group.layer_names
        }
        layer_specs = getattr(self, "_layer_specs", {}) or {}
        caches: list[torch.Tensor] = []
        for name, cache in self.device_kv_caches.items():
            if name in mamba_layers:
                continue
            spec = layer_specs.get(name)
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs.get(name)
            if isinstance(spec, KpoolTailSpec) and cache.shape[0] != self.num_blocks:
                # Covered by its co-located kpool indexer cache region.
                continue
            if (
                getattr(self, "use_mla", False)
                and self.num_blocks > 0
                and cache.shape[0] != self.num_blocks
                and cache.is_contiguous()
                and cache.nbytes % self.num_blocks == 0
            ):
                caches.append(cache.view(torch.uint8).view(self.num_blocks, -1))
            else:
                caches.append(cache)
        return caches

    def export_fp8_wire_staging(self, metadata) -> None:
        if not self.fp8_wire_staging_enabled:
            return
        if self.fp8_wire_staging is None:
            if getattr(self, "_empty_producer_rank", False):
                return
            raise RuntimeError("IluNixl FP8 DS MLA wire staging is not initialized")
        self.fp8_wire_staging.export_requests(
            [
                (
                    req_meta.local_block_ids,
                    req_meta.num_tokens,
                    getattr(req_meta, "sync_save", True),
                )
                for req_meta in metadata.reqs_to_save.values()
            ]
        )
