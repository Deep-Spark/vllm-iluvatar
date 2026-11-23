# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache registration mixin for IluNixlPushConnector (vLLM 0.31.0)."""

from __future__ import annotations

from typing import Any

import msgspec
import numpy as np
import torch
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    _is_attention_spec,
    _is_ssm_spec,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from ..ilu_nixl_common.dsv41 import (
    resolve_indexer_kv_layout_name,
    resolve_main_kv_format_name,
    wire_dsv41_layout,
)
from ..ilu_nixl_common.fp8_hetero.wire_layout import subpage_ratio_for
from ..ilu_nixl_common.semantic_pairing import regions_by_id
from ..ilu_nixl_common.virtual_packed_reg import (
    append_paired_scale_regions,
    build_packed_virtual_layout,
    build_region_ids_by_group,
    exclusive_cumsum,
    is_virtual_packed_layout,
    transfer_groups_of,
    virtual_slot_desc,
)
from ..skip_layers import get_skip_layers
from .metadata import (
    IluNixlPushAgentMetadata,
    IluNixlPushHandshakePayload,
    compute_ilu_push_compatibility_hash,
)

logger = init_logger(__name__)


class RegistrationMixin:
    """VRAM/DRAM KV cache registration, virtual packed layout, and host-wire pool setup."""

    @property
    def _worker_device_id(self) -> int:
        return self.device_id if self._cuda_device_id is None else self._cuda_device_id

    def _wire_cache_dtype(self) -> str:
        if self._hetero_cfg.enabled:
            return self._hetero_cfg.decode_cache_dtype
        return str(self.vllm_config.cache_config.cache_dtype)

    def _advertised_block_size(self) -> int:
        if self._hetero_cfg.enabled and self._hetero_cfg.decode_block_size > 0:
            return self._hetero_cfg.decode_block_size
        if self._wire_plan is None:
            return self.block_size
        return min(
            int(spec.block_size) for spec in self._wire_plan.wire_specs.values()
        )

    def _is_packed_kv_cache(self, kv_caches: dict[str, "torch.Tensor"]) -> bool:
        if not kv_caches or self._has_mamba or self.use_host_buffer:
            return False
        return is_virtual_packed_layout(self.kv_cache_config, kv_caches)

    def _compute_push_compat_hash(self) -> str:
        if getattr(self, "dsv41_layout", None) is not None or getattr(
            self, "_has_mamba", False
        ):
            return compute_ilu_push_compatibility_hash(
                self.vllm_config,
                transfer_mode=getattr(self, "_TRANSFER_MODE", "push"),
            )
        return compute_nixl_compatibility_hash(
            self.vllm_config,
            self.backend_name,
            transfer_mode=getattr(self, "_TRANSFER_MODE", "push"),
        )

    def _new_push_transfer_topology(self) -> TransferTopology:
        return TransferTopology(
            tp_rank=getattr(self, "transfer_tp_rank", self.tp_rank),
            tp_size=getattr(self, "transfer_tp_size", self.world_size),
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            total_num_kv_heads=(
                1 if self.use_mla else self.model_config.get_total_num_kv_heads()
            ),
            attn_backends=self.attn_backends,
            dcp_size=getattr(self, "dcp_size", 1),
            tensor_shape=None,
            is_mamba=self._has_mamba,
        )

    def _set_push_handshake_metadata(self) -> None:
        dsv41 = wire_dsv41_layout(
            getattr(self, "dsv41_layout", None),
            host_staging_producer=getattr(self, "_host_staging_producer", False),
            hetero_enabled=bool(
                getattr(getattr(self, "_hetero_cfg", None), "enabled", False)
            ),
        )
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
        block_strides = getattr(self, "block_stride_per_layer", None)
        if block_strides is None:
            block_strides = self.block_len_per_layer
        group_spec_types = getattr(self, "_group_spec_types", ())
        group_is_ssm = (
            [bool(_is_ssm_spec(t)) for t in group_spec_types]
            if group_spec_types
            else None
        )
        agent_metadata = IluNixlPushAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=(
                self.kv_caches_base_addr[self.engine_id][self.tp_rank]
            ),
            num_blocks=self.num_blocks,
            block_lens=list(self.block_len_per_layer),
            block_strides=list(block_strides),
            kv_cache_layout=(
                self.kv_cache_layout
                if not self.use_host_buffer
                else self.host_buffer_kv_cache_layout
            ),
            block_size=self._advertised_block_size(),
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
            wire_cache_dtype=self._wire_cache_dtype(),
            kv_buffer_memory_type=self.nixl_memory_type,
            layer_region_ids=self.layer_region_ids,
            region_ids_by_group=[
                region_ids.tolist() for region_ids in self.region_ids_by_group
            ],
            transfer_regions=self.transfer_regions,
            packed_block_stride=self.packed_block_stride,
            virtual_packed=self._virtual_packed_regions,
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
            group_is_ssm=group_is_ssm,
        )
        assert self.compat_hash is not None
        self.xfer_handshake_metadata = IluNixlPushHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=msgspec.msgpack.encode(agent_metadata),
            main_kv_format=main_kv_fmt,
            indexer_kv_layout=idx_kv_layout,
        )

    def _register_empty_rank(self, original_kv_caches: dict[str, Any]) -> None:
        """Handle a PP stage where all local KV views are skipped (e.g. V4.1 SWA/forwarder)."""
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
        self.packed_block_stride = 0
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
        self.region_ids_by_group = build_region_ids_by_group(
            self.kv_cache_config, {}, 0
        )
        self.block_len_per_layer = []
        self.block_stride_per_layer = []
        self.region_num_blocks = []
        self.region_group_ids = []
        self.region_names = []
        self.region_mem_types = []
        self.num_regions = 0
        self.num_descs = 0
        self._local_desc_cumsum = np.zeros(0, dtype=np.int64)
        self.transfer_topo = self._new_push_transfer_topology()
        self.compat_hash = self._compute_push_compat_hash()
        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = []
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        self.dst_region_mem_types[self.engine_id] = []
        self.device_kv_caches = original_kv_caches
        self._normalize_single_pp_split(self.layer_region_ids)
        self._set_push_handshake_metadata()
        self._start_push_writer()

    def _register_virtual_packed_kv_cache(
        self,
        kv_caches: dict[str, "torch.Tensor"],
        *,
        kv_cache_scales: dict[str, "torch.Tensor"] | None = None,
    ) -> None:
        logical_num_blocks = int(
            getattr(self, "_logical_num_blocks", self.num_blocks)
        )
        if getattr(self, "_physical_blocks_per_logical_kv_block", 1) != 1:
            self.num_blocks = logical_num_blocks
            self.block_size = int(
                getattr(self, "_scheduler_block_size", self.block_size)
            )
            self._physical_blocks_per_logical_kv_block = 1

        if self.kv_cache_config.num_blocks != self.num_blocks:
            raise ValueError(
                "Packed layout and connector block counts differ: "
                f"{self.kv_cache_config.num_blocks} != {self.num_blocks}"
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
                    "Packed virtual IluNixlPush device backing has no device index"
                )
        self.nixl_memory_type = backing_memory_type
        self.device_id = device_id
        self.packed_block_stride = 0
        self._virtual_packed_regions = True

        (
            self.layer_region_ids,
            self.region_ids_by_group,
            self.transfer_regions,
            self._region_is_mla,
        ) = build_packed_virtual_layout(
            kv_caches,
            self.kv_cache_config,
            self.kv_cache_config,
            is_dsv41=getattr(self, "dsv41_layout", None) is not None,
        )

        physical_data = [(base_addr, storage.nbytes(), device_id, "")]
        base_addresses = [base_addr]
        self.region_ids_by_group = append_paired_scale_regions(
            vllm_config=self.vllm_config,
            device_id=device_id,
            kv_caches=kv_caches,
            kv_cache_scales=kv_cache_scales,
            layout_config=self.kv_cache_config,
            block_id_config=self.kv_cache_config,
            physical_data=physical_data,
            base_addresses=base_addresses,
            layer_region_ids=self.layer_region_ids,
            transfer_regions=self.transfer_regions,
            region_is_mla=self._region_is_mla,
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
        self.region_group_ids = [
            region.group_ids[0] if len(region.group_ids) == 1 else -1
            for region in self.transfer_regions
        ]
        self.region_names = [
            region.shared_by[0] if region.shared_by else f"region_{region.region_id}"
            for region in self.transfer_regions
        ]
        self.num_regions = len(self.transfer_regions)
        self.num_descs = sum(self.region_num_blocks)
        self._local_desc_cumsum = exclusive_cumsum(self.region_num_blocks)
        self.region_mem_types = [backing_memory_type] * self.num_regions
        self._uses_region_group_mapping = len(set(self.region_group_ids)) > 1
        self._mixed_mem_types = False

        self.transfer_topo = self._new_push_transfer_topology()
        self.compat_hash = self._compute_push_compat_hash()

        descs = self.nixl_wrapper.get_reg_descs(
            physical_data, backing_memory_type
        )
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        self._registered_descs.append(descs)
        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = base_addresses
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        if hasattr(self, "dst_region_num_blocks"):
            self.dst_region_num_blocks[self.engine_id] = list(
                self.region_num_blocks
            )
            self.dst_region_group_ids[self.engine_id] = list(
                self.region_group_ids
            )
            self.dst_uses_region_group_mapping[self.engine_id] = (
                self._uses_region_group_mapping
            )
            self.dst_region_mem_types[self.engine_id] = list(
                self.region_mem_types
            )
        (
            self.src_xfer_handles_by_block_size[self.block_size],
            self.src_blocks_data,
        ) = self.register_local_xfer_handler(self.block_size)

    def _setup_host_wire(
        self,
        kv_caches: dict[str, "torch.Tensor"],
        kv_cache_scales: dict[str, "torch.Tensor"] | None = None,
    ) -> dict[str, "torch.Tensor"]:
        from ..ilu_nixl_common.fp8_hetero.host_pool import HostKVPool
        from ..ilu_nixl_common.fp8_hetero.stager import HostWireStager
        from ..ilu_nixl_common.fp8_hetero.staging import StagingPipeline
        from ..ilu_nixl_common.fp8_hetero.wire_layout import (
            WirePlan,
            stage_rows,
            stage_unit_bytes,
        )

        if self._physical_blocks_per_logical_kv_block != 1:
            raise ValueError(
                "Host wire staging requires one physical block per logical KV "
                f"block; got {self._physical_blocks_per_logical_kv_block}"
            )
        plan = WirePlan.build(self.vllm_config, self.kv_cache_config)
        pool = HostKVPool(plan, self.num_blocks, pinned=True)
        self.host_xfer_buffers = pool.layer_views
        device = next(iter(kv_caches.values())).device
        self._cuda_device_id = max(device.index or 0, 0)
        resolved_scales = dict(kv_cache_scales or {})
        compilation_cfg = getattr(self.vllm_config, "compilation_config", None)
        forward_context = (
            getattr(compilation_cfg, "static_forward_context", None)
            if compilation_cfg is not None
            else None
        ) or {}
        for layer_name in kv_caches:
            if layer_name not in resolved_scales:
                layer_obj = forward_context.get(layer_name)
                scale = getattr(layer_obj, "kv_cache_scales", None)
                if isinstance(scale, torch.Tensor):
                    resolved_scales[layer_name] = scale
        pipeline = StagingPipeline(
            plan,
            pool,
            kv_caches,
            rows=stage_rows(
                self.vllm_config,
                self.kv_cache_config.kv_cache_groups,
                False,
                self._hetero_cfg.staging_blocks,
            ),
            unit_bytes=stage_unit_bytes(
                self.kv_cache_config.kv_cache_groups,
                self.kv_cache_config.wire_kv_cache_config.kv_cache_groups,
                False,
            ),
            device=device,
            compute_scales=resolved_scales,
        )
        self._wire_plan = plan
        self._pool = pool
        self._pipeline = pipeline
        self.nixl_memory_type = "DRAM"
        self._stager = HostWireStager(
            pipeline,
            pool,
            self._stage_inbox,
            self._chunk_inbox,
            self._push_writer_wake,
            self._cuda_device_id,
        )
        self._stager.start()
        return pool.layer_views

    def _init_packed_slab_registration(self, storage: Any) -> None:
        self.transfer_topo = self._new_push_transfer_topology()
        self.compat_hash = self._compute_push_compat_hash()
        if self._pool is not None:
            base_addr, total_size, self.device_id, _ = self._pool.nixl_reg_entry()
            assert total_size == storage.nbytes()
        else:
            total_size = storage.nbytes()
            base_addr = storage.data_ptr()
            if storage.device.type == "cpu":
                self.device_id = 0
                self.nixl_memory_type = "DRAM"
            else:
                assert storage.device.index is not None
                self.device_id = storage.device.index

        block_stride = total_size // self.num_blocks
        self.block_len_per_layer = [block_stride]
        self.block_stride_per_layer = [block_stride]
        self.num_regions = 1
        self.num_descs = self.num_blocks
        self.region_num_blocks = [self.num_blocks]
        self.region_mem_types = [self.nixl_memory_type]
        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = [base_addr]

        descs = self.nixl_wrapper.get_reg_descs(
            [(base_addr, total_size, self.device_id, "")], self.nixl_memory_type
        )
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        self._registered_descs.append(descs)
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        (
            self.src_xfer_handles_by_block_size[self.block_size],
            self.src_blocks_data,
        ) = self.register_local_xfer_handler(self.block_size)

    def _register_packed_kv_cache(
        self,
        storage: Any,
        kv_caches: dict[str, "torch.Tensor"] | None = None,
        active_layers: set[str] | None = None,
    ) -> None:
        parent_packed = getattr(super(), "_register_packed_kv_cache", None)
        if self._wire_plan is None:
            if parent_packed is not None:
                parent_packed(storage)
            else:
                self._init_packed_slab_registration(storage)
            packed_tensors = self.kv_cache_config.kv_cache_tensors
            layer_specs = self._layer_specs
        else:
            self._init_packed_slab_registration(storage)
            packed_tensors = self._wire_plan.wire_config.kv_cache_tensors
            layer_specs = self._wire_plan.wire_specs

        storage_base = storage.data_ptr()
        block_stride = (
            packed_tensors[0].block_stride
            if packed_tensors and getattr(packed_tensors[0], "block_stride", 0) > 0
            else self.block_len_per_layer[0]
        )
        assert all(tensor.block_stride == block_stride for tensor in packed_tensors)
        packing: dict[str, int] = {}
        for tensor in packed_tensors:
            layer_stride = int(getattr(tensor, "layer_stride", 0) or 0)
            members = getattr(tensor, "layers", None)
            if members is None:
                members = getattr(tensor, "shared_by", ())
            for idx, layer_name in enumerate(members):
                packing[layer_name] = (
                    tensor.offset + idx * layer_stride
                    if len(members) > 1 and layer_stride > 0
                    else tensor.offset
                )
        if kv_caches:
            for layer_name, cache in kv_caches.items():
                if layer_name in packing:
                    actual_offset = cache.data_ptr() - storage_base
                    if actual_offset != packing[layer_name]:
                        raise ValueError(
                            f"Packed cache view offset mismatch for {layer_name}: "
                            f"view={actual_offset}, planned={packing[layer_name]}"
                        )

        base_addrs: list[int] = []
        block_lens: list[int] = []
        region_page_ratios: list[int] = []
        layer_region_ids: dict[str, list[int]] = {}
        region_is_mla: list[bool] = []
        region_group_ids: list[int] = []
        region_names: list[str] = []

        for group_idx, group in enumerate(transfer_groups_of(self.kv_cache_config)):
            for layer_name in group.layer_names:
                if active_layers is not None and layer_name not in active_layers:
                    continue
                spec = layer_specs[layer_name]
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[layer_name]
                offset = packing[layer_name]
                if self._wire_plan is not None:
                    base_ratio = self._wire_plan.page_ratios[layer_name]
                    sub_ratio = subpage_ratio_for(
                        spec, self._wire_plan.decode_block_size
                    )
                    ratio = base_ratio * sub_ratio
                    page_len = (
                        spec.page_size_bytes
                        // self._physical_blocks_per_logical_kv_block
                    )
                    assert page_len % ratio == 0, (
                        f"Wire page {layer_name} ({page_len}B) is not divisible "
                        f"by subpage ratio {ratio}"
                    )
                    block_len = page_len // ratio
                    assert offset + page_len <= block_stride
                else:
                    ratio = 1
                    block_len = (
                        spec.page_size_bytes
                        // self._physical_blocks_per_logical_kv_block
                    )
                    assert offset + block_len <= block_stride
                layer_region_ids[layer_name] = [len(base_addrs)]
                base_addrs.append(storage_base + offset)
                block_lens.append(block_len)
                region_page_ratios.append(ratio)
                region_is_mla.append(
                    isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec))
                )
                region_group_ids.append(group_idx)
                region_names.append(layer_name)

        old_handle = self.src_xfer_handles_by_block_size.pop(self.block_size, None)
        if old_handle is not None:
            self.nixl_wrapper.release_dlist_handle(old_handle)
        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = base_addrs
        self.block_len_per_layer = block_lens
        self.block_stride_per_layer = [block_stride] * len(base_addrs)
        self.packed_block_stride = block_stride
        self.layer_region_ids = layer_region_ids
        self._region_is_mla = region_is_mla
        self._region_page_ratios = region_page_ratios
        counts = [self.num_blocks * ratio for ratio in self._region_page_ratios]
        self._region_desc_offsets = (
            np.cumsum([0, *counts[:-1]], dtype=np.int64).tolist()
        )
        self.num_regions = len(base_addrs)
        self.num_descs = sum(counts)
        self.region_num_blocks = counts
        self.region_group_ids = region_group_ids
        self._uses_region_group_mapping = len(set(region_group_ids)) > 1
        self.region_names = region_names
        mem_type = getattr(self, "nixl_memory_type", "VRAM")
        self.region_mem_types = [mem_type] * self.num_regions
        if hasattr(self, "dst_region_num_blocks"):
            self.dst_region_num_blocks[self.engine_id] = list(counts)
            self.dst_region_group_ids[self.engine_id] = list(region_group_ids)
            self.dst_uses_region_group_mapping[self.engine_id] = (
                self._uses_region_group_mapping
            )
            self.dst_region_mem_types[self.engine_id] = list(self.region_mem_types)
        (
            self.src_xfer_handles_by_block_size[self.block_size],
            self.src_blocks_data,
        ) = self.register_local_xfer_handler(self.block_size)

    def _build_fa_local(
        self, base_addresses: list[int], block_size_ratio: int
    ) -> np.ndarray:
        if getattr(self, "_virtual_packed_regions", False):
            if block_size_ratio != 1:
                raise ValueError(
                    "Packed virtual IluNixlPush descriptors do not support a "
                    "remote logical block-size mismatch"
                )
            regions = regions_by_id(self.transfer_regions)
            blocks_data: list[tuple[int, int, int]] = []
            for region_id in range(self.num_regions):
                region = regions[region_id]
                if region.physical_backing_id >= len(base_addresses):
                    raise ValueError(
                        "Packed virtual region references an unregistered "
                        f"physical backing: region={region_id}, "
                        f"backing={region.physical_backing_id}"
                    )
                backing_base = base_addresses[region.physical_backing_id]
                region_blocks = region.num_blocks or self.num_blocks
                for block_id in range(region_blocks):
                    blocks_data.append(
                        virtual_slot_desc(
                            backing_base,
                            block_id,
                            region.block_stride,
                            region.byte_offset,
                            region.block_len,
                            self.device_id,
                        )
                    )
            return (
                np.asarray(blocks_data, dtype=np.uint64)
                if blocks_data
                else np.empty((0, 3), dtype=np.uint64)
            )
        if not self.packed_block_stride:
            return super()._build_fa_local(base_addresses, block_size_ratio)
        region_page_ratios = getattr(self, "_region_page_ratios", ())
        custom_page_ratios = len(region_page_ratios) == len(base_addresses)
        if not custom_page_ratios:
            logical_blocks = np.repeat(
                np.arange(self.num_blocks, dtype=np.uint64), block_size_ratio
            )
            ratios = [block_size_ratio] * len(base_addresses)
        else:
            logical_blocks = None
            ratios = region_page_ratios
        parts = []
        for region_id, (base_addr, ratio) in enumerate(
            zip(base_addresses, ratios)
        ):
            region_blocks = (
                logical_blocks
                if logical_blocks is not None
                else np.repeat(np.arange(self.num_blocks, dtype=np.uint64), ratio)
            )
            block_len = self.block_len_per_layer[region_id] // (
                1 if custom_page_ratios else block_size_ratio
            )
            subblock_offsets = np.tile(
                np.arange(ratio, dtype=np.uint64) * block_len,
                self.num_blocks,
            )
            addrs = (
                base_addr
                + region_blocks * self.packed_block_stride
                + subblock_offsets
            )
            parts.append(self._stack_descs(addrs, block_len, self.device_id))
        return np.concatenate(parts)

    def _build_fa_remote(
        self,
        plan: Any,
        nixl_agent_meta: IluNixlPushAgentMetadata,
        block_size_ratio: int,
    ) -> np.ndarray:
        if not self.packed_block_stride:
            return super()._build_fa_remote(plan, nixl_agent_meta, block_size_ratio)
        fa_group_idx = next(
            i
            for i, spec_type in enumerate(self._group_spec_types)
            if _is_attention_spec(spec_type)
        )
        split_reads = len(plan.source_ranks_per_group[fa_group_idx])
        block_arange = np.arange(nixl_agent_meta.num_blocks, dtype=np.uint64)
        parts = []
        for region_id, base_addr in enumerate(
            nixl_agent_meta.kv_caches_base_addr
        ):
            replicated = self._is_region_replicated(region_id)
            block_len = self.block_len_per_layer[region_id]
            remote_block_len = block_len // block_size_ratio
            if block_size_ratio > 1:
                block_len = remote_block_len
            transfer_len = block_len // (1 if replicated else split_reads)
            rank_offset = (
                0 if replicated else plan.rank_offset_factor * remote_block_len
            )
            addrs = (
                base_addr
                + rank_offset
                + block_arange * nixl_agent_meta.packed_block_stride
            )
            parts.append(
                self._stack_descs(addrs, transfer_len, nixl_agent_meta.device_id)
            )
        return np.concatenate(parts)

    def initialize_host_xfer_buffer(
        self, kv_caches: dict[str, "torch.Tensor"]
    ) -> None:
        if self._host_staging_producer:
            return
        super().initialize_host_xfer_buffer(kv_caches)

    def register_kv_caches(
        self,
        kv_caches: dict[str, "torch.Tensor"],
        kv_cache_scales: dict[str, "torch.Tensor"] | None = None,
    ) -> None:
        compute_caches = kv_caches
        skip_layers = get_skip_layers()
        skip_mtp_name = getattr(self, "dsv41_layout", None) is not None
        active_compute_caches = {
            layer_name: cache
            for layer_name, cache in compute_caches.items()
            if layer_name not in skip_layers
            and not (skip_mtp_name and "mtp" in layer_name)
        }
        self._skipped_layers = sorted(
            layer_name
            for layer_name in compute_caches
            if layer_name not in active_compute_caches
        )
        if not active_compute_caches:
            self._register_empty_rank(compute_caches)
            return

        if self._host_staging_producer:
            self._setup_host_wire(
                compute_caches, kv_cache_scales=kv_cache_scales
            )
            assert self._pool is not None
            self._register_packed_kv_cache(
                self._pool.storage(),
                active_layers=set(active_compute_caches),
            )
            self.device_kv_caches = compute_caches
            self._normalize_single_pp_split(self.layer_region_ids)
            self.region_ids_by_group = self._build_region_ids_by_group(
                self.layer_region_ids
            )
            self.transfer_regions = self._build_hma_transfer_regions(
                self.layer_region_ids, self.region_ids_by_group
            )
            self._local_desc_cumsum = exclusive_cumsum(
                list(self.region_num_blocks)
            )
            self.compat_hash = self._compute_push_compat_hash()
            self._set_push_handshake_metadata()
            self._start_push_writer()
            return

        active_kv_caches = active_compute_caches

        if self._is_packed_kv_cache(active_kv_caches):
            self._register_virtual_packed_kv_cache(
                active_kv_caches,
                kv_cache_scales=kv_cache_scales,
            )
            self.device_kv_caches = compute_caches
            self._normalize_single_pp_split(self.layer_region_ids)
            self._set_push_handshake_metadata()
            self._start_push_writer()
            return

        if getattr(self, "pp_size", 1) > 1 or getattr(
            self, "_bypass_upstream_pp_hma_guard", False
        ):
            original_pp_size = self.pp_size
            self.pp_size = 1
            try:
                super().register_kv_caches(active_kv_caches)
            finally:
                self.pp_size = original_pp_size
        else:
            super().register_kv_caches(active_kv_caches)

        self.device_kv_caches = compute_caches
        self.layer_region_ids = (
            getattr(self, "layer_region_ids", None)
            or self._build_layer_region_ids_from_kv_caches(active_kv_caches)
        )
        self._normalize_single_pp_split(self.layer_region_ids)
        self.region_ids_by_group = self._build_region_ids_by_group(
            self.layer_region_ids
        )
        self.transfer_regions = self._build_hma_transfer_regions(
            self.layer_region_ids, self.region_ids_by_group
        )
        self._local_desc_cumsum = exclusive_cumsum(
            [
                region.num_blocks or self.num_blocks
                for region in self.transfer_regions
            ]
        )
        self.compat_hash = self._compute_push_compat_hash()
        self._set_push_handshake_metadata()
