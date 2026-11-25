# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side PP, GLM-5.3-Flash, and DeepSeek-V4.1 registration & pairing mixin."""

from __future__ import annotations

import queue
from collections import defaultdict
from contextlib import suppress
from typing import Any

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.utils import EngineTransferInfo
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    _is_attention_spec,
    compute_tp_mapping,
)
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index

from ..ilu_nixl_common.dsv41 import (
    effective_pp_producers,
    is_predictably_skipped_dsv41_stage,
    validate_dsv41_peer_handshake,
    wire_dsv41_layout,
)
from ..ilu_nixl_common.memory_registration import MemoryRegistrationPlanner
from ..ilu_nixl_common.semantic_pairing import (
    build_mamba_hma_transfer_routes,
    cache_kind_from_layer_name,
    pp_layer_range,
    region_ids_for_layer_range,
    region_ids_for_mtp_layers,
    regions_by_id,
    select_pp_regions_and_pairings,
    target_pp_ranks,
    transfer_identity,
    validate_pp_hma_region_lengths,
    validate_pp_region_lengths,
)
from ..ilu_nixl_common.virtual_packed_reg import (
    build_region_ids_by_group,
    exclusive_cumsum,
    layer_specs_for_config,
    layer_to_groups,
    metadata_uses_virtual_packed_regions,
)
from .metadata import (
    HmaTransferRoute,
    IluNixlPushAgentMetadata,
    IluNixlPushTransferRegion,
    RegionPairing,
)
from .xfer_plan import CompiledXferPlan, compile_xfer_plan, nixl_block_size_ratio

logger = init_logger(__name__)


class IluNixlPushWorkerMixin:
    """Heterogeneous PP/HMA/V4.1/Mamba region mapping shared by the push worker."""

    @staticmethod
    def _pp_layer_range(
        pp_split: str,
        pp_rank: int,
        layer_region_ids: dict[str, list[int]] | None = None,
    ) -> tuple[int, int]:
        return pp_layer_range(pp_split, pp_rank, layer_region_ids)

    def _mtp_enabled(self) -> bool:
        kv_cache_config = getattr(self, "kv_cache_config", None)
        for group in getattr(kv_cache_config, "kv_cache_groups", ()):
            layer_names = getattr(group, "layer_names", ())
            if getattr(group, "is_eagle_group", False) or any(
                "draft_model" in layer_name
                or ".mtp_block." in layer_name
                or ".mtp.layers." in layer_name
                for layer_name in layer_names
            ):
                return True
        speculative_config = getattr(
            getattr(self, "vllm_config", None), "speculative_config", None
        )
        return speculative_config is not None and getattr(
            speculative_config, "method", None
        ) in ("mtp", "dspark")

    @classmethod
    def _target_pp_ranks(
        cls,
        local_pp_split: str,
        local_pp_rank: int,
        remote_pp_split: str,
        local_layer_region_ids: dict[str, list[int]] | None = None,
    ) -> list[int]:
        return target_pp_ranks(
            local_pp_split,
            local_pp_rank,
            remote_pp_split,
            local_layer_region_ids,
        )

    def _is_predictable_empty_producer(self) -> bool:
        return is_predictably_skipped_dsv41_stage(getattr(self, "dsv41_layout", None))

    def _effective_pp_producers_for_remote(
        self, dst_engine_id: str, remote_pp_rank: int
    ) -> int:
        remote_pp_split = self._remote_pp_split_by_engine[dst_engine_id]
        dsv41 = getattr(self, "dsv41_layout", None)
        return effective_pp_producers(
            consumer_pp_rank=remote_pp_rank,
            consumer_pp_split=remote_pp_split,
            producer_pp_split=self.pp_split,
            dsv41_layout=dsv41,
            kv_layer_end=dsv41.kv_layer_end if dsv41 is not None else None,
        )

    @staticmethod
    def _cache_kind_from_layer_name(layer_name: str) -> str:
        return cache_kind_from_layer_name(layer_name)

    @classmethod
    def _transfer_identity(cls, layer_name: str, group_idx: int | None = None) -> str:
        return transfer_identity(layer_name, group_idx)

    @staticmethod
    def _layer_specs_for_config(kv_cache_config) -> dict[str, Any]:
        return layer_specs_for_config(kv_cache_config)

    def _should_split_virtual_region(self) -> bool:
        return bool(
            self.transfer_topo is not None
            and getattr(self.transfer_topo, "virtually_split_kv_in_blocks", False)
            and not self._has_mamba
        )

    def _actual_to_virtual_region_ids(self) -> dict[int, list[int]]:
        mapping: dict[int, list[int]] = {}
        next_region_id = 0
        for actual_region_id in range(len(self.block_len_per_layer)):
            mapping[actual_region_id] = [next_region_id]
            next_region_id += 1
            if IluNixlPushWorkerMixin._should_split_virtual_region(
                self
            ) and not self._is_region_replicated(actual_region_id):
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
            if self._layer_specs.get(layer_name) is None:
                continue
            actual_region_id = base_addr_to_actual_region.get(cache.data_ptr())
            if actual_region_id is None:
                continue
            for region_id in actual_to_virtual[actual_region_id]:
                if region_id not in layer_region_ids[layer_name]:
                    layer_region_ids[layer_name].append(region_id)

        return dict(layer_region_ids)

    def _build_region_ids_by_group(
        self, layer_region_ids: dict[str, list[int]]
    ) -> list[np.ndarray]:
        return build_region_ids_by_group(
            self.kv_cache_config,
            layer_region_ids,
            int(getattr(self, "num_regions", 0)),
        )

    def _block_len_for_virtual_region(self, region_id: int) -> int:
        next_region_id = 0
        for actual_region_id, block_len in enumerate(self.block_len_per_layer):
            if region_id == next_region_id:
                if IluNixlPushWorkerMixin._should_split_virtual_region(
                    self
                ) and not self._is_region_replicated(actual_region_id):
                    return block_len // 2
                return block_len
            next_region_id += 1
            if IluNixlPushWorkerMixin._should_split_virtual_region(
                self
            ) and not self._is_region_replicated(actual_region_id):
                if region_id == next_region_id:
                    return block_len // 2
                next_region_id += 1
        return self.block_len_per_layer[region_id]

    def _build_hma_transfer_regions(
        self,
        layer_region_ids: dict[str, list[int]],
        region_ids_by_group: list[np.ndarray],
    ) -> list[IluNixlPushTransferRegion]:
        region_to_layers: dict[int, list[str]] = defaultdict(list)
        for layer_name, region_ids in layer_region_ids.items():
            for region_id in region_ids:
                region_to_layers[int(region_id)].append(layer_name)

        region_to_groups: dict[int, list[int]] = defaultdict(list)
        for group_idx, group_region_ids in enumerate(region_ids_by_group):
            for region_id in group_region_ids:
                region_to_groups[int(region_id)].append(group_idx)

        layer_groups = layer_to_groups(self.kv_cache_config)
        layer_specs = layer_specs_for_config(self.kv_cache_config)
        region_num_blocks_list = getattr(self, "region_num_blocks", None)
        default_num_blocks = int(getattr(self, "num_blocks", 0))
        kv_caches = getattr(self, "device_kv_caches", {}) or {}
        logical_num_blocks = int(
            getattr(self, "_logical_num_blocks", default_num_blocks)
            or default_num_blocks
        )

        transfer_regions: list[IluNixlPushTransferRegion] = []
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
                for group_idx in layer_groups.get(layer_name, []):
                    if group_idx not in identity_groups[ident]:
                        identity_groups[ident].append(group_idx)

            attn_pair = next(
                (
                    (layer_specs[name], kv_caches.get(name), name)
                    for name in shared_by
                    if name in layer_specs
                    and _is_attention_spec(type(layer_specs[name]))
                ),
                None,
            )
            if attn_pair is not None:
                sample_spec, sample_cache, sample_name = attn_pair
            else:
                sample_name = next(
                    (name for name in shared_by if name in layer_specs),
                    shared_by[0],
                )
                sample_spec = layer_specs.get(sample_name)
                sample_cache = kv_caches.get(sample_name)
            block_tokens = int(getattr(sample_spec, "block_size", 0) or 0)
            reg_num_blocks = (
                int(region_num_blocks_list[region_id])
                if region_num_blocks_list is not None
                and region_id < len(region_num_blocks_list)
                else default_num_blocks
            )
            kernel_blocks = 1
            if (
                sample_cache is not None
                and sample_spec is not None
                and logical_num_blocks > 0
            ):
                with suppress(ValueError):
                    kernel_blocks = MemoryRegistrationPlanner.kernel_blocks_per_block(
                        layer_name=sample_name,
                        tensor=sample_cache,
                        spec=sample_spec,
                        num_blocks=logical_num_blocks,
                    )
            physical_per_logical = int(
                getattr(self, "_physical_blocks_per_logical_kv_block", 1) or 1
            )
            kernel_page_tokens = int(getattr(self, "block_size", 0) or 0)
            region_page_ratios = getattr(self, "_region_page_ratios", ())
            if (
                kernel_blocks == 1
                and region_id < len(region_page_ratios)
                and int(region_page_ratios[region_id]) > 1
            ):
                kernel_blocks = int(region_page_ratios[region_id])
            elif (
                kernel_blocks == 1
                and physical_per_logical > 1
                and kernel_page_tokens > 0
                and block_tokens == kernel_page_tokens * physical_per_logical
                and sample_spec is not None
                and _is_attention_spec(type(sample_spec))
            ):
                kernel_blocks = physical_per_logical
            block_len = IluNixlPushWorkerMixin._block_len_for_virtual_region(
                self, region_id
            )
            block_stride = (
                int(self.block_stride_per_layer[region_id])
                if hasattr(self, "block_stride_per_layer")
                and region_id < len(self.block_stride_per_layer)
                else block_len
            )
            transfer_regions.append(
                IluNixlPushTransferRegion(
                    region_id=region_id,
                    block_len=block_len,
                    shared_by=shared_by,
                    layer_indices=sorted(set(layer_indices)),
                    group_ids=sorted(set(region_to_groups.get(region_id, []))),
                    cache_kinds=sorted(cache_kinds),
                    identities=sorted(identities),
                    physical_backing_id=region_id,
                    byte_offset=0,
                    block_stride=block_stride,
                    kernel_blocks=kernel_blocks,
                    block_tokens=block_tokens,
                    num_blocks=reg_num_blocks,
                    identity_groups={k: sorted(v) for k, v in identity_groups.items()},
                )
            )
        if transfer_regions:
            r0 = transfer_regions[0]
            logger.info(
                "[push-reg] regions=%d r0=(id=%s tokens=%s kblocks=%s blen=%s "
                "phys_per_log=%s bsize=%s ident=%s)",
                len(transfer_regions),
                r0.region_id,
                r0.block_tokens,
                r0.kernel_blocks,
                r0.block_len,
                getattr(self, "_physical_blocks_per_logical_kv_block", None),
                getattr(self, "block_size", None),
                r0.identities,
            )
        return transfer_regions

    def _normalize_single_pp_split(
        self, layer_region_ids: dict[str, list[int]]
    ) -> None:
        if getattr(self, "pp_world_size", 1) > 1 or self.pp_split not in ("", "0"):
            return
        model_config = getattr(self, "model_config", None)
        total_layers = (
            model_config.get_total_num_hidden_layers()
            if model_config is not None
            and hasattr(model_config, "get_total_num_hidden_layers")
            else 0
        )
        if total_layers <= 0:
            max_layer_idx = -1
            for layer_name in layer_region_ids:
                try:
                    max_layer_idx = max(max_layer_idx, extract_layer_index(layer_name))
                except (AssertionError, ValueError):
                    continue
            total_layers = max_layer_idx + 1
        if total_layers <= 0:
            total_layers = len(self.block_len_per_layer)
        if total_layers > 0:
            self.pp_split = str(total_layers)

    def _validate_pp_selected_region_lengths(
        self,
        nixl_agent_meta: IluNixlPushAgentMetadata,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
        remote_tp_size: int,
    ) -> None:
        assert self.transfer_topo is not None
        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )
        model_replicated = self.use_mla or self.transfer_topo.is_kv_replicated(
            nixl_agent_meta.engine_id
        )
        replicated_region_ids = {
            region_id
            for region_id in selected_local_region_ids
            if self._is_region_replicated(region_id)
        }
        validate_pp_region_lengths(
            local_block_lens=self.block_len_per_layer,
            remote_block_lens=nixl_agent_meta.block_lens,
            selected_local_region_ids=selected_local_region_ids,
            selected_remote_region_ids=selected_remote_region_ids,
            tp_ratio=tp_ratio,
            block_size_ratio=block_size_ratio,
            replicated_region_ids=replicated_region_ids,
            model_replicated=model_replicated,
        )
        assert self.dst_num_blocks[nixl_agent_meta.engine_id] == (
            nixl_agent_meta.num_blocks
        )

    def _validate_pp_hma_region_lengths(
        self,
        nixl_agent_meta: IluNixlPushAgentMetadata,
        local_region_ids_by_group: list[np.ndarray],
        remote_region_ids_by_group: list[np.ndarray],
        selected_remote_region_ids: list[int],
        remote_tp_size: int = 1,
        routes: list[HmaTransferRoute] | None = None,
        pairings: list[RegionPairing] | None = None,
    ) -> None:
        if getattr(self, "_hetero_cfg", None) is not None and self._hetero_cfg.enabled:
            validate_pp_hma_region_lengths(
                local_block_lens=self.block_len_per_layer,
                remote_block_lens=nixl_agent_meta.block_lens,
                local_region_ids_by_group=local_region_ids_by_group,
                remote_region_ids_by_group=remote_region_ids_by_group,
                selected_remote_region_ids=selected_remote_region_ids,
                block_size_ratio=1,
                routes=routes,
                pairings=pairings,
            )
            assert self.dst_num_blocks[nixl_agent_meta.engine_id] == (
                nixl_agent_meta.num_blocks
            )
            return

        assert self.transfer_topo is not None
        block_size_ratio = nixl_block_size_ratio(
            self.transfer_topo.block_size, nixl_agent_meta.block_size
        )
        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        replicated_region_ids = {
            rid
            for rid in range(len(self.block_len_per_layer))
            if self._is_region_replicated(rid)
        }
        hybrid_mla_mamba = bool(
            getattr(self, "_has_mamba", False) and getattr(self, "use_mla", False)
        )
        validate_pp_hma_region_lengths(
            local_block_lens=self.block_len_per_layer,
            remote_block_lens=nixl_agent_meta.block_lens,
            local_region_ids_by_group=local_region_ids_by_group,
            remote_region_ids_by_group=remote_region_ids_by_group,
            selected_remote_region_ids=selected_remote_region_ids,
            block_size_ratio=block_size_ratio,
            tp_ratio=tp_ratio,
            replicated_region_ids=replicated_region_ids,
            hybrid_mla_mamba=hybrid_mla_mamba,
            group_spec_types=getattr(self, "_group_spec_types", ()),
            local_transfer_regions=getattr(self, "transfer_regions", None),
            remote_transfer_regions=getattr(nixl_agent_meta, "transfer_regions", None),
            routes=routes,
            pairings=pairings,
        )
        assert self.dst_num_blocks[nixl_agent_meta.engine_id] == (
            nixl_agent_meta.num_blocks
        )

    def _select_pp_regions_and_pairings(
        self,
        nixl_agent_meta: IluNixlPushAgentMetadata,
        remote_pp_rank: int,
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[int],
        list[int],
        list[RegionPairing] | None,
        bool,
    ]:
        assert nixl_agent_meta.layer_region_ids is not None
        assert nixl_agent_meta.region_ids_by_group is not None
        dsv41 = getattr(self, "dsv41_layout", None)
        local_kv_layer_end = dsv41.kv_layer_end if dsv41 is not None else None
        allow_empty = bool(
            dsv41 is not None
            or getattr(self, "_skipped_layers", None)
            or getattr(nixl_agent_meta, "skip_layers", None)
        )
        return select_pp_regions_and_pairings(
            local_pp_split=self.pp_split,
            local_pp_rank=self.pp_rank,
            local_layer_region_ids=self.layer_region_ids,
            local_region_ids_by_group=self.region_ids_by_group,
            remote_pp_split=nixl_agent_meta.pp_split,
            remote_pp_rank=remote_pp_rank,
            remote_layer_region_ids=nixl_agent_meta.layer_region_ids,
            remote_region_ids_by_group=nixl_agent_meta.region_ids_by_group,
            is_hma_required=self._is_hma_required,
            local_transfer_regions=self.transfer_regions,
            remote_transfer_regions=nixl_agent_meta.transfer_regions,
            include_final_stage_extra_layers=IluNixlPushWorkerMixin._mtp_enabled(self),
            local_kv_layer_end=local_kv_layer_end,
            remote_kv_layer_end=getattr(nixl_agent_meta, "kv_layer_end", None),
            allow_empty_skipped_overlap=allow_empty,
        )

    def _remote_transfer_regions(
        self,
        engine_id: str,
        nixl_agent_meta: IluNixlPushAgentMetadata,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
    ) -> list[tuple[int, int, int]]:
        if not selected_local_region_ids:
            return []
        if self._is_hma_required and not getattr(self, "_has_mamba", False):
            return [
                (region_id, nixl_agent_meta.block_lens[region_id], 0)
                for region_id in selected_remote_region_ids
            ]

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
        nixl_agent_meta: IluNixlPushAgentMetadata,
        selected_remote_region_ids: list[int],
        remote_tp_size: int,
    ) -> np.ndarray:
        if not selected_remote_region_ids:
            return np.empty((0, 3), dtype=np.uint64)
        assert self.transfer_topo is not None and self._conv_decomp is not None
        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        local_offset = self.tp_rank % max(tp_ratio, 1)
        conv_offsets = self._conv_decomp.remote_conv_offsets(local_offset, tp_ratio)
        ssm_size = (
            self._mamba_ssm_size[1] if tp_ratio >= 1 else nixl_agent_meta.ssm_sizes[1]
        )
        remote_ratio = nixl_agent_meta.physical_blocks_per_logical_kv_block
        num_blocks = nixl_agent_meta.num_blocks // remote_ratio
        block_arange = np.arange(num_blocks, dtype=np.uint64)
        block_strides = nixl_agent_meta.block_strides or nixl_agent_meta.block_lens
        virtual_packed = metadata_uses_virtual_packed_regions(nixl_agent_meta)
        remote_regions = regions_by_id(nixl_agent_meta.transfer_regions)
        parts: list[np.ndarray] = []

        for region_id in selected_remote_region_ids:
            region = remote_regions.get(region_id)
            if virtual_packed and region is not None:
                base_addr = (
                    nixl_agent_meta.kv_caches_base_addr[region.physical_backing_id]
                    + region.byte_offset
                )
                page_stride = region.block_stride * remote_ratio
            else:
                base_addr = nixl_agent_meta.kv_caches_base_addr[region_id]
                page_stride = block_strides[region_id] * remote_ratio
            block_addrs = base_addr + block_arange * page_stride
            for offset, size in conv_offsets:
                parts.append(
                    self._stack_descs(
                        block_addrs + offset, size, nixl_agent_meta.device_id
                    )
                )
            ssm_addrs = (
                block_addrs + nixl_agent_meta.ssm_sizes[0] + local_offset * ssm_size
            )
            parts.append(
                self._stack_descs(ssm_addrs, ssm_size, nixl_agent_meta.device_id)
            )
        return np.concatenate(parts) if parts else np.empty((0, 3), dtype=np.uint64)

    def _register_pp_remote_xfer_handle(
        self,
        engine_id: str,
        remote_key: int,
        remote_agent_name: str,
        nixl_agent_meta: IluNixlPushAgentMetadata,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
        selected_ssm_remote_region_ids: list[int] | None = None,
    ) -> None:
        blocks_data: list[tuple[int, int, int]] = []
        remote_regions = regions_by_id(nixl_agent_meta.transfer_regions)
        virtual_packed = metadata_uses_virtual_packed_regions(nixl_agent_meta)
        block_strides = (
            getattr(nixl_agent_meta, "block_strides", None)
            or nixl_agent_meta.block_lens
        )
        remote_region_counts: list[int] = []
        for region_id, transfer_len, rank_offset in self._remote_transfer_regions(
            engine_id,
            nixl_agent_meta,
            selected_local_region_ids,
            selected_remote_region_ids,
        ):
            region = remote_regions.get(region_id)
            region_blocks = (
                int(region.num_blocks)
                if region is not None and region.num_blocks > 0
                else (
                    int(nixl_agent_meta.region_num_blocks[region_id])
                    if getattr(nixl_agent_meta, "region_num_blocks", None)
                    and region_id < len(nixl_agent_meta.region_num_blocks)
                    else int(nixl_agent_meta.num_blocks)
                )
            )
            remote_region_counts.append(region_blocks)
            if virtual_packed:
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
                physical_base = nixl_agent_meta.kv_caches_base_addr[
                    region.physical_backing_id
                ]
                slot_base = physical_base + region.byte_offset
                page_stride = region.block_stride
            elif getattr(nixl_agent_meta, "packed_block_stride", 0) > 0:
                slot_base = nixl_agent_meta.kv_caches_base_addr[region_id]
                page_stride = nixl_agent_meta.packed_block_stride
            else:
                slot_base = nixl_agent_meta.kv_caches_base_addr[region_id]
                page_stride = block_strides[region_id]
            for block_id in range(region_blocks):
                blocks_data.append(
                    (
                        slot_base + block_id * page_stride + rank_offset,
                        transfer_len,
                        nixl_agent_meta.device_id,
                    )
                )

        if not hasattr(self, "_remote_desc_cumsum"):
            self._remote_desc_cumsum = {}
        self._remote_desc_cumsum.setdefault(engine_id, {})[remote_key] = (
            exclusive_cumsum(remote_region_counts)
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
            remote_tp_size = self.transfer_topo.get_engine_info(
                engine_id
            ).remote_tp_size
            blocks_data = np.concatenate(
                [
                    fa_blocks,
                    IluNixlPushWorkerMixin._build_pp_mamba_remote(
                        self,
                        nixl_agent_meta,
                        ssm_remote_ids,
                        remote_tp_size,
                    ),
                ]
            )

        if len(blocks_data) == 0:
            self.dst_xfer_side_handles[engine_id][remote_key] = None
            return
        mem_type = getattr(nixl_agent_meta, "kv_buffer_memory_type", None) or "VRAM"
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, mem_type)
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
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        remote_block_size = remote_info.remote_block_size
        split_key = (tp_ratio, remote_block_size)
        if (
            not IluNixlPushWorkerMixin._needs_split_local_xfer_handles(
                self, tp_ratio, plan
            )
            or split_key in self.src_xfer_handles_by_tp_ratio
        ):
            return

        block_size_ratio = self.transfer_topo.block_size_ratio(remote_block_size)
        src_blocks_data = self.src_blocks_data
        if block_size_ratio > 1:
            if remote_block_size not in self.src_xfer_handles_by_block_size:
                handle, blocks_data = self.register_local_xfer_handler(
                    remote_block_size
                )
                self.src_xfer_handles_by_block_size[remote_block_size] = handle
                self.src_blocks_data_by_block_size[remote_block_size] = blocks_data
            src_blocks_data = self.src_blocks_data_by_block_size[remote_block_size]

        self.src_xfer_handles_by_tp_ratio[split_key] = []
        for handle_data in self._build_local_splits_from_plan(
            plan,
            src_blocks_data,
            self.num_descs * block_size_ratio,
            block_size_ratio,
        ):
            local_descs = self.nixl_wrapper.get_xfer_descs(
                handle_data, self.nixl_memory_type
            )
            self.src_xfer_handles_by_tp_ratio[split_key].append(
                self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", local_descs)
            )

    def _compile_and_cache_xfer_plan(
        self,
        engine_id: str,
        remote_key: int,
        block_size_ratio: int,
    ) -> CompiledXferPlan:
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        routes = (
            getattr(self, "_mamba_hma_routes", {}).get(engine_id, {}).get(remote_key)
        )
        pairings = (
            getattr(self, "region_pairings", {}).get(engine_id, {}).get(remote_key)
        )
        local_groups = (
            getattr(self, "local_region_ids_by_group", {})
            .get(engine_id, {})
            .get(remote_key)
        )
        remote_groups = (
            getattr(self, "remote_region_ids_by_group", {})
            .get(engine_id, {})
            .get(remote_key)
        )
        num_remote_fa = (
            getattr(self, "_remote_fa_num_regions", {})
            .get(engine_id, {})
            .get(remote_key)
        )
        compiled = compile_xfer_plan(
            local_num_blocks=self.dst_num_blocks[self.engine_id],
            remote_num_blocks=self.dst_num_blocks[engine_id],
            local_physical_per_logical=self._physical_blocks_per_logical_kv_block,
            remote_physical_per_logical=remote_info.remote_physical_blocks_per_logical,
            block_size_ratio=block_size_ratio,
            num_local_fa_regions=int(getattr(self, "num_regions", 0)),
            num_remote_fa_regions=num_remote_fa,
            group_spec_types=getattr(self, "_group_spec_types", ()),
            conv_decomp=getattr(self, "_conv_decomp", None),
            local_cumsum=getattr(self, "_local_desc_cumsum", None),
            remote_cumsum=getattr(self, "_remote_desc_cumsum", {})
            .get(engine_id, {})
            .get(remote_key),
            wire_desc_offsets=(
                getattr(self, "_region_desc_offsets", ())
                if getattr(self, "_wire_plan", None) is not None
                else ()
            ),
            local_region_page_ratios=(
                getattr(self, "_region_page_ratios", ())
                if getattr(self, "_wire_plan", None) is not None
                else ()
            ),
            routes=routes,
            pairings=pairings,
            local_region_ids_by_group=local_groups,
            remote_region_ids_by_group=remote_groups,
        )
        if not hasattr(self, "_compiled_xfer_plans"):
            self._compiled_xfer_plans = {}
        self._compiled_xfer_plans.setdefault(engine_id, {})[remote_key] = compiled
        return compiled

    def add_remote_agent(
        self,
        nixl_agent_meta: IluNixlPushAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
        remote_pp_rank: int = 0,
        remote_pp_size: int = 1,
    ) -> str:
        if (
            nixl_agent_meta.layer_region_ids is None
            or nixl_agent_meta.region_ids_by_group is None
        ):
            raise RuntimeError(
                "Heterogeneous PP requires layer_region_ids and "
                "region_ids_by_group in the "
                "IluNixlPush handshake metadata."
            )
        if self._is_hma_required and nixl_agent_meta.transfer_regions is None:
            raise RuntimeError(
                "Heterogeneous PP with HMA requires transfer_regions in the "
                "IluNixlPush handshake metadata."
            )

        engine_id = nixl_agent_meta.engine_id
        validate_dsv41_peer_handshake(
            wire_dsv41_layout(
                getattr(self, "dsv41_layout", None),
                host_staging_producer=getattr(self, "_host_staging_producer", False),
                hetero_enabled=bool(
                    getattr(getattr(self, "_hetero_cfg", None), "enabled", False)
                ),
            ),
            remote_engine_id=engine_id,
            remote_swa_bounded_replay=getattr(
                nixl_agent_meta, "swa_bounded_replay", None
            ),
            remote_kv_source_layers=getattr(nixl_agent_meta, "kv_source_layers", None),
            remote_index_source_layers=getattr(
                nixl_agent_meta, "index_source_layers", None
            ),
            remote_compress_ratios=getattr(nixl_agent_meta, "compress_ratios", None),
            remote_main_kv_format=getattr(nixl_agent_meta, "main_kv_format", "none"),
            remote_indexer_kv_layout=getattr(
                nixl_agent_meta, "indexer_kv_layout", "legacy"
            ),
        )

        remote_key = remote_pp_rank * remote_tp_size + remote_tp_rank
        if remote_key in self._remote_agents.get(engine_id, {}):
            return self._remote_agents[engine_id][remote_key]

        assert self.transfer_topo is not None
        physical_blocks_per_logical = (
            nixl_agent_meta.physical_blocks_per_logical_kv_block
        )
        remote_dcp_size = getattr(nixl_agent_meta, "dcp_size", 1)
        remote_first_len = (
            nixl_agent_meta.block_lens[0]
            if nixl_agent_meta.block_lens
            else (self.block_len_per_layer[0] if self.block_len_per_layer else 0)
        )
        transfer_info = EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=nixl_agent_meta.block_size,
            remote_block_len=remote_first_len,
            remote_physical_blocks_per_logical=physical_blocks_per_logical,
            remote_dcp_size=remote_dcp_size,
        )
        self.transfer_topo.register_remote_engine(engine_id, transfer_info)
        self.tp_mappings[engine_id] = compute_tp_mapping(
            transfer_topology=self.transfer_topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
            remote_dcp_size=remote_dcp_size,
        )
        self._remote_pp_size_by_engine[engine_id] = remote_pp_size
        self._remote_pp_split_by_engine[engine_id] = nixl_agent_meta.pp_split
        if not hasattr(self, "_remote_kv_layer_end_by_engine"):
            self._remote_kv_layer_end_by_engine = {}
        self._remote_kv_layer_end_by_engine[engine_id] = getattr(
            nixl_agent_meta, "kv_layer_end", None
        )
        if not hasattr(self, "_remote_supports_block_starts"):
            self._remote_supports_block_starts = {}
        self._remote_supports_block_starts[engine_id] = bool(
            getattr(nixl_agent_meta, "supports_remote_block_starts", False)
        )
        if not hasattr(self, "_remote_group_is_ssm_by_engine"):
            self._remote_group_is_ssm_by_engine = {}
        self._remote_group_is_ssm_by_engine[engine_id] = getattr(
            nixl_agent_meta, "group_is_ssm", None
        )

        if getattr(self, "_empty_producer_rank", False):
            if not hasattr(self, "_inactive_remote_keys"):
                self._inactive_remote_keys = {}
            self._inactive_remote_keys.setdefault(engine_id, set()).add(remote_key)
            num_groups = len(self.region_ids_by_group)
            empty_groups = [np.asarray([], dtype=np.int64) for _ in range(num_groups)]
            self.local_region_ids_by_group.setdefault(engine_id, {})[remote_key] = (
                empty_groups
            )
            self.remote_region_ids_by_group.setdefault(engine_id, {})[remote_key] = (
                empty_groups
            )
            if engine_id not in self.dst_num_blocks:
                self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks
            return self.nixl_wrapper.add_remote_agent(nixl_agent_meta.agent_metadata)

        routes: list[HmaTransferRoute] | None = None
        pairings: list[RegionPairing] | None = None
        is_inactive_overlap = False
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
            raw_selected_local = region_ids_for_layer_range(
                self.layer_region_ids, overlap_start, overlap_end
            )
            raw_selected_remote = region_ids_for_layer_range(
                nixl_agent_meta.layer_region_ids, overlap_start, overlap_end
            )
            extra_overlap = False
            if (
                IluNixlPushWorkerMixin._mtp_enabled(self)
                and self.pp_rank == len(self.pp_split.split(",")) - 1
                and remote_pp_rank == len(nixl_agent_meta.pp_split.split(",")) - 1
            ):
                local_total = sum(int(x) for x in self.pp_split.split(","))
                remote_total = sum(int(x) for x in nixl_agent_meta.pp_split.split(","))
                local_extra = region_ids_for_mtp_layers(
                    self.layer_region_ids, local_total
                )
                remote_extra = region_ids_for_mtp_layers(
                    nixl_agent_meta.layer_region_ids, remote_total
                )
                if local_extra and remote_extra:
                    extra_overlap = True
                    raw_selected_local.extend(
                        r for r in local_extra if r not in raw_selected_local
                    )
                    raw_selected_remote.extend(
                        r for r in remote_extra if r not in raw_selected_remote
                    )
            assert nixl_agent_meta.transfer_regions is not None
            num_groups = len(self.region_ids_by_group)
            (
                routes,
                selected_local_region_ids,
                selected_remote_region_ids,
                selected_ssm_remote_ids,
            ) = build_mamba_hma_transfer_routes(
                raw_selected_local,
                raw_selected_remote,
                self.transfer_regions,
                nixl_agent_meta.transfer_regions,
                num_local_groups=num_groups,
                group_spec_types=getattr(self, "_group_spec_types", None),
                ssm_region_indices=getattr(self, "_ssm_region_indices", None),
                num_local_regions=getattr(
                    self, "num_regions", len(self.transfer_regions)
                ),
                overlap_start=None if extra_overlap else overlap_start,
                overlap_end=None if extra_overlap else overlap_end,
            )
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
                pairings,
                is_inactive_overlap,
            ) = IluNixlPushWorkerMixin._select_pp_regions_and_pairings(
                self, nixl_agent_meta, remote_pp_rank
            )

        if engine_id not in self.dst_num_blocks:
            num_remote_regions = len(selected_remote_region_ids)
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks
            self.dst_region_num_blocks[engine_id] = [
                nixl_agent_meta.num_blocks
            ] * num_remote_regions
            self.dst_region_group_ids[engine_id] = [0] * num_remote_regions
            self.dst_uses_region_group_mapping[engine_id] = False
            self.dst_region_mem_types[engine_id] = [
                getattr(nixl_agent_meta, "kv_buffer_memory_type", "VRAM")
            ] * num_remote_regions

        if is_inactive_overlap:
            if not hasattr(self, "_inactive_remote_keys"):
                self._inactive_remote_keys = {}
            self._inactive_remote_keys.setdefault(engine_id, set()).add(remote_key)
            if pairings is not None:
                if not hasattr(self, "region_pairings"):
                    self.region_pairings = {}
                self.region_pairings.setdefault(engine_id, {})[remote_key] = []
            self.local_region_ids_by_group.setdefault(engine_id, {})[remote_key] = (
                local_region_ids_by_group
            )
            self.remote_region_ids_by_group.setdefault(engine_id, {})[remote_key] = (
                remote_region_ids_by_group
            )
            return self.nixl_wrapper.add_remote_agent(nixl_agent_meta.agent_metadata)

        if self._is_hma_required:
            self._validate_pp_hma_region_lengths(
                nixl_agent_meta,
                local_region_ids_by_group,
                remote_region_ids_by_group,
                selected_remote_region_ids,
                remote_tp_size=remote_tp_size,
                routes=routes,
                pairings=pairings,
            )
        else:
            self._validate_pp_selected_region_lengths(
                nixl_agent_meta,
                selected_local_region_ids,
                selected_remote_region_ids,
                remote_tp_size,
            )

        if pairings is not None:
            if not hasattr(self, "region_pairings"):
                self.region_pairings = {}
            self.region_pairings.setdefault(engine_id, {})[remote_key] = pairings
        self.local_region_ids_by_group.setdefault(engine_id, {})[remote_key] = (
            local_region_ids_by_group
        )
        self.remote_region_ids_by_group.setdefault(engine_id, {})[remote_key] = (
            remote_region_ids_by_group
        )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )
        remote_virtual_packed = metadata_uses_virtual_packed_regions(nixl_agent_meta)
        if remote_virtual_packed:
            self.kv_caches_base_addr.setdefault(engine_id, {})[remote_key] = list(
                nixl_agent_meta.kv_caches_base_addr
            )
        else:
            self.kv_caches_base_addr.setdefault(engine_id, {})[remote_key] = [
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

        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )
        if self.engine_id in self.dst_num_blocks:
            IluNixlPushWorkerMixin._compile_and_cache_xfer_plan(
                self, engine_id, remote_key, block_size_ratio
            )
        return remote_agent_name

    def _base_push_transfer_results(self) -> KVConnectorTransferResults:
        """Push completion without pull-side KV post-processing."""
        assert self.transfer_topo is not None
        done_sending = self._get_new_notifs()
        recv_poll = self._pop_done_transfers(self._recving_transfers)
        if isinstance(recv_poll, tuple):
            done_recving, newly_failed = recv_poll
        else:
            done_recving, newly_failed = recv_poll, set()
        if newly_failed:
            self._recv_failures.update(newly_failed)

        if hasattr(self, "_replicated_pcp_done_sending"):
            done_sending.update(self._replicated_pcp_done_sending)
            self._replicated_pcp_done_sending.clear()

        while not self._failed_recv_reqs.empty():
            try:
                req_id = self._failed_recv_reqs.get_nowait()
            except queue.Empty:
                break
            self._handle_failed_transfer(
                req_id, None, self._recv_failures, record_failed_transfer=False
            )

        failed_recv_reqs: set[str] = set()
        if self._recv_failures:
            for req_id in tuple(self._recv_failures):
                if req_id not in self._recving_metadata:
                    self._recv_failures.remove(req_id)
                elif req_id not in self._recving_transfers:
                    self._recv_failures.remove(req_id)
                    failed_recv_reqs.add(req_id)
        done_recving.update(failed_recv_reqs)

        direct_device_recving: set[str] = set()
        for req_id in done_recving:
            meta = self._recving_metadata.pop(req_id, None)
            if hasattr(self, "_recv_meta_index"):
                self._recv_meta_index.discard(req_id)
            assert meta is not None, f"{req_id} not found in recving_metadata list"
            if req_id in failed_recv_reqs:
                self._pending_recv_notifs.pop(req_id, None)
                if not self._is_hma_required and meta.local_block_ids:
                    self._invalid_block_ids.put(set(meta.local_block_ids[0]))
                logger.warning(
                    "Skipping KV post-processing for failed request %s",
                    req_id,
                )
                continue
            self._send_pending_recv_notifs(req_id)
            if self.use_host_buffer:
                self.sync_recved_kv_to_device(req_id, meta)
            direct_device_recving.add(req_id)

        self._sync_device_after_direct_recv(direct_device_recving)
        self._reap_expired_send_leases(done_sending)
        self._cleanup_replaced_remote_engines()

        return KVConnectorTransferResults(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed_recv_reqs,
        )

    def get_transfer_results(self) -> KVConnectorTransferResults:
        return self._base_push_transfer_results()

    def get_finished(self) -> tuple[set[str], set[str]]:
        results = self.get_transfer_results()
        return results.finished_sending, results.finished_recving
