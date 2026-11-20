# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Semantic region pairing, vectorized dlist preparation, and broadcast READs."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.utils import EngineTransferInfo
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    _is_attention_spec,
    _is_ssm_spec,
    compute_tp_mapping,
)
from vllm.logger import init_logger

from ..ilu_nixl_common.memory_registration import validate_64bit_span
from ..ilu_nixl_common.semantic_pairing import (
    align_hma_transfer_regions,
    build_mamba_hma_transfer_routes,
    cache_kind_from_layer_name,
    effective_overlap_pp_count,
    filter_group_region_ids,
    iter_region_identity_groups,
    overlap_pp_count,
    pair_hma_transfer_regions,
    pp_layer_range,
    region_ids_for_layer_range,
    regions_by_id,
    target_pp_ranks,
    transfer_identity,
    validate_region_geometry_pair,
)
from ..ilu_nixl_common.types import HmaTransferRoute, RegionPairing
from ..ilu_nixl_common.virtual_packed_reg import (
    metadata_uses_virtual_packed_regions,
    transfer_groups_of,
)
from ..ilu_nixl_common.dsv41 import validate_dsv41_peer_handshake
from .metadata import (
    IluNixlAgentMetadata,
    IluNixlTransferRegion,
    encode_structured_notif,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class _CompiledDescBucket:
    """Precomputed 1D descriptor base offsets for ``(local_gid, remote_gid, is_ssm)``."""

    local_group_idx: int
    remote_group_idx: int
    is_ssm: bool
    local_bases: np.ndarray
    remote_bases: np.ndarray


class IluNixlXferPlanMixin:
    """Pairing, vectorized NIXL dlist handles, and broadcast READ execution."""

    _pp_layer_range = staticmethod(pp_layer_range)
    _cache_kind_from_layer_name = staticmethod(cache_kind_from_layer_name)
    _regions_by_id = staticmethod(regions_by_id)
    _filter_group_region_ids = staticmethod(filter_group_region_ids)
    _validate_region_geometry_pair = staticmethod(validate_region_geometry_pair)
    _metadata_uses_virtual_packed_regions = staticmethod(
        metadata_uses_virtual_packed_regions
    )

    @classmethod
    def _target_pp_ranks(
        cls, local_pp_split: str, local_pp_rank: int, remote_pp_split: str
    ) -> list[int]:
        return target_pp_ranks(local_pp_split, local_pp_rank, remote_pp_split)

    @classmethod
    def _overlap_pp_count(
        cls, producer_pp_rank: int, producer_pp_split: str, consumer_pp_split: str
    ) -> int:
        return overlap_pp_count(
            producer_pp_rank, producer_pp_split, consumer_pp_split
        )

    @staticmethod
    def _effective_overlap_pp_count(
        producer_pp_rank: int,
        producer_pp_split: str,
        consumer_pp_split: str,
        kv_layer_end: int | None = None,
    ) -> int:
        return effective_overlap_pp_count(
            producer_pp_rank, producer_pp_split, consumer_pp_split, kv_layer_end
        )

    @classmethod
    def _transfer_identity(
        cls, layer_name: str, group_idx: int | None = None
    ) -> str:
        return transfer_identity(layer_name, group_idx)

    @classmethod
    def _region_ids_for_layer_range(
        cls,
        layer_region_ids: dict[str, list[int]],
        layer_start: int,
        layer_end: int,
    ) -> list[int]:
        return region_ids_for_layer_range(layer_region_ids, layer_start, layer_end)

    @classmethod
    def _pair_hma_transfer_regions(
        cls,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
        local_transfer_regions: list[IluNixlTransferRegion],
        remote_transfer_regions: list[IluNixlTransferRegion],
        num_groups: int | None = None,
        *,
        overlap_start: int | None = None,
        overlap_end: int | None = None,
    ) -> tuple[list[RegionPairing], list[int], list[int]]:
        return pair_hma_transfer_regions(
            selected_local_region_ids,
            selected_remote_region_ids,
            local_transfer_regions,
            remote_transfer_regions,
            num_groups=num_groups,
            overlap_start=overlap_start,
            overlap_end=overlap_end,
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
        return align_hma_transfer_regions(
            selected_local_region_ids,
            selected_remote_region_ids,
            local_transfer_regions,
            remote_transfer_regions,
            num_groups,
        )

    def _build_mamba_hma_transfer_routes(
        self,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
        local_transfer_regions: list[IluNixlTransferRegion],
        remote_transfer_regions: list[IluNixlTransferRegion],
        overlap_start: int | None = None,
        overlap_end: int | None = None,
    ) -> tuple[list[HmaTransferRoute], list[int], list[int], list[int]]:
        num_local_groups = len(
            getattr(self, "_group_spec_types", None)
            or getattr(self, "region_ids_by_group", ())
        )
        return build_mamba_hma_transfer_routes(
            selected_local_region_ids,
            selected_remote_region_ids,
            local_transfer_regions,
            remote_transfer_regions,
            num_local_groups=num_local_groups,
            group_spec_types=getattr(self, "_group_spec_types", None),
            ssm_region_indices=getattr(self, "_ssm_region_indices", None),
            num_local_regions=getattr(
                self, "num_regions", len(local_transfer_regions)
            ),
            overlap_start=overlap_start,
            overlap_end=overlap_end,
        )

    @staticmethod
    def _stack_descs_static(
        addrs: np.ndarray, length: int, device_id: int
    ) -> np.ndarray:
        out = np.empty((addrs.shape[0], 3), dtype=np.uint64)
        out[:, 0] = addrs
        out[:, 1] = length
        out[:, 2] = device_id
        return out

    def _emit_region_descs(
        self,
        slot_base: int,
        num_blocks: int,
        page_stride: int,
        page_len: int,
        device_id: int,
        *,
        label: str = "region",
    ) -> np.ndarray:
        """Vectorized Nx3 uint64 descriptor builder with 64-bit span validation (P1-2)."""
        if num_blocks <= 0:
            return np.empty((0, 3), dtype=np.uint64)
        span = (num_blocks - 1) * page_stride + page_len
        validate_64bit_span(slot_base, span, label=label)
        addrs = np.uint64(slot_base) + (
            np.arange(num_blocks, dtype=np.uint64) * np.uint64(page_stride)
        )
        stack_fn = getattr(self, "_stack_descs", self._stack_descs_static)
        return stack_fn(addrs, page_len, device_id)

    def _build_fa_local(
        self, base_addresses: list[int], block_size_ratio: int
    ) -> np.ndarray:
        if not getattr(self, "_virtual_packed_regions", False):
            return super()._build_fa_local(base_addresses, block_size_ratio)
        if block_size_ratio != 1:
            raise ValueError(
                "Packed virtual IluNixl descriptors do not support a remote "
                "logical block-size mismatch"
            )
        regions = self._regions_by_id(self.transfer_regions)
        parts: list[np.ndarray] = []
        for region_id in range(self.num_regions):
            region = regions[region_id]
            if region.physical_backing_id >= len(base_addresses):
                raise ValueError(
                    "Packed virtual region references an unregistered physical "
                    f"backing: region={region_id}, backing={region.physical_backing_id}"
                )
            backing_base = base_addresses[region.physical_backing_id]
            region_blocks = region.num_blocks or self.num_blocks
            self._virtual_slot_desc(
                backing_base,
                0,
                region.block_stride,
                region.byte_offset,
                region.block_len,
                self.device_id,
            )
            parts.append(
                self._emit_region_descs(
                    backing_base + region.byte_offset,
                    region_blocks,
                    region.block_stride,
                    region.block_len,
                    self.device_id,
                    label=f"local_r{region_id}",
                )
            )
        return np.concatenate(parts) if parts else np.empty((0, 3), dtype=np.uint64)

    def _remote_transfer_regions(
        self,
        engine_id: str,
        nixl_agent_meta: IluNixlAgentMetadata,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
    ) -> list[tuple[int, int, int]]:
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
            self._mamba_ssm_size[1]
            if tp_ratio >= 1
            else nixl_agent_meta.ssm_sizes[1]
        )
        remote_ratio = nixl_agent_meta.physical_blocks_per_logical_kv_block
        num_blocks = nixl_agent_meta.num_blocks // remote_ratio
        block_arange = np.arange(num_blocks, dtype=np.uint64)
        block_strides = nixl_agent_meta.block_strides or nixl_agent_meta.block_lens
        stack_fn = getattr(self, "_stack_descs", self._stack_descs_static)
        parts: list[np.ndarray] = []
        for region_id in selected_remote_region_ids:
            page_stride = block_strides[region_id] * remote_ratio
            base_addr = nixl_agent_meta.kv_caches_base_addr[region_id]
            validate_64bit_span(
                base_addr,
                max(0, num_blocks * page_stride),
                label=f"remote_mamba_r{region_id}",
            )
            block_addrs = np.uint64(base_addr) + block_arange * np.uint64(page_stride)
            for offset, size in conv_offsets:
                parts.append(
                    stack_fn(block_addrs + np.uint64(offset), size, nixl_agent_meta.device_id)
                )
            ssm_addrs = block_addrs + np.uint64(
                nixl_agent_meta.ssm_sizes[0] + local_offset * ssm_read_size
            )
            parts.append(
                stack_fn(ssm_addrs, ssm_read_size, nixl_agent_meta.device_id)
            )
        return np.concatenate(parts) if parts else np.empty((0, 3), dtype=np.uint64)

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
        remote_regions = self._regions_by_id(nixl_agent_meta.transfer_regions)
        virtual_packed = self._metadata_uses_virtual_packed_regions(nixl_agent_meta)
        block_strides = nixl_agent_meta.block_strides or nixl_agent_meta.block_lens
        remote_region_counts: list[int] = []
        fa_parts: list[np.ndarray] = []

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
                try:
                    physical_base = nixl_agent_meta.kv_caches_base_addr[
                        region.physical_backing_id
                    ]
                except IndexError as exc:
                    raise RuntimeError(
                        "Packed remote region references an unavailable physical "
                        f"backing: region={region_id}, backing={region.physical_backing_id}"
                    ) from exc
                slot_base = physical_base + region.byte_offset
                page_stride = region.block_stride
            else:
                slot_base = nixl_agent_meta.kv_caches_base_addr[region_id]
                page_stride = block_strides[region_id]

            fa_parts.append(
                self._emit_region_descs(
                    slot_base + rank_offset,
                    region_blocks,
                    page_stride,
                    transfer_len,
                    nixl_agent_meta.device_id,
                    label=f"remote_{engine_id}_r{region_id}",
                )
            )

        if not hasattr(self, "_remote_desc_cumsum"):
            self._remote_desc_cumsum = {}
        self._remote_desc_cumsum.setdefault(engine_id, {})[remote_key] = (
            self._exclusive_cumsum(remote_region_counts)
        )
        if not hasattr(self, "_remote_fa_num_regions"):
            self._remote_fa_num_regions = {}
        self._remote_fa_num_regions.setdefault(engine_id, {})[remote_key] = len(
            selected_remote_region_ids
        )

        fa_blocks = (
            np.concatenate(fa_parts)
            if fa_parts
            else np.empty((0, 3), dtype=np.uint64)
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
            blocks_arr = np.concatenate(
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
                blocks_arr, nixl_agent_meta.memory_type
            )
        else:
            # Pass numpy array directly when large; preserve tuple list for tiny
            # unit-test mocks that compare ``blocks_data == [(addr, len, dev), ...]``.
            payload_descs: Any = (
                [tuple(int(v) for v in row) for row in fa_blocks.tolist()]
                if len(fa_blocks) <= 64
                else fa_blocks
            )
            descs = self.nixl_wrapper.get_xfer_descs(
                payload_descs, nixl_agent_meta.memory_type
            )

        dlist_handle = self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        self.dst_xfer_side_handles[engine_id][remote_key] = dlist_handle
        sessions = getattr(self, "_active_sessions", None)
        if isinstance(sessions, dict) and engine_id in sessions:
            sessions[engine_id].dlist_handles[remote_key] = dlist_handle

    def _needs_split_local_xfer_handles(self, tp_ratio: int, plan) -> bool:
        return tp_ratio < 0 and (
            not getattr(self, "use_mla", False)
            or len(getattr(plan, "all_source_ranks", ())) > 1
        )

    def _ensure_pp_local_split_handles(
        self, engine_id: str, remote_tp_size: int
    ) -> None:
        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        remote_block_size = remote_info.remote_block_size
        block_size_ratio = self.transfer_topo.block_size_ratio(remote_block_size)

        src_blocks_data = getattr(self, "src_blocks_data", None)
        if block_size_ratio > 1:
            if not hasattr(self, "src_blocks_data_by_block_size"):
                self.src_blocks_data_by_block_size = {}
            if remote_block_size not in self.src_xfer_handles_by_block_size:
                handle, bdata = self.register_local_xfer_handler(remote_block_size)
                self.src_xfer_handles_by_block_size[remote_block_size] = handle
                self.src_blocks_data_by_block_size[remote_block_size] = bdata
            src_blocks_data = self.src_blocks_data_by_block_size.get(
                remote_block_size, src_blocks_data
            )

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
            src_blocks_data,
            self.num_descs * block_size_ratio,
            block_size_ratio,
        ):
            local_descs = self.nixl_wrapper.get_xfer_descs(
                handle_data, self.nixl_memory_type
            )
            self.src_xfer_handles_by_tp_ratio[tp_ratio].append(
                self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", local_descs)
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
        Example: heads=4, Decode TP8, Prefill TP1 -> ratio 4, not 8.
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
            transfer_cfg = getattr(self, "kv_transfer_config", None)
            is_homo_pp = getattr(nixl_agent_meta, "pp_split", "") in (
                "",
                getattr(self, "pp_split", ""),
            )
            if (
                is_homo_pp
                and transfer_cfg is not None
                and getattr(transfer_cfg, "enable_permute_local_kv", False)
                and nixl_agent_meta.kv_cache_layout == "HND"
                and not getattr(self, "_is_hma_required", False)
            ):
                self.enable_permute_local_kv = True
            else:
                raise RuntimeError(
                    "Heterogeneous PP expects the same non-MLA KV cache layout: "
                    f"local={local_layout}, remote={nixl_agent_meta.kv_cache_layout}"
                )
        if (
            getattr(nixl_agent_meta, "attn_backend_name", None)
            and nixl_agent_meta.attn_backend_name != getattr(self, "backend_name", None)
            and getattr(self, "backend_name", None) in ("CPU_ATTN",)
            and not getattr(self, "_is_hma_required", False)
        ):
            self.enable_heterogeneous_attn_post_process = True

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
        local_regions = self._regions_by_id(getattr(self, "transfer_regions", None))
        remote_regions = self._regions_by_id(
            getattr(nixl_agent_meta, "transfer_regions", None)
        )
        for pair_idx, (local_region_id, remote_region_id) in enumerate(
            zip(selected_local_region_ids, selected_remote_region_ids)
        ):
            self._validate_region_geometry_pair(
                local_regions.get(local_region_id),
                remote_regions.get(remote_region_id),
                identity=f"pair{pair_idx}",
                local_region_id=local_region_id,
                remote_region_id=remote_region_id,
            )
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
                "local_heads="
                f"{max(1, self.transfer_topo.total_num_kv_heads // self.transfer_topo.tp_size)}, "
                "remote_heads="
                f"{max(1, self.transfer_topo.total_num_kv_heads // remote_tp_size)}"
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
        routes: list[HmaTransferRoute] | None = None,
        pairings: list[RegionPairing] | None = None,
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
        local_regions = self._regions_by_id(getattr(self, "transfer_regions", None))
        remote_regions = self._regions_by_id(
            getattr(nixl_agent_meta, "transfer_regions", None)
        )

        def check_pair(
            l_rid: int, r_rid: int, *, is_ssm: bool, identity: str, context: str
        ) -> None:
            self._validate_region_geometry_pair(
                local_regions.get(l_rid),
                remote_regions.get(r_rid),
                identity=identity,
                local_region_id=l_rid,
                remote_region_id=r_rid,
            )
            local_block_len = self.block_len_per_layer[l_rid]
            remote_block_len = nixl_agent_meta.block_lens[r_rid]
            replicated = hybrid_mla_mamba or (
                not is_ssm and self._is_region_replicated(l_rid)
            )
            expected_remote_len = local_block_len // block_size_ratio
            if not replicated:
                if is_ssm:
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
                f"when using HMA with heterogeneous PP at {context}: "
                f"local_region={l_rid} local={local_block_len}, "
                f"remote_region={r_rid} remote={remote_block_len}, "
                f"expected_remote={expected_remote_len}"
            )

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
                ident = f"group{route.local_group_idx}->group{route.remote_group_idx}"
                for pair_idx, (local_region_id, remote_region_id) in enumerate(
                    zip(route.raw_local_region_ids, route.raw_remote_region_ids)
                ):
                    check_pair(
                        int(local_region_id),
                        int(remote_region_id),
                        is_ssm=route.is_ssm,
                        identity=ident,
                        context=f"{ident}, pair={pair_idx}",
                    )
        elif pairings is not None:
            group_spec_types = getattr(self, "_group_spec_types", ())
            for pair_idx, pairing in enumerate(pairings):
                is_ssm = (
                    pairing.local_group_id < len(group_spec_types)
                    and _is_ssm_spec(group_spec_types[pairing.local_group_id])
                )
                check_pair(
                    pairing.local_region_id,
                    pairing.remote_region_id,
                    is_ssm=is_ssm,
                    identity=pairing.identity,
                    context=(
                        f"{pairing.identity} "
                        f"(group={pairing.local_group_id}->{pairing.remote_group_id}, "
                        f"pair={pair_idx})"
                    ),
                )
        else:
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
                is_ssm = _is_ssm_spec(self._group_spec_types[group_idx])
                for pair_idx, (local_region_id, remote_compact_id) in enumerate(
                    zip(local_group, remote_group)
                ):
                    check_pair(
                        int(local_region_id),
                        selected_remote_region_ids[int(remote_compact_id)],
                        is_ssm=is_ssm,
                        identity=f"group{group_idx}",
                        context=f"group={group_idx}, pair={pair_idx}",
                    )
        assert self.dst_num_blocks[nixl_agent_meta.engine_id] == (
            nixl_agent_meta.num_blocks
        )

    def _select_pp_regions_and_pairings(
        self, nixl_agent_meta: IluNixlAgentMetadata, remote_pp_rank: int
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[int],
        list[int],
        list[RegionPairing] | None,
        bool,
    ]:
        local_start, local_end = self._pp_layer_range(
            self.pp_split, self.pp_rank, self.layer_region_ids
        )
        remote_pp_split_eff = nixl_agent_meta.pp_split or self.pp_split
        remote_start, remote_end = self._pp_layer_range(
            remote_pp_split_eff, remote_pp_rank, nixl_agent_meta.layer_region_ids
        )
        if max(local_start, remote_start) >= min(local_end, remote_end):
            raise RuntimeError(
                "Heterogeneous PP handshake selected a remote PP stage "
                "with no overlapping layers: "
                f"local=[{local_start}, {local_end}), "
                f"remote=[{remote_start}, {remote_end})."
            )

        kv_layer_end = getattr(nixl_agent_meta, "kv_layer_end", None)
        if kv_layer_end is not None:
            remote_end = min(remote_end, kv_layer_end)
        overlap_start = max(local_start, remote_start)
        overlap_end = min(local_end, remote_end)
        num_groups = len(self.region_ids_by_group)
        empty_groups = [np.asarray([], dtype=np.int64) for _ in range(num_groups)]
        if overlap_start >= overlap_end:
            return empty_groups, empty_groups, [], [], [], True

        selected_local_region_ids = self._region_ids_for_layer_range(
            self.layer_region_ids, overlap_start, overlap_end
        )
        assert nixl_agent_meta.layer_region_ids is not None
        selected_remote_region_ids = self._region_ids_for_layer_range(
            nixl_agent_meta.layer_region_ids, overlap_start, overlap_end
        )
        if not selected_remote_region_ids:
            remote_skip_layers = getattr(nixl_agent_meta, "skip_layers", None)
            if not selected_local_region_ids and (
                getattr(self, "dsv41_layout", None) is not None
                or getattr(self, "_skipped_layers", None)
                or remote_skip_layers
            ):
                return empty_groups, empty_groups, [], [], [], True
            if remote_skip_layers and selected_local_region_ids:
                skip_identities = {
                    transfer_identity(name) for name in remote_skip_layers
                }
                local_regions_map = regions_by_id(
                    getattr(self, "transfer_regions", None)
                )
                local_overlap_identities = {
                    ident
                    for rid in selected_local_region_ids
                    if rid in local_regions_map
                    for ident, _ in iter_region_identity_groups(
                        local_regions_map[rid]
                    )
                    if (
                        (layer_idx := int(ident.split(":", 1)[0])) < 0
                        or overlap_start <= layer_idx < overlap_end
                    )
                }
                if local_overlap_identities and local_overlap_identities.issubset(
                    skip_identities
                ):
                    return empty_groups, empty_groups, [], [], [], True

        if self._is_hma_required:
            assert nixl_agent_meta.transfer_regions is not None
            pairings, selected_local_ids, selected_remote_ids = (
                self._pair_hma_transfer_regions(
                    selected_local_region_ids,
                    selected_remote_region_ids,
                    self.transfer_regions,
                    nixl_agent_meta.transfer_regions,
                    num_groups=num_groups,
                    overlap_start=overlap_start,
                    overlap_end=overlap_end,
                )
            )
            compact_remote_map = {
                region_id: compact_id
                for compact_id, region_id in enumerate(selected_remote_ids)
            }
            local_by_group: list[list[int]] = [[] for _ in range(num_groups)]
            remote_by_group: list[list[int]] = [[] for _ in range(num_groups)]
            for pairing in pairings:
                gid = pairing.local_group_id
                if (
                    gid < num_groups
                    and pairing.local_region_id not in local_by_group[gid]
                ):
                    local_by_group[gid].append(pairing.local_region_id)
                    remote_by_group[gid].append(
                        compact_remote_map[pairing.remote_region_id]
                    )
            return (
                [np.asarray(g, dtype=np.int64) for g in local_by_group],
                [np.asarray(g, dtype=np.int64) for g in remote_by_group],
                selected_local_ids,
                selected_remote_ids,
                pairings,
                False,
            )

        assert nixl_agent_meta.region_ids_by_group is not None
        return (
            self._filter_group_region_ids(
                self.region_ids_by_group, selected_local_region_ids, compact=False
            ),
            self._filter_group_region_ids(
                nixl_agent_meta.region_ids_by_group,
                selected_remote_region_ids,
                compact=True,
            ),
            selected_local_region_ids,
            selected_remote_region_ids,
            None,
            False,
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
            nixl_agent_meta.layer_region_ids is None
            or nixl_agent_meta.region_ids_by_group is None
        ):
            if (
                remote_pp_size == self.pp_world_size
                and getattr(nixl_agent_meta, "pp_split", "") in ("", self.pp_split)
                and not getattr(self, "_virtual_packed_regions", False)
                and not remote_virtual_packed
                and getattr(self, "dsv41_layout", None) is None
                and getattr(nixl_agent_meta, "kv_layer_end", None) is None
                and not getattr(nixl_agent_meta, "skip_layers", None)
            ):
                return super().add_remote_agent(
                    nixl_agent_meta, remote_tp_rank, remote_tp_size
                )
            raise RuntimeError(
                "Heterogeneous PP requires layer_region_ids and "
                "region_ids_by_group in the IluNixl handshake metadata."
            )
        if self._is_hma_required and nixl_agent_meta.transfer_regions is None:
            raise RuntimeError(
                "Heterogeneous PP with HMA requires transfer_regions in the "
                "IluNixl handshake metadata."
            )

        engine_id = nixl_agent_meta.engine_id
        wire_staging = getattr(self, "fp8_wire_staging_enabled", False)
        validate_dsv41_peer_handshake(
            getattr(self, "dsv41_layout", None),
            remote_engine_id=engine_id,
            remote_swa_bounded_replay=getattr(
                nixl_agent_meta, "swa_bounded_replay", None
            ),
            remote_kv_source_layers=getattr(
                nixl_agent_meta, "kv_source_layers", None
            ),
            remote_index_source_layers=getattr(
                nixl_agent_meta, "index_source_layers", None
            ),
            remote_compress_ratios=getattr(
                nixl_agent_meta, "compress_ratios", None
            ),
            remote_main_kv_format=getattr(
                nixl_agent_meta, "main_kv_format", "none"
            ),
            remote_indexer_kv_layout=getattr(
                nixl_agent_meta, "indexer_kv_layout", "legacy"
            ),
            local_main_kv_format="none" if wire_staging else None,
            local_indexer_kv_layout="legacy" if wire_staging else None,
        )
        is_hetero_pp_topology = remote_pp_size != self.pp_world_size or (
            bool(nixl_agent_meta.pp_split)
            and nixl_agent_meta.pp_split != self.pp_split
        )
        remote_key = (
            remote_pp_rank * remote_tp_size + remote_tp_rank
            if is_hetero_pp_topology
            else remote_tp_rank
        )
        if remote_key in self._remote_agents.get(engine_id, {}):
            return self._remote_agents[engine_id][remote_key]

        assert self.transfer_topo is not None
        remote_first_len = (
            nixl_agent_meta.block_lens[0]
            if nixl_agent_meta.block_lens
            else (self.block_len_per_layer[0] if self.block_len_per_layer else 0)
        )
        transfer_info = EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=nixl_agent_meta.block_size,
            remote_block_len=remote_first_len,
            remote_physical_blocks_per_logical=(
                nixl_agent_meta.physical_blocks_per_logical_kv_block
            ),
        )
        self.transfer_topo.register_remote_engine(engine_id, transfer_info)
        self.tp_mappings[engine_id] = compute_tp_mapping(
            transfer_topology=self.transfer_topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
        )
        self._remote_pp_size_by_engine[engine_id] = remote_pp_size
        self._remote_pp_split_by_engine[engine_id] = (
            nixl_agent_meta.pp_split or self.pp_split
        )
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
        if not hasattr(self, "_remote_supports_structured_notif"):
            self._remote_supports_structured_notif = {}
        self._remote_supports_structured_notif[engine_id] = bool(
            getattr(nixl_agent_meta, "supports_structured_notif", False)
        )
        self._semantic_region_engines.add(engine_id)

        routes: list[HmaTransferRoute] | None = None
        pairings: list[RegionPairing] | None = None
        is_inactive_overlap = False
        selected_ssm_remote_ids: list[int] | None = None
        if self._is_hma_required and getattr(self, "_has_mamba", False):
            local_start, local_end = self._pp_layer_range(
                self.pp_split, self.pp_rank, self.layer_region_ids
            )
            remote_pp_split_eff = nixl_agent_meta.pp_split or self.pp_split
            remote_start, remote_end = self._pp_layer_range(
                remote_pp_split_eff, remote_pp_rank, nixl_agent_meta.layer_region_ids
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
                pairings,
                is_inactive_overlap,
            ) = self._select_pp_regions_and_pairings(
                nixl_agent_meta, remote_pp_rank
            )

        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

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
            remote_agent_name = self.nixl_wrapper.add_remote_agent(
                nixl_agent_meta.agent_metadata
            )
            return remote_agent_name

        if self._is_hma_required:
            self._validate_pp_hma_region_lengths(
                nixl_agent_meta,
                local_region_ids_by_group,
                remote_region_ids_by_group,
                selected_remote_region_ids,
                remote_tp_size,
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
        self._compile_desc_buckets_for_remote_key(
            engine_id,
            remote_key,
            routes=routes,
            pairings=pairings,
            local_region_ids_by_group=local_region_ids_by_group,
            remote_region_ids_by_group=remote_region_ids_by_group,
        )
        return remote_agent_name

    def _compile_desc_buckets_for_remote_key(
        self,
        engine_id: str,
        remote_key: int,
        *,
        routes: list[HmaTransferRoute] | None = None,
        pairings: list[RegionPairing] | None = None,
        local_region_ids_by_group: list[np.ndarray] | None = None,
        remote_region_ids_by_group: list[np.ndarray] | None = None,
    ) -> list[_CompiledDescBucket]:
        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        local_cumsum = getattr(self, "_local_desc_cumsum", None)
        remote_cumsum = (
            getattr(self, "_remote_desc_cumsum", {})
            .get(engine_id, {})
            .get(remote_key)
        )
        local_num_blocks = self.dst_num_blocks[self.engine_id]
        remote_num_blocks = self.dst_num_blocks[engine_id]

        def fa_bases(
            region_ids: np.ndarray,
            num_blocks: int,
            ratio: int | None,
            cumsum: np.ndarray | None,
        ) -> np.ndarray:
            if len(region_ids) == 0:
                return np.zeros(0, dtype=np.int64)
            if cumsum is not None and len(cumsum) > 0:
                scale = int(ratio) if ratio is not None else 1
                return np.asarray(cumsum[region_ids] * scale, dtype=np.int64)
            scaled = int(num_blocks * ratio) if ratio is not None else num_blocks
            return np.asarray(region_ids * scaled, dtype=np.int64)

        def ssm_bases(
            region_ids: np.ndarray,
            num_phys_blocks: int,
            phys_per_logical: int,
            num_fa_regions: int,
        ) -> np.ndarray:
            if len(region_ids) == 0:
                return np.zeros(0, dtype=np.int64)
            assert self._conv_decomp is not None
            num_logical = num_phys_blocks // phys_per_logical
            regions_per_layer = len(self._conv_decomp.local_conv_offsets) + 1
            return (
                num_fa_regions * num_phys_blocks
                + (
                    region_ids[:, None] * regions_per_layer
                    + np.arange(regions_per_layer, dtype=np.int64)[None, :]
                )
                * num_logical
            ).reshape(-1)

        if routes is None:
            routes = (
                getattr(self, "_mamba_hma_routes", {})
                .get(engine_id, {})
                .get(remote_key)
            )
        if pairings is None:
            pairings = (
                getattr(self, "region_pairings", {})
                .get(engine_id, {})
                .get(remote_key)
            )
        if local_region_ids_by_group is None:
            local_region_ids_by_group = (
                getattr(self, "local_region_ids_by_group", {})
                .get(engine_id, {})
                .get(remote_key)
            )
        if remote_region_ids_by_group is None:
            remote_region_ids_by_group = (
                getattr(self, "remote_region_ids_by_group", {})
                .get(engine_id, {})
                .get(remote_key)
            )

        compiled: list[_CompiledDescBucket] = []
        if routes is not None:
            num_remote_fa = (
                getattr(self, "_remote_fa_num_regions", {})
                .get(engine_id, {})
                .get(remote_key)
            )
            if num_remote_fa is None:
                num_remote_fa = max(
                    (
                        int(r.remote_desc_region_ids.max()) + 1
                        for r in routes
                        if not r.is_ssm and len(r.remote_desc_region_ids)
                    ),
                    default=0,
                )
            for route in routes:
                if len(route.local_desc_region_ids) == 0:
                    continue
                if route.is_ssm:
                    lb = ssm_bases(
                        route.local_desc_region_ids,
                        local_num_blocks,
                        self._physical_blocks_per_logical_kv_block,
                        self.num_regions,
                    )
                    rb = ssm_bases(
                        route.remote_desc_region_ids,
                        remote_num_blocks,
                        remote_info.remote_physical_blocks_per_logical,
                        num_remote_fa,
                    )
                else:
                    lb = fa_bases(
                        route.local_desc_region_ids,
                        local_num_blocks,
                        block_size_ratio,
                        local_cumsum,
                    )
                    rb = fa_bases(
                        route.remote_desc_region_ids,
                        remote_num_blocks,
                        None,
                        remote_cumsum,
                    )
                compiled.append(
                    _CompiledDescBucket(
                        local_group_idx=route.local_group_idx,
                        remote_group_idx=route.remote_group_idx,
                        is_ssm=route.is_ssm,
                        local_bases=lb,
                        remote_bases=rb,
                    )
                )
        elif pairings is not None:
            selected_remote_order: list[int] = []
            for p in pairings:
                if p.remote_region_id not in selected_remote_order:
                    selected_remote_order.append(p.remote_region_id)
            compact_remote_map = {
                rid: idx for idx, rid in enumerate(selected_remote_order)
            }
            bucketed: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
            for p in pairings:
                entry = (p.local_region_id, compact_remote_map[p.remote_region_id])
                if entry not in bucketed[(p.local_group_id, p.remote_group_id)]:
                    bucketed[(p.local_group_id, p.remote_group_id)].append(entry)
            group_spec_types = getattr(self, "_group_spec_types", ())
            num_remote_fa = (
                getattr(self, "_remote_fa_num_regions", {})
                .get(engine_id, {})
                .get(remote_key, len(selected_remote_order))
            )
            for (local_gid, remote_gid), reg_pairs in sorted(bucketed.items()):
                if not reg_pairs:
                    continue
                is_ssm = (
                    local_gid < len(group_spec_types)
                    and _is_ssm_spec(group_spec_types[local_gid])
                )
                l_arr = np.asarray([pair[0] for pair in reg_pairs], dtype=np.int64)
                r_arr = np.asarray([pair[1] for pair in reg_pairs], dtype=np.int64)
                if is_ssm:
                    lb = ssm_bases(
                        l_arr,
                        local_num_blocks,
                        self._physical_blocks_per_logical_kv_block,
                        self.num_regions,
                    )
                    rb = ssm_bases(
                        r_arr,
                        remote_num_blocks,
                        remote_info.remote_physical_blocks_per_logical,
                        num_remote_fa,
                    )
                else:
                    lb = fa_bases(
                        l_arr, local_num_blocks, block_size_ratio, local_cumsum
                    )
                    rb = fa_bases(r_arr, remote_num_blocks, None, remote_cumsum)
                compiled.append(
                    _CompiledDescBucket(
                        local_group_idx=local_gid,
                        remote_group_idx=remote_gid,
                        is_ssm=is_ssm,
                        local_bases=lb,
                        remote_bases=rb,
                    )
                )
        elif local_region_ids_by_group and remote_region_ids_by_group:
            group_spec_types = getattr(self, "_group_spec_types", ())
            num_remote_fa = (
                getattr(self, "_remote_fa_num_regions", {})
                .get(engine_id, {})
                .get(remote_key)
            )
            if num_remote_fa is None:
                num_remote_fa = max(
                    (
                        int(ids.max()) + 1
                        for ids in remote_region_ids_by_group
                        if len(ids)
                    ),
                    default=0,
                )
            for gid, (l_arr, r_arr) in enumerate(
                zip(local_region_ids_by_group, remote_region_ids_by_group)
            ):
                if len(l_arr) == 0:
                    continue
                is_ssm = (
                    gid < len(group_spec_types)
                    and _is_ssm_spec(group_spec_types[gid])
                )
                if is_ssm:
                    lb = ssm_bases(
                        l_arr,
                        local_num_blocks,
                        self._physical_blocks_per_logical_kv_block,
                        self.num_regions,
                    )
                    rb = ssm_bases(
                        r_arr,
                        remote_num_blocks,
                        remote_info.remote_physical_blocks_per_logical,
                        num_remote_fa,
                    )
                else:
                    lb = fa_bases(
                        l_arr, local_num_blocks, block_size_ratio, local_cumsum
                    )
                    rb = fa_bases(r_arr, remote_num_blocks, None, remote_cumsum)
                compiled.append(
                    _CompiledDescBucket(
                        local_group_idx=gid,
                        remote_group_idx=gid,
                        is_ssm=is_ssm,
                        local_bases=lb,
                        remote_bases=rb,
                    )
                )

        if not hasattr(self, "_compiled_desc_buckets"):
            self._compiled_desc_buckets = {}
        self._compiled_desc_buckets.setdefault(engine_id, {})[remote_key] = compiled
        return compiled

    def _record_stat_failed_transfer(self) -> None:
        stats = getattr(self, "xfer_stats", None)
        if stats is not None and hasattr(stats, "record_failed_transfer"):
            stats.record_failed_transfer()

    def _make_pp_aware_notif_id(
        self,
        remote_request_id: str,
        dst_engine_id: str,
        remote_key: int,
        *,
        status: str = "ok",
    ) -> bytes:
        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        remote_pp_rank = remote_key // remote_info.remote_tp_size
        remote_pp_split = self._remote_pp_split_by_engine[dst_engine_id]
        kv_layer_end = getattr(self, "_remote_kv_layer_end_by_engine", {}).get(
            dst_engine_id
        )
        pp_consumers = self._effective_overlap_pp_count(
            remote_pp_rank,
            remote_pp_split,
            self.pp_split,
            kv_layer_end=kv_layer_end,
        )
        d_tp_size = getattr(self, "transfer_tp_size", self.world_size)
        if getattr(self, "_remote_supports_structured_notif", {}).get(
            dst_engine_id, False
        ):
            tp_consumers = (
                d_tp_size // remote_info.remote_tp_size
                if d_tp_size > remote_info.remote_tp_size
                else 1
            )
            return encode_structured_notif(
                remote_request_id,
                status=status,
                remote_key=remote_key,
                consumer_pp_rank=getattr(self, "pp_rank", 0),
                consumer_tp_rank=getattr(
                    self, "transfer_tp_rank", getattr(self, "tp_rank", 0)
                ),
                expected_consumers=tp_consumers * pp_consumers,
            )
        return f"{remote_request_id}:{d_tp_size}:{pp_consumers}".encode()

    def _notify_remote_read_failed(
        self, dst_engine_id: str, remote_request_id: str, remote_key: int
    ) -> None:
        if not getattr(self, "_remote_supports_structured_notif", {}).get(
            dst_engine_id, False
        ):
            return
        agent_name = (
            getattr(self, "_remote_agents", {}).get(dst_engine_id, {}).get(remote_key)
        )
        if agent_name is None:
            return
        try:
            fail_msg = self._make_pp_aware_notif_id(
                remote_request_id, dst_engine_id, remote_key, status="fail"
            )
            self.nixl_wrapper.send_notif(agent_name, notif_msg=fail_msg)
        except Exception:
            logger.debug(
                "Failed to send fail notification to engine %s rank %s for req %s",
                dst_engine_id,
                remote_key,
                remote_request_id,
                exc_info=True,
            )

    def _handle_failed_transfer(self, req_id: str, handle: int | None) -> None:
        """P0-2: Mark ALL non-empty HMA local block groups invalid on READ failure."""
        meta = getattr(self, "_recving_metadata", {}).get(req_id)
        invalid_queue = getattr(self, "_invalid_block_ids", None)
        if meta is not None and invalid_queue is not None:
            dirty_blocks: set[int] = set()
            for grp in getattr(meta, "local_block_ids", ()) or ():
                if grp:
                    dirty_blocks.update(int(b) for b in grp)
            if dirty_blocks:
                invalid_queue.put(dirty_blocks)
        if hasattr(self, "_failed_recv_reqs"):
            self._failed_recv_reqs.put(req_id)
        if handle is not None and getattr(self, "nixl_wrapper", None) is not None:
            self.nixl_wrapper.release_xfer_handle(handle)
        self._record_stat_failed_transfer()

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
        block_ids: list[int], region_ids: np.ndarray, num_blocks: int
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

    def _align_group_blocks_with_start(
        self,
        local_blocks: list[int],
        remote_blocks: list[int],
        *,
        local_group_idx: int,
        num_local_groups: int,
        remote_info,
        start_offset: int | None,
    ) -> tuple[list[int], list[int]]:
        if not local_blocks or not remote_blocks:
            return [], []
        d_ppl = int(
            getattr(self, "_physical_blocks_per_logical_kv_block", 1) or 1
        )
        p_ppl = int(
            getattr(remote_info, "remote_physical_blocks_per_logical", 1) or 1
        )
        if start_offset is not None and start_offset >= 0:
            if d_ppl == 1 and p_ppl == 1:
                if len(remote_blocks) < start_offset + len(local_blocks):
                    raise ValueError(
                        f"Remote KV group has {len(remote_blocks)} blocks, cannot "
                        f"slice [{start_offset}, {start_offset + len(local_blocks)}) "
                        f"for local group {local_group_idx}"
                    )
                return list(local_blocks), list(
                    remote_blocks[start_offset : start_offset + len(local_blocks)]
                )
            if start_offset > 0:
                page_start = start_offset * d_ppl
                if page_start >= len(remote_blocks):
                    raise ValueError(
                        f"Remote KV group has {len(remote_blocks)} kernel pages, "
                        f"cannot start at page {page_start} "
                        f"(start_offset={start_offset}, d_ppl={d_ppl}) "
                        f"for local group {local_group_idx}"
                    )
                rem_slice = list(remote_blocks[page_start:])
                if d_ppl == p_ppl and len(local_blocks) <= len(rem_slice):
                    return list(local_blocks), rem_slice[: len(local_blocks)]
                max_padding = d_ppl + p_ppl
                assert abs(len(local_blocks) - len(rem_slice)) <= max_padding, (
                    f"Group {local_group_idx}: |{len(local_blocks)} - "
                    f"{len(rem_slice)}| > {max_padding}"
                )
                nb = min(len(local_blocks), len(rem_slice))
                return list(local_blocks[:nb]), rem_slice[:nb]
        padded_local = [[] for _ in range(max(num_local_groups, local_group_idx + 1))]
        padded_remote = [[] for _ in range(max(num_local_groups, local_group_idx + 1))]
        padded_local[local_group_idx] = list(local_blocks)
        padded_remote[local_group_idx] = list(remote_blocks)
        aligned_local, aligned_remote = self._apply_prefix_caching(
            decode_block_ids=padded_local,
            prefill_block_ids=padded_remote,
            decode_physical_per_logical=d_ppl,
            prefill_physical_per_logical=p_ppl,
        )
        return aligned_local[local_group_idx], aligned_remote[local_group_idx]

    def _logical_to_kernel_block_ids(self, block_ids, ratio: int):
        if ratio == 1:
            return block_ids
        t_groups = transfer_groups_of(getattr(self, "kv_cache_config", None))
        if not t_groups or len(block_ids) <= len(t_groups):
            return super()._logical_to_kernel_block_ids(block_ids, ratio)
        prefix = super()._logical_to_kernel_block_ids(
            block_ids[: len(t_groups)], ratio
        )
        return list(prefix) + [list(group) for group in block_ids[len(t_groups) :]]

    def _is_heterogeneous_pp_engine(self, engine_id: str) -> bool:
        return (
            getattr(self, "_remote_pp_size_by_engine", {}).get(
                engine_id, self.pp_world_size
            )
            != self.pp_world_size
            or engine_id in getattr(self, "_semantic_region_engines", ())
        )

    def _read_blocks_for_req(self, req_id: str, meta):
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        if not self._is_heterogeneous_pp_engine(engine_id):
            return super()._read_blocks_for_req(req_id, meta)

        self._engine_last_active[engine_id] = time.perf_counter()
        if getattr(self, "_bidirectional_kv_xfer_enabled", False) and hasattr(
            self, "_is_turn2_read_expired"
        ):
            if self._is_turn2_read_expired(meta):
                logger.warning(
                    "Declining expired remote read for %s from engine %s.",
                    req_id,
                    engine_id,
                )
                if hasattr(self, "xfer_stats"):
                    self.xfer_stats.record_kv_expired_req()
                self._handle_failed_transfer(req_id, None)
                return

        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        inactive_keys = getattr(self, "_inactive_remote_keys", {}).get(
            engine_id, set()
        )
        active_remote_keys = [
            remote_key
            for remote_key in sorted(self._remote_agents.get(engine_id, ()))
            if remote_key not in inactive_keys
        ]

        if not any(local_block_ids) or not any(remote_block_ids):
            for remote_key in active_remote_keys:
                notif_id = self._make_pp_aware_notif_id(
                    meta.remote.request_id, engine_id, remote_key
                )
                self.nixl_wrapper.send_notif(
                    self._remote_agents[engine_id][remote_key], notif_msg=notif_id
                )
            if getattr(meta, "awaiting_kvs", False):
                self._recving_transfers.setdefault(req_id, [])
            return

        if not active_remote_keys:
            if getattr(meta, "awaiting_kvs", False):
                self._recving_transfers.setdefault(req_id, [])
            return

        remote_block_starts = getattr(meta, "remote_block_starts", None)
        if (
            remote_block_starts is not None
            and getattr(self, "dsv41_layout", None) is not None
            and getattr(self, "_remote_supports_block_starts", {}).get(engine_id)
            is False
        ):
            raise RuntimeError(
                f"Remote engine {engine_id} does not support per-group "
                "remote_block_starts required by DeepSeek-V4.1 bounded replay; "
                "upgrade peer IluNixl."
            )

        engine_routes = getattr(self, "_mamba_hma_routes", {}).get(engine_id)
        engine_pairings = getattr(self, "region_pairings", {}).get(engine_id)
        transfer_group_ids = list(range(len(local_block_ids)))
        if engine_routes:
            for routes in engine_routes.values():
                for route in routes:
                    if (
                        route.local_group_idx >= len(local_block_ids)
                        or route.remote_group_idx >= len(remote_block_ids)
                    ):
                        raise RuntimeError(
                            "Cannot align producer and consumer KV cache groups "
                            f"for Mamba HMA route: local_group={route.local_group_idx}/"
                            f"{len(local_block_ids)}, remote_group="
                            f"{route.remote_group_idx}/{len(remote_block_ids)}"
                        )
        elif engine_pairings:
            for key_pairings in engine_pairings.values():
                for pairing in key_pairings:
                    if (
                        pairing.local_group_id >= len(local_block_ids)
                        or pairing.remote_group_id >= len(remote_block_ids)
                    ):
                        raise RuntimeError(
                            "Cannot align producer and consumer KV cache groups "
                            f"for region pairing {pairing.identity}: "
                            f"local_group={pairing.local_group_id}/{len(local_block_ids)}, "
                            f"remote_group={pairing.remote_group_id}/{len(remote_block_ids)}"
                        )
        elif len(remote_block_ids) < len(local_block_ids):
            t_groups = transfer_groups_of(self.kv_cache_config)
            transfer_group_ids = [
                group_id
                for group_id, group in enumerate(t_groups)
                if not group.is_eagle_group
            ]
            if len(transfer_group_ids) != len(remote_block_ids):
                raise RuntimeError(
                    "Cannot align producer and consumer KV cache groups: "
                    f"producer={len(remote_block_ids)}, consumer={len(local_block_ids)}, "
                    f"transferable_consumer={len(transfer_group_ids)}"
                )

        notified_remote_pp_ranks: set[int] = set()
        for remote_key in active_remote_keys:
            remote_tp_rank = remote_key % remote_info.remote_tp_size
            remote_pp_rank = remote_key // remote_info.remote_tp_size
            if engine_routes or engine_pairings:
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
                        if remote_tp_rank
                        in plan.source_ranks_per_group[local_group_id]
                        else []
                        for local_group_id in transfer_group_ids
                    ],
                    remote_block_ids=[
                        list(remote_block_ids[remote_group_id])
                        if remote_tp_rank
                        in plan.source_ranks_per_group[local_group_id]
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
                        self._remote_agents[engine_id][remote_key], notif_msg=notif_id
                    )
                    notified_remote_pp_ranks.add(remote_pp_rank)
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
                awaiting_kvs=getattr(meta, "awaiting_kvs", False),
                remote_block_starts=remote_block_starts,
            )
            notified_remote_pp_ranks.add(remote_pp_rank)

        if (
            getattr(self, "use_mla", False)
            and tp_ratio < 0
            and not getattr(self, "_has_mamba", False)
            and getattr(self, "dcp_size", 1) == 1
        ):
            chosen_tp_ranks = set(plan.all_source_ranks)
            for remote_key in active_remote_keys:
                remote_tp_rank = remote_key % remote_info.remote_tp_size
                remote_pp_rank = remote_key // remote_info.remote_tp_size
                if (
                    remote_pp_rank in notified_remote_pp_ranks
                    and remote_tp_rank not in chosen_tp_ranks
                ):
                    notif_id = self._make_pp_aware_notif_id(
                        meta.remote.request_id, engine_id, remote_key
                    )
                    self.nixl_wrapper.send_notif(
                        self._remote_agents[engine_id][remote_key], notif_msg=notif_id
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
        remote_block_starts: list[int] | None = None,
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
        if block_size_ratio > 1:
            local_block_ids, remote_block_ids = (
                self._map_block_ids_for_block_size_ratio(
                    local_block_ids, remote_block_ids, block_size_ratio
                )
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

        compiled_buckets = (
            getattr(self, "_compiled_desc_buckets", {})
            .get(dst_engine_id, {})
            .get(remote_key)
        )
        if compiled_buckets is None:
            compiled_buckets = self._compile_desc_buckets_for_remote_key(
                dst_engine_id, remote_key
            )

        num_local_groups = len(local_block_ids)
        aligned_pair_cache: dict[
            tuple[int, int, bool], tuple[np.ndarray, np.ndarray]
        ] = {}
        routes = (
            getattr(self, "_mamba_hma_routes", {})
            .get(dst_engine_id, {})
            .get(remote_key)
        )
        pairings = (
            getattr(self, "region_pairings", {})
            .get(dst_engine_id, {})
            .get(remote_key)
        )
        if routes is None and pairings is None and remote_block_starts is None:
            local_block_ids, remote_block_ids = self._apply_prefix_caching(
                decode_block_ids=local_block_ids,
                prefill_block_ids=remote_block_ids,
                decode_physical_per_logical=self._physical_blocks_per_logical_kv_block,
                prefill_physical_per_logical=remote_info.remote_physical_blocks_per_logical,
            )
            for bucket in compiled_buckets:
                lg, rg = bucket.local_group_idx, bucket.remote_group_idx
                key = (lg, rg, bucket.is_ssm)
                if key not in aligned_pair_cache:
                    grp_l = local_block_ids[lg] if lg < len(local_block_ids) else ()
                    grp_r = remote_block_ids[rg] if rg < len(remote_block_ids) else ()
                    aligned_pair_cache[key] = (
                        np.asarray(grp_l, dtype=np.int64),
                        np.asarray(grp_r, dtype=np.int64),
                    )
        else:
            for bucket in compiled_buckets:
                lg, rg = bucket.local_group_idx, bucket.remote_group_idx
                key = (lg, rg, bucket.is_ssm)
                if key in aligned_pair_cache:
                    continue
                grp_local = (
                    list(local_block_ids[lg]) if lg < len(local_block_ids) else []
                )
                if not grp_local:
                    aligned_pair_cache[key] = (
                        np.zeros(0, dtype=np.int64),
                        np.zeros(0, dtype=np.int64),
                    )
                    continue
                grp_remote = (
                    list(remote_block_ids[rg]) if rg < len(remote_block_ids) else []
                )
                start_offset = (
                    remote_block_starts[lg]
                    if (
                        not bucket.is_ssm
                        and remote_block_starts is not None
                        and lg < len(remote_block_starts)
                    )
                    else None
                )
                a_local, a_remote = self._align_group_blocks_with_start(
                    grp_local,
                    grp_remote,
                    local_group_idx=lg,
                    num_local_groups=num_local_groups,
                    remote_info=remote_info,
                    start_offset=start_offset,
                )
                aligned_pair_cache[key] = (
                    np.asarray(a_local, dtype=np.int64),
                    np.asarray(a_remote, dtype=np.int64),
                )

        active_entries: list[tuple[_CompiledDescBucket, np.ndarray, np.ndarray]] = []
        uniform_len: int | None = None
        can_single_broadcast = True
        for bucket in compiled_buckets:
            l_arr, r_arr = aligned_pair_cache[
                (bucket.local_group_idx, bucket.remote_group_idx, bucket.is_ssm)
            ]
            if len(l_arr) == 0:
                continue
            active_entries.append((bucket, l_arr, r_arr))
            if uniform_len is None:
                uniform_len = len(l_arr)
            elif len(l_arr) != uniform_len:
                can_single_broadcast = False

        if not active_entries:
            agent_name = self._remote_agents[dst_engine_id][remote_key]
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            if awaiting_kvs:
                self._recving_transfers.setdefault(request_id, [])
            return

        if len(active_entries) == 1:
            bucket, l_arr, r_arr = active_entries[0]
            local_block_descs_ids = (
                bucket.local_bases[:, None] + l_arr[None, :]
            ).reshape(-1)
            remote_block_descs_ids = (
                bucket.remote_bases[:, None] + r_arr[None, :]
            ).reshape(-1)
        elif can_single_broadcast:
            all_local_bases = np.concatenate(
                [b.local_bases for b, _, _ in active_entries]
            )
            all_remote_bases = np.concatenate(
                [b.remote_bases for b, _, _ in active_entries]
            )
            local_rows = np.repeat(
                np.stack([l_arr for _, l_arr, _ in active_entries], axis=0),
                [len(b.local_bases) for b, _, _ in active_entries],
                axis=0,
            )
            remote_rows = np.repeat(
                np.stack([r_arr for _, _, r_arr in active_entries], axis=0),
                [len(b.remote_bases) for b, _, _ in active_entries],
                axis=0,
            )
            local_block_descs_ids = (
                all_local_bases[:, None] + local_rows
            ).reshape(-1)
            remote_block_descs_ids = (
                all_remote_bases[:, None] + remote_rows
            ).reshape(-1)
        else:
            local_block_descs_ids = np.concatenate(
                [
                    (b.local_bases[:, None] + l_arr[None, :]).reshape(-1)
                    for b, l_arr, _ in active_entries
                ]
            )
            remote_block_descs_ids = np.concatenate(
                [
                    (b.remote_bases[:, None] + r_arr[None, :]).reshape(-1)
                    for b, _, r_arr in active_entries
                ]
            )
        assert len(local_block_descs_ids) == len(remote_block_descs_ids), (
            "Heterogeneous PP HMA descriptor count mismatch: "
            f"local={len(local_block_descs_ids)}, remote={len(remote_block_descs_ids)}"
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
            if not hasattr(self, "_handle_remote_info"):
                self._handle_remote_info = {}
            self._handle_remote_info[handle] = (
                dst_engine_id,
                remote_request_id,
                remote_key,
            )
            sessions = getattr(self, "_active_sessions", None)
            if isinstance(sessions, dict) and dst_engine_id in sessions:
                sessions[dst_engine_id].add_inflight(handle)
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Marking blocks as invalid",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_key,
            )
            self._notify_remote_read_failed(
                dst_engine_id, remote_request_id, remote_key
            )
            self._handle_failed_transfer(request_id, handle)


