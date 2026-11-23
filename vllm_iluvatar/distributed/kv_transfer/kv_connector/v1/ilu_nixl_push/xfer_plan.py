# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precompiled descriptor plans and block-range alignment for IluNixlPushConnector."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import _is_ssm_spec

from ..ilu_nixl_common.types import HmaTransferRoute, RegionPairing


def concat_descriptor_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """Avoid an extra allocation when a WRITE contains descriptors from one bucket."""
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays)


def expand_logical_group(block_ids: list[int], ratio: int) -> list[int]:
    if ratio <= 1 or not block_ids:
        return list(block_ids)
    return [b * ratio + offset for b in block_ids for offset in range(ratio)]


def nixl_block_size_ratio(local_block_size: int, remote_block_size: int) -> int:
    """Local/remote connector block ratio.

    ``local >= remote`` and divisible returns ``local // remote`` (one local
    block splits into that many remote blocks). The other direction — a kernel
    page facing a larger remote scheduler block — stays 1; those pages are
    paired with ``physical_blocks_per_logical`` instead.
    """
    if local_block_size <= 0 or remote_block_size <= 0:
        raise AssertionError(
            "block sizes must be positive: "
            f"local={local_block_size}, remote={remote_block_size}"
        )
    if local_block_size % remote_block_size == 0:
        return local_block_size // remote_block_size
    if remote_block_size % local_block_size == 0:
        return 1
    raise AssertionError(
        f"Local block size {local_block_size} is not divisible "
        f"by remote block size {remote_block_size} or vice versa."
    )


def pair_kernel_pages(
    local_blocks: list[int],
    remote_blocks: list[int],
    *,
    local_ratio: int,
    remote_ratio: int,
    local_page_offset: int = 0,
    remote_page_offset: int = 0,
) -> tuple[list[int], list[int]]:
    """Pair attention kernel pages that cover the same token prefix.

    Scheduler blocks can differ (4480 tokens = 70 pages, 640 tokens = 10
    pages). Expanding each side by its own ratio and intersecting the page
    ranges keeps token order. Mamba state is not passed through here.
    """
    local_pages = expand_logical_group(local_blocks, max(local_ratio, 1))
    remote_pages = expand_logical_group(remote_blocks, max(remote_ratio, 1))
    if local_page_offset < 0:
        local_page_offset = 0
    if remote_page_offset < 0:
        remote_page_offset = 0
    local_lo = local_page_offset
    local_hi = local_page_offset + len(local_pages)
    remote_lo = remote_page_offset
    remote_hi = remote_page_offset + len(remote_pages)
    start = max(local_lo, remote_lo)
    end = min(local_hi, remote_hi)
    if start >= end:
        return [], []
    return (
        local_pages[start - local_lo : end - local_lo],
        remote_pages[start - remote_lo : end - remote_lo],
    )


def align_push_group_blocks(
    p_blocks: list[int],
    d_blocks: list[int],
    *,
    p_chunk_offset: int | None,
    d_start_offset: int | None,
    is_ssm: bool,
    req_id: str = "",
    local_group_idx: int = 0,
) -> tuple[list[int], list[int]]:
    """Align one ``(P group, D group)`` pair for chunked-bulk or whole-request push.

    * ``p_chunk_offset >= 0``: token-indexed chunk slice ``[p_off, p_off + len(p))``,
      intersected with D's ``[d_start, d_start + len(d))`` when ``d_start_offset >= 0``.
    * ``p_chunk_offset is None`` or ``p_chunk_offset < 0``: whole/tail-only clipped
      group (Mamba running state, KpoolTail ring, CircularBuffer, or terminal push).
    """
    if not p_blocks or not d_blocks:
        return [], []

    if p_chunk_offset is not None and p_chunk_offset >= 0:
        if d_start_offset is not None and d_start_offset >= 0:
            p_start = p_chunk_offset
            p_end = p_chunk_offset + len(p_blocks)
            d_start = d_start_offset
            d_end = d_start_offset + len(d_blocks)
            inter_start = max(p_start, d_start)
            inter_end = min(p_end, d_end)
            if inter_start >= inter_end:
                return [], []
            return (
                list(p_blocks[inter_start - p_start : inter_end - p_start]),
                list(d_blocks[inter_start - d_start : inter_end - d_start]),
            )
        if p_chunk_offset >= len(d_blocks):
            return [], []
        sel_d = list(d_blocks[p_chunk_offset : p_chunk_offset + len(p_blocks)])
        if len(sel_d) != len(p_blocks):
            raise RuntimeError(
                "decode PUSH_REG does not cover the complete chunked "
                f"block range for {req_id} group={local_group_idx}: "
                f"offset={p_chunk_offset}, requested={len(p_blocks)}, "
                f"registered={len(d_blocks)}, selected={len(sel_d)}"
            )
        return list(p_blocks), sel_d

    if not is_ssm and d_start_offset is not None and d_start_offset >= 0:
        if len(p_blocks) < d_start_offset + len(d_blocks):
            raise ValueError(
                f"Prefill KV group {local_group_idx} has {len(p_blocks)} "
                f"blocks, cannot slice [{d_start_offset}, "
                f"{d_start_offset + len(d_blocks)}) for {req_id}"
            )
        return (
            list(p_blocks[d_start_offset : d_start_offset + len(d_blocks)]),
            list(d_blocks),
        )

    aligned = min(len(p_blocks), len(d_blocks))
    if aligned <= 0:
        return [], []
    return list(p_blocks[-aligned:]), list(d_blocks[-aligned:])


@dataclass(frozen=True)
class CompiledGroupBucket:
    """Precompiled ``(R, 1)`` descriptor base matrices for one ``(local_gid, remote_gid)``."""

    local_group_idx: int
    remote_group_idx: int
    is_ssm: bool
    local_bases: np.ndarray
    remote_bases: np.ndarray
    local_page_ratio: int = 1

    def compute_desc_pair(
        self, local_blocks: list[int], remote_blocks: list[int]
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if (
            not local_blocks
            or not remote_blocks
            or self.local_bases.size == 0
            or self.remote_bases.size == 0
        ):
            return None
        l_arr = np.asarray(local_blocks, dtype=np.int64)[None, :]
        r_arr = np.asarray(remote_blocks, dtype=np.int64)[None, :]
        return (
            (self.local_bases + l_arr).ravel(),
            (self.remote_bases + r_arr).ravel(),
        )


@dataclass(frozen=True)
class CompiledXferPlan:
    """Static per-``(engine_id, remote_key)`` descriptor plan built once at handshake."""

    mode: str  # "routes" | "pairings" | "groups"
    buckets: tuple[CompiledGroupBucket, ...]


def _fa_base_column(
    region_ids: np.ndarray,
    num_blocks: int,
    block_size_ratio: float | None,
    *,
    cumsum_offsets: np.ndarray | None = None,
    wire_desc_offsets: list[int] | tuple[int, ...] = (),
) -> np.ndarray:
    if len(region_ids) == 0:
        return np.empty((0, 1), dtype=np.int64)
    if wire_desc_offsets and block_size_ratio is not None:
        return np.asarray(
            [wire_desc_offsets[int(r)] for r in region_ids], dtype=np.int64
        )[:, None]
    if cumsum_offsets is not None and len(cumsum_offsets) > 0:
        scale = int(block_size_ratio) if block_size_ratio is not None else 1
        return (cumsum_offsets[region_ids] * scale)[:, None]
    nb = (
        int(num_blocks * block_size_ratio)
        if block_size_ratio is not None
        else int(num_blocks)
    )
    return (np.asarray(region_ids, dtype=np.int64) * nb)[:, None]


def _ssm_base_column(
    region_ids: np.ndarray,
    num_physical_blocks: int,
    physical_blocks_per_logical: int,
    num_fa_regions: int,
    regions_per_layer: int,
    num_fa_components: int = 1,
) -> np.ndarray:
    if len(region_ids) == 0:
        return np.empty((0, 1), dtype=np.int64)
    num_logical_blocks = num_physical_blocks // physical_blocks_per_logical
    sub = np.arange(regions_per_layer, dtype=np.int64)[None, :]
    bases = (
        num_fa_regions * num_fa_components * num_physical_blocks
        + (np.asarray(region_ids, dtype=np.int64)[:, None] * regions_per_layer + sub)
        * num_logical_blocks
    )
    return bases.reshape(-1, 1)


def _bucket_local_page_ratio(
    region_ids: np.ndarray,
    local_region_page_ratios: tuple[int, ...] | list[int],
) -> int:
    if not local_region_page_ratios or len(region_ids) == 0:
        return 1
    ratios = {
        int(local_region_page_ratios[int(rid)])
        for rid in region_ids
        if int(rid) < len(local_region_page_ratios)
    }
    if not ratios:
        return 1
    if len(ratios) != 1:
        raise ValueError(
            "All regions in one compiled transfer bucket must share a local "
            f"sub-page ratio, got regions={region_ids.tolist()} ratios={sorted(ratios)}"
        )
    return max(ratios.pop(), 1)


def compile_xfer_plan(
    *,
    local_num_blocks: int,
    remote_num_blocks: int,
    local_physical_per_logical: int,
    remote_physical_per_logical: int,
    block_size_ratio: int,
    num_local_fa_regions: int,
    num_remote_fa_regions: int | None,
    group_spec_types: tuple[Any, ...],
    conv_decomp: Any | None,
    local_cumsum: np.ndarray | None = None,
    remote_cumsum: np.ndarray | None = None,
    wire_desc_offsets: list[int] | tuple[int, ...] = (),
    local_region_page_ratios: tuple[int, ...] | list[int] = (),
    routes: list[HmaTransferRoute] | None = None,
    pairings: list[RegionPairing] | None = None,
    local_region_ids_by_group: list[np.ndarray] | None = None,
    remote_region_ids_by_group: list[np.ndarray] | None = None,
) -> CompiledXferPlan:
    """Compile static descriptor base matrices for one ``(engine_id, remote_key)``."""
    regions_per_ssm_layer = (
        len(conv_decomp.local_conv_offsets) + 1 if conv_decomp is not None else 0
    )
    buckets: list[CompiledGroupBucket] = []

    if routes is not None:
        eff_remote_fa = num_remote_fa_regions
        if eff_remote_fa is None:
            eff_remote_fa = max(
                (
                    int(r.remote_desc_region_ids.max()) + 1
                    for r in routes
                    if not r.is_ssm and len(r.remote_desc_region_ids)
                ),
                default=0,
            )
        for route in routes:
            if route.is_ssm:
                l_bases = _ssm_base_column(
                    route.local_desc_region_ids,
                    local_num_blocks,
                    local_physical_per_logical,
                    num_local_fa_regions,
                    regions_per_ssm_layer,
                )
                r_bases = _ssm_base_column(
                    route.remote_desc_region_ids,
                    remote_num_blocks,
                    remote_physical_per_logical,
                    eff_remote_fa,
                    regions_per_ssm_layer,
                )
            else:
                l_bases = _fa_base_column(
                    route.local_desc_region_ids,
                    local_num_blocks,
                    block_size_ratio,
                    cumsum_offsets=local_cumsum,
                    wire_desc_offsets=wire_desc_offsets,
                )
                r_bases = _fa_base_column(
                    route.remote_desc_region_ids,
                    remote_num_blocks,
                    None,
                    cumsum_offsets=remote_cumsum,
                )
            buckets.append(
                CompiledGroupBucket(
                    local_group_idx=route.local_group_idx,
                    remote_group_idx=route.remote_group_idx,
                    is_ssm=route.is_ssm,
                    local_bases=l_bases,
                    remote_bases=r_bases,
                    local_page_ratio=(
                        1
                        if route.is_ssm
                        else _bucket_local_page_ratio(
                            route.local_desc_region_ids,
                            local_region_page_ratios,
                        )
                    ),
                )
            )
        return CompiledXferPlan(mode="routes", buckets=tuple(buckets))

    if pairings is not None:
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

        eff_remote_fa = (
            num_remote_fa_regions
            if num_remote_fa_regions is not None
            else len(selected_remote_order)
        )
        for (local_gid, remote_gid), reg_pairs in sorted(bucketed.items()):
            is_ssm = (
                local_gid < len(group_spec_types)
                and _is_ssm_spec(group_spec_types[local_gid])
            )
            local_reg_arr = np.asarray(
                [pair[0] for pair in reg_pairs], dtype=np.int64
            )
            remote_reg_arr = np.asarray(
                [pair[1] for pair in reg_pairs], dtype=np.int64
            )
            if is_ssm:
                l_bases = _ssm_base_column(
                    local_reg_arr,
                    local_num_blocks,
                    local_physical_per_logical,
                    num_local_fa_regions,
                    regions_per_ssm_layer,
                )
                r_bases = _ssm_base_column(
                    remote_reg_arr,
                    remote_num_blocks,
                    remote_physical_per_logical,
                    eff_remote_fa,
                    regions_per_ssm_layer,
                )
            else:
                l_bases = _fa_base_column(
                    local_reg_arr,
                    local_num_blocks,
                    block_size_ratio,
                    cumsum_offsets=local_cumsum,
                    wire_desc_offsets=wire_desc_offsets,
                )
                r_bases = _fa_base_column(
                    remote_reg_arr,
                    remote_num_blocks,
                    None,
                    cumsum_offsets=remote_cumsum,
                )
            buckets.append(
                CompiledGroupBucket(
                    local_group_idx=local_gid,
                    remote_group_idx=remote_gid,
                    is_ssm=is_ssm,
                    local_bases=l_bases,
                    remote_bases=r_bases,
                    local_page_ratio=(
                        1
                        if is_ssm
                        else _bucket_local_page_ratio(
                            local_reg_arr, local_region_page_ratios
                        )
                    ),
                )
            )
        return CompiledXferPlan(mode="pairings", buckets=tuple(buckets))

    local_groups = local_region_ids_by_group or []
    remote_groups = remote_region_ids_by_group or []
    eff_remote_fa = num_remote_fa_regions
    if eff_remote_fa is None:
        eff_remote_fa = max(
            (int(ids.max()) + 1 for ids in remote_groups if len(ids)),
            default=0,
        )
    for group_idx in range(min(len(local_groups), len(remote_groups))):
        is_ssm = (
            group_idx < len(group_spec_types)
            and _is_ssm_spec(group_spec_types[group_idx])
        )
        l_regs = local_groups[group_idx]
        r_regs = remote_groups[group_idx]
        if is_ssm:
            l_bases = _ssm_base_column(
                l_regs,
                local_num_blocks,
                local_physical_per_logical,
                num_local_fa_regions,
                regions_per_ssm_layer,
            )
            r_bases = _ssm_base_column(
                r_regs,
                remote_num_blocks,
                remote_physical_per_logical,
                eff_remote_fa,
                regions_per_ssm_layer,
            )
        else:
            l_bases = _fa_base_column(
                l_regs,
                local_num_blocks,
                block_size_ratio,
                cumsum_offsets=local_cumsum,
                wire_desc_offsets=wire_desc_offsets,
            )
            r_bases = _fa_base_column(
                r_regs,
                remote_num_blocks,
                None,
                cumsum_offsets=remote_cumsum,
            )
        buckets.append(
            CompiledGroupBucket(
                local_group_idx=group_idx,
                remote_group_idx=group_idx,
                is_ssm=is_ssm,
                local_bases=l_bases,
                remote_bases=r_bases,
                local_page_ratio=(
                    1
                    if is_ssm
                    else _bucket_local_page_ratio(
                        l_regs, local_region_page_ratios
                    )
                ),
            )
        )
    return CompiledXferPlan(mode="groups", buckets=tuple(buckets))
