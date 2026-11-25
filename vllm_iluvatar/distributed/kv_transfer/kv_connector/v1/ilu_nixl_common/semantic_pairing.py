# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared PP/HMA/Mamba semantic region selection and pairing helpers."""

from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from typing import Any

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import _is_ssm_spec
from vllm.model_executor.models.utils import extract_layer_index

from .types import (
    HmaTransferRoute,
    RegionPairing,
    SemanticTransferRegion,
)


def pp_layer_range(
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


def target_pp_ranks(
    local_pp_split: str,
    local_pp_rank: int,
    remote_pp_split: str,
    local_layer_region_ids: dict[str, list[int]] | None = None,
) -> list[int]:
    local_start, local_end = pp_layer_range(
        local_pp_split, local_pp_rank, local_layer_region_ids
    )
    remote_split = [int(x) for x in remote_pp_split.split(",")]
    result: list[int] = []
    remote_start = 0
    for remote_pp_rank, num_layers in enumerate(remote_split):
        remote_end = remote_start + num_layers
        eff_remote_end = (
            max(remote_end, local_end)
            if remote_pp_rank == len(remote_split) - 1
            else remote_end
        )
        if max(local_start, remote_start) < min(local_end, eff_remote_end):
            result.append(remote_pp_rank)
        remote_start = remote_end
    return result


def overlap_pp_count(
    producer_pp_rank: int,
    producer_pp_split: str,
    consumer_pp_split: str,
) -> int:
    return len(
        target_pp_ranks(
            producer_pp_split,
            producer_pp_rank,
            consumer_pp_split,
        )
    )


def effective_overlap_pp_count(
    producer_pp_rank: int,
    producer_pp_split: str,
    consumer_pp_split: str,
    kv_layer_end: int | None = None,
) -> int:
    if kv_layer_end is None:
        return overlap_pp_count(producer_pp_rank, producer_pp_split, consumer_pp_split)
    p_start, p_end = pp_layer_range(producer_pp_split, producer_pp_rank)
    p_end = min(p_end, kv_layer_end)
    if p_start >= p_end:
        return 0
    consumer_counts = [int(x) for x in consumer_pp_split.split(",")]
    d_start = 0
    count = 0
    for num_layers in consumer_counts:
        d_end = d_start + num_layers
        if max(p_start, d_start) < min(p_end, d_end):
            count += 1
        d_start = d_end
    return count


def cache_kind_from_layer_name(layer_name: str) -> str:
    is_scale = layer_name.endswith(".kv_cache_scales")
    base_name = layer_name.removesuffix(".kv_cache_scales") if is_scale else layer_name
    if ".swa_cache" in base_name:
        kind = "swa_cache"
    elif ".indexer.k_cache" in base_name:
        kind = "indexer_k_cache"
    elif ".indexer.tail_cache" in base_name:
        kind = "kpool_tail_cache"
    elif ".indexer.compressor" in base_name:
        kind = "indexer_compressor"
    elif ".compressor.state_cache" in base_name:
        kind = "compressor_state_cache"
    elif (
        ".ilu_deepseek_v4_multi_head_latent_attention" in base_name
        or ".ilu_glm_moe_dsa_attention" in base_name
        or base_name.endswith(".mla_attn")
        or base_name.endswith(".attn")
    ):
        kind = "main_mla"
    elif base_name.endswith(".linear_attn") or base_name.endswith(".self_attn"):
        kind = "mamba_cache"
    else:
        kind = "unknown"
    return f"{kind}_scales" if is_scale else kind


def transfer_identity(layer_name: str, group_idx: int | None = None) -> str:
    del group_idx
    try:
        layer_idx = extract_layer_index(layer_name)
    except (AssertionError, ValueError):
        layer_idx = -1
    return f"{layer_idx}:{cache_kind_from_layer_name(layer_name)}"


def regions_by_id(
    transfer_regions: list[SemanticTransferRegion] | None,
) -> dict[int, SemanticTransferRegion]:
    return {region.region_id: region for region in transfer_regions or []}


def iter_region_identity_groups(
    region: SemanticTransferRegion,
) -> list[tuple[str, list[int]]]:
    """Yield ``(base_identity, group_ids)`` for both group-free and legacy regions."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for raw_identity in region.identities:
        parts = raw_identity.split(":", 2)
        if len(parts) == 3 and parts[2].startswith("group"):
            base_ident = f"{parts[0]}:{parts[1]}"
            gid = int(parts[2].removeprefix("group"))
            if gid not in grouped[base_ident]:
                grouped[base_ident].append(gid)
        else:
            base_ident = raw_identity
            if region.identity_groups and base_ident in region.identity_groups:
                gids = region.identity_groups[base_ident]
            elif region.group_ids:
                gids = region.group_ids
            else:
                gids = [0]
            for gid in gids:
                if gid not in grouped[base_ident]:
                    grouped[base_ident].append(gid)
    return [(ident, sorted(gids)) for ident, gids in grouped.items()]


def region_ids_for_layer_range(
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


def region_ids_for_mtp_layers(
    layer_region_ids: dict[str, list[int]], target_total_layers: int
) -> list[int]:
    """Return MTP draft KV regions appended outside the target PP partition."""
    selected: list[int] = []
    layers: list[tuple[int, str, list[int]]] = []
    for layer_name, region_ids in layer_region_ids.items():
        try:
            layer_idx = extract_layer_index(layer_name)
        except (AssertionError, ValueError):
            continue
        if (
            layer_idx >= target_total_layers
            or "draft_model" in layer_name
            or ".mtp_block." in layer_name
            or ".mtp.layers." in layer_name
        ):
            layers.append((layer_idx, layer_name, region_ids))
    for _, _, region_ids in sorted(layers):
        for region_id in region_ids:
            if region_id not in selected:
                selected.append(region_id)
    return selected


def filter_group_region_ids(
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


def pair_hma_transfer_regions(
    selected_local_region_ids: list[int],
    selected_remote_region_ids: list[int],
    local_transfer_regions: list[SemanticTransferRegion],
    remote_transfer_regions: list[SemanticTransferRegion],
    num_groups: int | None = None,
    *,
    overlap_start: int | None = None,
    overlap_end: int | None = None,
    strict_identity_set: bool = False,
) -> tuple[list[RegionPairing], list[int], list[int]]:
    """Pair local and remote semantic regions into 4-tuple ``RegionPairing``s."""
    local_regions = regions_by_id(local_transfer_regions)
    remote_regions = regions_by_id(remote_transfer_regions)

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

    def in_overlap(ident: str) -> bool:
        if overlap_start is None or overlap_end is None:
            return True
        layer_str = ident.split(":", 1)[0]
        layer_idx = int(layer_str)
        if layer_idx < 0:
            return True
        return overlap_start <= layer_idx < overlap_end

    local_by_identity: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for region in selected_local:
        for ident, gids in iter_region_identity_groups(region):
            if not in_overlap(ident):
                continue
            for gid in gids:
                if num_groups is not None and gid >= num_groups:
                    continue
                entry = (gid, region.region_id)
                if entry not in local_by_identity[ident]:
                    local_by_identity[ident].append(entry)

    remote_by_identity: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for region in selected_remote:
        for ident, gids in iter_region_identity_groups(region):
            if not in_overlap(ident):
                continue
            for gid in gids:
                entry = (gid, region.region_id)
                if entry not in remote_by_identity[ident]:
                    remote_by_identity[ident].append(entry)

    local_identities = set(local_by_identity)
    remote_identities = set(remote_by_identity)
    if strict_identity_set and local_identities != remote_identities:
        raise RuntimeError(
            "Heterogeneous PP HMA semantic alignment requires identical "
            "cache identities on both peers; refusing a partial intersection. "
            f"local_only={sorted(local_identities - remote_identities)[:8]}, "
            f"remote_only={sorted(remote_identities - local_identities)[:8]}, "
            f"local_regions={selected_local_region_ids}, "
            f"remote_regions={selected_remote_region_ids}"
        )

    def identity_sort_key(identity: str) -> tuple[int, str]:
        layer_idx, kind = identity.split(":", 1)
        return (int(layer_idx), kind)

    common_identities = sorted(
        local_identities.intersection(remote_identities),
        key=identity_sort_key,
    )
    if not common_identities:
        raise RuntimeError(
            "Heterogeneous PP HMA semantic alignment found no shared "
            "cache identities between local and remote selected regions. "
            f"local_regions={selected_local_region_ids}, "
            f"remote_regions={selected_remote_region_ids}"
        )

    conflicts: list[dict[str, Any]] = []
    raw_pairings: list[RegionPairing] = []
    for ident in common_identities:
        local_entries = sorted(local_by_identity[ident])
        remote_entries = sorted(remote_by_identity[ident])
        if len(local_entries) != len(remote_entries):
            conflicts.append(
                {
                    "identity": ident,
                    "local_entries": local_entries,
                    "remote_entries": remote_entries,
                }
            )
            continue
        for (local_gid, local_rid), (remote_gid, remote_rid) in zip(
            local_entries, remote_entries
        ):
            raw_pairings.append(
                RegionPairing(
                    local_region_id=local_rid,
                    remote_region_id=remote_rid,
                    local_group_id=local_gid,
                    remote_group_id=remote_gid,
                    identity=ident,
                )
            )

    by_group_pair: dict[tuple[int, int], list[RegionPairing]] = defaultdict(list)
    for pairing in raw_pairings:
        by_group_pair[(pairing.local_group_id, pairing.remote_group_id)].append(pairing)

    dedup_pairings: list[RegionPairing] = []
    selected_local_ids: list[int] = []
    selected_remote_ids: list[int] = []
    global_local_to_remote: dict[int, int] = {}
    global_remote_to_local: dict[int, int] = {}

    for (local_gid, remote_gid), group_list in sorted(by_group_pair.items()):
        local_to_remote: dict[int, int] = {}
        remote_to_local: dict[int, int] = {}
        seen_in_bucket: set[tuple[int, int]] = set()
        for pairing in group_list:
            l_rid = pairing.local_region_id
            r_rid = pairing.remote_region_id
            prior_remote = local_to_remote.get(l_rid)
            prior_local = remote_to_local.get(r_rid)
            if prior_remote is not None and prior_remote != r_rid:
                conflicts.append(
                    {
                        "group": (local_gid, remote_gid),
                        "local_region": l_rid,
                        "remote_regions": sorted({prior_remote, r_rid}),
                        "identity": pairing.identity,
                    }
                )
                continue
            if prior_local is not None and prior_local != l_rid:
                conflicts.append(
                    {
                        "group": (local_gid, remote_gid),
                        "remote_region": r_rid,
                        "local_regions": sorted({prior_local, l_rid}),
                        "identity": pairing.identity,
                    }
                )
                continue
            prior_global_remote = global_local_to_remote.get(l_rid)
            prior_global_local = global_remote_to_local.get(r_rid)
            if (prior_global_remote is not None and prior_global_remote != r_rid) or (
                prior_global_local is not None and prior_global_local != l_rid
            ):
                conflicts.append(
                    {
                        "group": (local_gid, remote_gid),
                        "local_region": l_rid,
                        "remote_region": r_rid,
                        "identity": pairing.identity,
                    }
                )
                continue
            local_to_remote[l_rid] = r_rid
            remote_to_local[r_rid] = l_rid
            global_local_to_remote[l_rid] = r_rid
            global_remote_to_local[r_rid] = l_rid
            if l_rid not in selected_local_ids:
                selected_local_ids.append(l_rid)
            if r_rid not in selected_remote_ids:
                selected_remote_ids.append(r_rid)
            if (l_rid, r_rid) not in seen_in_bucket:
                seen_in_bucket.add((l_rid, r_rid))
                dedup_pairings.append(pairing)

    if conflicts:
        raise RuntimeError(
            "Heterogeneous PP HMA needs slice/scatter transfer inside a "
            "single KV group because one semantic owner maps to multiple "
            f"physical regions. conflicts={conflicts[:8]}"
        )

    return dedup_pairings, selected_local_ids, selected_remote_ids


def align_hma_transfer_regions(
    selected_local_region_ids: list[int],
    selected_remote_region_ids: list[int],
    local_transfer_regions: list[SemanticTransferRegion],
    remote_transfer_regions: list[SemanticTransferRegion],
    num_groups: int,
    *,
    overlap_start: int | None = None,
    overlap_end: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]:
    """Align HMA regions while deferring TP-scaled block_len checks to handshake validation."""
    pairings, selected_local_ids, selected_remote_ids = pair_hma_transfer_regions(
        selected_local_region_ids,
        selected_remote_region_ids,
        local_transfer_regions,
        remote_transfer_regions,
        num_groups=num_groups,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
    )
    compact_remote_map = {
        region_id: compact_id
        for compact_id, region_id in enumerate(selected_remote_ids)
    }
    local_groups: list[list[int]] = [[] for _ in range(num_groups)]
    remote_groups: list[list[int]] = [[] for _ in range(num_groups)]
    for pairing in pairings:
        gid = pairing.local_group_id
        if gid >= num_groups:
            continue
        if pairing.local_region_id not in local_groups[gid]:
            local_groups[gid].append(pairing.local_region_id)
            remote_groups[gid].append(compact_remote_map[pairing.remote_region_id])
    return (
        [np.asarray(g, dtype=np.int64) for g in local_groups],
        [np.asarray(g, dtype=np.int64) for g in remote_groups],
        selected_local_ids,
        selected_remote_ids,
    )


def build_mamba_hma_transfer_routes(
    selected_local_region_ids: list[int],
    selected_remote_region_ids: list[int],
    local_transfer_regions: list[SemanticTransferRegion],
    remote_transfer_regions: list[SemanticTransferRegion],
    *,
    num_local_groups: int = 0,
    group_spec_types: tuple[Any, ...] | None = None,
    ssm_region_indices: list[int] | None = None,
    num_local_regions: int | None = None,
    overlap_start: int | None = None,
    overlap_end: int | None = None,
) -> tuple[
    list[HmaTransferRoute],
    list[int],
    list[int],
    list[int],
]:
    """Build hybrid MLA+Mamba HMA routes separating FA and SSM descriptor channels."""
    local_regions = regions_by_id(local_transfer_regions)
    remote_regions = regions_by_id(remote_transfer_regions)

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
        for base_ident, gids in iter_region_identity_groups(region):
            layer_str, kind = base_ident.split(":", 1)
            layer_idx = int(layer_str)
            if in_overlap(layer_idx):
                local_by_key[(layer_idx, kind)] = (region.region_id, gids[0])

    remote_by_key: dict[tuple[int, str], tuple[int, int]] = {}
    for region in selected_remote:
        for base_ident, gids in iter_region_identity_groups(region):
            layer_str, kind = base_ident.split(":", 1)
            layer_idx = int(layer_str)
            if in_overlap(layer_idx):
                remote_by_key[(layer_idx, kind)] = (region.region_id, gids[0])

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
                    prior_fa_remote is not None and prior_fa_remote != remote_region_id
                ) or (prior_fa_local is not None and prior_fa_local != local_region_id):
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
        reg_id: compact_id for compact_id, reg_id in enumerate(selected_fa_remote_ids)
    }

    total_local_regions = (
        num_local_regions
        if num_local_regions is not None
        else len(local_transfer_regions)
    )
    local_ssm_regions = (
        list(ssm_region_indices)
        if ssm_region_indices
        else list(range(total_local_regions))
    )
    local_ssm_slot_map = {
        int(reg_id): slot_idx for slot_idx, reg_id in enumerate(local_ssm_regions)
    }
    remote_ssm_slot_map = {
        int(reg_id): slot_idx for slot_idx, reg_id in enumerate(selected_ssm_remote_ids)
    }

    routes: list[HmaTransferRoute] = []
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
            HmaTransferRoute(
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


def select_pp_regions_and_pairings(
    *,
    local_pp_split: str,
    local_pp_rank: int,
    local_layer_region_ids: dict[str, list[int]],
    local_region_ids_by_group: list[list[int]] | list[np.ndarray],
    remote_pp_split: str,
    remote_pp_rank: int,
    remote_layer_region_ids: dict[str, list[int]],
    remote_region_ids_by_group: list[list[int]] | list[np.ndarray],
    is_hma_required: bool = False,
    local_transfer_regions: list[SemanticTransferRegion] | None = None,
    remote_transfer_regions: list[SemanticTransferRegion] | None = None,
    include_final_stage_extra_layers: bool = False,
    local_kv_layer_end: int | None = None,
    remote_kv_layer_end: int | None = None,
    allow_empty_skipped_overlap: bool = False,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[int],
    list[int],
    list[RegionPairing] | None,
    bool,
]:
    """Select overlapping regions and optional HMA ``RegionPairing``s."""
    local_start, local_end = pp_layer_range(
        local_pp_split, local_pp_rank, local_layer_region_ids
    )
    remote_start, remote_end = pp_layer_range(
        remote_pp_split, remote_pp_rank, remote_layer_region_ids
    )
    raw_overlap_start = max(local_start, remote_start)
    raw_overlap_end = min(local_end, remote_end)
    if raw_overlap_start >= raw_overlap_end:
        raise RuntimeError(
            "Heterogeneous PP handshake selected a remote PP stage "
            "with no overlapping layers: "
            f"local=[{local_start}, {local_end}), "
            f"remote=[{remote_start}, {remote_end})."
        )

    if local_kv_layer_end is not None:
        local_end = min(local_end, local_kv_layer_end)
    if remote_kv_layer_end is not None:
        remote_end = min(remote_end, remote_kv_layer_end)
    overlap_start = max(local_start, remote_start)
    overlap_end = min(local_end, remote_end)
    num_groups = len(local_region_ids_by_group)
    empty_groups = [np.asarray([], dtype=np.int64) for _ in range(num_groups)]
    if overlap_start >= overlap_end:
        return empty_groups, empty_groups, [], [], [], True

    selected_local_region_ids = region_ids_for_layer_range(
        local_layer_region_ids, overlap_start, overlap_end
    )
    selected_remote_region_ids = region_ids_for_layer_range(
        remote_layer_region_ids, overlap_start, overlap_end
    )

    extra_overlap = False
    if (
        include_final_stage_extra_layers
        and local_pp_rank == len(local_pp_split.split(",")) - 1
        and remote_pp_rank == len(remote_pp_split.split(",")) - 1
    ):
        local_total_layers = sum(int(x) for x in local_pp_split.split(","))
        remote_total_layers = sum(int(x) for x in remote_pp_split.split(","))
        local_extra_region_ids = region_ids_for_mtp_layers(
            local_layer_region_ids, local_total_layers
        )
        remote_extra_region_ids = region_ids_for_mtp_layers(
            remote_layer_region_ids, remote_total_layers
        )
        if local_extra_region_ids and remote_extra_region_ids:
            extra_overlap = True
            selected_local_region_ids.extend(
                region_id
                for region_id in local_extra_region_ids
                if region_id not in selected_local_region_ids
            )
            selected_remote_region_ids.extend(
                region_id
                for region_id in remote_extra_region_ids
                if region_id not in selected_remote_region_ids
            )

    if allow_empty_skipped_overlap and (
        not selected_local_region_ids or not selected_remote_region_ids
    ):
        return empty_groups, empty_groups, [], [], [], True

    if is_hma_required:
        assert local_transfer_regions is not None
        assert remote_transfer_regions is not None
        pairings, selected_local_ids, selected_remote_ids = pair_hma_transfer_regions(
            selected_local_region_ids,
            selected_remote_region_ids,
            local_transfer_regions,
            remote_transfer_regions,
            num_groups=num_groups,
            overlap_start=None if extra_overlap else overlap_start,
            overlap_end=None if extra_overlap else overlap_end,
        )
        if allow_empty_skipped_overlap and not pairings:
            return empty_groups, empty_groups, [], [], [], True
        compact_remote_map = {
            region_id: compact_id
            for compact_id, region_id in enumerate(selected_remote_ids)
        }
        local_by_group: list[list[int]] = [[] for _ in range(num_groups)]
        remote_by_group: list[list[int]] = [[] for _ in range(num_groups)]
        for pairing in pairings:
            gid = pairing.local_group_id
            if gid < num_groups and pairing.local_region_id not in local_by_group[gid]:
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

    return (
        filter_group_region_ids(
            local_region_ids_by_group,
            selected_local_region_ids,
            compact=False,
        ),
        filter_group_region_ids(
            remote_region_ids_by_group,
            selected_remote_region_ids,
            compact=True,
        ),
        selected_local_region_ids,
        selected_remote_region_ids,
        None,
        False,
    )


def select_pp_regions(
    *,
    local_pp_split: str,
    local_pp_rank: int,
    local_layer_region_ids: dict[str, list[int]],
    local_region_ids_by_group: list[list[int]] | list[np.ndarray],
    remote_pp_split: str,
    remote_pp_rank: int,
    remote_layer_region_ids: dict[str, list[int]],
    remote_region_ids_by_group: list[list[int]] | list[np.ndarray],
    is_hma_required: bool = False,
    local_transfer_regions: list[SemanticTransferRegion] | None = None,
    remote_transfer_regions: list[SemanticTransferRegion] | None = None,
    include_final_stage_extra_layers: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]:
    local_groups, remote_groups, sel_local, sel_remote, _, _ = (
        select_pp_regions_and_pairings(
            local_pp_split=local_pp_split,
            local_pp_rank=local_pp_rank,
            local_layer_region_ids=local_layer_region_ids,
            local_region_ids_by_group=local_region_ids_by_group,
            remote_pp_split=remote_pp_split,
            remote_pp_rank=remote_pp_rank,
            remote_layer_region_ids=remote_layer_region_ids,
            remote_region_ids_by_group=remote_region_ids_by_group,
            is_hma_required=is_hma_required,
            local_transfer_regions=local_transfer_regions,
            remote_transfer_regions=remote_transfer_regions,
            include_final_stage_extra_layers=include_final_stage_extra_layers,
        )
    )
    return local_groups, remote_groups, sel_local, sel_remote


def _divisible_span_ratio(larger: int, smaller: int) -> int | None:
    """How many ``smaller`` spans fit in ``larger``. ``None`` when not integral."""
    if larger <= 0 or smaller <= 0:
        return 1
    if larger == smaller:
        return 1
    if larger < smaller:
        larger, smaller = smaller, larger
    if larger % smaller != 0:
        return None
    return larger // smaller


def validate_region_geometry_pair(
    local_region: SemanticTransferRegion | None,
    remote_region: SemanticTransferRegion | None,
    *,
    identity: str,
    local_region_id: int,
    remote_region_id: int,
    is_ssm: bool = False,
    allow_divisible_ratio: bool = True,
) -> int:
    """Return the scheduler-block ratio (>= 1) between the two regions.

    Attention pages may differ by an integer ratio (P PP8 block 4480 vs D TP8
    block 640). Mamba state is one slot and is copied 1:1; its ``block_tokens``
    is only the scheduler span, so a ratio there does not split the state.
    """
    if local_region is None or remote_region is None:
        return 1
    if not allow_divisible_ratio:
        if local_region.kernel_blocks != remote_region.kernel_blocks:
            raise ValueError(
                f"{identity} (local r{local_region_id}, remote r{remote_region_id}): "
                "kernel blocks per scheduler block differ "
                f"(local={local_region.kernel_blocks}, "
                f"remote={remote_region.kernel_blocks})"
            )
        if (
            local_region.block_tokens > 0
            and remote_region.block_tokens > 0
            and local_region.block_tokens != remote_region.block_tokens
        ):
            raise ValueError(
                f"{identity} (local r{local_region_id}, remote r{remote_region_id}): "
                "scheduler block_tokens differ "
                f"(local={local_region.block_tokens}, "
                f"remote={remote_region.block_tokens})"
            )
        return 1
    where = f"{identity} (local r{local_region_id}, remote r{remote_region_id})"
    if is_ssm and local_region.kernel_blocks == remote_region.kernel_blocks:
        # Real mamba state is one slot on both sides; its block_tokens is only
        # the scheduler span, so non-divisible spans (e.g. 4480 vs 2304) still
        # copy 1:1.
        if local_region.block_tokens > 0 and remote_region.block_tokens > 0:
            ratio = _divisible_span_ratio(
                local_region.block_tokens, remote_region.block_tokens
            )
            return max(ratio or 1, 1)
        return 1
    token_ratio = 1
    if local_region.block_tokens > 0 and remote_region.block_tokens > 0:
        token_ratio = _divisible_span_ratio(
            local_region.block_tokens, remote_region.block_tokens
        )
        if token_ratio is None:
            # When both sides expose multiple kernel pages (e.g. P 4480=70*64
            # vs D 2304=36*64), page pairing aligns at the kernel-page level.
            if (
                local_region.kernel_blocks > 1
                and remote_region.kernel_blocks > 1
                and local_region.block_tokens % local_region.kernel_blocks == 0
                and remote_region.block_tokens % remote_region.kernel_blocks == 0
            ):
                local_page_tokens = (
                    local_region.block_tokens // local_region.kernel_blocks
                )
                remote_page_tokens = (
                    remote_region.block_tokens // remote_region.kernel_blocks
                )
                page_ratio = _divisible_span_ratio(
                    local_page_tokens, remote_page_tokens
                )
                if page_ratio is not None:
                    return max(page_ratio, 1)
            raise ValueError(
                f"{where}: scheduler block_tokens differ and are not divisible "
                f"(local={local_region.block_tokens}, "
                f"remote={remote_region.block_tokens}, "
                f"local_kblocks={local_region.kernel_blocks}, "
                f"remote_kblocks={remote_region.kernel_blocks}, "
                f"local_blen={local_region.block_len}, "
                f"remote_blen={remote_region.block_len}, "
                f"is_ssm={is_ssm})"
            )
    kernel_ratio = _divisible_span_ratio(
        local_region.kernel_blocks, remote_region.kernel_blocks
    )
    if kernel_ratio is None:
        raise ValueError(
            f"{where}: kernel blocks per scheduler block differ and are not "
            f"divisible (local={local_region.kernel_blocks}, "
            f"remote={remote_region.kernel_blocks})"
        )
    if (
        token_ratio != 1
        and kernel_ratio != 1
        and token_ratio != kernel_ratio
        # P stores 4480 tokens as 70 kernel pages of 64. D's 640-token block
        # is one scheduler slot (kernel_blocks=1) even after the connector
        # resyncs to the same 64-token page, so 70 == 7 * 10. The page pairing
        # uses physical_blocks_per_logical; this check only requires that the
        # kernel-page ratio is an integer multiple of the scheduler-block ratio.
        and kernel_ratio % token_ratio != 0
    ):
        raise ValueError(
            f"{where}: scheduler block ratio {token_ratio} does not match "
            f"kernel-block ratio {kernel_ratio} "
            f"(block_tokens local={local_region.block_tokens}, "
            f"remote={remote_region.block_tokens}; "
            f"kernel_blocks local={local_region.kernel_blocks}, "
            f"remote={remote_region.kernel_blocks})"
        )
    return max(token_ratio, 1)


def validate_pp_region_lengths(
    *,
    local_block_lens: list[int],
    remote_block_lens: list[int],
    selected_local_region_ids: list[int],
    selected_remote_region_ids: list[int],
    tp_ratio: int,
    block_size_ratio: int,
    replicated_region_ids: set[int],
    model_replicated: bool,
) -> None:
    assert len(selected_local_region_ids) == len(selected_remote_region_ids), (
        "Heterogeneous PP selected region count mismatch: "
        f"local={len(selected_local_region_ids)}, "
        f"remote={len(selected_remote_region_ids)}"
    )
    for pair_idx, (local_region_id, remote_region_id) in enumerate(
        zip(selected_local_region_ids, selected_remote_region_ids)
    ):
        local_len = local_block_lens[local_region_id]
        remote_len = remote_block_lens[remote_region_id]
        replicated = model_replicated or local_region_id in replicated_region_ids
        if replicated:
            expected_remote_len = local_len // block_size_ratio
        elif tp_ratio > 0:
            expected_remote_len = (local_len * tp_ratio) // block_size_ratio
        else:
            assert block_size_ratio == 1, (
                "Different local/remote block sizes are not supported when P TP > D TP."
            )
            expected_remote_len = local_len // (-tp_ratio)
        assert remote_len == expected_remote_len, (
            "Heterogeneous PP selected region size mismatch at "
            f"pair={pair_idx}: local_region={local_region_id}, "
            f"remote_region={remote_region_id}, local={local_len}, "
            f"remote={remote_len}, expected_remote={expected_remote_len}, "
            f"tp_ratio={tp_ratio}, block_size_ratio={block_size_ratio}"
        )


def validate_pp_hma_region_lengths(
    *,
    local_block_lens: list[int],
    remote_block_lens: list[int],
    local_region_ids_by_group: list[np.ndarray],
    remote_region_ids_by_group: list[np.ndarray],
    selected_remote_region_ids: list[int],
    block_size_ratio: int,
    tp_ratio: int = 1,
    replicated_region_ids: set[int] | None = None,
    hybrid_mla_mamba: bool = False,
    group_spec_types: tuple[Any, ...] = (),
    local_transfer_regions: list[SemanticTransferRegion] | None = None,
    remote_transfer_regions: list[SemanticTransferRegion] | None = None,
    routes: list[HmaTransferRoute] | None = None,
    pairings: list[RegionPairing] | None = None,
) -> None:
    if block_size_ratio < 1:
        raise AssertionError(
            f"block_size_ratio must be positive, got {block_size_ratio}"
        )
    local_regions = regions_by_id(local_transfer_regions)
    remote_regions = regions_by_id(remote_transfer_regions)

    def check_pair(
        l_rid: int, r_rid: int, *, is_ssm: bool, identity: str, context: str
    ) -> None:
        token_ratio = validate_region_geometry_pair(
            local_regions.get(l_rid),
            remote_regions.get(r_rid),
            identity=identity,
            local_region_id=l_rid,
            remote_region_id=r_rid,
            is_ssm=is_ssm,
        )
        local_block_len = local_block_lens[l_rid]
        remote_block_len = remote_block_lens[r_rid]
        if replicated_region_ids is None:
            is_rep = not is_ssm
        else:
            is_rep = l_rid in replicated_region_ids
        replicated = hybrid_mla_mamba or (not is_ssm and is_rep)
        expected_remote_len = local_block_len // block_size_ratio
        if not replicated:
            expected_remote_len = (
                local_block_len * tp_ratio
                if tp_ratio >= 1
                else local_block_len // (-tp_ratio)
            )
        acceptable = {expected_remote_len, local_block_len}
        # Connector block sizes already match (kernel page). Scheduler blocks
        # can still be 4480 vs 640, and one side may register that whole
        # scheduler slot as a single page. Accept the token ratio and the
        # kernel-page ratio (70/7=10) so either registration shape passes.
        scales = {1, token_ratio}
        local_region = local_regions.get(l_rid)
        remote_region = remote_regions.get(r_rid)
        if (
            not is_ssm
            and local_region is not None
            and remote_region is not None
            and remote_region.kernel_blocks > 0
            and local_region.kernel_blocks > 0
        ):
            kernels = (
                local_region.kernel_blocks,
                remote_region.kernel_blocks,
            )
            big_k, small_k = max(kernels), min(kernels)
            if small_k > 0 and big_k % small_k == 0:
                scales.add(big_k // small_k)
                if token_ratio > 1 and (big_k // small_k) % token_ratio == 0:
                    scales.add((big_k // small_k) // token_ratio)
        for scale in scales:
            if scale <= 1:
                continue
            if local_block_len % scale == 0:
                acceptable.add(local_block_len // scale)
            acceptable.add(local_block_len * scale)
            if expected_remote_len % scale == 0:
                acceptable.add(expected_remote_len // scale)
            acceptable.add(expected_remote_len * scale)
        assert remote_block_len in acceptable, (
            "KV cache selected region size must match between P and D "
            f"when using HMA with heterogeneous PP at {context}: "
            f"local_region={l_rid} local={local_block_len}, "
            f"remote_region={r_rid} remote={remote_block_len}, "
            f"expected_remote={expected_remote_len}, "
            f"token_ratio={token_ratio}"
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
        return

    if pairings is not None:
        for pair_idx, pairing in enumerate(pairings):
            is_ssm = pairing.local_group_id < len(group_spec_types) and _is_ssm_spec(
                group_spec_types[pairing.local_group_id]
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
        is_ssm = group_idx < len(group_spec_types) and _is_ssm_spec(
            group_spec_types[group_idx]
        )
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
