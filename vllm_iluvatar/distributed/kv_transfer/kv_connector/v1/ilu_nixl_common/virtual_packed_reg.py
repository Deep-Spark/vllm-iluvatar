# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared virtual-packed KV registration and paired V4.1 scale plane builders."""

from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from typing import Any

import numpy as np
import torch
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from .dsv41 import resolve_quantized_kv_format
from .memory_registration import (
    MemoryRegistrationPlanner,
    validate_registered_allocation_alias,
)
from .semantic_pairing import cache_kind_from_layer_name, transfer_identity
from .types import SemanticTransferRegion


def exclusive_cumsum(counts: list[int]) -> np.ndarray:
    offsets = np.zeros(len(counts), dtype=np.int64)
    if counts:
        np.cumsum(counts[:-1], out=offsets[1:])
    return offsets


def layer_specs_for_config(kv_cache_config: Any) -> dict[str, Any]:
    specs: dict[str, Any] = {}
    for group in getattr(kv_cache_config, "kv_cache_groups", ()):
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            specs.update(group_spec.kv_cache_specs)
        else:
            specs.update(dict.fromkeys(group.layer_names, group_spec))
    return specs


def transfer_groups_of(kv_cache_config: Any) -> Any:
    return getattr(
        kv_cache_config,
        "transfer_groups",
        getattr(kv_cache_config, "kv_cache_groups", ()),
    )


def layer_to_groups(kv_cache_config: Any) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for group_idx, group in enumerate(transfer_groups_of(kv_cache_config)):
        for layer_name in group.layer_names:
            result[layer_name].append(group_idx)
    return dict(result)


def build_region_ids_by_group(
    kv_cache_config: Any,
    layer_region_ids: dict[str, list[int]],
    num_regions: int = 0,
) -> list[np.ndarray]:
    region_ids_by_group: list[np.ndarray] = []
    for group in transfer_groups_of(kv_cache_config):
        group_region_ids: list[int] = []
        for layer_name in group.layer_names:
            group_region_ids.extend(layer_region_ids.get(layer_name, []))
        region_ids_by_group.append(
            np.asarray(list(dict.fromkeys(group_region_ids)), dtype=np.int64)
        )
    if not region_ids_by_group and num_regions > 0:
        region_ids_by_group = [np.arange(num_regions, dtype=np.int64)]
    return region_ids_by_group


def is_virtual_packed_layout(
    kv_cache_config: Any, kv_caches: dict[str, Any]
) -> bool:
    tensors = getattr(kv_cache_config, "kv_cache_tensors", ()) or ()
    if not tensors or not kv_caches:
        return False
    if not any(getattr(tensor, "block_stride", 0) > 0 for tensor in tensors):
        return False
    storage_ptrs = {
        cache.untyped_storage().data_ptr() for cache in kv_caches.values()
    }
    return len(storage_ptrs) == 1


def virtual_slot_desc(
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
            "Packed virtual slot offset must be non-negative: "
            f"offset={slot_offset}"
        )
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


def metadata_uses_virtual_packed_regions(nixl_agent_meta: Any) -> bool:
    if getattr(nixl_agent_meta, "virtual_packed", False):
        return True
    transfer_regions = getattr(nixl_agent_meta, "transfer_regions", None)
    return bool(
        transfer_regions
        and any(
            region.block_stride > 0
            and (
                region.byte_offset != 0
                or region.block_stride != region.block_len
                or region.physical_backing_id != region.region_id
            )
            for region in transfer_regions
        )
    )


def build_packed_virtual_layout(
    kv_caches: dict[str, torch.Tensor],
    layout_config: Any,
    block_id_config: Any,
    *,
    is_dsv41: bool = False,
    region_cls: type[Any] = SemanticTransferRegion,
) -> tuple[
    dict[str, list[int]],
    list[np.ndarray],
    list[Any],
    list[bool],
]:
    specs = layer_specs_for_config(layout_config)
    layer_groups = layer_to_groups(block_id_config)

    first_cache = next(iter(kv_caches.values()))
    backing_ptr = first_cache.untyped_storage().data_ptr()
    backing_nbytes = first_cache.untyped_storage().nbytes()
    layer_region_ids: dict[str, list[int]] = defaultdict(list)
    transfer_regions: list[Any] = []
    region_is_mla: list[bool] = []

    winning_tensor_idx: dict[str, int] = {}
    for t_idx, tensor_config in enumerate(layout_config.kv_cache_tensors):
        members = getattr(tensor_config, "layers", None)
        if members is None:
            members = getattr(tensor_config, "shared_by", ())
        for layer_name in members:
            if layer_name in kv_caches:
                winning_tensor_idx[layer_name] = t_idx

    for t_idx, tensor_config in enumerate(layout_config.kv_cache_tensors):
        members = getattr(tensor_config, "layers", None)
        if members is None:
            members = getattr(tensor_config, "shared_by", ())
        shared_by = [
            layer_name
            for layer_name in members
            if layer_name in kv_caches
            and winning_tensor_idx.get(layer_name) == t_idx
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

        layer_positions = {
            layer_name: index for index, layer_name in enumerate(members)
        }
        layer_stride = int(getattr(tensor_config, "layer_stride", 0) or 0)
        tensor_offset = int(tensor_config.offset)
        num_blocks = int(layout_config.num_blocks)

        def layer_byte_offset(layer_name: str) -> int:
            return tensor_offset + layer_positions[layer_name] * layer_stride

        missing_specs = [
            layer_name for layer_name in shared_by if layer_name not in specs
        ]
        if missing_specs:
            raise ValueError(
                "Packed IluNixl layout has no KV cache spec for layers: "
                + ", ".join(sorted(missing_specs))
            )

        block_span = tensor_stride * num_blocks
        is_block_outer = layer_stride == 0 or layer_stride < tensor_stride
        if is_block_outer and (
            block_span <= 0 or backing_nbytes % block_span != 0
        ):
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
            cache = kv_caches[layer_name]
            layer_num_blocks = num_blocks if is_block_outer else int(cache.shape[0])
            page_size = int(specs[layer_name].page_size_bytes)
            last_end = (
                layer_byte_offset(layer_name)
                + (layer_num_blocks - 1) * tensor_stride
                + page_size
            )
            if last_end > backing_nbytes:
                raise ValueError(
                    "Packed IluNixl layer page exceeds registered backing: "
                    f"{layer_name} end={last_end} > backing={backing_nbytes}"
                )

        semantic_slots: dict[
            tuple[int, bool, str, int, int, int, int], list[str]
        ] = defaultdict(list)
        for layer_name in shared_by:
            cache = kv_caches[layer_name]
            spec = specs[layer_name]
            layer_num_blocks = num_blocks if is_block_outer else int(cache.shape[0])
            if cache.untyped_storage().data_ptr() != backing_ptr:
                raise ValueError(
                    f"Packed cache layer {layer_name} uses another backing"
                )
            slot_offset = layer_byte_offset(layer_name)
            actual_offset = cache.data_ptr() - backing_ptr
            if actual_offset != slot_offset:
                raise ValueError(
                    "Packed cache view disagrees with KV planning metadata "
                    f"for {layer_name}: offset={actual_offset}/{slot_offset}"
                )
            kernel_blocks = MemoryRegistrationPlanner.kernel_blocks_per_block(
                layer_name=layer_name,
                tensor=cache,
                spec=spec,
                num_blocks=layer_num_blocks,
            )
            if is_dsv41 or kernel_blocks > 1:
                payload_bytes = (
                    MemoryRegistrationPlanner.scheduler_block_payload_bytes(
                        layer_name=layer_name,
                        tensor=cache,
                        spec=spec,
                        kernel_blocks=kernel_blocks,
                        block_stride=tensor_stride,
                    )
                )
            else:
                payload_bytes = int(spec.page_size_bytes)
            MemoryRegistrationPlanner.validate_block_geometry(
                layer_name=layer_name,
                tensor=cache,
                block_stride=tensor_stride,
                num_blocks=layer_num_blocks,
                payload_segments=(
                    (slot_offset if is_block_outer else 0, payload_bytes),
                ),
                kernel_blocks=kernel_blocks,
            )
            is_mla_region = isinstance(
                spec, (MLAAttentionSpec, SlidingWindowMLASpec)
            )
            cache_kind = cache_kind_from_layer_name(layer_name)
            block_tokens = int(getattr(spec, "block_size", 0) or 0)
            semantic_slots[
                (
                    payload_bytes,
                    is_mla_region,
                    cache_kind,
                    slot_offset,
                    kernel_blocks,
                    block_tokens,
                    layer_num_blocks,
                )
            ].append(layer_name)

        for (
            page_size,
            is_mla_region,
            cache_kind,
            slot_offset,
            kernel_blocks,
            block_tokens,
            layer_num_blocks,
        ), region_layers in sorted(semantic_slots.items()):
            virtual_slot_desc(
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
                    for group_idx in layer_groups.get(layer_name, [])
                }
            )
            if not group_ids:
                raise ValueError(
                    "Packed virtual slot has no block-ID owner group: "
                    + ", ".join(region_layers)
                )
            identities: set[str] = set()
            identity_groups: dict[str, list[int]] = defaultdict(list)
            for layer_name in region_layers:
                ident = transfer_identity(layer_name)
                identities.add(ident)
                for group_idx in layer_groups.get(layer_name, []):
                    if group_idx not in identity_groups[ident]:
                        identity_groups[ident].append(group_idx)
            layer_indices: list[int] = []
            for layer_name in region_layers:
                with suppress(AssertionError, ValueError):
                    layer_indices.append(extract_layer_index(layer_name))
                layer_region_ids[layer_name].append(region_id)
            transfer_regions.append(
                region_cls(
                    region_id=region_id,
                    block_len=page_size,
                    shared_by=sorted(region_layers),
                    layer_indices=sorted(set(layer_indices)),
                    group_ids=group_ids,
                    cache_kinds=[cache_kind],
                    identities=sorted(identities),
                    physical_backing_id=0,
                    byte_offset=slot_offset,
                    block_stride=tensor_stride,
                    kernel_blocks=kernel_blocks,
                    block_tokens=block_tokens,
                    num_blocks=layer_num_blocks,
                    identity_groups={
                        k: sorted(v) for k, v in identity_groups.items()
                    },
                )
            )
            region_is_mla.append(is_mla_region)

    if not transfer_regions:
        raise ValueError("Packed IluNixl layout has no local semantic slots")

    region_ids_by_group = build_region_ids_by_group(
        block_id_config, dict(layer_region_ids), len(transfer_regions)
    )
    return (
        dict(layer_region_ids),
        region_ids_by_group,
        transfer_regions,
        region_is_mla,
    )


def append_paired_scale_regions(
    *,
    vllm_config: Any,
    device_id: int,
    kv_caches: dict[str, torch.Tensor],
    kv_cache_scales: dict[str, torch.Tensor] | None,
    layout_config: Any,
    block_id_config: Any,
    physical_data: list[tuple[int, int, int, str]],
    base_addresses: list[int],
    layer_region_ids: dict[str, list[int]],
    transfer_regions: list[Any],
    region_is_mla: list[bool],
    region_cls: type[Any] = SemanticTransferRegion,
) -> list[np.ndarray]:
    from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
        is_quantized_main_spec,
    )
    from vllm_iluvatar.models.deepseek_v41.indexer_kv import (
        is_split_indexer_spec,
    )

    specs = layer_specs_for_config(layout_config)
    paired_layers = [
        layer_name
        for layer_name in kv_caches
        if layer_name in specs
        and (
            is_quantized_main_spec(specs[layer_name])
            or is_split_indexer_spec(specs[layer_name])
        )
    ]
    if not paired_layers:
        return build_region_ids_by_group(
            block_id_config, layer_region_ids, len(transfer_regions)
        )

    planner = MemoryRegistrationPlanner(layout_config)
    layer_groups = layer_to_groups(block_id_config)
    num_blocks = int(layout_config.num_blocks)
    compilation_cfg = getattr(vllm_config, "compilation_config", None)
    forward_context = (
        getattr(compilation_cfg, "static_forward_context", {})
        if compilation_cfg is not None
        else {}
    )

    seen_addr_to_idx = {addr: idx for idx, addr in enumerate(base_addresses)}
    seen_addr_sizes = {entry[0]: entry[1] for entry in physical_data}
    scale_slots: dict[
        tuple[int, int, int, int, str, int, int], list[str]
    ] = defaultdict(list)

    for layer_name in paired_layers:
        spec = specs[layer_name]
        scales = (kv_cache_scales or {}).get(layer_name)
        if scales is None and layer_name in forward_context:
            scales = getattr(forward_context[layer_name], "kv_cache_scales", None)
        if not isinstance(scales, torch.Tensor):
            raise ValueError(
                f"{layer_name}: compressed KV registration requires its scale plane"
            )
        payload_reg = planner.describe(
            layer_name=layer_name,
            tensor=kv_caches[layer_name],
            spec=spec,
            num_blocks=num_blocks,
        )
        if is_split_indexer_spec(spec):
            scale_reg = planner.describe_indexer_scale_plane(
                layer_name=layer_name,
                scales=scales,
                payload=payload_reg,
            )
        else:
            fmt = resolve_quantized_kv_format(spec.cache_dtype_str)
            scale_reg = planner.describe_scale_plane(
                layer_name=layer_name,
                scales=scales,
                payload=payload_reg,
                scale_page_bytes=fmt.scale_page_bytes(spec.head_size),
            )

        scale_base = scale_reg.registration_base
        scale_nbytes = scale_reg.registration_nbytes
        if scale_base in seen_addr_to_idx:
            backing_id = seen_addr_to_idx[scale_base]
            validate_registered_allocation_alias(
                base_address=scale_base,
                registered_nbytes=seen_addr_sizes[scale_base],
                expected_nbytes=scale_nbytes,
                layer_name=f"{layer_name}.kv_cache_scales",
            )
        else:
            backing_id = len(base_addresses)
            seen_addr_to_idx[scale_base] = backing_id
            seen_addr_sizes[scale_base] = scale_nbytes
            base_addresses.append(scale_base)
            physical_data.append(
                (scale_base, scale_nbytes, scale_reg.device_id, "")
            )

        ((scale_offset, scale_payload_len),) = scale_reg.payload_segments
        scale_name = f"{layer_name}.kv_cache_scales"
        scale_kind = cache_kind_from_layer_name(scale_name)
        block_tokens = int(getattr(spec, "block_size", 0) or 0)
        scale_slots[
            (
                backing_id,
                scale_offset,
                scale_reg.block_stride,
                scale_payload_len,
                scale_kind,
                scale_reg.kernel_blocks,
                block_tokens,
            )
        ].append(layer_name)

    for (
        backing_id,
        scale_offset,
        scale_stride,
        scale_payload_len,
        scale_kind,
        kernel_blocks,
        block_tokens,
    ), region_layers in sorted(scale_slots.items()):
        virtual_slot_desc(
            base_addresses[backing_id],
            0,
            scale_stride,
            scale_offset,
            scale_payload_len,
            device_id,
        )
        region_id = len(transfer_regions)
        group_ids = sorted(
            {
                group_idx
                for layer_name in region_layers
                for group_idx in layer_groups.get(layer_name, [])
            }
        )
        identities: set[str] = set()
        identity_groups: dict[str, list[int]] = defaultdict(list)
        layer_indices: list[int] = []
        scale_shared_by: list[str] = []
        for layer_name in sorted(region_layers):
            scale_name = f"{layer_name}.kv_cache_scales"
            scale_shared_by.append(scale_name)
            ident = transfer_identity(scale_name)
            identities.add(ident)
            for group_idx in layer_groups.get(layer_name, []):
                if group_idx not in identity_groups[ident]:
                    identity_groups[ident].append(group_idx)
            with suppress(AssertionError, ValueError):
                layer_indices.append(extract_layer_index(layer_name))
            layer_region_ids.setdefault(layer_name, []).append(region_id)

        transfer_regions.append(
            region_cls(
                region_id=region_id,
                block_len=scale_payload_len,
                shared_by=scale_shared_by,
                layer_indices=sorted(set(layer_indices)),
                group_ids=group_ids,
                cache_kinds=[scale_kind],
                identities=sorted(identities),
                physical_backing_id=backing_id,
                byte_offset=scale_offset,
                block_stride=scale_stride,
                kernel_blocks=kernel_blocks,
                block_tokens=block_tokens,
                num_blocks=num_blocks,
                identity_groups={
                    k: sorted(v) for k, v in identity_groups.items()
                },
            )
        )
        region_is_mla.append(True)

    return build_region_ids_by_group(
        block_id_config, layer_region_ids, len(transfer_regions)
    )
