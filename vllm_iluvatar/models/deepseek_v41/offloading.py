"""Canonical CPU offload pages for paired DeepSeek V4.1 KV storage.

Planning and runtime registration share the same region description. Only
physical layer pages are copied: holes in the mirrored main-scale allocation
are not flattened into unrelated manager blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .compressed_kv import is_quantized_main_spec, layer_specs, plan_scale_layout
from .indexer_kv import is_split_indexer_spec, plan_indexer_scales


@dataclass(frozen=True)
class OffloadRegion:
    plane: str
    layer: str
    offset_bytes: int
    block_stride_bytes: int
    page_bytes: int


@dataclass(frozen=True)
class OffloadLayout:
    regions: tuple[OffloadRegion, ...]
    group_regions: tuple[tuple[int, ...], ...]

    @property
    def bytes_per_block(self) -> int:
        return sum(region.page_bytes for region in self.regions)


def has_paired_pages(config) -> bool:
    groups = getattr(config, "kv_cache_groups", ())
    return any(
        is_quantized_main_spec(spec) or is_split_indexer_spec(spec)
        for spec in layer_specs(groups).values()
    )


def plan_offload_layout(config) -> OffloadLayout:
    """Describe every selected group's payload and scales using manager IDs."""
    from ixformer.inference.functions.deepseek.quantized_kv_format import (
        resolve_quant_format,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        get_offloading_group_ids,
    )

    specs = layer_specs(config.kv_cache_groups)
    main_formats = {
        spec.cache_dtype_str for spec in specs.values() if is_quantized_main_spec(spec)
    }
    if len(main_formats) > 1:
        raise ValueError("A worker cannot mix main KV quantization formats")
    main = (
        plan_scale_layout(config, resolve_quant_format(next(iter(main_formats))))
        if main_formats
        else None
    )
    index = plan_indexer_scales(config)
    payloads = {}
    for tensor in config.kv_cache_tensors:
        for position, name in enumerate(tensor.layers):
            placement = (
                tensor.offset + position * tensor.layer_stride,
                tensor.block_stride,
                specs[name].page_size_bytes,
            )
            if name in payloads and payloads[name] != placement:
                raise ValueError(f"{name}: inconsistent payload offload placement")
            payloads[name] = placement

    regions, aliases, group_regions = [], {}, []

    def add(plane, name, offset, stride, size):
        if offset < 0 or size <= 0 or stride < size:
            raise ValueError(f"{name}: invalid {plane} offload geometry")
        key = (plane, offset, stride, size)
        if key not in aliases:
            aliases[key] = len(regions)
            regions.append(OffloadRegion(plane, name, offset, stride, size))
        return aliases[key]

    for group_id in get_offloading_group_ids(config):
        references = []
        for name in config.kv_cache_groups[group_id].layer_names:
            references.append(add("payload", name, *payloads[name]))
            if main is not None and name in main.placements:
                p = main.placements[name]
                references.append(
                    add(
                        "main_scale",
                        name,
                        p.offset_elements * 2,
                        p.block_stride_elements * p.blocks_per_page * 2,
                        p.rows * p.groups * p.blocks_per_page * 2,
                    )
                )
            if name in index.placements:
                p = index.placements[name]
                references.append(
                    add(
                        "index_scale",
                        name,
                        p.offset * 4,
                        p.rows_per_manager_block * 4,
                        p.rows_per_manager_block * 4,
                    )
                )
        group_regions.append(tuple(dict.fromkeys(references)))
    return OffloadLayout(tuple(regions), tuple(group_regions))


def canonical_paired_pages(config, kv_caches, forward_context):
    """Bind the planned regions to real storage, validating all addresses."""
    from vllm.v1.kv_cache_interface import group_kernel_blocks
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
    )

    layout = plan_offload_layout(config)
    tensors = []
    storage_by_plane = {}
    for region in layout.regions:
        source = (
            kv_caches[region.layer]
            if region.plane == "payload"
            else getattr(forward_context[region.layer], "kv_cache_scales", None)
        )
        if not isinstance(source, torch.Tensor):
            raise ValueError(f"{region.layer}: missing {region.plane} offload owner")
        physical = group_kernel_blocks(source, config.num_blocks)
        element = source.element_size()
        if (
            physical.storage_offset() * element != region.offset_bytes
            or physical.stride(0) * element != region.block_stride_bytes
        ):
            raise ValueError(
                f"{region.layer}: {region.plane} differs from planned pages"
            )
        # Kernel pages within one manager page must be dense. Stride(0) may
        # have a gap between manager pages, which the copy engine preserves.
        if (
            not physical[0].is_contiguous()
            or physical[0].numel() * element != region.page_bytes
        ):
            raise ValueError(f"{region.layer}: {region.plane} is not one dense page")
        storage = source.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes(), source.device)
        previous = storage_by_plane.setdefault(region.plane, key)
        if previous != key:
            raise ValueError(f"{region.plane}: layers do not share the planned pool")
        view = torch.empty(0, dtype=torch.int8, device=source.device).set_(
            storage,
            region.offset_bytes,
            (config.num_blocks, region.page_bytes),
            (region.block_stride_bytes, 1),
        )
        tensors.append(CanonicalKVCacheTensor(view, region.page_bytes))
    return CanonicalKVCaches(
        tensors,
        [
            [
                CanonicalKVCacheRef(index, layout.regions[index].page_bytes)
                for index in group
            ]
            for group in layout.group_regions
        ],
    )
