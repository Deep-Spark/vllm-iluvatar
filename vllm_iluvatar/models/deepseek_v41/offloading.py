"""Canonical CPU offload pages for paired DeepSeek V4.1 KV storage.

Planning and runtime registration share the same region description. Only
physical layer pages are copied: holes in the mirrored main-scale allocation
are not flattened into unrelated manager blocks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from vllm_iluvatar.ilu_patch.dsv41_utils import is_ilu_dsv41_config

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
    from ixformer.inference.functions.models.deepseek_v41.kv.common.quantized_kv_format import (
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


# Rank-local PP geometry shared by planner, config and spec adapters.
_GEOMETRY_ATTR = "_ilu_offload_worker_bytes_by_rank"
_RANK_SAFE_REQUIRED_ATTR = "_ilu_rank_safe_offload_required"
_REVIEWED_PARTITIONS = {
    (4, 2, 8): ((5, 5, 5, 25),),
    # The baseline 2,3,3,3,3,2,3,21 cut and both existing PP8 cuts are reviewed. 3,3,3,3,2,3,2,21 is the 256K profile and is
    # 0.19 GiB cheaper per 1M request (PP5 starts on KV source 14, so it needs no
    # 7th forwarder replica). 3,3,3,3,3,2,2,21 is what
    # prefill_state/configuration.py:layer_partitions derives by default for
    # depth=21 over PP8, and is the 1M profile; accepting only the former made
    # a legitimate cut fail with "expected the reviewed ... partition".
    (8, 1, 8): (
        (3, 3, 3, 3, 2, 3, 2, 21),
        (3, 3, 3, 3, 3, 2, 2, 21),
        # 4.4.4 KV producer stops at layer 20: PP7 runs only L19 and L20.
        (2, 3, 3, 3, 3, 2, 3, 21),
    ),
    (8, 2, 16): (
        (3, 3, 3, 3, 2, 3, 2, 21),
        (3, 3, 3, 3, 3, 2, 2, 21),
        (2, 3, 3, 3, 3, 2, 3, 21),
    ),
}


def _has_native_offloading(vllm_config) -> bool:
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    if transfer is None:
        return False
    if transfer.kv_connector == "OffloadingConnector":
        return True
    if transfer.kv_connector != "MultiConnector":
        return False
    extra_config = getattr(transfer, "kv_connector_extra_config", None) or {}
    return any(
        child.get("kv_connector") == "OffloadingConnector"
        for child in extra_config.get("connectors", ())
    )


def _needs_rank_safe_geometry(vllm_config) -> bool:
    return (
        is_ilu_dsv41_config(vllm_config)
        and vllm_config.parallel_config.pipeline_parallel_size > 1
        and _has_native_offloading(vllm_config)
    )


def _worker_bytes_per_block(config) -> int:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        get_offloading_group_ids,
    )

    # A PP stage containing only bounded-replay SWA still allocates GPU pages,
    # but none of those pages belongs to the CPU prefix cache. Do not charge
    # that backing allocation to the offload layout.
    groups = getattr(config, "kv_cache_groups", None)
    if groups is not None:
        group_ids = (
            get_offloading_group_ids(config)
            if hasattr(config, "hisparse_host_num_blocks")
            else range(len(groups))
        )
        if not any(groups[group_id].layer_names for group_id in group_ids):
            return 0
    if has_paired_pages(config):
        return plan_offload_layout(config).bytes_per_block
    if config.num_blocks <= 0 or not config.kv_cache_tensors:
        raise ValueError("PP CPU offload requires an allocated GPU KV cache.")
    total_sizes = {tensor.size for tensor in config.kv_cache_tensors}
    if len(total_sizes) != 1:
        raise ValueError(
            "PP CPU offload requires every KV tensor descriptor to reference "
            f"one backing allocation size, got {sorted(total_sizes)}."
        )
    (total_size,) = total_sizes
    if total_size % config.num_blocks:
        raise ValueError(
            f"KV backing size {total_size} is not divisible by "
            f"num_blocks={config.num_blocks}."
        )
    return total_size // config.num_blocks


def _validate_target_geometry(vllm_config, geometry: tuple[int, ...]) -> None:
    parallel = vllm_config.parallel_config
    world_size = parallel.world_size
    pp_size = parallel.pipeline_parallel_size
    tp_size = parallel.tensor_parallel_size
    if len(geometry) != world_size:
        raise ValueError(
            "PP CPU offload expected one KV geometry per worker: "
            f"got {len(geometry)} for world_size={world_size}."
        )
    if world_size != pp_size * tp_size:
        raise ValueError(
            "DSV4.1 PP CPU offload requires world_size == pp_size * tp_size: "
            f"{world_size} != {pp_size} * {tp_size}."
        )
    if any(value < 0 for value in geometry) or not any(geometry):
        raise ValueError(
            "DSV4.1 PP CPU offload geometry must contain nonnegative byte "
            "sizes and at least one positive size: "
            f"{geometry}."
        )

    for pp_rank in range(pp_size):
        start = pp_rank * tp_size
        stage_values = geometry[start : start + tp_size]
        if len(set(stage_values)) != 1:
            raise ValueError(
                "DSV4.1 PP CPU offload TP ranks within one PP stage must have "
                f"identical geometry: pp_rank={pp_rank}, values={stage_values}."
            )

    expected_partitions = _REVIEWED_PARTITIONS.get((pp_size, tp_size, world_size))
    if expected_partitions is None:
        return
    partition_text = os.environ.get("VLLM_PP_LAYER_PARTITION", "")
    try:
        partition = tuple(int(value) for value in partition_text.split(","))
    except ValueError as exc:
        raise ValueError(
            "DSV4.1 PP CPU offload requires an integer VLLM_PP_LAYER_PARTITION."
        ) from exc
    if partition not in expected_partitions:
        expected_text = " or ".join(
            ",".join(str(value) for value in candidate)
            for candidate in expected_partitions
        )
        raise ValueError(
            f"DSV4.1 PP CPU offload expected the reviewed {expected_text} "
            f"partition, got {partition_text!r}."
        )
