"""Paired INT8 index-key payload and FP32 scale allocation.

The scheduler owns the payload block IDs. Scales follow those IDs without a
second block table or a second cache policy. Only actual index-key owners and
PP forwarders receive storage; aliases of identical physical pages share it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

SPLIT_INDEXER_FORMAT = "int8_indexer_split"


def indexer_kv_layout() -> str:
    layout = os.getenv("VLLM_ILU_DS41_INDEXER_KV_LAYOUT", "legacy")
    if layout not in ("legacy", "split"):
        raise ValueError("Indexer KV layout must be 'legacy' or 'split'")
    if layout == "split":
        from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import indexer_kv_dtype

        if indexer_kv_dtype() != "int8":
            raise ValueError("Split indexer KV requires INT8 index keys")
    return layout


def is_split_indexer_spec(spec) -> bool:
    return getattr(spec, "cache_dtype_str", None) == SPLIT_INDEXER_FORMAT


def indexer_format_factors() -> dict:
    if indexer_kv_layout() != "split":
        return {}
    return {"indexer_kv_layout": "int8_values128+fp32_scale_v1"}


@dataclass(frozen=True)
class IndexerScalePlacement:
    offset: int
    rows_per_manager_block: int


@dataclass(frozen=True)
class IndexerScaleLayout:
    num_blocks: int
    nbytes: int
    placements: dict[str, IndexerScalePlacement]


def plan_indexer_scales(config) -> IndexerScaleLayout:
    """Plan compact sideplanes from the allocator's actual physical aliases."""
    from vllm_iluvatar.models.deepseek_v41.compressed_kv import layer_specs

    specs = layer_specs(config.kv_cache_groups)
    aliases = {}
    placements = {}
    elements = 0
    physical_regions = []
    for tensor in config.kv_cache_tensors:
        for index, name in enumerate(tensor.layers):
            spec = specs[name]
            if not is_split_indexer_spec(spec):
                continue
            if (
                spec.dtype is not torch.uint8
                or spec.head_size != 128
                or spec.state_content_size_bytes != 128
                or spec.page_size_bytes != spec.num_states * 128
            ):
                raise ValueError(f"{name}: invalid split indexer payload geometry")
            offset = tensor.offset + index * tensor.layer_stride
            key = (offset, tensor.block_stride, spec.page_size_bytes)
            if key not in aliases:
                if config.num_blocks <= 0 or tensor.block_stride < spec.page_size_bytes:
                    raise ValueError(f"{name}: overlapping indexer manager pages")
                last = offset + (config.num_blocks - 1) * tensor.block_stride
                if offset < 0 or last + spec.page_size_bytes > tensor.size:
                    raise ValueError(f"{name}: indexer payload exceeds its allocation")
                placement = IndexerScalePlacement(elements, spec.num_states)
                aliases[key] = placement
                elements += config.num_blocks * spec.num_states
                physical_regions.extend(
                    (
                        offset + b * tensor.block_stride,
                        offset + b * tensor.block_stride + spec.page_size_bytes,
                    )
                    for b in range(config.num_blocks)
                )
            placements[name] = aliases[key]
    # Distinct sideplanes must not describe partially aliased payloads. The
    # common-page allocator provides either disjoint or identical page views.
    regions = sorted(physical_regions)
    if any(left[1] > right[0] for left, right in zip(regions, regions[1:])):
        raise ValueError("Split indexer payloads have unsupported partial aliases")
    return IndexerScaleLayout(config.num_blocks, elements * 4, placements)


class IndexerScaleState:
    """Worker-owned planes with addresses stable across graph capture/replay."""

    def __init__(self, layout, device, context):
        self.layout = layout
        self.context = dict(context)
        self.scales = {}
        for name in layout.placements:
            if getattr(context[name], "kv_cache_scales", None) is not None:
                raise ValueError(f"{name}: indexer scales are already bound")
        self.storage = torch.zeros(
            layout.nbytes // 4, dtype=torch.float32, device=device
        )
        for name, placement in layout.placements.items():
            size = layout.num_blocks * placement.rows_per_manager_block
            view = self.storage[placement.offset : placement.offset + size].view(
                layout.num_blocks, placement.rows_per_manager_block
            )
            layer = context[name]
            self.scales[name] = view
            layer.kv_cache_scales = view
            layer._indexer_scale_owner = self

    def bind(self, name, cache):
        scale = self.scales[name]
        if scale.numel() != cache.shape[0] * cache.shape[1]:
            raise ValueError(f"{name}: indexer payload/scale page ranges differ")
        view = scale.view(cache.shape[0], cache.shape[1])
        self.scales[name] = view
        self.context[name].kv_cache_scales = view

    def release(self):
        for name in self.scales:
            layer = self.context[name]
            if getattr(layer, "_indexer_scale_owner", None) is self:
                layer.kv_cache_scales = None
                layer._indexer_scale_owner = None
        self.scales.clear()
        self.storage = None
        self.context.clear()


def extend_kv_zeroer(zeroer, num_blocks, planes, workspace=None):
    """Add paired cache planes to the existing physical-block zero operation."""
    from vllm.v1.kv_cache_interface import group_kernel_blocks

    addresses, strides, sizes = [], [], []
    if zeroer._meta is not None:
        old_addresses, old_strides, old_sizes, *_ = zeroer._meta
        addresses.extend(old_addresses.cpu().tolist())
        strides.extend(old_strides.cpu().tolist())
        sizes.extend(old_sizes.cpu().tolist())
    seen = set(addresses)
    for plane in planes:
        if plane.data_ptr() in seen:
            continue
        seen.add(plane.data_ptr())
        physical = group_kernel_blocks(plane, num_blocks)
        stride = physical.stride(0) * plane.element_size()
        size = physical[0].numel() * plane.element_size()
        if stride % 4 or size % 4:
            raise ValueError("Sideplane zeroing requires whole int32 words")
        addresses.append(plane.data_ptr())
        strides.append(stride // 4)
        sizes.append(size // 4)
    if not addresses:
        return
    count = len(addresses)
    metadata = (
        workspace.reserve_metadata(count * 24)
        if workspace is not None
        else torch.empty(count * 24, dtype=torch.uint8, device=zeroer.device)
    )
    addr = metadata[: count * 8].view(torch.uint64)
    stride = metadata[count * 8 : count * 16].view(torch.int64)
    size = metadata[count * 16 :].view(torch.int64)
    for tensor, values in ((addr, addresses), (stride, strides), (size, sizes)):
        tensor.copy_(torch.tensor(values, dtype=tensor.dtype, device="cpu"))
    block_size = min(1 << (max(sizes) - 1).bit_length(), 1024)
    zeroer._meta = (
        addr,
        stride,
        size,
        (max(sizes) + block_size - 1) // block_size,
        block_size,
        count,
    )


def reserve_indexer_scale_memory(original, vllm_config, specs, available):
    """Reserve actual per-rank sideplanes before the block count is finalized.

    The first pass only plans descriptors; it allocates no GPU memory. Its
    per-block costs include global PP grouping and physical payload aliases.
    Replanning with the paired cost avoids guessing a global scale ratio from
    the largest cache group or allocating scales for unrelated layer pages.
    """
    if not any(
        is_split_indexer_spec(spec) for worker in specs for spec in worker.values()
    ):
        return original(vllm_config, specs, available)
    planned = original(vllm_config, specs, available)
    reserved = []
    for config, budget in zip(planned, available, strict=True):
        layout = plan_indexer_scales(config)
        if not layout.nbytes:
            reserved.append(budget)
            continue
        sizes = {tensor.size for tensor in config.kv_cache_tensors}
        if len(sizes) != 1:
            raise ValueError("Split indexer requires one common payload allocation")
        payload = sizes.pop()
        if payload % config.num_blocks or layout.nbytes % config.num_blocks:
            raise ValueError(
                "Split indexer allocation must have integral per-block costs"
            )
        payload_per_block = payload // config.num_blocks
        scales_per_block = layout.nbytes // config.num_blocks
        blocks = budget // (payload_per_block + scales_per_block)
        reserved.append(blocks * payload_per_block)
    return original(vllm_config, specs, reserved)
