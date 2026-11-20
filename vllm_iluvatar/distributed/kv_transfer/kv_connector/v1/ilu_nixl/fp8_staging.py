# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-side compute KV -> packed FP8 wire staging (DSv4 / DSv4.1)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    quantize_and_insert_k_cache,
)
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    group_kernel_blocks,
)

from ..ilu_nixl_common.fp8_hetero import scatter_staging_rows_to_pool
from .fp8_layout import (
    FP8_WIRE_SCRATCH_BUFFER_COUNT,
    build_fp8_ds_mla_wire_config,
    get_fp8_wire_scratch_blocks,
    page_states,
    wire_sub_blocks_per_compute_block,
)

logger = init_logger("vllm." + __name__)

ByteSpan = tuple[int, int]
_COMPRESSED_MAIN_FORMATS = frozenset({"i8g64", "i4g32"})
_SPLIT_INDEXER_FORMAT = "int8_indexer_split"


def _layer_specs(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
    specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            specs.update(group_spec.kv_cache_specs)
        else:
            specs.update(dict.fromkeys(group.layer_names, group_spec))
    return specs


def _is_fp8_ds_mla_wire_page(spec: KVCacheSpec) -> bool:
    return isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)) and (
        spec.cache_dtype_str == "fp8_ds_mla"
    )


def _is_indexer_cache(layer_name: str) -> bool:
    return ".indexer." in layer_name and layer_name.endswith(".k_cache")


def _dequant_grouped_main_rows(
    payload: torch.Tensor,
    scales: torch.Tensor,
    *,
    format_name: str,
    head_dim: int = 512,
) -> torch.Tensor:
    """Dequant ``i8g64`` / ``i4g32`` rows to BF16 for FP8 wire requantization."""
    if format_name == "i8g64":
        group_size, groups, payload_width = 64, 8, 512
    elif format_name == "i4g32":
        group_size, groups, payload_width = 32, 16, 256
    else:
        raise ValueError(f"Unsupported compressed main KV format: {format_name}")

    flat = payload.reshape(-1, payload_width)
    if flat.numel() == 0:
        return payload.new_empty((0, head_dim), dtype=torch.bfloat16)
    if format_name == "i8g64":
        codes = flat.view(torch.int8).to(torch.float32)
    else:
        low = ((flat & 0xF).to(torch.int8) ^ 8) - 8
        high = (((flat >> 4) & 0xF).to(torch.int8) ^ 8) - 8
        codes = torch.empty(
            flat.shape[0], head_dim, dtype=torch.float32, device=payload.device
        )
        codes[:, 0::2] = low.to(torch.float32)
        codes[:, 1::2] = high.to(torch.float32)

    scale = scales.reshape(-1, groups).to(torch.float32)
    if codes.shape[0] != scale.shape[0]:
        raise ValueError(
            f"Compressed KV payload/scale row mismatch: {codes.shape[0]} vs "
            f"{scale.shape[0]}"
        )
    group_ids = torch.arange(head_dim, device=payload.device) // group_size
    return (codes * scale[:, group_ids]).to(torch.bfloat16)


def merge_byte_intervals(intervals: Sequence[ByteSpan]) -> tuple[ByteSpan, ...]:
    """Sort and merge overlapping or adjacent half-open byte intervals."""
    if not intervals:
        return ()
    ordered = sorted(intervals)
    if any(start < 0 or end <= start for start, end in ordered):
        raise ValueError(f"Invalid FP8 wire byte interval(s): {ordered!r}")

    merged: list[ByteSpan] = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def build_hma_copy_spans(
    compute_config: KVCacheConfig,
    wire_config: KVCacheConfig,
) -> tuple[tuple[ByteSpan, ...], ...]:
    """Build the packed wire byte spans owned by every compute/HMA group."""
    if not wire_config.kv_cache_tensors:
        raise ValueError("FP8 DS MLA wire staging has no cache tensors")
    block_stride = wire_config.kv_cache_tensors[0].block_stride
    if block_stride <= 0:
        raise ValueError("FP8 DS MLA wire staging must use a packed layout")

    wire_specs = _layer_specs(wire_config)
    layer_to_group: dict[str, int] = {}
    for group_idx, group in enumerate(compute_config.kv_cache_groups):
        for layer_name in group.layer_names:
            if layer_name in layer_to_group:
                raise ValueError(
                    f"Compute cache layer {layer_name} belongs to multiple groups"
                )
            layer_to_group[layer_name] = group_idx

    intervals: list[list[ByteSpan]] = [[] for _ in compute_config.kv_cache_groups]
    covered_layers: set[str] = set()
    for tensor in wire_config.kv_cache_tensors:
        if tensor.block_stride != block_stride:
            raise ValueError("FP8 wire tensors have inconsistent block strides")
        for layer_name in tensor.layers:
            group_idx = layer_to_group.get(layer_name)
            if group_idx is None:
                continue
            spec = wire_specs.get(layer_name)
            if spec is None:
                raise ValueError(f"Wire cache has no spec for layer {layer_name}")
            start = tensor.offset
            end = start + spec.page_size_bytes
            if start < 0 or end <= start or end > block_stride:
                raise ValueError(
                    "FP8 wire span is outside the packed block: "
                    f"layer={layer_name}, span=({start}, {end}), "
                    f"block_stride={block_stride}"
                )
            intervals[group_idx].append((start, end))
            covered_layers.add(layer_name)

    missing_layers = set(layer_to_group) - covered_layers
    if missing_layers:
        raise ValueError(
            "FP8 wire tensors do not cover compute cache layers: "
            + ", ".join(sorted(missing_layers))
        )

    spans = tuple(
        merge_byte_intervals(group_intervals) for group_intervals in intervals
    )
    for group_idx, (group, group_spans) in enumerate(
        zip(compute_config.kv_cache_groups, spans)
    ):
        if group.layer_names and not group_spans:
            raise ValueError(
                f"Non-empty compute cache group {group_idx} has no FP8 wire spans"
            )
    return spans


def select_scratch_buffer(chunk_index: int) -> int:
    if chunk_index < 0:
        raise ValueError(f"Invalid scratch selection: chunk={chunk_index}")
    return chunk_index % FP8_WIRE_SCRATCH_BUFFER_COUNT


def _coalesce_contiguous_runs(
    source_block_ids: Sequence[int],
) -> list[tuple[int, int, int]]:
    """Group consecutive ``source_block_ids`` into ``(dense_offset, dst_start, length)``."""
    if not source_block_ids:
        return []
    runs: list[tuple[int, int, int]] = []
    dense_start = 0
    dst_start = int(source_block_ids[0])
    run_len = 1
    prev = dst_start
    for idx in range(1, len(source_block_ids)):
        cur = int(source_block_ids[idx])
        if cur == prev + 1:
            run_len += 1
        else:
            runs.append((dense_start, dst_start, run_len))
            dense_start = idx
            dst_start = cur
            run_len = 1
        prev = cur
    runs.append((dense_start, dst_start, run_len))
    return runs


def copy_wire_rows_to_host(
    scratch_rows: torch.Tensor,
    wire_rows: torch.Tensor,
    source_block_ids: Sequence[int],
    spans: Sequence[ByteSpan],
    *,
    non_blocking: bool = False,
    h2h_threads: int = 4,
) -> None:
    """Copy selected HMA spans from dense scratch rows into host mirror rows."""
    n_blocks = len(source_block_ids)
    if n_blocks > scratch_rows.shape[0]:
        raise ValueError("FP8 wire scratch does not contain all requested rows")
    effective_spans = tuple(spans)
    if not effective_spans:
        raise ValueError("Exported FP8 wire blocks have no effective HMA byte spans")
    for start, end in effective_spans:
        if start < 0 or end <= start or end > scratch_rows.shape[1]:
            raise ValueError(
                f"FP8 wire copy span ({start}, {end}) is outside the scratch row"
            )
        if end > wire_rows.shape[1]:
            raise ValueError(
                f"FP8 wire copy span ({start}, {end}) is outside the host row"
            )
    if n_blocks == 0:
        return

    # Host-to-host path: use OpenMP C++ h2h_scatter (or vectorized span index_copy_).
    if scratch_rows.device.type == "cpu" and wire_rows.device.type == "cpu":
        scatter_staging_rows_to_pool(
            scratch_rows,
            wire_rows,
            source_block_ids,
            effective_spans,
            num_threads=h2h_threads,
        )
        return

    # Device-to-pinned-host path: coalesce consecutive block IDs into bulk 2D DMA copies
    # on the active CUDA copy_stream without breaking stream ordering!
    runs = _coalesce_contiguous_runs(source_block_ids)
    for dense_off, dst_off, run_len in runs:
        for start, end in effective_spans:
            wire_rows[dst_off : dst_off + run_len, start:end].copy_(
                scratch_rows[dense_off : dense_off + run_len, start:end],
                non_blocking=non_blocking,
            )


@triton.jit
def _quantize_indexer_wire_kernel(
    source_ptr,
    destination_ptr,
    slot_mapping_ptr,
    source_stride,
    destination_block_stride,
    cache_block_size,
    HEAD_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, HEAD_SIZE)
    values = tl.load(source_ptr + row_idx * source_stride + offsets).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    exponent = tl.ceil(tl.log2(tl.maximum(amax, 1e-4) / 448.0))
    scale = tl.exp2(exponent)
    quantized = tl.clamp(values / scale, -448.0, 448.0)
    fp8_bytes = quantized.to(tl.float8e4nv).to(tl.uint8, bitcast=True)

    slot = tl.load(slot_mapping_ptr + row_idx)
    block_id = (slot // cache_block_size).to(tl.int64)
    block_offset = slot % cache_block_size
    block_base = block_id * destination_block_stride
    data_offset = block_base + block_offset * HEAD_SIZE
    tl.store(destination_ptr + data_offset + offsets, fp8_bytes)

    scale_offset = block_base + cache_block_size * HEAD_SIZE + block_offset * 4
    scale_ptr = (destination_ptr + scale_offset).to(tl.pointer_type(tl.float32))
    tl.store(scale_ptr, scale)


@triton.jit
def _transcode_int8_indexer_wire_kernel(
    source_ptr,
    destination_ptr,
    source_block_ids_ptr,
    destination_block_ids_ptr,
    valid_entries_ptr,
    source_block_stride,
    destination_block_stride,
    cache_block_size,
    HEAD_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    page_idx = row_idx // cache_block_size
    block_offset = row_idx % cache_block_size
    valid = block_offset < tl.load(valid_entries_ptr + page_idx)
    offsets = tl.arange(0, HEAD_SIZE)

    source_block_id = tl.load(source_block_ids_ptr + page_idx).to(tl.int64)
    source_block_base = source_block_id * source_block_stride
    source_data_offset = source_block_base + block_offset * HEAD_SIZE
    int8_bytes = tl.load(
        source_ptr + source_data_offset + offsets,
        mask=valid,
        other=0,
    )
    int8_values = int8_bytes.to(tl.int8, bitcast=True).to(tl.float32)
    source_scale_offset = (
        source_block_base + cache_block_size * HEAD_SIZE + block_offset * 4
    )
    source_scale_ptr = (source_ptr + source_scale_offset).to(
        tl.pointer_type(tl.float32)
    )
    source_scale = tl.load(source_scale_ptr, mask=valid, other=0.0)
    values = int8_values * source_scale

    amax = tl.max(tl.abs(values), axis=0)
    exponent = tl.ceil(tl.log2(tl.maximum(amax, 1e-4) / 448.0))
    wire_scale = tl.exp2(exponent)
    quantized = tl.clamp(values / wire_scale, -448.0, 448.0)
    fp8_bytes = quantized.to(tl.float8e4nv).to(tl.uint8, bitcast=True)

    destination_block_id = tl.load(destination_block_ids_ptr + page_idx).to(tl.int64)
    destination_block_base = destination_block_id * destination_block_stride
    destination_data_offset = destination_block_base + block_offset * HEAD_SIZE
    tl.store(
        destination_ptr + destination_data_offset + offsets,
        fp8_bytes,
        mask=valid,
    )
    destination_scale_offset = (
        destination_block_base + cache_block_size * HEAD_SIZE + block_offset * 4
    )
    destination_scale_ptr = (destination_ptr + destination_scale_offset).to(
        tl.pointer_type(tl.float32)
    )
    tl.store(destination_scale_ptr, wire_scale, mask=valid)


@triton.jit
def _transcode_split_int8_indexer_wire_kernel(
    source_ptr,
    scale_ptr,
    destination_ptr,
    source_block_ids_ptr,
    destination_block_ids_ptr,
    valid_entries_ptr,
    source_block_stride,
    scale_block_stride,
    destination_block_stride,
    cache_block_size,
    HEAD_SIZE: tl.constexpr,
    SUB_BLOCKS: tl.constexpr = 1,
):
    """Split INT8 indexer (uint8[128] + FP32 scale plane) -> D wire uint8[132]."""
    row_idx = tl.program_id(0)
    wire_page_idx = row_idx // cache_block_size
    page_idx = wire_page_idx // SUB_BLOCKS
    sub_idx = wire_page_idx % SUB_BLOCKS
    block_offset = row_idx % cache_block_size
    source_row_offset = sub_idx * cache_block_size + block_offset
    valid = source_row_offset < tl.load(valid_entries_ptr + page_idx)
    offsets = tl.arange(0, HEAD_SIZE)

    source_block_id = tl.load(source_block_ids_ptr + page_idx).to(tl.int64)
    source_block_base = source_block_id * source_block_stride
    source_data_offset = source_block_base + source_row_offset * HEAD_SIZE
    int8_bytes = tl.load(
        source_ptr + source_data_offset + offsets,
        mask=valid,
        other=0,
    )
    int8_values = int8_bytes.to(tl.int8, bitcast=True).to(tl.float32)
    source_scale = tl.load(
        scale_ptr + source_block_id * scale_block_stride + source_row_offset,
        mask=valid,
        other=0.0,
    )
    values = int8_values * source_scale

    amax = tl.max(tl.abs(values), axis=0)
    exponent = tl.ceil(tl.log2(tl.maximum(amax, 1e-4) / 448.0))
    wire_scale = tl.exp2(exponent)
    quantized = tl.clamp(values / wire_scale, -448.0, 448.0)
    fp8_bytes = quantized.to(tl.float8e4nv).to(tl.uint8, bitcast=True)

    destination_block_id = tl.load(destination_block_ids_ptr + wire_page_idx).to(
        tl.int64
    )
    destination_block_base = destination_block_id * destination_block_stride
    destination_data_offset = destination_block_base + block_offset * HEAD_SIZE
    tl.store(
        destination_ptr + destination_data_offset + offsets,
        fp8_bytes,
        mask=valid,
    )
    destination_scale_offset = (
        destination_block_base + cache_block_size * HEAD_SIZE + block_offset * 4
    )
    destination_scale_ptr = (destination_ptr + destination_scale_offset).to(
        tl.pointer_type(tl.float32)
    )
    tl.store(destination_scale_ptr, wire_scale, mask=valid)


def _is_layer_outermost_wire_config(wire_config: KVCacheConfig) -> bool:
    return any(
        int(getattr(tensor, "layer_stride", 0) or 0) > int(tensor.block_stride)
        for tensor in wire_config.kv_cache_tensors
    )


def allocate_fp8_ds_mla_wire_caches(
    wire_config: KVCacheConfig,
    device: torch.device,
    *,
    num_blocks: int | None = None,
    pin_memory: bool = False,
) -> dict[str, torch.Tensor]:
    """Allocate one packed backing and create all D-layout layer views."""
    if not wire_config.kv_cache_tensors:
        raise ValueError("FP8 DS MLA wire staging has no cache tensors")

    view_num_blocks = wire_config.num_blocks if num_blocks is None else num_blocks
    if view_num_blocks <= 0:
        raise ValueError("FP8 DS MLA wire staging must allocate at least one block")
    wire_specs = _layer_specs(wire_config)

    if _is_layer_outermost_wire_config(wire_config):
        layer_plans: list[tuple[str, int, int, int]] = []
        offset = 0
        for tensor_config in wire_config.kv_cache_tensors:
            if tensor_config.block_stride <= 0:
                raise ValueError("FP8 DS MLA wire staging must use a packed layout")
            if len(tensor_config.layers) != 1:
                raise ValueError(
                    "Layer-outermost FP8 wire tensor must own one layer"
                )
            layer_name = tensor_config.layers[0]
            spec = wire_specs[layer_name]
            unit_bytes = wire_config.num_blocks * tensor_config.block_stride
            if unit_bytes <= 0 or tensor_config.layer_stride % unit_bytes != 0:
                raise ValueError(
                    f"Unexpected layer_stride for {layer_name}: "
                    f"{tensor_config.layer_stride}"
                )
            sub_blocks = tensor_config.layer_stride // unit_bytes
            layer_bytes = view_num_blocks * sub_blocks * tensor_config.block_stride
            layer_plans.append(
                (layer_name, sub_blocks, int(spec.page_size_bytes), offset)
            )
            offset += layer_bytes
        backing = torch.zeros(
            offset,
            dtype=torch.uint8,
            device=device,
            pin_memory=pin_memory,
        )
        wire_caches: dict[str, torch.Tensor] = {}
        for layer_name, sub_blocks, page_bytes, layer_offset in layer_plans:
            wire_caches[layer_name] = torch.as_strided(
                backing,
                size=(view_num_blocks * sub_blocks, page_bytes),
                stride=(page_bytes, 1),
                storage_offset=layer_offset,
            )
        return wire_caches

    first = wire_config.kv_cache_tensors[0]
    if first.block_stride <= 0:
        raise ValueError("FP8 DS MLA wire staging must use a packed layout")
    if any(
        tensor.size != first.size or tensor.block_stride != first.block_stride
        for tensor in wire_config.kv_cache_tensors
    ):
        raise ValueError(
            "FP8 DS MLA wire staging tensors must share one packed allocation"
        )

    backing_size = (
        first.size if num_blocks is None else first.block_stride * view_num_blocks
    )
    backing = torch.zeros(
        backing_size,
        dtype=torch.uint8,
        device=device,
        pin_memory=pin_memory,
    )
    wire_caches = {}
    for tensor_config in wire_config.kv_cache_tensors:
        for layer_name in tensor_config.layers:
            spec = wire_specs[layer_name]
            wire_caches[layer_name] = torch.as_strided(
                backing,
                size=(view_num_blocks, spec.page_size_bytes),
                stride=(first.block_stride, 1),
                storage_offset=tensor_config.offset,
            )
    return wire_caches


def allocate_fp8_ds_mla_wire_scratch_caches(
    wire_config: KVCacheConfig,
    device: torch.device,
    scratch_blocks: int,
) -> list[dict[str, torch.Tensor]]:
    """Allocate exactly two independent packed GPU scratch backings."""
    return [
        allocate_fp8_ds_mla_wire_caches(
            wire_config,
            device,
            num_blocks=scratch_blocks,
        )
        for _ in range(FP8_WIRE_SCRATCH_BUFFER_COUNT)
    ]


def _filter_groups_for_active_layers(
    groups: list[KVCacheGroupSpec],
    active_layers: set[str],
) -> list[KVCacheGroupSpec]:
    filtered: list[KVCacheGroupSpec] = []
    for group in groups:
        kept = [name for name in group.layer_names if name in active_layers]
        if len(kept) == len(group.layer_names):
            filtered.append(group)
        elif not kept:
            filtered.append(replace(group, layer_names=[]))
        else:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = UniformTypeKVCacheSpecs.from_specs(
                    {name: spec.kv_cache_specs[name] for name in kept}
                )
                assert spec is not None
            filtered.append(replace(group, layer_names=kept, kv_cache_spec=spec))
    return filtered


def _normalize_manager_blocks(
    tensor: torch.Tensor,
    num_blocks: int,
    *,
    name: str,
) -> torch.Tensor:
    """View kernel-block-split storage with manager block IDs on axis 0."""
    if num_blocks <= 0 or tensor.shape[0] == num_blocks:
        return tensor
    grouped = group_kernel_blocks(tensor, num_blocks)
    if not grouped[0].is_contiguous():
        raise ValueError(
            "FP8 wire staging requires contiguous kernel blocks within each "
            f"manager block for {name}, got shape={tuple(grouped.shape)} "
            f"stride={grouped.stride()}"
        )
    return grouped


class FP8DSMLAWireStaging:
    """Pinned-host mirror of producer block IDs in D's packed FP8 wire layout.

    Existing NIXL request metadata remains unchanged because the mirror uses
    compute block IDs. GPU quantization alternates between two independent
    scratch backings while a dedicated stream copies the current HMA group's
    merged padded byte spans into pinned host memory.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        compute_config: KVCacheConfig,
        compute_caches: dict[str, torch.Tensor],
        kv_cache_scales: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.compute_config = compute_config
        self.num_blocks = compute_config.num_blocks
        self.compute_caches = {
            layer_name: _normalize_manager_blocks(
                cache, self.num_blocks, name=layer_name
            )
            for layer_name, cache in compute_caches.items()
        }
        resolved_scales = dict(kv_cache_scales) if kv_cache_scales else {}
        compilation_cfg = getattr(vllm_config, "compilation_config", None)
        forward_context = (
            getattr(compilation_cfg, "static_forward_context", {})
            if compilation_cfg is not None
            else {}
        )
        for layer_name in compute_caches:
            if layer_name not in resolved_scales and layer_name in forward_context:
                scales = getattr(forward_context[layer_name], "kv_cache_scales", None)
                if isinstance(scales, torch.Tensor):
                    resolved_scales[layer_name] = scales
        self.kv_cache_scales = {
            layer_name: _normalize_manager_blocks(
                scale, self.num_blocks, name=f"{layer_name}.scales"
            )
            for layer_name, scale in resolved_scales.items()
        }
        raw_wire_config = getattr(compute_config, "wire_kv_cache_config", None)
        if raw_wire_config is None:
            raise ValueError(
                "FP8 DS MLA wire staging was not planned with the compute cache"
            )
        if raw_wire_config.num_blocks != self.num_blocks:
            raise ValueError(
                "Compute and wire KV cache block counts must match: "
                f"{self.num_blocks} != {raw_wire_config.num_blocks}"
            )

        if not raw_wire_config.kv_cache_tensors:
            raise ValueError("FP8 DS MLA wire staging has no cache tensors")
        if not compute_caches:
            raise ValueError("FP8 DS MLA wire staging has no compute caches")

        raw_wire_specs = _layer_specs(raw_wire_config)
        active_layers = set(compute_caches)
        if active_layers < set(raw_wire_specs):
            filtered_compute_groups = _filter_groups_for_active_layers(
                compute_config.kv_cache_groups, active_layers
            )
            filtered_wire_groups = _filter_groups_for_active_layers(
                raw_wire_config.kv_cache_groups, active_layers
            )
            self.wire_config = build_fp8_ds_mla_wire_config(
                filtered_compute_groups, filtered_wire_groups, self.num_blocks
            )
        else:
            self.wire_config = raw_wire_config

        first = self.wire_config.kv_cache_tensors[0]
        self.block_stride = first.block_stride
        if self.block_stride <= 0:
            raise ValueError("FP8 DS MLA wire staging must use a packed layout")

        wire_specs = _layer_specs(self.wire_config)
        self.compute_specs = _layer_specs(compute_config)
        self.wire_specs = wire_specs
        self._is_layer_outermost = _is_layer_outermost_wire_config(self.wire_config)
        self.sub_blocks_by_layer = {
            layer_name: wire_sub_blocks_per_compute_block(
                self.compute_specs[layer_name], wire_specs[layer_name]
            )
            for layer_name in wire_specs
            if layer_name in self.compute_specs
        }
        compute_device = next(iter(compute_caches.values())).device
        self.scratch_blocks = get_fp8_wire_scratch_blocks(vllm_config)
        self.wire_caches = allocate_fp8_ds_mla_wire_caches(
            self.wire_config,
            torch.device("cpu"),
            pin_memory=True,
        )
        self.scratch_wire_cache_buffers = allocate_fp8_ds_mla_wire_scratch_caches(
            self.wire_config,
            compute_device,
            self.scratch_blocks,
        )
        expected_layers: set[str] = set()
        for tensor_config in self.wire_config.kv_cache_tensors:
            for layer_name in tensor_config.layers:
                expected_layers.add(layer_name)
                spec = wire_specs[layer_name]
                cache = self.wire_caches.get(layer_name)
                if cache is None:
                    raise ValueError(
                        f"Preallocated wire cache is missing layer {layer_name}"
                    )
                sub_blocks = self.sub_blocks_by_layer.get(layer_name, 1)
                expected_shape = (
                    self.num_blocks * sub_blocks,
                    spec.page_size_bytes,
                )
                if cache.shape != expected_shape:
                    raise ValueError(
                        f"Wire cache shape for {layer_name} is {cache.shape}, "
                        f"expected {expected_shape}"
                    )
                expected_stride = (
                    spec.page_size_bytes
                    if self._is_layer_outermost
                    else self.block_stride
                )
                if cache.stride(0) != expected_stride:
                    raise ValueError(
                        f"Wire cache block stride for {layer_name} is "
                        f"{cache.stride(0)}, expected {expected_stride}"
                    )
        unexpected_layers = set(self.wire_caches) - expected_layers
        if unexpected_layers:
            raise ValueError(
                "Preallocated wire cache contains unexpected layers: "
                + ", ".join(sorted(unexpected_layers))
            )
        storage_ptrs = {
            cache.untyped_storage().data_ptr() for cache in self.wire_caches.values()
        }
        if len(storage_ptrs) != 1:
            raise ValueError(
                "Preallocated FP8 DS MLA wire caches must share one backing"
            )
        self.backing = next(iter(self.wire_caches.values()))
        self.scratch_backings = [
            next(iter(caches.values())) for caches in self.scratch_wire_cache_buffers
        ]
        if self._is_layer_outermost:
            backing_ptr = self.backing.untyped_storage().data_ptr()
            self.wire_layer_rows: dict[str, torch.Tensor] = {}
            self.scratch_layer_rows: list[dict[str, torch.Tensor]] = [
                {} for _ in range(FP8_WIRE_SCRATCH_BUFFER_COUNT)
            ]
            self.layer_compute_bytes: dict[str, int] = {}
            for layer_name, wire_cache in self.wire_caches.items():
                sub_blocks = self.sub_blocks_by_layer.get(layer_name, 1)
                row_bytes = sub_blocks * int(wire_specs[layer_name].page_size_bytes)
                self.layer_compute_bytes[layer_name] = row_bytes
                self.wire_layer_rows[layer_name] = torch.as_strided(
                    self.backing,
                    size=(self.num_blocks, row_bytes),
                    stride=(row_bytes, 1),
                    storage_offset=wire_cache.data_ptr() - backing_ptr,
                )
            for buf_idx, caches in enumerate(self.scratch_wire_cache_buffers):
                scratch_ptr = self.scratch_backings[buf_idx].untyped_storage().data_ptr()
                for layer_name, sc in caches.items():
                    row_bytes = self.layer_compute_bytes[layer_name]
                    self.scratch_layer_rows[buf_idx][layer_name] = torch.as_strided(
                        sc,
                        size=(self.scratch_blocks, row_bytes),
                        stride=(row_bytes, 1),
                        storage_offset=sc.data_ptr() - scratch_ptr,
                    )
            self.wire_rows = None
            self.scratch_row_buffers = []
            if self.storage.nbytes() != first.size:
                raise ValueError("Pinned CPU FP8 wire backing has an unexpected size")
        else:
            self.wire_layer_rows = {}
            self.scratch_layer_rows = []
            self.layer_compute_bytes = {}
            self.wire_rows = torch.as_strided(
                self.backing,
                size=(self.num_blocks, self.block_stride),
                stride=(self.block_stride, 1),
                storage_offset=0,
            )
            self.scratch_row_buffers = [
                torch.as_strided(
                    scratch,
                    size=(self.scratch_blocks, self.block_stride),
                    stride=(self.block_stride, 1),
                    storage_offset=0,
                )
                for scratch in self.scratch_backings
            ]
            if self.storage.nbytes() != self.num_blocks * self.block_stride:
                raise ValueError("Pinned CPU FP8 wire backing has an unexpected size")
            if any(
                scratch.untyped_storage().nbytes()
                != self.scratch_blocks * self.block_stride
                for scratch in self.scratch_backings
            ):
                raise ValueError("GPU FP8 wire scratch has an unexpected size")
        if self.backing.device.type != "cpu" or not self.backing.is_pinned():
            raise ValueError("CPU FP8 wire backing must be pinned host memory")
        if any(scratch.device != compute_device for scratch in self.scratch_backings):
            raise ValueError("FP8 wire scratch must use the compute device")
        scratch_storage_ptrs = {
            scratch.untyped_storage().data_ptr() for scratch in self.scratch_backings
        }
        if len(scratch_storage_ptrs) != FP8_WIRE_SCRATCH_BUFFER_COUNT:
            raise ValueError("FP8 wire scratch buffers must use distinct backings")
        with torch.cuda.device(compute_device):
            self.copy_stream = torch.cuda.Stream(device=compute_device)
            self.quant_done_events = [
                torch.cuda.Event(enable_timing=False)
                for _ in range(FP8_WIRE_SCRATCH_BUFFER_COUNT)
            ]
            self.copy_done_events = [
                torch.cuda.Event(enable_timing=False)
                for _ in range(FP8_WIRE_SCRATCH_BUFFER_COUNT)
            ]
        self._next_scratch_chunk = 0
        self._dense_arange = torch.arange(
            self.scratch_blocks, dtype=torch.int64, device=compute_device
        )
        max_sub_blocks = max(self.sub_blocks_by_layer.values(), default=1)
        self._dense_sub_arange = (
            torch.arange(
                self.scratch_blocks * max_sub_blocks,
                dtype=torch.int64,
                device=compute_device,
            )
            if max_sub_blocks > 1
            else self._dense_arange
        )

        self.layer_to_group: dict[str, int] = {}
        self.group_active_layers: list[list[str]] = [
            [] for _ in compute_config.kv_cache_groups
        ]
        for group_idx, group in enumerate(compute_config.kv_cache_groups):
            for layer_name in group.layer_names:
                self.layer_to_group[layer_name] = group_idx
                if layer_name in compute_caches:
                    self.group_active_layers[group_idx].append(layer_name)

        missing = set(compute_caches) - set(self.wire_caches)
        if missing:
            raise ValueError(
                "Wire staging planner omitted compute cache layers: "
                + ", ".join(sorted(missing))
            )
        if self._is_layer_outermost:
            self.hma_copy_spans = tuple(
                tuple(
                    (0, self.layer_compute_bytes[layer_name])
                    for layer_name in layers
                )
                for layers in self.group_active_layers
            )
        else:
            self.hma_copy_spans = build_hma_copy_spans(
                self.compute_config, self.wire_config
            )
        for group_idx, spans in enumerate(self.hma_copy_spans):
            covered_bytes = sum(end - start for start, end in spans)
            logger.info(
                "FP8 wire HMA copy spans: group=%d, span_count=%d, "
                "covered_bytes=%d, block_stride=%d, coverage_ratio=%.6f",
                group_idx,
                len(spans),
                covered_bytes,
                self.block_stride,
                covered_bytes / self.block_stride,
            )
        logger.info(
            "Using preallocated packed FP8 DS MLA wire staging: device=pinned_cpu, "
            "num_blocks=%d, block_stride=%d, total_bytes=%d, "
            "scratch_buffers=%d, scratch_bytes=%d",
            self.num_blocks,
            self.block_stride,
            self.storage.nbytes(),
            FP8_WIRE_SCRATCH_BUFFER_COUNT,
            sum(
                scratch.untyped_storage().nbytes() for scratch in self.scratch_backings
            ),
        )

    @property
    def storage(self) -> torch.UntypedStorage:
        return self.backing.untyped_storage()

    def _copy_repacked_pages(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        source_block_ids: torch.Tensor,
        destination_block_ids: torch.Tensor,
        source_spec: KVCacheSpec,
        wire_spec: KVCacheSpec,
        *,
        already_zeroed: bool = False,
    ) -> None:
        if source_spec.real_page_size_bytes != wire_spec.real_page_size_bytes:
            raise ValueError(
                "REPACK_ONLY page contents differ in size for "
                f"{type(source_spec).__name__}: "
                f"{source_spec.real_page_size_bytes} != "
                f"{wire_spec.real_page_size_bytes}"
            )
        selected = source.index_select(0, source_block_ids).contiguous()
        selected_bytes = selected.view(torch.uint8).reshape(
            source_block_ids.numel(), -1
        )
        real_size = source_spec.real_page_size_bytes
        if selected_bytes.shape[1] != real_size:
            raise ValueError(
                "Unexpected source cache shape for REPACK_ONLY: "
                f"got {selected_bytes.shape[1]} bytes, expected {real_size}"
            )
        if not already_zeroed and real_size < wire_spec.page_size_bytes:
            self._zero_destination_pages(destination, destination_block_ids)
        destination[:, :real_size].index_copy_(
            0, destination_block_ids, selected_bytes
        )

    @staticmethod
    def _zero_destination_pages(
        destination: torch.Tensor,
        destination_block_ids: torch.Tensor,
    ) -> None:
        n_pages = destination_block_ids.numel()
        if n_pages == 0:
            return
        # Fast contiguous slice zeroing when destination_block_ids is 0..n-1.
        if (
            destination_block_ids.device.type == "cpu"
            and destination_block_ids.tolist() == list(range(n_pages))
        ):
            destination[:n_pages].zero_()
            return
        destination.index_fill_(0, destination_block_ids, 0)

    @staticmethod
    def _valid_entries_for_chunk(
        source_spec: MLAAttentionSpec | SlidingWindowMLASpec,
        num_tokens: int,
        total_block_count: int,
        chunk_offset: int,
        chunk_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        num_logical_pages = (
            num_tokens + source_spec.block_size - 1
        ) // source_spec.block_size
        first_logical_page = num_logical_pages - total_block_count
        if first_logical_page < 0:
            raise ValueError(
                "More staging blocks were supplied than the token count requires"
            )
        logical_pages = torch.arange(
            first_logical_page + chunk_offset,
            first_logical_page + chunk_offset + chunk_size,
            dtype=torch.int64,
            device=device,
        )
        valid_tokens = torch.clamp(
            num_tokens - logical_pages * source_spec.block_size,
            min=0,
            max=source_spec.block_size,
        )
        return valid_tokens // int(source_spec.tokens_per_state)

    @staticmethod
    def _valid_token_count_for_chunk(
        source_spec: MLAAttentionSpec | SlidingWindowMLASpec,
        num_tokens: int,
        total_block_count: int,
        chunk_offset: int,
        chunk_size: int,
    ) -> int:
        """Pure-Python exact valid row count without GPU->CPU boolean mask sync."""
        block_size = int(source_spec.block_size)
        tps = int(source_spec.tokens_per_state)
        num_logical_pages = (num_tokens + block_size - 1) // block_size
        first_logical_page = num_logical_pages - total_block_count
        if first_logical_page < 0:
            raise ValueError(
                "More staging blocks were supplied than the token count requires"
            )
        total_valid = 0
        for idx in range(chunk_size):
            lp = first_logical_page + chunk_offset + idx
            rem = num_tokens - lp * block_size
            valid_tok = 0 if rem <= 0 else (block_size if rem >= block_size else rem)
            total_valid += valid_tok // tps
        return total_valid

    def _quantize_pages(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        source_block_ids: torch.Tensor,
        destination_block_ids: torch.Tensor,
        source_spec: MLAAttentionSpec | SlidingWindowMLASpec,
        num_tokens: int,
        total_block_count: int,
        chunk_offset: int,
        *,
        already_zeroed: bool = False,
        source_scales: torch.Tensor | None = None,
        layer_name: str | None = None,
        wire_spec: MLAAttentionSpec | SlidingWindowMLASpec | None = None,
    ) -> None:
        format_name = source_spec.cache_dtype_str
        compressed = format_name in _COMPRESSED_MAIN_FORMATS
        if compressed:
            if source.dtype != torch.uint8:
                raise ValueError(
                    f"Compressed main KV requires uint8 payload, got {source.dtype}"
                )
            if source_scales is None:
                raise ValueError(
                    "Compressed main KV wire staging needs paired FP16 scales"
                    + (f" for {layer_name}" if layer_name else "")
                )
            if source_scales.dtype != torch.float16:
                raise ValueError(
                    f"Compressed main KV scales must be float16, got {source_scales.dtype}"
                )
        elif source.dtype != torch.bfloat16:
            raise ValueError(
                f"QUANT_PACK requires a BF16 producer cache, got {source.dtype}"
            )
        if source_spec.head_size != 512:
            raise ValueError(
                "FP8 DS MLA wire quantization requires head_size=512, "
                f"got {source_spec.head_size}"
            )
        tokens_per_page = page_states(source_spec)
        wire_tokens_per_page = (
            page_states(wire_spec) if wire_spec is not None else tokens_per_page
        )
        chunk_size = source_block_ids.numel()
        total_valid = self._valid_token_count_for_chunk(
            source_spec,
            num_tokens,
            total_block_count,
            chunk_offset,
            chunk_size,
        )
        if not already_zeroed:
            self._zero_destination_pages(destination, destination_block_ids)
        if total_valid == 0:
            return
        selected = source.index_select(0, source_block_ids).contiguous()
        if compressed:
            assert format_name is not None
            payload_width = 512 if format_name == "i8g64" else 256
            flat_payload = selected.reshape(
                chunk_size * tokens_per_page, payload_width
            )[:total_valid]
            selected_scales = source_scales.index_select(0, source_block_ids)
            # Scales are [blocks, rows, groups] (or an equivalent strided view).
            flat_scales = selected_scales.reshape(
                chunk_size * tokens_per_page, selected_scales.shape[-1]
            )[:total_valid]
            flat_rows = _dequant_grouped_main_rows(
                flat_payload,
                flat_scales,
                format_name=format_name,
                head_dim=source_spec.head_size,
            )
        else:
            flat_rows = selected.reshape(
                chunk_size * tokens_per_page, source_spec.head_size
            )[:total_valid]
        entry_offsets = torch.arange(
            wire_tokens_per_page, dtype=torch.int64, device=source_block_ids.device
        )
        flat_slots = (
            destination_block_ids[:, None] * wire_tokens_per_page
            + entry_offsets[None, :]
        ).reshape(-1)[:total_valid]
        quantize_and_insert_k_cache(
            flat_rows,
            destination,
            flat_slots,
            block_size=wire_tokens_per_page,
            is_ue8m0=True,
            use_fnuz=False,
        )

    def _stage_indexer_pages(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        source_block_ids: torch.Tensor,
        destination_block_ids: torch.Tensor,
        source_spec: MLAAttentionSpec,
        wire_spec: MLAAttentionSpec,
        num_tokens: int,
        total_block_count: int,
        chunk_offset: int,
        *,
        already_zeroed: bool = False,
        source_scales: torch.Tensor | None = None,
        layer_name: str | None = None,
    ) -> None:
        wire_page_states = page_states(wire_spec)
        source_page_states = page_states(source_spec)
        if (
            wire_spec.dtype != torch.uint8
            or wire_spec.head_size != 132
            or source_page_states < wire_page_states
            or source_page_states % wire_page_states != 0
        ):
            raise ValueError(
                "Indexer wire cache must use D's uint8 [block, S, 132] "
                f"layout, got dtype={wire_spec.dtype}, "
                f"head_size={wire_spec.head_size}, "
                f"page_states={wire_page_states} (source {source_page_states})"
            )
        sub_blocks = source_page_states // wire_page_states

        split_indexer = source_spec.cache_dtype_str == _SPLIT_INDEXER_FORMAT
        if split_indexer:
            if (
                source.dtype != torch.uint8
                or source_spec.head_size != 128
                or source_spec.dtype != torch.uint8
            ):
                raise ValueError(
                    "Split INT8 indexer requires uint8 head_size=128 payload, "
                    f"got dtype={source.dtype}, head_size={source_spec.head_size}"
                )
            if source_scales is None or source_scales.dtype != torch.float32:
                raise ValueError(
                    "Split INT8 indexer wire staging needs paired float32 scales"
                    + (f" for {layer_name}" if layer_name else "")
                )
            valid_entries = self._valid_entries_for_chunk(
                source_spec,
                num_tokens,
                total_block_count,
                chunk_offset,
                source_block_ids.numel(),
                source_block_ids.device,
            )
            if not already_zeroed:
                self._zero_destination_pages(destination, destination_block_ids)
            scale_block_stride = source_scales.stride(0)
            _transcode_split_int8_indexer_wire_kernel[
                (source_block_ids.numel() * source_page_states,)
            ](
                source.view(torch.uint8),
                source_scales,
                destination,
                source_block_ids,
                destination_block_ids,
                valid_entries,
                source.view(torch.uint8).stride(0),
                scale_block_stride,
                destination.stride(0),
                wire_page_states,
                HEAD_SIZE=128,
                SUB_BLOCKS=sub_blocks,
            )
            return

        if source_spec.dtype == torch.uint8 and source_spec.head_size == 132:
            if source.dtype != torch.uint8:
                raise ValueError(
                    "INT8 indexer cache tensor must use uint8 storage, "
                    f"got {source.dtype}"
                )
            tokens_per_page = source_page_states
            valid_entries = self._valid_entries_for_chunk(
                source_spec,
                num_tokens,
                total_block_count,
                chunk_offset,
                source_block_ids.numel(),
                source_block_ids.device,
            )
            if not already_zeroed:
                self._zero_destination_pages(destination, destination_block_ids)
            _transcode_int8_indexer_wire_kernel[
                (source_block_ids.numel() * tokens_per_page,)
            ](
                source,
                destination,
                source_block_ids,
                destination_block_ids,
                valid_entries,
                source.stride(0),
                destination.stride(0),
                tokens_per_page,
                HEAD_SIZE=128,
            )
            return

        if source.dtype != torch.bfloat16 or source_spec.head_size != 128:
            raise ValueError(
                "Indexer wire staging requires BF16 head_size=128, "
                "uint8 head_size=132, or split INT8 head_size=128 producer cache, "
                f"got dtype={source.dtype}, head_size={source_spec.head_size}"
            )

        tokens_per_page = source_page_states
        chunk_size = source_block_ids.numel()
        total_valid = self._valid_token_count_for_chunk(
            source_spec,
            num_tokens,
            total_block_count,
            chunk_offset,
            chunk_size,
        )
        if not already_zeroed:
            self._zero_destination_pages(destination, destination_block_ids)
        if total_valid == 0:
            return

        selected = (
            source.index_select(0, source_block_ids)
            .contiguous()
            .reshape(
                chunk_size * tokens_per_page,
                source_spec.head_size,
            )[:total_valid]
        )
        entry_offsets = torch.arange(
            wire_page_states, dtype=torch.int64, device=source_block_ids.device
        )
        slot_mapping = (
            destination_block_ids[:, None] * wire_page_states + entry_offsets[None, :]
        ).reshape(-1)[:total_valid]

        # Native D physically stores all 128-byte FP8 rows first, followed by
        # one FP32 UE8M0 scale per token. The final 192 bytes of each 8640-byte
        # wire page remain zero padding.
        _quantize_indexer_wire_kernel[(total_valid,)](
            selected,
            destination,
            slot_mapping,
            selected.stride(0),
            destination.stride(0),
            wire_page_states,
            HEAD_SIZE=128,
        )


    def _stage_group_chunk(
        self,
        group_idx: int,
        source_block_ids: torch.Tensor,
        destination_block_ids: torch.Tensor,
        destination_caches: dict[str, torch.Tensor],
        num_tokens: int,
        total_block_count: int,
        chunk_offset: int,
        *,
        already_zeroed: bool = False,
    ) -> None:
        for layer_name, source in self.compute_caches.items():
            if self.layer_to_group[layer_name] != group_idx:
                continue
            source_spec = self.compute_specs[layer_name]
            wire_spec = self.wire_specs[layer_name]
            destination = destination_caches[layer_name]
            source_scales = self.kv_cache_scales.get(layer_name)
            sub_blocks = getattr(self, "sub_blocks_by_layer", {}).get(layer_name, 1)
            if sub_blocks > 1:
                dst_ids = self._dense_sub_arange[
                    : destination_block_ids.numel() * sub_blocks
                ]
            else:
                dst_ids = destination_block_ids
            if _is_indexer_cache(layer_name):
                if not isinstance(source_spec, MLAAttentionSpec) or not isinstance(
                    wire_spec, MLAAttentionSpec
                ):
                    raise ValueError(
                        f"Indexer cache {layer_name} must use MLAAttentionSpec"
                    )
                self._stage_indexer_pages(
                    source,
                    destination,
                    source_block_ids,
                    dst_ids,
                    source_spec,
                    wire_spec,
                    num_tokens,
                    total_block_count,
                    chunk_offset,
                    already_zeroed=already_zeroed,
                    source_scales=source_scales,
                    layer_name=layer_name,
                )
            elif _is_fp8_ds_mla_wire_page(wire_spec):
                assert isinstance(source_spec, (MLAAttentionSpec, SlidingWindowMLASpec))
                assert isinstance(wire_spec, (MLAAttentionSpec, SlidingWindowMLASpec))
                self._quantize_pages(
                    source,
                    destination,
                    source_block_ids,
                    dst_ids,
                    source_spec,
                    num_tokens,
                    total_block_count,
                    chunk_offset,
                    already_zeroed=already_zeroed,
                    source_scales=source_scales,
                    layer_name=layer_name,
                    wire_spec=wire_spec,
                )
            else:
                self._copy_repacked_pages(
                    source,
                    destination,
                    source_block_ids,
                    dst_ids,
                    source_spec,
                    wire_spec,
                    already_zeroed=already_zeroed,
                )

    def _enqueue_group_chunk(
        self,
        group_idx: int,
        source_id_list: Sequence[int],
        source_ids: torch.Tensor,
        dense_ids: torch.Tensor,
        num_tokens: int,
        total_block_count: int,
        chunk_offset: int,
    ) -> None:
        chunk_index = self._next_scratch_chunk
        buffer_idx = select_scratch_buffer(chunk_index)
        self._next_scratch_chunk += 1
        scratch_backing = self.scratch_backings[buffer_idx]
        scratch_caches = self.scratch_wire_cache_buffers[buffer_idx]
        quant_done = self.quant_done_events[buffer_idx]
        copy_done = self.copy_done_events[buffer_idx]
        quant_stream = torch.cuda.current_stream(scratch_backing.device)

        if chunk_index >= FP8_WIRE_SCRATCH_BUFFER_COUNT:
            quant_stream.wait_event(copy_done)
        if getattr(self, "_is_layer_outermost", False):
            for layer_name in self.group_active_layers[group_idx]:
                self.scratch_layer_rows[buffer_idx][layer_name][
                    : len(source_id_list)
                ].zero_()
        else:
            scratch_rows = self.scratch_row_buffers[buffer_idx]
            scratch_rows[: len(source_id_list)].zero_()
        self._stage_group_chunk(
            group_idx,
            source_ids,
            dense_ids,
            scratch_caches,
            num_tokens,
            total_block_count,
            chunk_offset,
            already_zeroed=True,
        )
        quant_done.record(quant_stream)

        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(quant_done)
            if getattr(self, "_is_layer_outermost", False):
                for layer_name in self.group_active_layers[group_idx]:
                    row_bytes = self.layer_compute_bytes[layer_name]
                    copy_wire_rows_to_host(
                        self.scratch_layer_rows[buffer_idx][layer_name],
                        self.wire_layer_rows[layer_name],
                        source_id_list,
                        ((0, row_bytes),),
                        non_blocking=True,
                    )
            else:
                copy_wire_rows_to_host(
                    self.scratch_row_buffers[buffer_idx],
                    self.wire_rows,
                    source_id_list,
                    self.hma_copy_spans[group_idx],
                    non_blocking=True,
                )
            copy_done.record(self.copy_stream)

    def export(self, block_ids: BlockIds, num_tokens: int) -> None:
        """Synchronously enqueue export of the selected compute blocks."""
        if len(block_ids) != len(self.compute_config.kv_cache_groups):
            raise ValueError(
                "Number of block-id groups does not match the compute cache config"
            )

        compute_device = next(iter(self.compute_caches.values())).device
        export_flags: list[str] = []
        for group_idx, ordered_ids in enumerate(block_ids):
            if not ordered_ids:
                export_flags.append("skip=empty_ids")
                continue
            # PP projection keeps empty group slots so block-id indices stay
            # aligned across ranks. The scheduler still allocates those slots;
            # this rank has no local layers and therefore no HMA spans to copy.
            if not self.hma_copy_spans[group_idx]:
                export_flags.append("skip=empty_span")
                continue
            export_flags.append("export")
        for group_idx, ordered_ids in enumerate(block_ids):
            if export_flags[group_idx] != "export":
                continue
            total_block_count = len(ordered_ids)
            for chunk_offset in range(0, total_block_count, self.scratch_blocks):
                source_id_list = ordered_ids[
                    chunk_offset : chunk_offset + self.scratch_blocks
                ]
                source_ids = torch.as_tensor(
                    source_id_list, dtype=torch.int64, device=compute_device
                )
                dense_arange = getattr(self, "_dense_arange", None)
                if (
                    dense_arange is not None
                    and dense_arange.device == compute_device
                    and len(source_id_list) <= dense_arange.numel()
                ):
                    dense_ids = dense_arange[: len(source_id_list)]
                else:
                    dense_ids = torch.arange(
                        len(source_id_list),
                        dtype=torch.int64,
                        device=compute_device,
                    )
                self._enqueue_group_chunk(
                    group_idx,
                    source_id_list,
                    source_ids,
                    dense_ids,
                    num_tokens,
                    total_block_count,
                    chunk_offset,
                )

    def export_requests(
        self,
        requests: Sequence[
            tuple[BlockIds, int | None] | tuple[BlockIds, int | None, bool]
        ],
    ) -> None:
        if not requests:
            return
        need_sync = False
        for entry in requests:
            if len(entry) == 3:
                block_ids, num_tokens, sync_save = entry
            else:
                block_ids, num_tokens = entry
                sync_save = True
            if sync_save:
                need_sync = True
            if any(block_ids):
                if num_tokens is None:
                    raise ValueError("FP8 wire staging export is missing num_tokens")
                self.export(block_ids, num_tokens)
        # Intermediate chunk-pipelined exports set sync_save=False so the main
        # thread immediately enters the next forward chunk; the final chunk sets
        # sync_save=True as the publication barrier before remote_block_ids leaves P.
        if need_sync:
            self.copy_stream.synchronize()
