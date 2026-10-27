# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-side DeepSeek-V4 BF16 -> packed FP8 wire staging."""

from __future__ import annotations

from collections.abc import Sequence

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
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from .fp8_layout import (
    FP8_WIRE_SCRATCH_BUFFER_COUNT,
    get_fp8_wire_scratch_blocks,
    page_states,
)
from .metadata import format_kv_group_page_log

logger = init_logger("vllm." + __name__)

ByteSpan = tuple[int, int]


def _layer_specs(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
    specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            specs.update(group_spec.kv_cache_specs)
        else:
            specs.update(dict.fromkeys(group.layer_names, group_spec))
    return specs


def _is_quantized_ds_v4_page(spec: KVCacheSpec) -> bool:
    return isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)) and (
        spec.model_version == "deepseek_v4" and spec.cache_dtype_str == "fp8_ds_mla"
    )


def _is_indexer_cache(layer_name: str) -> bool:
    return ".indexer." in layer_name and layer_name.endswith(".k_cache")


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


def copy_wire_rows_to_host(
    scratch_rows: torch.Tensor,
    wire_rows: torch.Tensor,
    source_block_ids: Sequence[int],
    spans: Sequence[ByteSpan],
    *,
    non_blocking: bool = False,
) -> None:
    """Copy selected HMA spans from dense scratch rows into host mirror rows."""
    if len(source_block_ids) > scratch_rows.shape[0]:
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
    for dense_id, source_id in enumerate(source_block_ids):
        for start, end in effective_spans:
            wire_rows[source_id, start:end].copy_(
                scratch_rows[dense_id, start:end],
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

    view_num_blocks = wire_config.num_blocks if num_blocks is None else num_blocks
    if view_num_blocks <= 0:
        raise ValueError("FP8 DS MLA wire staging must allocate at least one block")
    backing_size = (
        first.size if num_blocks is None else first.block_stride * view_num_blocks
    )
    backing = torch.zeros(
        backing_size,
        dtype=torch.uint8,
        device=device,
        pin_memory=pin_memory,
    )
    wire_specs = _layer_specs(wire_config)
    wire_caches: dict[str, torch.Tensor] = {}
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
    ) -> None:
        self.compute_config = compute_config
        self.compute_caches = compute_caches
        self.num_blocks = compute_config.num_blocks
        self.wire_config = getattr(compute_config, "wire_kv_cache_config", None)
        if self.wire_config is None:
            raise ValueError(
                "FP8 DS MLA wire staging was not planned with the compute cache"
            )
        if self.wire_config.num_blocks != self.num_blocks:
            raise ValueError(
                "Compute and wire KV cache block counts must match: "
                f"{self.num_blocks} != {self.wire_config.num_blocks}"
            )

        if not self.wire_config.kv_cache_tensors:
            raise ValueError("FP8 DS MLA wire staging has no cache tensors")
        first = self.wire_config.kv_cache_tensors[0]
        self.block_stride = first.block_stride
        if self.block_stride <= 0:
            raise ValueError("FP8 DS MLA wire staging must use a packed layout")

        wire_specs = _layer_specs(self.wire_config)
        self.compute_specs = _layer_specs(compute_config)
        self.wire_specs = wire_specs
        if not compute_caches:
            raise ValueError("FP8 DS MLA wire staging has no compute caches")
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
                expected_shape = (self.num_blocks, spec.page_size_bytes)
                if cache.shape != expected_shape:
                    raise ValueError(
                        f"Wire cache shape for {layer_name} is {cache.shape}, "
                        f"expected {expected_shape}"
                    )
                if cache.stride(0) != self.block_stride:
                    raise ValueError(
                        f"Wire cache block stride for {layer_name} is "
                        f"{cache.stride(0)}, expected {self.block_stride}"
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
        self.wire_rows = torch.as_strided(
            self.backing,
            size=(self.num_blocks, self.block_stride),
            stride=(self.block_stride, 1),
            storage_offset=0,
        )
        self.scratch_backings = [
            next(iter(caches.values())) for caches in self.scratch_wire_cache_buffers
        ]
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

        self.layer_to_group: dict[str, int] = {}
        for group_idx, group in enumerate(compute_config.kv_cache_groups):
            for layer_name in group.layer_names:
                self.layer_to_group[layer_name] = group_idx

        missing = set(compute_caches) - set(self.wire_caches)
        if missing:
            raise ValueError(
                "Wire staging planner omitted compute cache layers: "
                + ", ".join(sorted(missing))
            )
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
        wire_rows = torch.zeros(
            (source_block_ids.numel(), wire_spec.page_size_bytes),
            dtype=torch.uint8,
            device=source.device,
        )
        wire_rows[:, :real_size].copy_(selected_bytes)
        destination.index_copy_(0, destination_block_ids, wire_rows)

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
    ) -> None:
        if source.dtype != torch.bfloat16:
            raise ValueError(
                f"QUANT_PACK requires a BF16 producer cache, got {source.dtype}"
            )
        tokens_per_page = page_states(source_spec)
        selected = source.index_select(0, source_block_ids).contiguous()
        selected = selected.reshape(
            source_block_ids.numel(), tokens_per_page, source_spec.head_size
        )
        valid_entries = self._valid_entries_for_chunk(
            source_spec,
            num_tokens,
            total_block_count,
            chunk_offset,
            source_block_ids.numel(),
            source_block_ids.device,
        )
        entry_offsets = torch.arange(
            tokens_per_page, dtype=torch.int64, device=source_block_ids.device
        )
        valid_mask = entry_offsets[None, :] < valid_entries[:, None]
        rows = selected[valid_mask]
        if rows.shape[1] != 512:
            raise ValueError(
                "DeepSeek-V4 FP8 wire quantization requires head_size=512, "
                f"got {rows.shape[1]}"
            )
        slot_mapping = (
            destination_block_ids[:, None] * tokens_per_page + entry_offsets[None, :]
        )[valid_mask]
        destination.index_copy_(
            0,
            destination_block_ids,
            torch.zeros(
                (source_block_ids.numel(), destination.shape[1]),
                dtype=torch.uint8,
                device=destination.device,
            ),
        )
        if rows.numel() == 0:
            return
        quantize_and_insert_k_cache(
            rows,
            destination,
            slot_mapping,
            block_size=tokens_per_page,
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
    ) -> None:
        wire_page_states = page_states(wire_spec)
        source_page_states = page_states(source_spec)
        if (
            wire_spec.dtype != torch.uint8
            or wire_spec.head_size != 132
            or wire_page_states != source_page_states
        ):
            raise ValueError(
                "Indexer wire cache must use D's uint8 [block, 64, 132] "
                f"layout, got dtype={wire_spec.dtype}, "
                f"head_size={wire_spec.head_size}, "
                f"page_states={wire_page_states} (source {source_page_states})"
            )

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
            destination.index_copy_(
                0,
                destination_block_ids,
                torch.zeros(
                    (source_block_ids.numel(), destination.shape[1]),
                    dtype=torch.uint8,
                    device=destination.device,
                ),
            )
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
                "Indexer wire staging requires a BF16 head_size=128 or "
                "uint8 head_size=132 producer cache, "
                f"got dtype={source.dtype}, head_size={source_spec.head_size}"
            )

        tokens_per_page = source_page_states
        selected = (
            source.index_select(0, source_block_ids)
            .contiguous()
            .reshape(
                source_block_ids.numel(),
                tokens_per_page,
                source_spec.head_size,
            )
        )
        valid_entries = self._valid_entries_for_chunk(
            source_spec,
            num_tokens,
            total_block_count,
            chunk_offset,
            source_block_ids.numel(),
            source_block_ids.device,
        )
        entry_offsets = torch.arange(
            tokens_per_page, dtype=torch.int64, device=source_block_ids.device
        )
        valid_mask = entry_offsets[None, :] < valid_entries[:, None]
        rows = selected[valid_mask]
        slot_mapping = (
            destination_block_ids[:, None] * tokens_per_page + entry_offsets[None, :]
        )[valid_mask]

        destination.index_copy_(
            0,
            destination_block_ids,
            torch.zeros(
                (source_block_ids.numel(), destination.shape[1]),
                dtype=torch.uint8,
                device=destination.device,
            ),
        )
        if rows.numel() == 0:
            return

        # Native D physically stores all 128-byte FP8 rows first, followed by
        # one FP32 UE8M0 scale per token. The final 192 bytes of each 8640-byte
        # wire page remain zero padding.
        _quantize_indexer_wire_kernel[(rows.shape[0],)](
            rows,
            destination,
            slot_mapping,
            rows.stride(0),
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
    ) -> None:
        for layer_name, source in self.compute_caches.items():
            if self.layer_to_group[layer_name] != group_idx:
                continue
            source_spec = self.compute_specs[layer_name]
            wire_spec = self.wire_specs[layer_name]
            destination = destination_caches[layer_name]
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
                    destination_block_ids,
                    source_spec,
                    wire_spec,
                    num_tokens,
                    total_block_count,
                    chunk_offset,
                )
            elif _is_quantized_ds_v4_page(wire_spec):
                assert isinstance(source_spec, (MLAAttentionSpec, SlidingWindowMLASpec))
                self._quantize_pages(
                    source,
                    destination,
                    source_block_ids,
                    destination_block_ids,
                    source_spec,
                    num_tokens,
                    total_block_count,
                    chunk_offset,
                )
            else:
                self._copy_repacked_pages(
                    source,
                    destination,
                    source_block_ids,
                    destination_block_ids,
                    source_spec,
                    wire_spec,
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
        scratch_rows = self.scratch_row_buffers[buffer_idx]
        scratch_caches = self.scratch_wire_cache_buffers[buffer_idx]
        quant_done = self.quant_done_events[buffer_idx]
        copy_done = self.copy_done_events[buffer_idx]
        quant_stream = torch.cuda.current_stream(scratch_backing.device)

        if chunk_index >= FP8_WIRE_SCRATCH_BUFFER_COUNT:
            quant_stream.wait_event(copy_done)
        scratch_rows[: len(source_id_list)].zero_()
        self._stage_group_chunk(
            group_idx,
            source_ids,
            dense_ids,
            scratch_caches,
            num_tokens,
            total_block_count,
            chunk_offset,
        )
        quant_done.record(quant_stream)

        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(quant_done)
            copy_wire_rows_to_host(
                scratch_rows,
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
        # logger.info(
        #     "[PD-P] export tokens=%s %s",
        #     num_tokens,
        #     format_kv_group_page_log(
        #         self.compute_config.kv_cache_groups,
        #         block_ids,
        #         extra=export_flags,
        #     ),
        # )
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

    def export_requests(self, requests: Sequence[tuple[BlockIds, int | None]]) -> None:
        if not requests:
            return
        for block_ids, num_tokens in requests:
            if num_tokens is None:
                raise ValueError("FP8 wire staging export is missing num_tokens")
            self.export(block_ids, num_tokens)
        # wait_for_save is a publication barrier: D may issue READ immediately
        # after the scheduler returns the remote block metadata.
        self.copy_stream.synchronize()
