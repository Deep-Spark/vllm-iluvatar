# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 producer KV -> D-layout FP8 wire page quantization (vLLM 0.31.0)."""

from __future__ import annotations

import numpy as np
import torch
from vllm.models.deepseek_v4.common.ops.cache_utils import quantize_and_insert_k_cache
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

from ..dsv41 import resolve_quantized_kv_format
from .wire_layout import (
    COPY_ONLY,
    QUANT_INDEXER,
    QUANT_PACK,
    QUANT_PACK_COMPRESSED,
    WirePlan,
    page_states,
    spec_tokens_per_state,
    subpage_ratio_for,
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
    SUBPAGE_RATIO: tl.constexpr = 1,
    SUBPAGE_STRIDE: tl.constexpr = 0,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, HEAD_SIZE)
    values = tl.load(source_ptr + row_idx * source_stride + offsets).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    exponent = tl.ceil(tl.log2(tl.maximum(amax, 1e-4) / 448.0))
    biased = tl.maximum(tl.minimum(exponent.to(tl.int32) + 127, 254), 1)
    scale = (biased << 23).to(tl.float32, bitcast=True)
    quantized = tl.clamp(values / scale, -448.0, 448.0)
    fp8_bytes = quantized.to(tl.float8e4nv).to(tl.uint8, bitcast=True)

    slot = tl.load(slot_mapping_ptr + row_idx)
    raw_block_id = (slot // cache_block_size).to(tl.int64)
    block_offset = slot % cache_block_size
    if SUBPAGE_RATIO > 1:
        p_block_id = raw_block_id // SUBPAGE_RATIO
        subpage_id = raw_block_id % SUBPAGE_RATIO
        block_base = (
            p_block_id * destination_block_stride + subpage_id * SUBPAGE_STRIDE
        )
    else:
        block_base = raw_block_id * destination_block_stride
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
    valid_entries_ptr,
    source_block_stride,
    destination_block_stride,
    cache_block_size,
    HEAD_SIZE: tl.constexpr,
):
    """Legacy packed INT8 indexer page -> D FP8 wire page (dense scratch rows)."""
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

    destination_block_base = page_idx * destination_block_stride
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


def _valid_entries(
    source_spec: MLAAttentionSpec | SlidingWindowMLASpec,
    num_blocks: int,
    first_page: int,
    num_valid_tokens: int | None,
) -> list[int]:
    """Storage entries holding computed tokens, per dense block."""
    if num_valid_tokens is None:
        return [page_states(source_spec)] * num_blocks
    block_size = source_spec.block_size
    compress_ratio = spec_tokens_per_state(source_spec)
    return [
        min(max(num_valid_tokens - page * block_size, 0), block_size)
        // compress_ratio
        for page in range(first_page, first_page + num_blocks)
    ]


def _run_lengths(counts: list[int]):
    """Collapse equal neighbours into ``(start, end, count)`` runs."""
    start = 0
    total = len(counts)
    while start < total:
        count = counts[start]
        end = start + 1
        while end < total and counts[end] == count:
            end += 1
        yield start, end, count
        start = end


def _group_manager_blocks(
    tensor: torch.Tensor, num_manager_blocks: int | None
) -> torch.Tensor:
    """Group kernel-granular dim-0 ``(num_blocks * r, ...)`` into ``(num_blocks, r, ...)``."""
    if (
        num_manager_blocks is None
        or num_manager_blocks <= 0
        or tensor.shape[0] == num_manager_blocks
    ):
        return tensor
    if tensor.shape[0] % num_manager_blocks != 0:
        raise ValueError(
            f"KV cache tensor dim-0 ({tensor.shape[0]}) is not a multiple of "
            f"manager num_blocks ({num_manager_blocks})"
        )
    return tensor.unflatten(0, (num_manager_blocks, -1))


def _dense_rows_and_slots(
    source: torch.Tensor,
    source_spec: MLAAttentionSpec | SlidingWindowMLASpec,
    source_block_ids: torch.Tensor,
    first_page: int,
    num_valid_tokens: int | None,
    num_manager_blocks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens_per_page = page_states(source_spec)
    num_blocks = source_block_ids.numel()
    device = source_block_ids.device
    grouped_source = _group_manager_blocks(source, num_manager_blocks)
    selected = (
        grouped_source.index_select(0, source_block_ids)
        .contiguous()
        .reshape(num_blocks, tokens_per_page, source_spec.head_size)
    )
    counts = _valid_entries(
        source_spec, num_blocks, first_page, num_valid_tokens
    )
    flat = selected.reshape(-1, source_spec.head_size)
    total = sum(counts)
    prefix = all(c == tokens_per_page for c in counts[:-1])
    if prefix:
        rows = flat[:total]
        slot_mapping = torch.arange(total, dtype=torch.int64, device=device)
        return rows, slot_mapping
    index = (
        np.concatenate(
            [
                np.arange(
                    block * tokens_per_page,
                    block * tokens_per_page + count,
                    dtype=np.int64,
                )
                for block, count in enumerate(counts)
                if count
            ]
        )
        if total
        else np.empty(0, dtype=np.int64)
    )
    slot_mapping = torch.from_numpy(index).to(device)
    rows = flat.index_select(0, slot_mapping)
    return rows, slot_mapping


def _dequant_compressed_rows(
    payload_rows: torch.Tensor,
    scale_rows: torch.Tensor,
    fmt_name: str,
    *,
    head_dim: int = 512,
) -> torch.Tensor:
    """Expand ``i8g64`` / ``i4g32`` page rows to BF16 ``[N, head_dim]``."""
    fmt = resolve_quantized_kv_format(fmt_name)
    group_size = int(fmt.group_size)
    if head_dim % group_size:
        raise ValueError(
            f"Compressed KV head_dim={head_dim} is not divisible by "
            f"group_size={group_size} for format {fmt_name}"
        )
    groups = head_dim // group_size
    if scale_rows.shape != (payload_rows.shape[0], groups):
        raise ValueError(
            "Compressed KV scale geometry mismatch: "
            f"payload_rows={tuple(payload_rows.shape)}, "
            f"scale_rows={tuple(scale_rows.shape)}, expected_groups={groups}"
        )
    if fmt.name == "i8g64":
        if payload_rows.shape[1] != head_dim:
            raise ValueError(
                f"i8g64 payload width must be {head_dim}, got {payload_rows.shape[1]}"
            )
        values = payload_rows.view(torch.int8).to(torch.float32)
    elif fmt.name == "i4g32":
        if payload_rows.shape[1] != head_dim // 2:
            raise ValueError(
                f"i4g32 payload width must be {head_dim // 2}, "
                f"got {payload_rows.shape[1]}"
            )
        packed = payload_rows.view(torch.uint8)
        low = (packed & 0x0F).to(torch.int8)
        high = ((packed >> 4) & 0x0F).to(torch.int8)
        # Signed nibble: values 8..15 are negative.
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
        values = torch.stack((low, high), dim=-1).reshape(
            payload_rows.shape[0], head_dim
        ).to(torch.float32)
    else:
        raise ValueError(
            f"Unsupported compressed KV format for FP8 wire: {fmt_name}"
        )
    scaled = values.view(-1, groups, group_size) * scale_rows.to(
        torch.float32
    ).unsqueeze(-1)
    return scaled.reshape(-1, head_dim).to(torch.bfloat16)


def _gather_compressed_rows(
    source: torch.Tensor,
    scales: torch.Tensor,
    source_spec: MLAAttentionSpec | SlidingWindowMLASpec,
    source_block_ids: torch.Tensor,
    first_page: int,
    num_valid_tokens: int | None,
    num_manager_blocks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dense payload/scale rows and slot mapping for compressed main KV."""
    tokens_per_page = page_states(source_spec)
    num_blocks = source_block_ids.numel()
    device = source_block_ids.device
    grouped_source = _group_manager_blocks(source, num_manager_blocks)
    grouped_scales = _group_manager_blocks(scales, num_manager_blocks)
    if grouped_source[0].numel() % tokens_per_page != 0:
        raise ValueError(
            "Compressed main KV manager page elements must divide "
            f"tokens_per_page={tokens_per_page}, got {tuple(source.shape)}"
        )
    payload_width = grouped_source[0].numel() // tokens_per_page
    if grouped_scales[0].numel() % tokens_per_page != 0:
        raise ValueError(
            "Compressed main KV scale page elements must divide "
            f"tokens_per_page={tokens_per_page}, got {tuple(scales.shape)}"
        )
    scale_groups = grouped_scales[0].numel() // tokens_per_page
    selected = (
        grouped_source.index_select(0, source_block_ids)
        .contiguous()
        .reshape(num_blocks, tokens_per_page, payload_width)
    )
    selected_scales = (
        grouped_scales.index_select(0, source_block_ids)
        .contiguous()
        .reshape(num_blocks, tokens_per_page, scale_groups)
    )
    counts = _valid_entries(
        source_spec, num_blocks, first_page, num_valid_tokens
    )
    flat_payload = selected.reshape(num_blocks * tokens_per_page, selected.shape[-1])
    flat_scales = selected_scales.reshape(
        num_blocks * tokens_per_page, selected_scales.shape[-1]
    )
    total = sum(counts)
    prefix = all(c == tokens_per_page for c in counts[:-1])
    if prefix:
        rows = flat_payload[:total]
        scale_rows = flat_scales[:total]
        slot_mapping = torch.arange(total, dtype=torch.int64, device=device)
        return rows, scale_rows, slot_mapping
    index = (
        np.concatenate(
            [
                np.arange(
                    block * tokens_per_page,
                    block * tokens_per_page + count,
                    dtype=np.int64,
                )
                for block, count in enumerate(counts)
                if count
            ]
        )
        if total
        else np.empty(0, dtype=np.int64)
    )
    slot_mapping = torch.from_numpy(index).to(device)
    rows = flat_payload.index_select(0, slot_mapping)
    scale_rows = flat_scales.index_select(0, slot_mapping)
    return rows, scale_rows, slot_mapping


def _insert_main_fp8_wire(
    rows: torch.Tensor,
    destination: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    states_per_page: int,
    subpage_ratio: int,
    wire_page_bytes: int,
) -> None:
    """Insert quantized main KV rows, splitting each P page into D sub-pages."""
    if subpage_ratio <= 1:
        quantize_and_insert_k_cache(
            rows,
            destination,
            slot_mapping,
            block_size=states_per_page,
            is_ue8m0=True,
            use_fnuz=False,
        )
        return
    if states_per_page % subpage_ratio or wire_page_bytes % subpage_ratio:
        raise ValueError(
            "Main KV wire sub-page split must be integral: "
            f"states={states_per_page}, bytes={wire_page_bytes}, "
            f"subpage_ratio={subpage_ratio}"
        )
    d_states = states_per_page // subpage_ratio
    d_page_bytes = wire_page_bytes // subpage_ratio
    if destination.stride(0) == wire_page_bytes and destination.stride(1) == 1:
        sub_dest = torch.as_strided(
            destination,
            (destination.shape[0] * subpage_ratio, d_page_bytes),
            (d_page_bytes, 1),
        )
        quantize_and_insert_k_cache(
            rows,
            sub_dest,
            slot_mapping,
            block_size=d_states,
            is_ue8m0=True,
            use_fnuz=False,
        )
        return
    subpage_ids = (slot_mapping // d_states) % subpage_ratio
    remapped_base = (slot_mapping // states_per_page) * d_states + (
        slot_mapping % d_states
    )
    for sp in range(subpage_ratio):
        mask = subpage_ids == sp
        if not mask.any():
            continue
        sp_dest = destination[:, sp * d_page_bytes : (sp + 1) * d_page_bytes]
        quantize_and_insert_k_cache(
            rows[mask],
            sp_dest,
            remapped_base[mask],
            block_size=d_states,
            is_ue8m0=True,
            use_fnuz=False,
        )


def quantize_layer(
    plan: WirePlan,
    layer_name: str,
    source: torch.Tensor,
    destination: torch.Tensor,
    source_block_ids: torch.Tensor,
    first_page: int,
    num_valid_tokens: int | None,
    source_scales: torch.Tensor | None = None,
) -> None:
    """Write one layer's wire pages for ``source_block_ids`` into dense scratch."""
    kind = plan.quant_kind[layer_name]
    source_spec = plan.compute_specs[layer_name]
    num_manager_blocks = getattr(getattr(plan, "wire_config", None), "num_blocks", None)

    if kind == COPY_ONLY:
        wire_spec = plan.wire_specs[layer_name]
        if source_spec != wire_spec:
            raise ValueError(
                f"COPY_ONLY requires identical compute/wire specs for {layer_name}"
            )
        real_size = source_spec.real_page_size_bytes
        page_size = source_spec.page_size_bytes
        source_page_bytes = source[0].numel() * source.element_size()
        if source_page_bytes != real_size:
            raise ValueError(
                "Unexpected source cache shape for COPY_ONLY: "
                f"got {source_page_bytes} bytes, expected {real_size}"
            )
        element_size = source.element_size()
        if page_size % element_size:
            raise ValueError(
                f"COPY_ONLY page size {page_size} is not aligned to {source.dtype}"
            )
        destination_view = torch.as_strided(
            destination.view(source.dtype),
            size=(source_block_ids.numel(), *source.shape[1:]),
            stride=(page_size // element_size, *source.stride()[1:]),
        )
        token_dim = source.ndim - 2
        entries_per_page = page_states(source_spec)
        if source.shape[token_dim] != entries_per_page:
            raise ValueError(
                f"COPY_ONLY cannot locate the storage-entry dimension for "
                f"{layer_name}: shape={tuple(source.shape)}, "
                f"entries={entries_per_page}"
            )
        counts = _valid_entries(
            source_spec, source_block_ids.numel(), first_page, num_valid_tokens
        )
        for start, end, count in _run_lengths(counts):
            if count:
                selected = source.index_select(
                    0, source_block_ids[start:end]
                ).narrow(token_dim, 0, count)
                destination_view[start:end].narrow(
                    token_dim, 0, count
                ).copy_(selected)
        return

    if kind == QUANT_PACK:
        if source.dtype != torch.bfloat16:
            raise ValueError(
                f"QUANT_PACK requires a BF16 producer cache, got {source.dtype}"
            )
        rows, slot_mapping = _dense_rows_and_slots(
            source, source_spec, source_block_ids, first_page, num_valid_tokens,
            num_manager_blocks=num_manager_blocks,
        )
        if rows.shape[1] != 512:
            raise ValueError(
                "DeepSeek-V4 FP8 wire quantization requires head_size=512, "
                f"got {rows.shape[1]}"
            )
        if rows.numel() == 0:
            return
        wire_spec = plan.wire_specs[layer_name]
        subpage_ratio = subpage_ratio_for(wire_spec, plan.decode_block_size)
        _insert_main_fp8_wire(
            rows,
            destination,
            slot_mapping,
            states_per_page=page_states(source_spec),
            subpage_ratio=subpage_ratio,
            wire_page_bytes=plan.page_bytes[layer_name],
        )
        return

    if kind == QUANT_PACK_COMPRESSED:
        fmt_name = getattr(source_spec, "cache_dtype_str", None)
        if source.dtype != torch.uint8 or fmt_name not in ("i8g64", "i4g32"):
            raise ValueError(
                "QUANT_PACK_COMPRESSED requires uint8 i8g64/i4g32 producer "
                f"cache, got dtype={source.dtype}, cache_dtype_str={fmt_name!r}"
            )
        if source_scales is None:
            raise ValueError(
                f"QUANT_PACK_COMPRESSED requires paired FP16 scales for {layer_name}"
            )
        payload_rows, scale_rows, slot_mapping = _gather_compressed_rows(
            source,
            source_scales,
            source_spec,
            source_block_ids,
            first_page,
            num_valid_tokens,
            num_manager_blocks=num_manager_blocks,
        )
        if payload_rows.numel() == 0:
            return
        rows = _dequant_compressed_rows(
            payload_rows, scale_rows, fmt_name, head_dim=512
        )
        wire_spec = plan.wire_specs[layer_name]
        subpage_ratio = subpage_ratio_for(wire_spec, plan.decode_block_size)
        _insert_main_fp8_wire(
            rows,
            destination,
            slot_mapping,
            states_per_page=page_states(source_spec),
            subpage_ratio=subpage_ratio,
            wire_page_bytes=plan.page_bytes[layer_name],
        )
        return

    if kind == QUANT_INDEXER:
        wire_spec = plan.wire_specs[layer_name]
        wire_entries = page_states(wire_spec)
        source_entries = page_states(source_spec)
        if (
            wire_spec.dtype != torch.uint8
            or wire_spec.head_size != 132
            or wire_entries != source_entries
        ):
            raise ValueError(
                "Indexer wire cache must use D's uint8 [block, 64, 132] layout, "
                f"got dtype={wire_spec.dtype}, head_size={wire_spec.head_size}, "
                f"page_states={wire_entries} (source {source_entries})"
            )

        # Legacy packed INT8 indexer: uint8 pages with head_size=132.
        if source_spec.dtype == torch.uint8 and source_spec.head_size == 132:
            if source.dtype != torch.uint8:
                raise ValueError(
                    "INT8 indexer cache tensor must use uint8 storage, "
                    f"got {source.dtype}"
                )
            counts = _valid_entries(
                source_spec,
                source_block_ids.numel(),
                first_page,
                num_valid_tokens,
            )
            valid_entries = torch.tensor(
                counts, dtype=torch.int32, device=source_block_ids.device
            )
            _transcode_int8_indexer_wire_kernel[
                (source_block_ids.numel() * source_entries,)
            ](
                source,
                destination,
                source_block_ids,
                valid_entries,
                source.stride(0),
                destination.stride(0),
                source_entries,
                HEAD_SIZE=128,
            )
            return

        # Split INT8 indexer: uint8 head_size=128 payload + FP32 scale plane.
        if (
            getattr(source_spec, "cache_dtype_str", None) == "int8_indexer_split"
            or (
                source.dtype == torch.uint8
                and source_spec.head_size == 128
                and source_scales is not None
            )
        ):
            if source_scales is None:
                raise ValueError(
                    f"Split INT8 indexer requires FP32 scales for {layer_name}"
                )
            tokens_per_page = page_states(source_spec)
            grouped_source = _group_manager_blocks(source, num_manager_blocks)
            grouped_scales = _group_manager_blocks(source_scales, num_manager_blocks)
            num_sel = source_block_ids.numel()
            selected = grouped_source.index_select(0, source_block_ids).contiguous()
            selected_scales = grouped_scales.index_select(
                0, source_block_ids
            ).contiguous()
            if selected.numel() != num_sel * tokens_per_page * 128:
                raise ValueError(
                    "Split indexer payload must align to [blocks, rows, 128]: "
                    f"got {tuple(selected.shape)} for tokens_per_page={tokens_per_page}"
                )
            if selected_scales.numel() != num_sel * tokens_per_page:
                raise ValueError(
                    "Split indexer scale pages must align to payload pages: "
                    f"payload={tuple(selected.shape)}, scales={tuple(selected_scales.shape)}"
                )
            selected = selected.reshape(num_sel, tokens_per_page, 128)
            selected_scales = selected_scales.reshape(num_sel, tokens_per_page)
            if selected.ndim != 3 or selected.shape[-1] != 128:
                raise ValueError(
                    "Split indexer payload must be [blocks, rows, 128], "
                    f"got {tuple(selected.shape)}"
                )
            if selected_scales.shape[:2] != selected.shape[:2]:
                raise ValueError(
                    "Split indexer scale pages must align with payload pages: "
                    f"payload={tuple(selected.shape)}, "
                    f"scales={tuple(selected_scales.shape)}"
                )
            counts = _valid_entries(
                source_spec,
                source_block_ids.numel(),
                first_page,
                num_valid_tokens,
            )
            flat = selected.reshape(-1, 128)
            flat_scales = selected_scales.reshape(-1)
            total = sum(counts)
            if total == 0:
                return
            if all(c == tokens_per_page for c in counts[:-1]):
                int8_rows = flat[:total]
                scale_vals = flat_scales[:total]
                slot_mapping = torch.arange(
                    total, dtype=torch.int64, device=source_block_ids.device
                )
            else:
                index = np.concatenate(
                    [
                        np.arange(
                            block * tokens_per_page,
                            block * tokens_per_page + count,
                            dtype=np.int64,
                        )
                        for block, count in enumerate(counts)
                        if count
                    ]
                )
                slot_mapping = torch.from_numpy(index).to(source_block_ids.device)
                int8_rows = flat.index_select(0, slot_mapping)
                scale_vals = flat_scales.index_select(0, slot_mapping)
            values = int8_rows.view(torch.int8).to(torch.float32) * scale_vals.to(
                torch.float32
            ).unsqueeze(-1)
            rows = values.to(torch.bfloat16)
            subpage_ratio = subpage_ratio_for(wire_spec, plan.decode_block_size)
            if wire_entries % subpage_ratio:
                raise ValueError(
                    "Indexer wire sub-page split must be integral: "
                    f"entries={wire_entries}, subpage_ratio={subpage_ratio}"
                )
            d_entries = wire_entries // subpage_ratio
            d_page_bytes = d_entries * 132
            _quantize_indexer_wire_kernel[(rows.shape[0],)](
                rows,
                destination,
                slot_mapping,
                rows.stride(0),
                destination.stride(0),
                d_entries,
                HEAD_SIZE=128,
                SUBPAGE_RATIO=subpage_ratio,
                SUBPAGE_STRIDE=d_page_bytes if subpage_ratio > 1 else 0,
            )
            return

        if source.dtype != torch.bfloat16 or source_spec.head_size != 128:
            raise ValueError(
                "Indexer wire quantization requires BF16 head_size=128, "
                "legacy INT8 head_size=132, or split INT8 payload+scales; "
                f"got dtype={source.dtype}, head_size={source_spec.head_size}"
            )
        rows, slot_mapping = _dense_rows_and_slots(
            source, source_spec, source_block_ids, first_page, num_valid_tokens,
            num_manager_blocks=num_manager_blocks,
        )
        if rows.numel() == 0:
            return
        subpage_ratio = subpage_ratio_for(wire_spec, plan.decode_block_size)
        if wire_entries % subpage_ratio:
            raise ValueError(
                "Indexer wire sub-page split must be integral: "
                f"entries={wire_entries}, subpage_ratio={subpage_ratio}"
            )
        d_entries = wire_entries // subpage_ratio
        d_page_bytes = d_entries * 132
        _quantize_indexer_wire_kernel[(rows.shape[0],)](
            rows,
            destination,
            slot_mapping,
            rows.stride(0),
            destination.stride(0),
            d_entries,
            HEAD_SIZE=128,
            SUBPAGE_RATIO=subpage_ratio,
            SUBPAGE_STRIDE=d_page_bytes if subpage_ratio > 1 else 0,
        )
        return

    # REPACK_ONLY: split one larger P compressor page into D-sized wire pages.
    wire_spec = plan.wire_specs[layer_name]
    ratio = plan.page_ratios[layer_name]
    source_real_size = source_spec.real_page_size_bytes
    target_real_size = wire_spec.real_page_size_bytes
    if source_real_size != ratio * target_real_size:
        raise ValueError(
            "REPACK_ONLY page contents differ in size for "
            f"{layer_name}: {source_real_size} != "
            f"{ratio} * {target_real_size}"
        )
    selected = _group_manager_blocks(source, num_manager_blocks).index_select(0, source_block_ids).contiguous()
    selected_bytes = selected.view(torch.uint8).reshape(source_block_ids.numel(), -1)
    if selected_bytes.shape[1] != source_real_size:
        raise ValueError(
            "Unexpected source cache shape for REPACK_ONLY: "
            f"got {selected_bytes.shape[1]} bytes, expected {source_real_size}"
        )
    target_page_size = plan.target_page_bytes[layer_name]
    target_entries = page_states(wire_spec)
    entry_size = target_real_size // target_entries
    source_entries = page_states(source_spec)
    if source_entries != ratio * target_entries:
        raise ValueError(
            "REPACK_ONLY entry counts do not match the page ratio for "
            f"{layer_name}: {source_entries} != {ratio} * {target_entries}"
        )
    selected_entries = selected_bytes.reshape(
        source_block_ids.numel(), ratio, target_entries, entry_size
    )
    destination_entries = torch.as_strided(
        destination,
        size=(
            source_block_ids.numel(),
            ratio,
            target_entries,
            entry_size,
        ),
        stride=(
            destination.stride(0),
            target_page_size,
            entry_size,
            1,
        ),
    )
    valid_entries = _valid_entries(
        source_spec, source_block_ids.numel(), first_page, num_valid_tokens
    )
    for subpage in range(ratio):
        counts = [
            min(max(entries - subpage * target_entries, 0), target_entries)
            for entries in valid_entries
        ]
        for start, end, count in _run_lengths(counts):
            if count:
                destination_entries[start:end, subpage, :count].copy_(
                    selected_entries[start:end, subpage, :count]
                )
