"""Allocation and lifetime support for signed grouped main-attention KV.

The scheduler owns physical block IDs.  The scale plane follows the payload
allocation's layer placement and uses those same IDs; it has no block table.
"""

from __future__ import annotations

import operator
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from vllm_iluvatar.models.deepseek_v41.compressed_kv_restore import RestorePlanTracker

PAYLOAD_PAGE_BYTES = 65536
WORKSPACE_BYTES = 128 * 1024 * 1024
PREFILL_WORKSPACE_BYTES = 512 * 1024 * 1024
ENCODING_VERSION = 1


def prefill_producer_for_config(vllm_config) -> bool:
    """Identify the producer role without inferring it from an arena size."""
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    return getattr(transfer, "kv_role", None) in ("kv_producer", "kv_both")


def workspace_bytes_for_config(vllm_config) -> int:
    """Reserve one arena by PD role; standalone and decode retain the default."""
    return (
        PREFILL_WORKSPACE_BYTES
        if prefill_producer_for_config(vllm_config)
        else WORKSPACE_BYTES
    )


def retained_mapper_bytes_for_config(vllm_config) -> int:
    """Credit one retained int32 CI/length pair inside the worker total.

    A family replaces the previous pair before mapping. Use configured token
    and top-k bounds rather than current request lengths, and round each
    tensor's logical allocation to 512 bytes. Graph reserved segments remain
    part of the runner's separate capture-memory accounting.
    """
    model_config = vllm_config.model_config
    text_config = getattr(model_config, "hf_text_config", None)
    if text_config is None:
        text_config = model_config.hf_config
    values = (
        vllm_config.scheduler_config.max_num_batched_tokens,
        text_config.index_topk,
    )
    sizes = []
    for value in values:
        try:
            size = operator.index(value)
        except TypeError as error:
            raise ValueError(
                "Compressed mapper bounds must be positive integers"
            ) from error
        if isinstance(value, bool) or size <= 0:
            raise ValueError("Compressed mapper bounds must be positive integers")
        sizes.append(size)
    tokens, topk = sizes
    return sum(((size * 4 + 511) // 512) * 512 for size in (tokens * topk, tokens))


def _workspace_bytes(value: int) -> int:
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(
            "Compressed KV workspace bytes must be a positive integer"
        ) from error
    if value <= 0:
        raise ValueError("Compressed KV workspace bytes must be a positive integer")
    return value


def main_kv_format():
    """Resolve the opt-in format without changing the global SWA cache dtype."""
    value = os.environ.get("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "none")
    if value == "none":
        return None
    from ixformer.inference.functions.deepseek.quantized_kv_format import (
        resolve_quant_format,
    )

    return resolve_quant_format(value)


def format_factors() -> dict[str, Any]:
    fmt = main_kv_format()
    if fmt is None:
        return {}
    return {
        "compressed_kv_encoding": ENCODING_VERSION,
        "compressed_kv_format": fmt.name,
        "compressed_kv_group_size": fmt.group_size,
        "compressed_kv_scale_dtype": "float16",
        "compressed_kv_layout": "page_row_payload+page_row_group",
        "compressed_kv_page_bytes": PAYLOAD_PAGE_BYTES,
    }


def is_quantized_main_spec(spec: Any) -> bool:
    return getattr(spec, "cache_dtype_str", None) in ("i8g64", "i4g32")


def layer_specs(groups) -> dict[str, Any]:
    result = {}
    for group in groups:
        per_layer = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        for name in group.layer_names:
            result[name] = per_layer[name] if per_layer else group.kv_cache_spec
    return result


def cache_group_geometry(group) -> dict[str, Any]:
    """Receipt geometry with codec identity for every owning main layer."""
    geometry = {
        "block_size": group.kv_cache_spec.block_size,
        "layer_names": list(group.layer_names),
        "dtype": str(group.kv_cache_spec.dtype),
    }
    formats = {
        name: spec.cache_dtype_str
        for name, spec in layer_specs([group]).items()
        if is_quantized_main_spec(spec)
    }
    if formats:
        geometry["compressed_main_kv"] = {
            "formats": formats,
            "encoding_version": ENCODING_VERSION,
            "scale_dtype": "float16",
            "layout": "page_row_payload+page_row_group",
        }
    return geometry


@dataclass(frozen=True)
class CacheBudget:
    total_bytes: int
    payload_bytes: int
    scale_page_bytes: int
    workspace_bytes: int = WORKSPACE_BYTES
    retained_mapper_bytes: int = 0
    prefill_producer: bool = False

    @property
    def arena_bytes(self) -> int:
        return self.workspace_bytes - self.retained_mapper_bytes

    @classmethod
    def create(
        cls,
        available_bytes: int,
        fmt,
        head_dim: int = 512,
        *,
        workspace_bytes: int = WORKSPACE_BYTES,
        retained_mapper_bytes: int = 0,
        prefill_producer: bool = False,
    ):
        from ixformer.inference.functions.deepseek.quantized_kv_format import (
            payload_budget_bytes,
        )

        workspace_bytes = _workspace_bytes(workspace_bytes)
        if not isinstance(prefill_producer, bool):
            raise ValueError("Compressed KV producer role must be a boolean")
        try:
            credit = operator.index(retained_mapper_bytes)
        except TypeError as error:
            raise ValueError(
                "Compressed mapper credit must be a nonnegative integer"
            ) from error
        if isinstance(retained_mapper_bytes, bool) or credit < 0:
            raise ValueError("Compressed mapper credit must be a nonnegative integer")
        if credit >= workspace_bytes:
            raise ValueError(
                "Compressed mapper credit leaves no usable workspace arena"
            )
        scale_page_bytes = fmt.scale_page_bytes(head_dim)
        payload_bytes = payload_budget_bytes(
            available_bytes,
            payload_block_bytes=PAYLOAD_PAGE_BYTES,
            scale_block_bytes=scale_page_bytes,
            fixed_overhead_bytes=workspace_bytes,
        )
        if payload_bytes == 0:
            raise ValueError("Compressed KV budget cannot fit workspace and one page")
        return cls(
            available_bytes,
            payload_bytes,
            scale_page_bytes,
            workspace_bytes,
            credit,
            prefill_producer,
        )

    def validate(self, payload_bytes: int, scale_bytes: int) -> None:
        required = payload_bytes + scale_bytes + self.workspace_bytes
        if required > self.total_bytes:
            raise ValueError(
                "Compressed KV allocation exceeds its profiled budget: "
                f"payload={payload_bytes}, scales={scale_bytes}, "
                f"workspace={self.workspace_bytes}, required={required}, "
                f"available={self.total_bytes}. Check num_gpu_blocks_override "
                "and kv_cache_memory_bytes."
            )


@dataclass(frozen=True)
class ScalePlacement:
    offset_elements: int
    block_stride_elements: int
    rows: int
    groups: int
    blocks_per_page: int = 1


@dataclass(frozen=True)
class ScaleLayout:
    payload_bytes: int
    scale_bytes: int
    num_blocks: int
    placements: dict[str, ScalePlacement]


def plan_scale_layout(config, fmt, head_dim: int = 512) -> ScaleLayout:
    """Mirror physical payload page placement, including overlay aliases."""
    if config.num_blocks <= 0:
        raise ValueError("Compressed KV needs a positive physical block count")
    sizes = {tensor.size for tensor in config.kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError("Compressed KV requires one shared payload allocation")
    payload_bytes = sizes.pop()
    if payload_bytes % (config.num_blocks * PAYLOAD_PAGE_BYTES):
        raise ValueError("Compressed KV payload pool is not a 64 KiB page pattern")
    specs = layer_specs(config.kv_cache_groups)
    rows = fmt.rows_per_page(head_dim)
    groups = fmt.scale_groups(head_dim)
    scale_page_elements = rows * groups
    scale_bytes = payload_bytes // PAYLOAD_PAGE_BYTES * scale_page_elements * 2
    placements = {}
    for tensor in config.kv_cache_tensors:
        for index, name in enumerate(tensor.layers):
            spec = specs[name]
            if not is_quantized_main_spec(spec):
                continue
            blocks_per_page = spec.page_size_bytes // PAYLOAD_PAGE_BYTES
            storage_block = getattr(spec, "storage_block_size", None)
            kernel_rows = (
                spec.get_num_kernel_states(storage_block)
                if storage_block is not None
                else spec.num_states
            )
            if (
                spec.cache_dtype_str != fmt.name
                or spec.dtype is not torch.uint8
                or spec.state_content_size_bytes != fmt.payload_bytes(head_dim)
                or blocks_per_page < 1
                or spec.page_size_bytes % PAYLOAD_PAGE_BYTES
                or spec.num_states != rows * blocks_per_page
                or kernel_rows != rows
            ):
                raise ValueError(
                    f"{name}: incompatible compressed KV page geometry: {spec}"
                )
            offset = tensor.offset + index * tensor.layer_stride
            if offset % PAYLOAD_PAGE_BYTES or tensor.block_stride % PAYLOAD_PAGE_BYTES:
                raise ValueError(
                    f"{name}: compressed payload placement is not page aligned"
                )
            # Match create_kv_cache_views: splitting a manager page into
            # codec pages requires dense pages without inter-layer gaps.
            if blocks_per_page > 1 and tensor.block_stride != spec.page_size_bytes:
                raise ValueError(
                    f"{name}: compressed manager pages must be dense "
                    "for kernel block splitting"
                )
            placement = ScalePlacement(
                offset // PAYLOAD_PAGE_BYTES * scale_page_elements,
                tensor.block_stride
                // blocks_per_page
                // PAYLOAD_PAGE_BYTES
                * scale_page_elements,
                rows,
                groups,
                blocks_per_page,
            )
            last = (
                placement.offset_elements
                + (config.num_blocks * blocks_per_page - 1)
                * placement.block_stride_elements
                + scale_page_elements
            )
            if last * 2 > scale_bytes:
                raise ValueError(
                    f"{name}: scale view exceeds the reserved sidecar pool"
                )
            placements[name] = placement
    return ScaleLayout(payload_bytes, scale_bytes, config.num_blocks, placements)


class WorkspacePool:
    """One worker allocation with ordered stream leases and no context banks."""

    def __init__(self, tensor: torch.Tensor, *, prefill_producer: bool = False):
        if not isinstance(prefill_producer, bool):
            raise ValueError("Compressed KV producer role must be a boolean")
        if (
            tensor.dtype is not torch.uint8
            or tensor.ndim != 1
            or not tensor.is_contiguous()
            or tensor.numel() == 0
        ):
            raise ValueError(
                "Compressed KV workspace must be one nonempty contiguous 1D uint8 span"
            )
        self.storage = tensor
        self.tensor = tensor
        self.prefill_producer = prefill_producer
        self._metadata_reserved = False
        self._event = torch.cuda.Event() if tensor.is_cuda else None
        self._last_recorded_stream = None
        if self._event is not None:
            self._last_recorded_stream = torch.cuda.current_stream(tensor.device)
            self._event.record(self._last_recorded_stream)
        self._leased = False
        self._stream = None
        self._thread = None
        self._forward_depth = 0
        self._lease_capturing = False
        self.capturing = False
        self.restore = RestorePlanTracker()

    def acquire(self, *, nested: bool = False):
        @contextmanager
        def lease():
            if self._leased:
                if nested:
                    if self._thread != threading.get_ident():
                        raise RuntimeError("Nested compressed KV lease changed threads")
                    if self.tensor.is_cuda:
                        self._lease_capturing = torch.cuda.is_current_stream_capturing()
                        if (
                            not self._lease_capturing
                            and torch.cuda.current_stream(self.tensor.device)
                            != self._stream
                        ):
                            raise RuntimeError(
                                "Nested compressed KV lease changed CUDA streams"
                            )
                    yield self.tensor
                    return
                raise RuntimeError(
                    "Compressed KV workspace already has an active lease"
                )
            self._leased = True
            self._thread = threading.get_ident()
            stream = None
            try:
                if self.tensor.is_cuda:
                    self._stream = torch.cuda.current_stream(self.tensor.device)
                    # A graph captures kernel use of this buffer. Replay is
                    # ordered by the outer worker lease around execute_model,
                    # so capture must not record an external-event dependency.
                    self._lease_capturing = torch.cuda.is_current_stream_capturing()
                    if not self._lease_capturing:
                        stream = self._stream
                        # The last writer's same-stream work is already FIFO.
                        # Keep the actual stream alive; handoffs still wait on
                        # the event recorded by every noncapture release.
                        previous = self._last_recorded_stream
                        if (
                            previous is None
                            or previous.device != stream.device
                            or previous != stream
                        ):
                            stream.wait_event(self._event)
                yield self.tensor
            finally:
                if stream is not None:
                    self._event.record(stream)
                    self._last_recorded_stream = stream
                self._leased = False
                self._stream = None
                self._thread = None
                self._lease_capturing = False
                self.restore.invalidate()

        return lease()

    @contextmanager
    def forward_scope(self):
        """Reset host reuse for one eager/warmup/captured local model forward.

        CUDA Graph replay runs the preparation kernels recorded by this scope.
        Actual replay remains ordered by the separate outer worker lease.
        """
        with self.acquire(nested=True) as workspace:
            previous_capture = self.capturing
            if not self._forward_depth:
                # The entering lease already queried this exact stream. Keep
                # one forward result without another API call in eager or in
                # every attention layer. Nested worker leases are also valid
                # when graph capture starts inside the outer lease.
                self.capturing = self._lease_capturing
            self.restore.begin_forward()
            self._forward_depth += 1
            try:
                yield workspace
            finally:
                self._forward_depth -= 1
                self.capturing = previous_capture
                self.restore.invalidate()

    def read_scope(self):
        """Standalone attention gets a scope; model consumers share the outer one."""
        return (
            self.acquire(nested=True) if self._forward_depth else self.forward_scope()
        )

    def main_cache_written(self):
        self.restore.main_cache_written()

    def reserve_metadata(self, byte_count: int):
        if self._metadata_reserved:
            raise RuntimeError("Compressed workspace metadata was already reserved")
        try:
            byte_count = operator.index(byte_count)
        except TypeError as error:
            raise ValueError(
                "Compressed KV metadata bytes must be nonnegative"
            ) from error
        if byte_count < 0:
            raise ValueError("Compressed KV metadata bytes must be nonnegative")
        aligned = ((byte_count + 255) // 256) * 256
        if aligned >= self.storage.numel():
            raise ValueError("Compressed KV metadata exceeds the fixed workspace")
        metadata = self.storage[:byte_count]
        self.tensor = self.storage[aligned:]
        self._metadata_reserved = True
        return metadata


def move_quantized_records(
    rows, scale_rows, cache, scales, slots, positions, compress_ratio, *, format
) -> None:
    """Copy raw payload/FP16 bytes into paired pages without dequantization.

    Validate both planes before the fused replay writer publishes them on
    the current stream, before the caller emits its layer-ready event.
    """
    from ixformer.inference.functions.deepseek.quantized_kv_cache import (
        move_paged_records,
    )
    from ixformer.inference.functions.deepseek.quantized_kv_format import (
        resolve_quant_format,
    )

    fmt = resolve_quant_format(format)
    expected_rows = fmt.rows_per_page(512)
    if (
        rows.dtype is not torch.uint8
        or cache.dtype is not torch.uint8
        or scale_rows.dtype is not torch.float16
        or scales.dtype is not torch.float16
        or rows.ndim != 2
        or scale_rows.ndim != 2
        or cache.ndim != 3
        or scales.ndim != 3
        or rows.shape[1] != fmt.payload_bytes(512)
        or scale_rows.shape[1] != fmt.scale_groups(512)
        or cache.shape[1:] != (expected_rows, rows.shape[1])
        or scales.shape != (cache.shape[0], expected_rows, scale_rows.shape[1])
        or rows.shape[0] != scale_rows.shape[0]
        or not rows.is_contiguous()
        or not scale_rows.is_contiguous()
        or cache.stride(2) != 1
        or scales.stride(2) != 1
        or slots.dtype is not torch.int64
        or slots.ndim != 1
        or not slots.is_contiguous()
        or not positions.is_contiguous()
        or positions.dtype not in (torch.int32, torch.int64)
        or slots.numel() > rows.shape[0]
        or positions.numel() < slots.numel()
        or compress_ratio not in (1, 2)
    ):
        raise ValueError("Incompatible paired compressed-KV PP records")
    devices = {
        value.device for value in (rows, scale_rows, cache, scales, slots, positions)
    }
    if len(devices) != 1:
        raise ValueError("Paired compressed-KV PP records must share a device")
    move_paged_records(
        rows, scale_rows, cache, scales, slots, positions, compress_ratio, format=fmt
    )


class CompressedKVState:
    """Worker-owned scale storage, paired views, and fixed workspace."""

    def __init__(
        self,
        layout: ScaleLayout,
        fmt,
        device,
        forward_context,
        *,
        workspace_bytes: int = WORKSPACE_BYTES,
        prefill_producer: bool = False,
    ):
        workspace_bytes = _workspace_bytes(workspace_bytes)
        if not isinstance(prefill_producer, bool):
            raise ValueError("Compressed KV producer role must be a boolean")
        self.layout = layout
        self.format = fmt
        for name in layout.placements:
            if getattr(forward_context[name], "compressed_kv_format", None) != fmt:
                raise ValueError(f"{name}: worker/layer compressed format mismatch")
        # The upstream runner clears its registry before worker State release.
        # Retain the original layer bindings without copying their tensors.
        self._forward_context = dict(forward_context)
        self.scale_storage = None
        self.workspace = None
        self.scales = {}
        try:
            self.scale_storage = torch.zeros(
                layout.scale_bytes // 2, dtype=torch.float16, device=device
            )
            self.workspace = WorkspacePool(
                torch.empty(workspace_bytes, dtype=torch.uint8, device=device),
                prefill_producer=prefill_producer,
            )
            aliases = {}
            for name, placement in layout.placements.items():
                key = (
                    placement.offset_elements,
                    placement.block_stride_elements,
                    placement.blocks_per_page,
                )
                if key not in aliases:
                    aliases[key] = torch.as_strided(
                        self.scale_storage,
                        (
                            layout.num_blocks * placement.blocks_per_page,
                            placement.rows,
                            placement.groups,
                        ),
                        (placement.block_stride_elements, placement.groups, 1),
                        placement.offset_elements,
                    )
                self.scales[name] = aliases[key]
                layer = forward_context[name]
                layer.kv_cache_scales = self.scales[name]
            for layer in forward_context.values():
                if getattr(layer, "compressed_kv_format", None) == fmt:
                    layer.compressed_kv_workspace = self.workspace
        except BaseException:
            self.release()
            raise

    def bind_zeroer(self, zeroer) -> None:
        """Extend the V2 zeroer using metadata inside the reserved workspace."""
        from vllm.v1.kv_cache_interface import group_kernel_blocks

        addresses, strides, sizes = [], [], []
        if zeroer._meta is not None:
            old_addresses, old_strides, old_sizes, *_ = zeroer._meta
            addresses.extend(old_addresses.cpu().tolist())
            strides.extend(old_strides.cpu().tolist())
            sizes.extend(old_sizes.cpu().tolist())
        seen = set(addresses)
        for scale in self.scales.values():
            address = scale.data_ptr()
            if address in seen:
                continue
            seen.add(address)
            physical = group_kernel_blocks(scale, self.layout.num_blocks)
            addresses.append(address)
            strides.append(physical.stride(0) * 2 // 4)
            sizes.append(physical[0].numel() * 2 // 4)
        count = len(addresses)
        metadata = self.workspace.reserve_metadata(count * 24)
        addr_view = metadata[: count * 8].view(torch.uint64)
        stride_view = metadata[count * 8 : count * 16].view(torch.int64)
        size_view = metadata[count * 16 :].view(torch.int64)
        for view, values in (
            (addr_view, addresses),
            (stride_view, strides),
            (size_view, sizes),
        ):
            view.copy_(torch.tensor(values, dtype=view.dtype, device="cpu"))
        max_page = max(sizes)
        block_size = min(1 << (max_page - 1).bit_length(), 1024)
        zeroer._meta = (
            addr_view,
            stride_view,
            size_view,
            (max_page + block_size - 1) // block_size,
            block_size,
            count,
        )

    def release(self) -> None:
        for name, scale in self.scales.items():
            layer = self._forward_context[name]
            if getattr(layer, "kv_cache_scales", None) is scale:
                layer.kv_cache_scales = None
        for layer in self._forward_context.values():
            if getattr(layer, "compressed_kv_workspace", None) is self.workspace:
                layer.compressed_kv_workspace = None
        self.scales.clear()
        self.scale_storage = None
        self.workspace = None
        self._forward_context.clear()
