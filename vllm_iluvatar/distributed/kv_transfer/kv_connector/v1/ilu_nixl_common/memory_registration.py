# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache memory registration geometry shared across Ilu NIXL connectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import MambaSpec

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

logger = init_logger("vllm." + __name__)


@dataclass(frozen=True)
class PackedLayerInfo:
    """First-block placement of a layer within a shared allocation."""

    offset: int
    block_stride: int


_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1


def validate_64bit_span(
    base_address: int,
    span_bytes: int,
    *,
    label: str = "region",
) -> None:
    """Ensure ``[base_address, base_address + span_bytes]`` fits in 64-bit descriptors."""
    if base_address < 0 or span_bytes < 0:
        raise ValueError(
            f"{label}: negative descriptor base ({base_address}) or span ({span_bytes})"
        )
    last_addr = base_address + span_bytes
    if last_addr > _U64_MAX:
        raise ValueError(
            f"{label}: last descriptor address {last_addr} exceeds 64-bit NIXL range"
        )
    if span_bytes > _U32_MAX:
        logger.debug(
            "%s spans %d bytes (> 4 GiB); using 64-bit NIXL descriptor arithmetic",
            label,
            span_bytes,
        )


@dataclass(frozen=True)
class TensorRegistration:
    """Physical registration and per-block transfer geometry for one view."""

    registration_base: int
    registration_nbytes: int
    block_stride: int
    payload_segments: tuple[tuple[int, int], ...]
    num_blocks: int
    kernel_blocks: int
    device_id: int

    @property
    def descriptor_span(self) -> int:
        """Bytes from the registration base to the end of the last block."""
        return (self.num_blocks - 1) * self.block_stride + max(
            offset + length for offset, length in self.payload_segments
        )

    def validate(self, layer_name: str) -> None:
        if self.registration_base <= 0:
            raise ValueError(f"{layer_name}: invalid registration base address")
        if self.registration_nbytes <= 0:
            raise ValueError(f"{layer_name}: registration size must be positive")
        if self.block_stride <= 0:
            raise ValueError(f"{layer_name}: block stride must be positive")
        if self.num_blocks <= 0:
            raise ValueError(f"{layer_name}: num_blocks must be positive")
        if not self.payload_segments:
            raise ValueError(f"{layer_name}: at least one payload segment is required")

        for offset, length in self.payload_segments:
            if offset < 0 or length <= 0:
                raise ValueError(
                    f"{layer_name}: invalid payload segment ({offset}, {length})"
                )
            if offset % self.block_stride + length > self.block_stride:
                raise ValueError(
                    f"{layer_name}: payload segment ({offset}, {length}) exceeds "
                    f"block stride {self.block_stride}"
                )

        max_end = self.descriptor_span
        if max_end > self.registration_nbytes:
            raise ValueError(
                f"{layer_name}: descriptor span {max_end} exceeds registered "
                f"allocation size {self.registration_nbytes}"
            )

        last_addr = self.registration_base + max_end
        if last_addr > _U64_MAX:
            raise ValueError(
                f"{layer_name}: last descriptor address {last_addr} does not "
                "fit in 64 bits"
            )


class MemoryRegistrationPlanner:
    """Build registration geometry from ``KVCacheConfig`` and tensor views."""

    def __init__(self, kv_cache_config: KVCacheConfig) -> None:
        from vllm_iluvatar.models.deepseek_v41.compressed_kv import layer_specs
        from vllm_iluvatar.models.deepseek_v41.indexer_kv import (
            is_split_indexer_spec,
            plan_indexer_scales,
        )

        groups = getattr(kv_cache_config, "kv_cache_groups", ())
        self._indexer_scale_layout = (
            plan_indexer_scales(kv_cache_config)
            if any(is_split_indexer_spec(spec) for spec in layer_specs(groups).values())
            else None
        )
        self._packing_by_layer: dict[str, PackedLayerInfo] = {}
        self._allocation_sizes_by_layer: dict[str, int] = {}
        for tensor_config in kv_cache_config.kv_cache_tensors:
            if tensor_config.block_stride <= 0:
                continue
            if tensor_config.offset < 0:
                raise ValueError("packed KV tensor offset must be non-negative")
            if tensor_config.offset >= tensor_config.size:
                raise ValueError(
                    "packed KV tensor offset must be smaller than allocation size"
                )
            layer_stride = getattr(tensor_config, "layer_stride", 0) or 0
            members = getattr(tensor_config, "layers", None)
            if members is None:
                members = getattr(tensor_config, "shared_by", ())
            for layer_idx, layer_name in enumerate(members):
                self._packing_by_layer[layer_name] = PackedLayerInfo(
                    offset=tensor_config.offset + layer_idx * layer_stride,
                    block_stride=tensor_config.block_stride,
                )
                self._allocation_sizes_by_layer[layer_name] = tensor_config.size

    def get_packing(self, layer_name: str) -> PackedLayerInfo | None:
        return self._packing_by_layer.get(layer_name)

    @staticmethod
    def kernel_blocks_per_block(
        *,
        layer_name: str,
        tensor: torch.Tensor,
        spec: KVCacheSpec,
        num_blocks: int,
    ) -> int:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if isinstance(spec, MambaSpec) or tensor.ndim < 2:
            return 1
        rows = tensor.shape[0]
        if rows % num_blocks:
            raise ValueError(
                f"{layer_name}: {rows} kernel blocks are not a multiple of "
                f"{num_blocks} scheduler blocks (shape={tuple(tensor.shape)}, "
                f"stride={tuple(tensor.stride())})"
            )
        return rows // num_blocks

    @staticmethod
    def kernel_row_bytes(*, layer_name: str, tensor: torch.Tensor) -> int:
        row = tensor[0]
        extent = 1 + sum(
            (size - 1) * stride
            for size, stride in zip(row.shape, row.stride())
            if size > 0
        )
        if any(stride < 0 for stride in row.stride()) or extent != row.numel():
            raise ValueError(
                f"{layer_name}: a kernel block must be one dense byte range to "
                f"be transferred as one payload (shape={tuple(tensor.shape)}, "
                f"stride={tuple(tensor.stride())})"
            )
        return extent * tensor.element_size()

    @staticmethod
    def scheduler_block_payload_bytes(
        *,
        layer_name: str,
        tensor: torch.Tensor,
        spec: KVCacheSpec,
        kernel_blocks: int,
        block_stride: int,
    ) -> int:
        if isinstance(spec, MambaSpec) or tensor.ndim < 2:
            return block_stride
        kernel_stride = tensor.stride(0) * tensor.element_size()
        row_bytes = MemoryRegistrationPlanner.kernel_row_bytes(
            layer_name=layer_name, tensor=tensor
        )
        return (kernel_blocks - 1) * kernel_stride + row_bytes

    @staticmethod
    def tensor_block_stride_bytes(tensor: torch.Tensor) -> int | None:
        if tensor.ndim < 2:
            return None
        return tensor.stride(0) * tensor.element_size()

    @staticmethod
    def validate_block_geometry(
        *,
        layer_name: str,
        tensor: torch.Tensor,
        block_stride: int,
        num_blocks: int,
        payload_segments: tuple[tuple[int, int], ...],
        kernel_blocks: int,
    ) -> None:
        if kernel_blocks <= 0:
            raise ValueError(f"{layer_name}: kernel_blocks must be positive")
        tensor_stride = MemoryRegistrationPlanner.tensor_block_stride_bytes(tensor)
        if tensor_stride is not None and tensor_stride * kernel_blocks != block_stride:
            raise ValueError(
                f"{layer_name}: block stride disagreement: registered "
                f"{block_stride} vs tensor-derived {tensor_stride} x "
                f"{kernel_blocks} kernel blocks "
                f"(shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}, "
                f"itemsize={tensor.element_size()}, num_blocks={num_blocks})"
            )

        for offset, length in payload_segments:
            if offset % block_stride + length > block_stride:
                raise ValueError(f"{layer_name}: payload crosses a block boundary")
        payload_end = max(offset + length for offset, length in payload_segments)
        descriptor_end = (num_blocks - 1) * block_stride + payload_end
        if descriptor_end > tensor.untyped_storage().nbytes():
            raise ValueError(
                f"{layer_name}: descriptor span exceeds the backing allocation"
            )

        span = num_blocks * block_stride
        if span > _U32_MAX:
            logger.warning_once(
                "NIXL region spans more than 4 GiB; every address, length "
                "and stride on the path to NIXL must be 64-bit"
            )

    def describe(
        self,
        *,
        layer_name: str,
        tensor: torch.Tensor,
        spec: KVCacheSpec,
        num_blocks: int,
    ) -> TensorRegistration:
        kernel_blocks = self.kernel_blocks_per_block(
            layer_name=layer_name, tensor=tensor, spec=spec, num_blocks=num_blocks
        )
        storage = tensor.untyped_storage()
        storage_base = storage.data_ptr()
        storage_nbytes = storage.nbytes()
        view_offset = tensor.data_ptr() - storage_base

        packing = self.get_packing(layer_name)
        if packing is None:
            block_stride = spec.page_size_bytes
            declared = f"spec page size {block_stride}"
            expected_nbytes = num_blocks * block_stride
        else:
            block_stride = packing.block_stride
            declared = f"KVCacheConfig block_stride {block_stride}"
            expected_nbytes = self._allocation_sizes_by_layer[layer_name]
            if view_offset != packing.offset:
                raise ValueError(
                    f"{layer_name}: KVCacheConfig offset {packing.offset} does "
                    f"not match tensor view offset {view_offset}"
                )
        if storage_nbytes != expected_nbytes:
            raise ValueError(
                f"{layer_name}: storage size {storage_nbytes} is not "
                f"declared allocation size {expected_nbytes} ({declared}) "
                f"(shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())})"
            )

        payload = self.scheduler_block_payload_bytes(
            layer_name=layer_name,
            tensor=tensor,
            spec=spec,
            kernel_blocks=kernel_blocks,
            block_stride=block_stride,
        )
        result = TensorRegistration(
            registration_base=storage_base,
            registration_nbytes=storage_nbytes,
            block_stride=block_stride,
            payload_segments=((view_offset, payload),),
            num_blocks=num_blocks,
            kernel_blocks=kernel_blocks,
            device_id=max(tensor.get_device(), 0),
        )
        result.validate(layer_name)
        self.validate_block_geometry(
            layer_name=layer_name,
            tensor=tensor,
            block_stride=block_stride,
            num_blocks=num_blocks,
            payload_segments=result.payload_segments,
            kernel_blocks=kernel_blocks,
        )
        return result

    def describe_indexer_scale_plane(
        self,
        *,
        layer_name: str,
        scales: torch.Tensor,
        payload: TensorRegistration,
    ) -> TensorRegistration:
        if (
            scales.dtype != torch.float32
            or scales.ndim != 2
            or scales.shape[0] != payload.num_blocks * payload.kernel_blocks
            or scales.stride(1) != 1
            or scales.stride(0) != scales.shape[1]
        ):
            raise ValueError(f"{layer_name}: invalid paired indexer scale geometry")
        page_bytes = scales.shape[1] * 4
        storage = scales.untyped_storage()
        offset = scales.data_ptr() - storage.data_ptr()
        stride = scales.stride(0) * 4 * payload.kernel_blocks
        layout = self._indexer_scale_layout
        placement = layout.placements.get(layer_name) if layout is not None else None
        if (
            placement is None
            or offset != placement.offset * 4
            or stride != placement.rows_per_manager_block * 4
            or storage.nbytes() != layout.nbytes
        ):
            raise ValueError(
                f"{layer_name}: indexer scales do not match payload ownership"
            )
        result = TensorRegistration(
            registration_base=storage.data_ptr(),
            registration_nbytes=storage.nbytes(),
            block_stride=stride,
            payload_segments=((offset, page_bytes * payload.kernel_blocks),),
            num_blocks=payload.num_blocks,
            kernel_blocks=payload.kernel_blocks,
            device_id=max(scales.get_device(), 0),
        )
        result.validate(layer_name)
        MemoryRegistrationPlanner.validate_block_geometry(
            layer_name=layer_name,
            tensor=scales,
            block_stride=stride,
            num_blocks=payload.num_blocks,
            payload_segments=result.payload_segments,
            kernel_blocks=payload.kernel_blocks,
        )
        return result

    @staticmethod
    def describe_scale_plane(
        *,
        layer_name: str,
        scales: torch.Tensor,
        payload: TensorRegistration,
        scale_page_bytes: int,
    ) -> TensorRegistration:
        from vllm_iluvatar.models.deepseek_v41.compressed_kv import PAYLOAD_PAGE_BYTES

        if scales.dtype is not torch.float16 or scales.ndim != 3:
            raise ValueError(
                f"{layer_name}: scale plane must be FP16[pages,rows,groups]"
            )
        if scales.shape[0] != payload.num_blocks * payload.kernel_blocks:
            raise ValueError(
                f"{layer_name}: scale/payload physical block ranges differ"
            )
        if (
            MemoryRegistrationPlanner.kernel_row_bytes(
                layer_name=layer_name, tensor=scales
            )
            != scale_page_bytes
        ):
            raise ValueError(f"{layer_name}: scale page size does not match its format")
        storage = scales.untyped_storage()
        offset = scales.data_ptr() - storage.data_ptr()
        stride = scales.stride(0) * scales.element_size() * payload.kernel_blocks
        ((payload_offset, payload_length),) = payload.payload_segments
        payload_block_bytes = PAYLOAD_PAGE_BYTES * payload.kernel_blocks
        scale_block_bytes = scale_page_bytes * payload.kernel_blocks
        if (
            payload_length != payload_block_bytes
            or payload_offset % PAYLOAD_PAGE_BYTES
            or payload.block_stride % PAYLOAD_PAGE_BYTES
            or offset != payload_offset // PAYLOAD_PAGE_BYTES * scale_page_bytes
            or stride != payload.block_stride // PAYLOAD_PAGE_BYTES * scale_page_bytes
            or storage.nbytes()
            != payload.registration_nbytes // PAYLOAD_PAGE_BYTES * scale_page_bytes
        ):
            raise ValueError(
                f"{layer_name}: scale plane does not mirror payload ownership/layout"
            )
        result = TensorRegistration(
            registration_base=storage.data_ptr(),
            registration_nbytes=storage.nbytes(),
            block_stride=stride,
            payload_segments=((offset, scale_block_bytes),),
            num_blocks=payload.num_blocks,
            kernel_blocks=payload.kernel_blocks,
            device_id=max(scales.get_device(), 0),
        )
        result.validate(layer_name)
        MemoryRegistrationPlanner.validate_block_geometry(
            layer_name=layer_name,
            tensor=scales,
            block_stride=stride,
            num_blocks=payload.num_blocks,
            payload_segments=result.payload_segments,
            kernel_blocks=payload.kernel_blocks,
        )
        return result


def validate_registered_allocation_alias(
    *,
    base_address: int,
    registered_nbytes: int,
    expected_nbytes: int,
    layer_name: str,
) -> None:
    """Validate two views that alias one NIXL registration."""
    if registered_nbytes != expected_nbytes:
        raise ValueError(
            f"{layer_name}: allocation at {base_address} was already registered "
            f"with {registered_nbytes} bytes, requested {expected_nbytes} bytes"
        )
