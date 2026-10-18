"""KV cache memory registration geometry.

Every KV view is registered the same way, whatever its layout:

* registration span: the whole backing storage, registered once per storage;
* block stride: bytes between scheduler block ``b`` and ``b + 1``;
* payload: the bytes of one scheduler block that are actually transferred,
  at ``view offset`` inside the block.

All three are read off the view itself and cross-checked against what vLLM
declared: ``KVCacheConfig`` for a layer placed in a shared per-block slab
(packed), otherwise the spec's page size.  Plain, packed and Mamba views only
differ in the numbers, not in the code path.

A registered block is always one *scheduler* block of the owning group, never
a kernel block.  When a backend views a scheduler block as ``r`` smaller
kernel blocks, the registered stride is ``r`` kernel strides and the payload
spans all ``r`` kernel rows, so scheduler block ids index the descriptors
without any remapping (see ``kernel_blocks_per_block``).
"""

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
    """Placement of a layer inside a packed per-block slab."""

    offset: int
    block_stride: int


#: NIXL descriptor addresses and lengths must stay exact in 64-bit.  A single
#: DeepSeek-V4.1 KV region is routinely larger than 4 GiB, so any 32-bit
#: truncation on the way to the NIXL C API silently corrupts the transfer
#: instead of failing.  Registration asserts the span fits in 64 bits and warns
#: once it no longer fits in 32, which is the point where a narrow field
#: anywhere in the path would start wrapping.
_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1


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
            if offset + length > self.block_stride:
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
        self._packing_by_layer: dict[str, PackedLayerInfo] = {}
        for tensor_config in kv_cache_config.kv_cache_tensors:
            if tensor_config.block_stride <= 0:
                continue
            if tensor_config.offset < 0:
                raise ValueError("packed KV tensor offset must be non-negative")
            if tensor_config.offset >= tensor_config.block_stride:
                raise ValueError(
                    "packed KV tensor offset must be smaller than block_stride"
                )
            # 0.30 renamed KVCacheTensor.shared_by to .layers and added a
            # third geometry field, .layer_stride. Each layer view starts at
            # its own position in the packed scheduler-block slab.
            for layer_index, layer_name in enumerate(tensor_config.layers):
                placement = PackedLayerInfo(
                    offset=(
                        tensor_config.offset + layer_index * tensor_config.layer_stride
                    ),
                    block_stride=tensor_config.block_stride,
                )
                previous = self._packing_by_layer.setdefault(layer_name, placement)
                if previous != placement:
                    raise ValueError(
                        f"{layer_name}: conflicting packed KV placements: "
                        f"{previous} vs {placement}"
                    )

    @property
    def has_packed_layers(self) -> bool:
        return bool(self._packing_by_layer)

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
        """Kernel blocks that make up one scheduler block of this view.

        The scheduler allocates each group in units of its own
        ``spec.block_size``, while the backend may view the same bytes as ``r``
        smaller kernel blocks.  DeepSeek-V4.1 at TP2/PP8 has ``r`` = 1, 2, 4
        and 8 in different groups of one rank (all views alias one storage of
        ``num_blocks`` pages).  Kernel block ``b * r + j`` (``j < r``) belongs
        to scheduler block ``b``, so ``r`` is read off the view instead of
        from a single worker-wide ratio.
        """

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
        """Bytes of one kernel block, which must be one dense byte range.

        The row may be permuted (e.g. an HND stride order) but must not have
        holes, otherwise one ``(offset, length)`` segment would carry bytes
        of a neighbouring row.
        """

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
        """Bytes one scheduler block covers in this view.

        ``r - 1`` whole kernel strides plus the last kernel row, so a padded
        kernel stride does not drag the padding after the last row along.
        A Mamba state page and a flat 1-D view are moved as a whole block.
        """

        if isinstance(spec, MambaSpec) or tensor.ndim < 2:
            return block_stride
        kernel_stride = tensor.stride(0) * tensor.element_size()
        row_bytes = MemoryRegistrationPlanner.kernel_row_bytes(
            layer_name=layer_name, tensor=tensor
        )
        return (kernel_blocks - 1) * kernel_stride + row_bytes

    @staticmethod
    def tensor_block_stride_bytes(tensor: torch.Tensor) -> int | None:
        """Byte distance between consecutive blocks, read off the tensor.

        Copied from the 0.30 upstream NIXL worker
        (``nixl/base_worker.py``, ``register_kv_caches``) rather than calling
        it, so that upstream changes show up as a diff here:

            if cache.ndim == 1:
                # Flat byte view: HMA tensors shared between layer types carry
                # no block dimension, so stride(0) is 1 byte.
                block_stride = physical_page_size
            else:
                block_stride = cache.stride(0) * cache.element_size()

        A 1-D flat byte view carries no block dimension, so its ``stride(0)``
        is one byte and the caller's spec-derived value is the only source.
        Returns ``None`` in that case.
        """

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
        """Cross-check the registered geometry against the tensor itself.

        The registered stride is one scheduler block, i.e. ``kernel_blocks``
        kernel strides, where the kernel stride is ``stride(0) *
        element_size()`` read straight off the tensor as upstream 0.30 does.
        The declared stride (spec page size, or ``KVCacheConfig.block_stride``
        for a packed layer) must land on that value; they disagree exactly
        where it matters (shared slabs, padded pages, a kernel/scheduler block
        size mismatch).  Registration is the only place where both numbers are
        available, so a mismatch is reported here with both values instead of
        turning into a wrong remote address later.
        """

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

        payload_end = max(offset + length for offset, length in payload_segments)
        if payload_end > block_stride:
            raise ValueError(
                f"{layer_name}: payload end {payload_end} exceeds block stride "
                f"{block_stride}"
            )

        span = num_blocks * block_stride
        if span > _U32_MAX:
            logger.warning_once(
                "%s: NIXL region spans %d bytes (> 4 GiB); every address, "
                "length and stride on the path to NIXL must be 64-bit "
                "(num_blocks=%d, block_stride=%d)",
                layer_name,
                span,
                num_blocks,
                block_stride,
            )

    def describe(
        self,
        *,
        layer_name: str,
        tensor: torch.Tensor,
        spec: KVCacheSpec,
        num_blocks: int,
    ) -> TensorRegistration:
        """Describe how a view is registered and indexed per scheduler block."""

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
        else:
            block_stride = packing.block_stride
            declared = f"KVCacheConfig block_stride {block_stride}"
            if view_offset != packing.offset:
                raise ValueError(
                    f"{layer_name}: KVCacheConfig offset {packing.offset} does "
                    f"not match tensor view offset {view_offset}"
                )
        if storage_nbytes != num_blocks * block_stride:
            raise ValueError(
                f"{layer_name}: storage size {storage_nbytes} is not "
                f"num_blocks {num_blocks} x {declared} "
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
