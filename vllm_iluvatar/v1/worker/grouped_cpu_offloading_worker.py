# SPDX-License-Identifier: Apache-2.0
"""Private group arenas using the unmodified native transfer handlers."""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import CanonicalKVCaches
from vllm.v1.kv_offload.cpu.gpu_worker import (
    PIN_MEMORY,
    CPUOffloadingWorker,
    SingleDirectionOffloadingHandler,
)
from vllm_iluvatar.v1.worker.pinned_tensor_storage import allocate_pinned_tensor_views

logger = init_logger(__name__)


class GroupCPUOffloadingWorker(CPUOffloadingWorker):
    """Allocate per-plane capacities; inherit native transfer and shutdown APIs."""

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        cpu_chunks_per_tensor: list[int],
    ):
        if blocks_per_chunk <= 0:
            raise ValueError("CPU blocks_per_chunk must be positive")
        if (
            not kv_caches.tensors
            or len(cpu_chunks_per_tensor) != len(kv_caches.tensors)
            or any(n <= 0 for n in cpu_chunks_per_tensor)
        ):
            raise ValueError("Invalid per-tensor CPU chunk capacities")
        self._mmap_region = None
        gpu_tensors, cpu_shapes = [], []
        for tensor, chunks in zip(kv_caches.tensors, cpu_chunks_per_tensor):
            page_bytes = tensor.page_size_bytes
            gpu_tensors.append(tensor.tensor.view(torch.int8).view(-1, page_bytes))
            cpu_shapes.append((chunks, page_bytes * blocks_per_chunk))
        cpu_tensors, layout = allocate_pinned_tensor_views(
            cpu_shapes, pin_memory=PIN_MEMORY
        )
        logger.info(
            "CPU group planes allocated in plugin: tensors=%d, logical_bytes=%d, "
            "torch_storage_bytes=%d, torch_arenas=%s, rows=%s",
            len(cpu_tensors),
            sum(t.numel() * t.element_size() for t in cpu_tensors),
            layout.storage_bytes,
            layout.arena_bytes,
            tuple(cpu_chunks_per_tensor),
        )
        handler_args = dict(
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            canonical_layout=False,
            host_memory_is_pinned=PIN_MEMORY,
        )
        # Shutdown clears each direction's tensor lists after draining it.
        # Keep distinct lists so one direction cannot drop the other's views
        # while that direction still has asynchronous copies in flight.
        self._store_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=list(gpu_tensors),
            cpu_tensors=list(cpu_tensors),
            gpu_to_cpu=True,
            **handler_args,
        )
        self._load_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=list(gpu_tensors),
            cpu_tensors=list(cpu_tensors),
            gpu_to_cpu=False,
            **handler_args,
        )
