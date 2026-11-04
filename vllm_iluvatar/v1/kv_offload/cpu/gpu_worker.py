# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/kv_offload/cpu/gpu_worker.py

Target   : CPUOffloadingWorker.__init__, CPUOffloadingWorker.shutdown
Baseline : v0.31.0 (341bf7d4d124)
Why      : Avoid CoreX shared-memory/power-of-two host allocations for private pages.
How      : Guarded worker method copies; delegate shared-mmap paths.
Scope    : Generic native CPU-offload private storage; no model environment switches.
Remove   : When community owns equivalent private pinned storage and cleanup.
"""

from __future__ import annotations

import contextlib
import errno
import mmap
import time
import weakref

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_iluvatar.ilu_util.mmap_utils import mmap_byte_array

logger = init_logger(__name__)

_ORIGINAL_WORKER_INIT = None
_ORIGINAL_WORKER_SHUTDOWN = None
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)


def _populate_anonymous_mmap(mmap_obj: mmap.mmap, length: int) -> None:
    if length <= 0:
        return
    try:
        mmap_obj.madvise(_MADV_POPULATE_WRITE, 0, length)
    except OSError as exc:
        if exc.errno != errno.EINVAL:
            raise
        arr = np.frombuffer(mmap_obj, dtype=np.uint8)
        arr[: length : mmap.PAGESIZE] = 0


def _cudart_host_register(address: int, size: int) -> int:
    cudart = torch.cuda.cudart()
    if cudart is None:
        raise RuntimeError("CUDA runtime is unavailable for host registration")
    result = cudart.cudaHostRegister(address, size, 0)
    return int(getattr(result, "value", result))


def _cudart_host_unregister(address: int) -> int:
    cudart = torch.cuda.cudart()
    if cudart is None:
        return 0
    result = cudart.cudaHostUnregister(address)
    return int(getattr(result, "value", result))


def _clear_cuda_last_error() -> None:
    with contextlib.suppress(Exception):
        cudart = torch.cuda.cudart()
        if cudart is not None and hasattr(cudart, "cudaGetLastError"):
            cudart.cudaGetLastError()


def _unregister_addresses(addresses: list[int]) -> None:
    while addresses:
        address = addresses.pop()
        try:
            code = _cudart_host_unregister(address)
        except Exception:
            logger.warning(
                "cudaHostUnregister failed for private offload address=%#x",
                address,
                exc_info=True,
            )
            continue
        if code != 0:
            _clear_cuda_last_error()
            logger.warning(
                "cudaHostUnregister failed for private offload address=%#x (code=%d)",
                address,
                code,
            )


def _close_mmap_quietly(mmap_obj: mmap.mmap | None) -> None:
    if mmap_obj is not None and not mmap_obj.closed:
        with contextlib.suppress(Exception):
            mmap_obj.close()


def _cleanup_private_mmap(
    addresses: list[int],
    mmap_obj: mmap.mmap | None,
) -> None:
    _unregister_addresses(addresses)
    _close_mmap_quietly(mmap_obj)


class _PrivateOffloadBuffer:
    """Page-aligned anonymous private mmap pinned in-place via cudaHostRegister.

    Bypasses PyTorch's CUDACachingHostAllocator (which rounds every
    `pin_memory=True` allocation up to the next power of 2 and allocates from
    POSIX shared memory on CoreX).
    """

    def __init__(
        self,
        mmap_obj: mmap.mmap | None,
        owner: np.ndarray | None,
        addresses: list[int],
        *,
        is_pinned: bool,
        tensor: torch.Tensor | None = None,
    ) -> None:
        self.mmap_obj = mmap_obj
        self.owner = owner
        self.pinned_addresses = addresses
        self.is_pinned = is_pinned
        self._tensor_ref = None
        if tensor is not None:
            with contextlib.suppress(TypeError):
                self._tensor_ref = weakref.ref(tensor)
        self._finalizer = None
        if owner is not None and mmap_obj is not None:
            self._finalizer = weakref.finalize(
                owner,
                _cleanup_private_mmap,
                addresses,
                mmap_obj,
            )
            self._finalizer.atexit = False

    def cleanup(self) -> None:
        self.is_pinned = False
        if self._tensor_ref is not None:
            tensor = self._tensor_ref()
            self._tensor_ref = None
            if tensor is not None and hasattr(tensor, "set_"):
                with contextlib.suppress(Exception):
                    tensor.set_(torch.empty(0, dtype=tensor.dtype, device="cpu"))
            tensor = None
        self.owner = None
        # The address-backed NumPy array deliberately has no mmap buffer
        # export. Explicit shutdown must obey the same storage ownership as
        # GC: derived Tensor views may still keep the pinned pages alive.
        if self._finalizer is not None and self._finalizer.alive:
            return
        _unregister_addresses(self.pinned_addresses)
        _close_mmap_quietly(self.mmap_obj)
        if self._finalizer is not None and (
            self.mmap_obj is None or self.mmap_obj.closed
        ):
            self._finalizer.detach()


def _allocate_private_cpu_tensor(
    num_cpu_chunks: int,
    cpu_page_size_bytes: int,
    *,
    pin_memory: bool,
    max_register_chunk_bytes: int,
) -> tuple[torch.Tensor, _PrivateOffloadBuffer]:
    from vllm.platforms import current_platform

    nbytes = int(num_cpu_chunks) * int(cpu_page_size_bytes)
    if nbytes <= 0:
        empty = torch.zeros(
            (num_cpu_chunks, cpu_page_size_bytes),
            dtype=torch.int8,
            device="cpu",
        )
        return empty, _PrivateOffloadBuffer(None, None, [], is_pinned=bool(pin_memory))

    page_size = mmap.PAGESIZE
    aligned_bytes = ((nbytes + page_size - 1) // page_size) * page_size
    mmap_obj = mmap.mmap(
        -1,
        aligned_bytes,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    owner = None
    tensor = None
    try:
        _populate_anonymous_mmap(mmap_obj, aligned_bytes)
        owner = mmap_byte_array(mmap_obj, nbytes).view(np.int8)
        tensor = torch.from_numpy(owner).view(num_cpu_chunks, cpu_page_size_bytes)
    except BaseException:
        tensor = None
        owner = None
        mmap_obj.close()
        raise

    addresses: list[int] = []
    is_pinned = False
    if pin_memory and current_platform.is_cuda_alike():
        base_ptr = tensor.data_ptr()
        chunk_bytes = max(
            (max_register_chunk_bytes // page_size) * page_size, page_size
        )
        try:
            for offset in range(0, aligned_bytes, chunk_bytes):
                address = base_ptr + offset
                size = min(chunk_bytes, aligned_bytes - offset)
                code = _cudart_host_register(address, size)
                if code != 0:
                    _clear_cuda_last_error()
                    logger.warning(
                        "cudaHostRegister failed at %.2f of %.2f GiB (code=%d); "
                        "private offload tensor stays pageable",
                        offset / (1 << 30),
                        aligned_bytes / (1 << 30),
                        code,
                    )
                    _unregister_addresses(addresses)
                    break
                addresses.append(address)
            else:
                is_pinned = True
        except Exception:
            logger.warning(
                "Could not register private offload tensor with CUDA runtime; "
                "the tensor stays pageable",
                exc_info=True,
            )
            _unregister_addresses(addresses)

    return tensor, _PrivateOffloadBuffer(
        mmap_obj,
        owner,
        addresses,
        is_pinned=is_pinned,
        tensor=tensor,
    )


# [ILU override] CPUOffloadingWorker.__init__ — delta: private allocation and lifetime cleanup.
# >>> ILU MODIFICATION >>>
def _ilu_cpu_offloading_worker_init(
    self,
    kv_caches,
    blocks_per_chunk: int,
    num_cpu_chunks: int,
    mmap_region=None,
    canonical_layout: bool = False,
) -> None:
    if mmap_region is not None:
        assert _ORIGINAL_WORKER_INIT is not None
        _ORIGINAL_WORKER_INIT(
            self,
            kv_caches=kv_caches,
            blocks_per_chunk=blocks_per_chunk,
            num_cpu_chunks=num_cpu_chunks,
            mmap_region=mmap_region,
            canonical_layout=canonical_layout,
        )
        return

    from vllm.v1.kv_offload.cpu import gpu_worker

    assert not canonical_layout
    self._mmap_region = None
    self._ilu_private_regions = []
    pin_memory = bool(gpu_worker.PIN_MEMORY)
    gpu_worker.logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))

    gpu_tensors: list[torch.Tensor] = []
    cpu_tensors: list[torch.Tensor] = []
    gpu_tensor = None
    cpu_tensor = None
    region = None
    try:
        for kv_cache_tensor in kv_caches.tensors:
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * blocks_per_chunk
            t0 = time.monotonic()
            cpu_tensor, region = _allocate_private_cpu_tensor(
                num_cpu_chunks,
                cpu_page_size_bytes,
                pin_memory=pin_memory,
                max_register_chunk_bytes=gpu_worker.MAX_HOST_REGISTER_CHUNK_BYTES,
            )
            gpu_worker.logger.debug(
                "anonymous mmap pinned tensor %d×%d (%.2f GiB, pinned=%s): %.3f s",
                num_cpu_chunks,
                cpu_page_size_bytes,
                num_cpu_chunks * cpu_page_size_bytes / (1 << 30),
                region.is_pinned,
                time.monotonic() - t0,
            )
            self._ilu_private_regions.append(region)
            gpu_tensors.append(gpu_tensor)
            cpu_tensors.append(cpu_tensor)
            gpu_tensor = None
            cpu_tensor = None
            region = None

        host_memory_is_pinned = pin_memory and all(
            r.is_pinned for r in self._ilu_private_regions
        )
        self._store_handler = gpu_worker.SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            canonical_layout=False,
            host_memory_is_pinned=host_memory_is_pinned,
        )
        self._load_handler = gpu_worker.SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=False,
            canonical_layout=False,
            host_memory_is_pinned=host_memory_is_pinned,
        )
    except BaseException:
        gpu_tensor = None
        cpu_tensor = None
        region = None
        gpu_tensors.clear()
        cpu_tensors.clear()
        for attr in ("_store_handler", "_load_handler"):
            handler = getattr(self, attr, None)
            if handler is not None:
                with contextlib.suppress(Exception):
                    handler.shutdown()
                for list_attr in (
                    "src_tensors",
                    "dst_tensors",
                    "gpu_tensors",
                    "cpu_tensors",
                ):
                    tensor_list = getattr(handler, list_attr, None)
                    if isinstance(tensor_list, list):
                        tensor_list.clear()
                setattr(self, attr, None)
        for priv_region in reversed(self._ilu_private_regions):
            priv_region.cleanup()
        self._ilu_private_regions.clear()
        raise


# <<< ILU MODIFICATION <<<


# [ILU override] CPUOffloadingWorker.shutdown — delta: private allocation and lifetime cleanup.
# >>> ILU MODIFICATION >>>
def _ilu_cpu_offloading_worker_shutdown(self) -> None:
    regions = getattr(self, "_ilu_private_regions", None)
    if not regions:
        assert _ORIGINAL_WORKER_SHUTDOWN is not None
        _ORIGINAL_WORKER_SHUTDOWN(self)
        return

    handler_failed = False
    try:
        store_handler = getattr(self, "_store_handler", None)
        if store_handler is not None:
            try:
                store_handler.shutdown()
            except Exception:
                logger.exception("Failed to shut down store offloading handler")
                handler_failed = True

        load_handler = getattr(self, "_load_handler", None)
        if load_handler is not None:
            try:
                load_handler.shutdown()
            except Exception:
                logger.exception("Failed to shut down load offloading handler")
                handler_failed = True
    except BaseException:
        handler_failed = True
        raise
    finally:
        for handler in (
            getattr(self, "_store_handler", None),
            getattr(self, "_load_handler", None),
        ):
            if handler is not None:
                for list_attr in (
                    "src_tensors",
                    "dst_tensors",
                    "gpu_tensors",
                    "cpu_tensors",
                ):
                    tensor_list = getattr(handler, list_attr, None)
                    if isinstance(tensor_list, list):
                        tensor_list.clear()
        if handler_failed:
            try:
                torch.accelerator.synchronize()
            except Exception:
                logger.warning(
                    "Device sync before private offload cleanup failed; "
                    "proceeding with cleanup anyway",
                    exc_info=True,
                )
        for region in reversed(regions):
            region.cleanup()
        regions.clear()


# <<< ILU MODIFICATION <<<
