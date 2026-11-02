# SPDX-License-Identifier: Apache-2.0
"""Configuration checks and shared checkpoint storage for Iluvatar Engram."""

import contextlib
import mmap
import os
import resource
import socket
import warnings
import weakref

import numpy as np
import torch

from vllm_iluvatar.ilu_util.mmap_utils import mmap_byte_array


def validate_engram_config(vllm_config):
    """Reject unsupported storage before workers allocate model weights."""
    model = vllm_config.model_config
    if model is None or os.getenv("VLLM_ILU_V41_DISABLE_ENGRAM", "0") != "0":
        return
    hf = model.hf_text_config
    if not getattr(hf, "engram_layer_ids", None):
        return
    config = vllm_config.engram_config
    if config is None:
        raise ValueError("Iluvatar Engram requires EngramConfig")
    if not config.cpu_offload or os.getenv("VLLM_PLE_CPU_OFFLOAD") == "0":
        raise ValueError(
            "Iluvatar Engram requires cpu_offload=True: the approximately "
            "189 GiB Flash tables do not fit in V300 HBM"
        )
    if config.use_thp:
        raise ValueError("Iluvatar Engram use_thp is not supported")
    dp_size = vllm_config.parallel_config.data_parallel_size
    if dp_size > 1 and not config.dp_shared_memory and not config.embedding_across_dp:
        raise ValueError(
            "Iluvatar Engram forbids per-replica host table copies; enable "
            "dp_shared_memory or embedding_across_dp"
        )
    shards = vllm_config.parallel_config.tensor_parallel_size
    if config.embedding_across_dp and not config.dp_shared_memory:
        shards *= dp_size
    columns = (hf.engram_max_ngram_size - 1) * hf.engram_n_heads
    assert (shards - 1) * ((columns + shards - 1) // shards) < columns, (
        "Engram ranks without hash heads"
    )


def check_shared_group(group):
    """Require all sharing peers to be on the same host and IPC namespace."""
    identity = (socket.gethostname(), os.readlink("/proc/self/ns/ipc"))
    peers = [None] * group.world_size
    torch.distributed.all_gather_object(peers, identity, group=group.cpu_group)
    if any(peer != identity for peer in peers):
        raise RuntimeError(
            "Engram dp_shared_memory requires same-node DP replicas with a shared "
            f"IPC namespace; peers={peers}"
        )


def map_shared_checkpoint_rows(path, name, start, end, group, entry):
    """Map checkpoint rows read-only (`MAP_SHARED`) so all peers share page cache.

    Mapping the safetensors file directly with `MAP_SHARED` and `PROT_READ`
    lets all DP replicas (and co-located Prefill workers on the same host) share
    one kernel page-cache copy instead of duplicating ~189 GiB into `/dev/shm`.
    """
    offset, shape, dtype = entry(path, name)
    if not 0 <= start < end <= shape[0]:
        raise ValueError(f"{name}: invalid row interval {start}:{end} for {shape}")
    itemsize = torch.empty((), dtype=dtype, device="cpu").element_size()
    row_bytes = int(np.prod(shape[1:])) * itemsize
    num_bytes = (end - start) * row_bytes
    offset += start * row_bytes
    map_offset = offset - offset % mmap.ALLOCATIONGRANULARITY
    skew = offset - map_offset
    mapping = owner = tensor = view = finalizer = None
    error = None
    try:
        with contextlib.suppress(OSError, ValueError):
            resource.setrlimit(
                resource.RLIMIT_MEMLOCK,
                (resource.RLIM_INFINITY, resource.RLIM_INFINITY),
            )
        with open(path, "rb") as source:
            mapping = mmap.mmap(
                source.fileno(),
                num_bytes + skew,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ,
                offset=map_offset,
            )
        owner = mmap_byte_array(
            mapping,
            num_bytes,
            offset=skew,
            readonly=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tensor = torch.from_numpy(owner).view(dtype).view(end - start, *shape[1:])
        pointer = tensor.data_ptr()
        page = mmap.PAGESIZE
        base = pointer - pointer % page
        aligned = ((num_bytes + (pointer - base) + page - 1) // page) * page
        result = torch.cuda.cudart().cudaHostRegister(base, aligned, 2)
        if result.value != 0:
            _clear_cuda_last_error()
            raise RuntimeError(f"Engram cudaHostRegister failed: {result}")
        finalizer = weakref.finalize(owner, _unregister, mapping, base)
        finalizer.atexit = False
        if not tensor.is_pinned():
            raise RuntimeError("Engram shared mapping was not registered as pinned")
        import ixformer.inference.functions as ops

        view = ops.get_cuda_view_from_cpu_tensor(tensor)
    except Exception as exc:
        _detach_tensor_storage(view)
        _detach_tensor_storage(tensor)
        view = None
        tensor = None
        owner = None
        _cleanup_failed_mapping(mapping, finalizer)
        mapping = None
        finalizer = None
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * group.world_size
    torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
    if any(errors):
        _detach_tensor_storage(view)
        _detach_tensor_storage(tensor)
        view = None
        tensor = None
        owner = None
        _cleanup_failed_mapping(mapping, finalizer)
        mapping = None
        finalizer = None
        raise RuntimeError(f"Engram shared mmap/UVA initialization failed: {errors}")
    return tensor, view, mapping


def _detach_tensor_storage(tensor):
    if tensor is not None and hasattr(tensor, "set_"):
        with contextlib.suppress(Exception):
            tensor.set_(torch.empty(0, dtype=tensor.dtype, device="cpu"))


def _clear_cuda_last_error():
    with contextlib.suppress(Exception):
        cudart = torch.cuda.cudart()
        if cudart is not None and hasattr(cudart, "cudaGetLastError"):
            cudart.cudaGetLastError()


def _cleanup_failed_mapping(mapping, finalizer):
    if finalizer is not None:
        with contextlib.suppress(Exception):
            finalizer()
    if mapping is not None and not mapping.closed:
        with contextlib.suppress(Exception):
            mapping.close()


def _unregister(mapping, pointer):
    # The numpy storage owner retains the mapping until CUDA releases its pages.
    result = torch.cuda.cudart().cudaHostUnregister(pointer)
    if getattr(result, "value", result) != 0:
        _clear_cuda_last_error()
    if mapping is not None and not mapping.closed:
        with contextlib.suppress(BufferError, Exception):
            mapping.close()
