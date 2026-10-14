# SPDX-License-Identifier: Apache-2.0
"""Configuration checks and shared checkpoint storage for Iluvatar Engram."""

import mmap
import os
import shutil
import socket
import tempfile
import weakref

import numpy as np
import torch


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
    if config.dp_shared_memory:
        required = sum(hf.engram_num_embeddings) * (
            hf.engram_head_dim + hf.engram_head_dim // 32
        )
        if (
            not os.path.isdir("/dev/shm")
            or shutil.disk_usage("/dev/shm").total < required
        ):
            raise ValueError(
                f"Iluvatar Engram shared tables require {required} bytes in /dev/shm; "
                "increase --shm-size/--ipc=host or set dp_shared_memory=False "
                "and embedding_across_dp=True"
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
    """One writer copies checkpoint rows into tmpfs; every peer maps those pages.

    Keep checkpoint access as MAP_SHARED and copy in bounded chunks. Unlink the
    tmpfs file after every peer has mapped and registered it, so process teardown
    releases storage even after a failed service start.
    """
    offset, shape, dtype = entry(path, name)
    if not 0 <= start < end <= shape[0]:
        raise ValueError(f"{name}: invalid row interval {start}:{end} for {shape}")
    itemsize = torch.empty((), dtype=dtype, device="cpu").element_size()
    row_bytes = int(np.prod(shape[1:])) * itemsize
    num_bytes = (end - start) * row_bytes
    offset += start * row_bytes
    backing = None
    shared_path = error = None
    if group.rank_in_group == 0:
        try:
            if shutil.disk_usage("/dev/shm").free < num_bytes:
                raise RuntimeError(f"/dev/shm needs {num_bytes} free bytes for {name}")
            # Its lifetime spans peer mapping/registration below.
            backing = tempfile.NamedTemporaryFile(  # noqa: SIM115
                prefix="ilu_engram_", dir="/dev/shm"
            )
            # Reserve capacity before touching pages (avoids tmpfs SIGBUS).
            os.posix_fallocate(backing.fileno(), 0, num_bytes)
            with mmap.mmap(backing.fileno(), num_bytes, flags=mmap.MAP_SHARED) as dst:
                map_offset = offset - offset % mmap.ALLOCATIONGRANULARITY
                skew = offset - map_offset
                with (
                    open(path, "rb") as source,
                    mmap.mmap(
                        source.fileno(),
                        num_bytes + skew,
                        flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ,
                        offset=map_offset,
                    ) as src,
                ):
                    for lo in range(0, num_bytes, 64 * 1024**2):
                        hi = min(lo + 64 * 1024**2, num_bytes)
                        dst[lo:hi] = src[skew + lo : skew + hi]
            shared_path = backing.name
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    shared_path, error = group.broadcast_object((shared_path, error))
    mapping = owner = tensor = view = None
    try:
        if error is not None:
            raise RuntimeError("Engram shared checkpoint creation failed: " + error)
        error = None
        try:
            with open(shared_path, "r+b") as shared_file:
                mapping = mmap.mmap(
                    shared_file.fileno(), num_bytes, flags=mmap.MAP_SHARED
                )
            owner = np.frombuffer(mapping, dtype=np.uint8)
            tensor = torch.from_numpy(owner).view(dtype).view(end - start, *shape[1:])
            pointer = tensor.data_ptr()
            result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 2)
            if result.value != 0:
                raise RuntimeError(f"Engram cudaHostRegister failed: {result}")
            finalizer = weakref.finalize(owner, _unregister, mapping, pointer)
            finalizer.atexit = False
            if not tensor.is_pinned():
                raise RuntimeError("Engram shared mapping was not registered as pinned")
            import ixformer.inference.functions as ops

            view = ops.get_cuda_view_from_cpu_tensor(tensor)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [None] * group.world_size
        torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
        if any(errors):
            raise RuntimeError(
                f"Engram shared mmap/UVA initialization failed: {errors}"
            )
        return tensor, view, mapping
    finally:
        if backing is not None:
            backing.close()


def _unregister(mapping, pointer):
    # The numpy storage owner retains the mapping until CUDA releases its pages.
    torch.cuda.cudart().cudaHostUnregister(pointer)
