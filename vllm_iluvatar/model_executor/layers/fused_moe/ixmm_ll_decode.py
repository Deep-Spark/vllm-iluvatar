"""IXMM split dispatch and grouped W4A8 for the DS4.1 EP16 decode graph."""
import ctypes as C
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch
import torch.distributed as dist

TOKENS = (6, 12, 18)
EXPERTS, RANKS, HIDDEN, INTERMEDIATE, TOPK = 24, 16, 5120, 2304, 6
_BUFFERS = {}
_STORAGE = []
_OPS = {}
_LIB = None
_SHMEM_READY = False

_caps_ptr = C.pythonapi.PyCapsule_GetPointer
_caps_ptr.argtypes = [C.py_object, C.c_char_p]
_caps_ptr.restype = C.c_void_p


class CudaBytes:
    def __init__(self, ptr, n):
        self.__cuda_array_interface__ = {
            "shape": (n,), "strides": None, "typestr": "|u1",
            "data": (ptr, False), "version": 3,
        }


def init_communicator(group):
    global _SHMEM_READY
    if _SHMEM_READY:
        return
    assert group.world_size == RANKS
    size = _LIB.ixmm_shmem_uid_size()
    if not size:
        raise RuntimeError("IXMM LL decode needs libixmm with SHMEM host support")
    uid = C.create_string_buffer(size)
    rank = group.rank_in_group
    values = [None]
    if rank == 0:
        status = _LIB.ixmm_shmem_get_uid(uid, size)
        values[0] = (status, uid.raw)
    control = group.cpu_group
    dist.broadcast_object_list(values, group=control, src=dist.get_global_rank(control, 0))
    status, uid.raw = values[0]
    if status:
        raise RuntimeError(f"IXMM SHMEM unique-ID allocation returned {status}")
    status = _LIB.ixmm_shmem_init(rank, group.world_size, uid, size)
    if status:
        raise RuntimeError(f"IXMM SHMEM initialization returned {status} on rank {rank}")
    _SHMEM_READY = True
    print(f"Micro-D IXMM LL NVSHMEM ready rank={rank}", flush=True)


def symm_bytes(n, device):
    ptr = C.c_void_p()
    status = _LIB.ixmm_shmem_alloc(n, C.byref(ptr))
    if status:
        raise RuntimeError(f"IXMM symmetric allocation of {n} bytes returned {status}")
    owner = CudaBytes(ptr.value, n)
    storage = torch.as_tensor(owner, device=device)
    _STORAGE.append((ptr, owner, storage))
    return storage


def native_ops():
    global _LIB
    path = Path(os.environ["VLLM_ILU_IXMM_LL_LIBRARY"])
    _LIB = C.CDLL(str(path), mode=os.RTLD_LOCAL | os.RTLD_DEEPBIND)
    _LIB.ixmm_shmem_uid_size.argtypes = []
    _LIB.ixmm_shmem_uid_size.restype = C.c_size_t
    _LIB.ixmm_shmem_get_uid.argtypes = [C.c_void_p, C.c_size_t]
    _LIB.ixmm_shmem_init.argtypes = [C.c_int, C.c_int, C.c_void_p, C.c_size_t]
    _LIB.ixmm_shmem_alloc.argtypes = [C.c_size_t, C.POINTER(C.c_void_p)]
    output = subprocess.check_output(["nm", "-D", "--defined-only", str(path)], text=True)
    requests = {
        "dispatch_low_latency_v2_send_a8i": (
            [C.c_void_p] * 3 + [C.c_int64, C.c_int, C.c_size_t, C.c_void_p], 6
        ),
        "dispatch_low_latency_v2_recv_a8i": (
            [C.c_void_p] * 8 + [C.c_int64, C.c_int, C.c_size_t, C.c_void_p], 11
        ),
        "dispatch_low_latency_v2_init": ([C.c_void_p, C.c_int, C.c_size_t], 2),
        "dispatch_low_latency_v2_workspace": ([C.c_int64] * 5 + [C.POINTER(C.c_uint64)], 5),
        "groupgemm_combine_token_workspace": ([C.c_int64] * 4 + [C.POINTER(C.c_uint64)], 4),
        "gemm_i4_i8_fp16_graph_group_v1": ([C.c_void_p] * 7 + [C.c_int64, C.c_float, C.c_size_t], 9),
        "groupgemm_combine_token_w4i_a8i_bf16": ([C.c_void_p] * 9 + [C.c_int, C.c_size_t, C.c_void_p], 11),
    }
    for op, (argtypes, commas) in requests.items():
        names = [line.split()[-1] for line in output.splitlines() if op in line]
        signatures = subprocess.check_output(["c++filt", *names], text=True).splitlines()
        (name,) = [name for name, sig in zip(names, signatures, strict=True)
                   if sig.startswith(f"ixmm::{op}(") and sig.count(",") == commas]
        function = getattr(_LIB, name)
        function.argtypes = argtypes
        function.restype = C.c_int
        _OPS[op] = function


def call(name, *args):
    capsules = []
    values = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            capsule = torch.utils.dlpack.to_dlpack(arg)
            capsules.append(capsule)
            values.append(_caps_ptr(capsule, b"dltensor"))
        else:
            values.append(arg)
    status = _OPS[name](*values)
    if status:
        raise RuntimeError(f"IXMM {name} returned {status}")


def workspace_size(name, *args):
    size = C.c_uint64()
    call(name, *args, C.byref(size))
    if not size.value:
        raise RuntimeError(f"IXMM {name} reported an empty workspace")
    return size.value


def prepare_layer(layer, code_multiplier, group):
    if tuple(layer.w13_weight.shape) != (EXPERTS, HIDDEN, INTERMEDIATE):
        raise ValueError("Unexpected IXMM LL FC1 packed weight shape")
    if tuple(layer.w2_weight.shape) != (EXPERTS, INTERMEDIATE, HIDDEN // 2):
        raise ValueError("Unexpected IXMM LL FC2 packed weight shape")
    layer._ixmm_ll_scales = (
        (layer.w13_weight_scale.reshape(EXPERTS, 2 * INTERMEDIATE) * code_multiplier).contiguous(),
        (layer.w2_weight_scale.reshape(EXPERTS, HIDDEN) * code_multiplier).contiguous(),
    )
    prepare(layer.w13_weight.device, group)


def prepare(device, group):
    if _BUFFERS:
        return
    native_ops()
    init_communicator(group)
    for tokens in TOKENS:
        rows = EXPERTS * RANKS * tokens
        dispatch_size = workspace_size(
            "dispatch_low_latency_v2_workspace", tokens, HIDDEN, RANKS, EXPERTS, TOPK)
        combine_size = workspace_size(
            "groupgemm_combine_token_workspace", rows, HIDDEN, tokens, TOPK)
        storage = symm_bytes(dispatch_size + combine_size + 8192, device)
        print(f"Micro-D IXMM LL symmetric workspace ready rank={group.rank_in_group} tokens={tokens}", flush=True)
        offset = (-storage.data_ptr()) % 4096
        combine_offset = offset + (dispatch_size + 4095) // 4096 * 4096
        dispatch = storage[offset:offset + dispatch_size].view(torch.int8)
        combine = storage[combine_offset:combine_offset + combine_size].view(torch.int8)
        call("dispatch_low_latency_v2_init", dispatch, 0, torch.cuda.current_stream().cuda_stream)
        combine.zero_()
        torch.cuda.synchronize()
        print(f"Micro-D IXMM LL workspace initialized rank={group.rank_in_group} tokens={tokens}", flush=True)

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        _BUFFERS[tokens] = SimpleNamespace(
            dispatch=dispatch, combine=combine, rank=group.rank_in_group,
            scale=empty((tokens,), torch.float32),
            recv=empty((rows, HIDDEN), torch.int8),
            recv_scale=empty((rows,), torch.float32),
            natural=empty((EXPERTS, 3), torch.int32),
            hot=empty((EXPERTS, 3), torch.int32),
            routes=torch.full((rows,), -1, dtype=torch.int32, device=device),
            fc1=empty((rows, 2 * INTERMEDIATE), torch.bfloat16),
            act=empty((rows, INTERMEDIATE), torch.int8),
            act_scale=empty((rows,), torch.float32),
            out=empty((tokens, HIDDEN), torch.bfloat16),
        )
    torch.cuda.synchronize()
    dist.barrier(group=group.cpu_group)
    print(f"Micro-D IXMM LL initialized rank={group.rank_in_group} token_bins={TOKENS}", flush=True)


def dispatch_send(x, ids, stream):
    b = _BUFFERS[x.shape[0]]
    call("dispatch_low_latency_v2_send_a8i", x, b.scale, ids, EXPERTS, 0, stream, b.dispatch)
    return b


def dispatch_recv(b, stream):
    assert stream == torch.cuda.current_stream().cuda_stream
    call("dispatch_low_latency_v2_recv_a8i", b.recv, b.recv_scale, b.natural,
         b.routes, None, None, b.hot, None, TOPK, 0, stream, b.dispatch)


def experts(b, layer, ids, topk_weights, limit):
    from ixformer.inference.functions import moe_group_gemm_swiglu_quant

    scale1, scale2 = layer._ixmm_ll_scales
    stream = torch.cuda.current_stream().cuda_stream
    call("gemm_i4_i8_fp16_graph_group_v1",
         b.recv, layer.w13_weight, b.recv_scale, scale1, b.hot, b.fc1, None, 0, 1.0, stream)
    moe_group_gemm_swiglu_quant(
        b.fc1, i8_output=b.act, output_scales=b.act_scale, limit=limit,
        group_layout=b.hot, backend="ixformer_native",
    )
    call("groupgemm_combine_token_w4i_a8i_bf16",
         b.act, b.act_scale, layer.w2_weight, scale2, b.natural, b.routes,
         ids, topk_weights, b.out, 0, stream, b.combine)
    return b.out
