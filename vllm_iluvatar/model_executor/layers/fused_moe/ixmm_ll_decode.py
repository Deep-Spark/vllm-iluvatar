# SPDX-License-Identifier: Apache-2.0
"""IXMM split dispatch and grouped W4A8 for the DS4.1 EP16 decode graph."""

import ctypes as C
import os
from bisect import bisect_left
from types import SimpleNamespace

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_iluvatar import envs

logger = init_logger(__name__)
EXPERTS, RANKS, HIDDEN, INTERMEDIATE, TOPK = 24, 16, 5120, 2304, 6

# Exact exported LP64 C++ signatures from the IXMM communication/GEMM headers.
# Binding the intended overload by name also makes an ABI mismatch detectable
# before any rank enters an IXMM collective.
_TENSOR = C.c_void_p
_SIZE_OUT = C.POINTER(C.c_uint64)
_BINDINGS = {
    "send": (
        "_ZN4ixmm32dispatch_low_latency_v2_send_a8iEPK8DLTensorPS0_S2_limS3_",
        [_TENSOR] * 3 + [C.c_int64, C.c_int, C.c_size_t, _TENSOR],
    ),
    "recv": (
        "_ZN4ixmm32dispatch_low_latency_v2_recv_a8iEP8DLTensorS1_S1_S1_S1_S1_S1_S1_limS1_",
        [_TENSOR] * 8 + [C.c_int64, C.c_int, C.c_size_t, _TENSOR],
    ),
    "init_dispatch": (
        "_ZN4ixmm28dispatch_low_latency_v2_initEP8DLTensorim",
        [_TENSOR, C.c_int, C.c_size_t],
    ),
    "dispatch_size": (
        "_ZN4ixmm33dispatch_low_latency_v2_workspaceElllllPm",
        [C.c_int64] * 5 + [_SIZE_OUT],
    ),
    "combine_size": (
        "_ZN4ixmm33groupgemm_combine_token_workspaceEllllPm",
        [C.c_int64] * 4 + [_SIZE_OUT],
    ),
    "fc1": (
        "_ZN4ixmm30gemm_i4_i8_fp16_graph_group_v1EPK8DLTensorS2_S2_S2_S2_PS0_S2_lfm",
        [_TENSOR] * 7 + [C.c_int64, C.c_float, C.c_size_t],
    ),
    "combine": (
        "_ZN4ixmm36groupgemm_combine_token_w4i_a8i_bf16EPK8DLTensorS2_S2_S2_S2_S2_S2_S2_PS0_imS3_",
        [_TENSOR] * 9 + [C.c_int, C.c_size_t, _TENSOR],
    ),
    "get_uid": ("ixmm_shmem_get_uid", [C.c_void_p, C.c_size_t]),
    "init_shmem": ("ixmm_shmem_init", [C.c_int, C.c_int, C.c_void_p, C.c_size_t]),
    "allocate": ("ixmm_shmem_alloc", [C.c_size_t, C.POINTER(C.c_void_p)]),
}

_caps_ptr = C.pythonapi.PyCapsule_GetPointer
_caps_ptr.argtypes = [C.py_object, C.c_char_p]
_caps_ptr.restype = C.c_void_p


class CudaBytes:
    def __init__(self, ptr, size):
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "strides": None,
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def _call(function, *args):
    capsules = []
    values = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            capsule = torch.utils.dlpack.to_dlpack(arg)
            capsules.append(capsule)
            values.append(_caps_ptr(capsule, b"dltensor"))
        else:
            values.append(arg)
    status = function(*values)
    if status:
        raise RuntimeError(f"IXMM {function.__name__} returned {status}")


def _workspace_size(function, *args):
    size = C.c_uint64()
    _call(function, *args, C.byref(size))
    if not size.value:
        raise ValueError(f"IXMM {function.__name__} returned an empty workspace")
    return size.value


class IxmmLL:
    """EP-group-owned communication buffers retain their addresses across graph replays."""

    def __init__(self, path, capacities):
        if not path:
            raise ValueError("VLLM_ILU_IXMM_LL_LIBRARY is unset")
        self.lib = C.CDLL(path, mode=os.RTLD_LOCAL | os.RTLD_DEEPBIND)
        self.lib.ixmm_shmem_uid_size.argtypes = []
        self.lib.ixmm_shmem_uid_size.restype = C.c_size_t
        self.uid_size = self.lib.ixmm_shmem_uid_size()
        if not self.uid_size:
            raise ValueError("IXMM was built without SHMEM host support")
        self.ops = SimpleNamespace()
        for name, (symbol, argtypes) in _BINDINGS.items():
            op = getattr(self.lib, symbol)
            op.argtypes = argtypes
            op.restype = C.c_int
            setattr(self.ops, name, op)
        self.capacities = capacities
        self.sizes = {
            tokens: (
                _workspace_size(
                    self.ops.dispatch_size, tokens, HIDDEN, RANKS, EXPERTS, TOPK
                ),
                _workspace_size(
                    self.ops.combine_size,
                    EXPERTS * RANKS * tokens,
                    HIDDEN,
                    tokens,
                    TOPK,
                ),
            )
            for tokens in capacities
        }
        self.buffers = {}

    def capacity_for(self, local_tokens):
        index = bisect_left(self.capacities, local_tokens)
        if local_tokens > 0 and index < len(self.capacities):
            return self.capacities[index]
        return None

    def initialize(self, device, group):
        uid = C.create_string_buffer(self.uid_size)
        rank = group.rank_in_group
        payload = [None]
        if rank == 0:
            payload[0] = (self.ops.get_uid(uid, self.uid_size), uid.raw)
        control = group.cpu_group
        dist.broadcast_object_list(
            payload, group=control, src=dist.get_global_rank(control, 0)
        )
        status, raw = payload[0]
        if status:
            if rank == 0:
                logger.warning(
                    "IXMM SHMEM UID unavailable (status %d); all EP ranks are using NVSM",
                    status,
                )
            return False
        uid.raw = raw
        _call(self.ops.init_shmem, rank, group.world_size, uid, self.uid_size)
        # Decode graphs run serially on this group. Scratch can overlap across
        # shapes; dispatch/combine signal words must remain private to each shape.
        max_rows = EXPERTS * RANKS * max(self.sizes)
        fc1 = torch.empty(
            (max_rows, 2 * INTERMEDIATE), dtype=torch.bfloat16, device=device
        )
        act = torch.empty((max_rows, INTERMEDIATE), dtype=torch.int8, device=device)
        act_scale = torch.empty(max_rows, dtype=torch.float32, device=device)
        recv = torch.empty((max_rows, HIDDEN), dtype=torch.int8, device=device)
        recv_scale = torch.empty(max_rows, dtype=torch.float32, device=device)
        for tokens, (dispatch_size, combine_size) in self.sizes.items():
            rows = EXPERTS * RANKS * tokens
            size = dispatch_size + combine_size + 8192
            ptr = C.c_void_p()
            _call(self.ops.allocate, size, C.byref(ptr))
            storage = torch.as_tensor(CudaBytes(ptr.value, size), device=device)
            offset = (-storage.data_ptr()) % 4096
            combine_offset = offset + (dispatch_size + 4095) // 4096 * 4096
            dispatch = storage[offset : offset + dispatch_size].view(torch.int8)
            combine = storage[combine_offset : combine_offset + combine_size].view(
                torch.int8
            )
            _call(
                self.ops.init_dispatch,
                dispatch,
                0,
                torch.cuda.current_stream().cuda_stream,
            )
            combine.zero_()
            # Complete the stream-side SHMEM barrier before the next
            # symmetric allocation enters a host-side SHMEM collective.
            torch.cuda.current_stream().synchronize()

            def empty(shape, dtype):
                return torch.empty(shape, dtype=dtype, device=device)

            self.buffers[tokens] = SimpleNamespace(
                ops=self.ops,
                dispatch=dispatch,
                combine=combine,
                rank=rank,
                scale=empty((tokens,), torch.float32),
                recv=recv[:rows],
                recv_scale=recv_scale[:rows],
                natural=empty((EXPERTS, 3), torch.int32),
                hot=empty((EXPERTS, 3), torch.int32),
                routes=torch.full((rows,), -1, dtype=torch.int32, device=device),
                fc1=fc1[:rows],
                act=act[:rows],
                act_scale=act_scale[:rows],
                out=empty((tokens, HIDDEN), torch.bfloat16),
            )
        torch.cuda.synchronize()
        dist.barrier(group=control)
        return True

    def dispatch_send(self, x, ids, stream):
        b = self.buffers[x.shape[0]]
        _call(self.ops.send, x, b.scale, ids, EXPERTS, 0, stream, b.dispatch)
        return b

    def dispatch_recv(self, b, stream):
        _call(
            self.ops.recv,
            b.recv,
            b.recv_scale,
            b.natural,
            b.routes,
            None,
            None,
            b.hot,
            None,
            TOPK,
            0,
            stream,
            b.dispatch,
        )


def prepare(device, group, capacities):
    # Cache both outcomes on the EP group. No library loads, object collectives
    # or host allocations occur inside CUDA graph capture or replay.
    if hasattr(group, "_ilu_v41_ixmm_ll"):
        return group._ilu_v41_ixmm_ll
    runtime = None
    enabled = envs.VLLM_ILU_V41_IXMM_LL_DECODE
    reason = None
    contract = None
    if enabled:
        try:
            runtime = IxmmLL(envs.VLLM_ILU_IXMM_LL_LIBRARY, capacities)
            contract = (runtime.uid_size, runtime.sizes)
        except (OSError, AttributeError, ValueError, RuntimeError) as exc:
            reason = str(exc)
    else:
        reason = "disabled"

    reports = [None] * group.world_size
    dist.all_gather_object(reports, (enabled, reason, contract), group=group.cpu_group)
    if any(report[0] for report in reports):
        problem = next(
            (
                f"rank {i}: {report[1]}"
                for i, report in enumerate(reports)
                if report[1] is not None
            ),
            None,
        )
        if problem is None and any(report[2] != reports[0][2] for report in reports):
            problem = "inconsistent SHMEM UID or workspace sizes across ranks"
        if problem is not None:
            if group.rank_in_group == 0:
                logger.warning(
                    "IXMM LL decode unavailable (%s); all %d EP ranks are using NVSM",
                    problem,
                    group.world_size,
                )
            runtime = None
        elif not runtime.initialize(device, group):
            runtime = None
    group._ilu_v41_ixmm_ll = runtime
    return runtime


def prepare_layer(layer, code_multiplier, group):
    runtime = prepare(
        layer.w13_weight.device, group, layer._ilu_ixmm_ll_token_capacities
    )
    if runtime is None:
        return
    if tuple(layer.w13_weight.shape) != (EXPERTS, HIDDEN, INTERMEDIATE):
        raise ValueError("Unexpected IXMM LL FC1 packed weight shape")
    if tuple(layer.w2_weight.shape) != (EXPERTS, INTERMEDIATE, HIDDEN // 2):
        raise ValueError("Unexpected IXMM LL FC2 packed weight shape")
    layer._ixmm_ll_scales = (
        (
            layer.w13_weight_scale.reshape(EXPERTS, 2 * INTERMEDIATE) * code_multiplier
        ).contiguous(),
        (layer.w2_weight_scale.reshape(EXPERTS, HIDDEN) * code_multiplier).contiguous(),
    )
    layer._ilu_ixmm_ll = runtime


def experts(b, layer, ids, topk_weights, limit):
    from ixformer.inference.functions import moe_group_gemm_swiglu_quant

    scale1, scale2 = layer._ixmm_ll_scales
    stream = torch.cuda.current_stream().cuda_stream
    _call(
        b.ops.fc1,
        b.recv,
        layer.w13_weight,
        b.recv_scale,
        scale1,
        b.hot,
        b.fc1,
        None,
        0,
        1.0,
        stream,
    )
    moe_group_gemm_swiglu_quant(
        b.fc1,
        i8_output=b.act,
        output_scales=b.act_scale,
        limit=limit,
        group_layout=b.hot,
        backend="ixformer_native",
    )
    _call(
        b.ops.combine,
        b.act,
        b.act_scale,
        layer.w2_weight,
        scale2,
        b.natural,
        b.routes,
        ids,
        topk_weights,
        b.out,
        0,
        stream,
        b.combine,
    )
    return b.out


def forward(
    runtime,
    capacity,
    hidden,
    ids,
    weights,
    layer,
    shared_layer,
    shared_input,
    routed_scale,
    tp_chunk_sizes,
    tp_group,
):
    """Overlap shared compute with IXMM transit, then join the two TP results."""
    ids = ids.to(torch.int32).contiguous()
    weights = weights.float().contiguous()
    padding = capacity - hidden.shape[0]
    if padding:
        # All PEs use identical IXMM strides. Inactive rows have no route.
        hidden = torch.nn.functional.pad(hidden, (0, 0, 0, padding))
        ids = torch.nn.functional.pad(ids, (0, 0, 0, padding), value=-1)
        weights = torch.nn.functional.pad(weights, (0, 0, 0, padding))

    stream = torch.cuda.current_stream().cuda_stream
    buffers = runtime.dispatch_send(hidden, ids, stream)
    shared = shared_layer(shared_input)
    runtime.dispatch_recv(buffers, stream)
    routed = experts(buffers, layer, ids, weights, layer.swiglu_limit)
    if routed_scale != 1.0:
        routed.mul_(routed_scale)
    rows = max(tp_chunk_sizes) if tp_chunk_sizes is not None else shared.shape[0]
    routed = routed[:rows]
    if tp_group.world_size > 1:
        if tp_chunk_sizes is not None:
            routed = tp_group.all_gather(routed, dim=0)
            if sum(tp_chunk_sizes) != routed.shape[0]:
                shards = routed.split(rows)
                routed = torch.cat(
                    [shard[:n] for shard, n in zip(shards, tp_chunk_sizes, strict=True)]
                )
        shared = tp_group.all_reduce(shared)
    return shared + routed
