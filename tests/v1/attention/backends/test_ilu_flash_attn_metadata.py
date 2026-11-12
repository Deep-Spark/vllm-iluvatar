# SPDX-License-Identifier: Apache-2.0
"""Decode bounds must be built without reading device scalars."""

import itertools

import pytest
import torch
from conftest import requires_cuda
from torch.utils._python_dispatch import TorchDispatchMode
from vllm.v1.attention.backend import CommonAttentionMetadata

from vllm_iluvatar.v1.attention.backends.ilu_flash_attn import (
    IluFlashAttentionMetadataBuilder,
)


class RejectDeviceScalar(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            assert args[0].device.type == "cpu", "metadata attempted a D2H scalar read"
        return func(*args, **(kwargs or {}))


def make_builder(device, graph, n_spec):
    # The tested build path needs no distributed process group or model weights.
    builder = IluFlashAttentionMetadataBuilder.__new__(IluFlashAttentionMetadataBuilder)
    builder.device = torch.device(device)
    builder.dcp_world_size = 1
    builder.has_spec = n_spec > 0
    builder.reorder_batch_threshold = n_spec + 1
    builder.use_full_cuda_graph = graph
    builder.max_cudagraph_size = 128
    builder.block_size = 128
    # This fixture bypasses __init__ and exercises the non-FP8 path.
    builder._fp8_swa_window = None
    builder.decode_cu_seqlens_q = torch.arange(17, dtype=torch.int32, device=device)
    builder.decode_cu_seqlens_k = torch.zeros(17, dtype=torch.int32, device=device)
    return builder


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("cpu_bound", [False, True])
@pytest.mark.parametrize(
    "lengths,seqs,n_spec,num_decodes",
    [
        ([1, 1, 1], [127, 256, 33], 0, 3),
        ([7, 7], [134, 263], 7, 2),  # DSpark draft
        ([8, 8, 0, 0], [135, 264, 0, 0], 7, 4),  # target/padded graph
        ([8, 8, 64], [135, 264, 1024], 7, 2),  # mixed batch
        ([64, 128], [64, 256], 7, 0),
    ],
)
def test_decode_bounds_and_device_lengths(
    device, graph, cpu_bound, lengths, seqs, n_spec, num_decodes
):
    builder = make_builder(device, graph, n_spec)
    cu_cpu = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32)
    # Deliberately optimistic CPU lengths exercise async-spec upper-bound semantics.
    upper = torch.tensor([s + 7 if s else 0 for s in seqs], dtype=torch.int32)
    common = CommonAttentionMetadata(
        num_reqs=len(lengths),
        num_actual_tokens=sum(lengths),
        query_start_loc=cu_cpu.to(device),
        query_start_loc_cpu=cu_cpu,
        seq_lens=torch.tensor(seqs, dtype=torch.int32, device=device),
        max_query_len=max(lengths),
        max_seq_len=max(seqs) + 64,
        block_table_tensor=torch.zeros(
            len(lengths), 16, dtype=torch.int32, device=device
        ),
        slot_mapping=torch.full((sum(lengths),), -1, dtype=torch.int64, device=device),
        seq_lens_cpu_upper_bound=upper if cpu_bound else None,
    )
    with RejectDeviceScalar():
        result = builder.build(0, common)
    assert result.num_decodes == num_decodes
    if not num_decodes:
        assert result.decode is None
        return
    decode = result.decode
    assert decode.max_query_len == max(lengths[:num_decodes])
    expected_bound = (
        2048
        if graph and sum(lengths) <= 128
        else max(upper[:num_decodes]).item()
        if cpu_bound
        else common.max_seq_len
    )
    assert decode.max_decode_seq_len == expected_bound
    assert decode.seq_lens.data_ptr() == common.seq_lens.data_ptr()
    torch.testing.assert_close(
        decode.seq_lens.cpu(), torch.tensor(seqs[:num_decodes], dtype=torch.int32)
    )
    torch.testing.assert_close(decode.cu_seqlens_q.cpu(), cu_cpu[: num_decodes + 1])
    expected_k = torch.tensor(
        [0, *itertools.accumulate(seqs[:num_decodes])], dtype=torch.int32
    )
    torch.testing.assert_close(decode.cu_seqlens_k.cpu(), expected_k)


@requires_cuda
def test_builder_capture_and_replay_does_not_read_device_scalars():
    builder = make_builder("cuda", True, 7)
    cu_cpu = torch.tensor([0, 7, 14, 14], dtype=torch.int32)
    common = CommonAttentionMetadata(
        num_reqs=3,
        num_actual_tokens=14,
        max_query_len=7,
        max_seq_len=512,
        query_start_loc_cpu=cu_cpu,
        query_start_loc=cu_cpu.cuda(),
        seq_lens=torch.tensor([127, 255, 0], dtype=torch.int32, device="cuda"),
        block_table_tensor=torch.zeros(3, 16, dtype=torch.int32, device="cuda"),
        slot_mapping=torch.full((14,), -1, dtype=torch.int64, device="cuda"),
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), RejectDeviceScalar():
        builder.build(0, common)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), RejectDeviceScalar():
        result = builder.build(0, common)
    for seqs in ([513, 1025, 0], [2047, 1999, 0]):
        common.seq_lens.copy_(torch.tensor(seqs, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        assert result.decode.max_decode_seq_len == 2048
        torch.testing.assert_close(
            result.decode.cu_seqlens_k.cpu(),
            torch.tensor([0, *itertools.accumulate(seqs)], dtype=torch.int32),
        )
