"""Small real-buffer tests; run only with an explicit allowed GPU lease."""

import os
from types import SimpleNamespace

import pytest
import torch
from ixformer.inference.functions.deepseek.quantized_kv_format import I4G32, I8G64
from test_compressed_kv_state import cache_config
from vllm.v1.worker.utils import KVBlockZeroer, copy_kv_cache_blocks_inplace

from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    WORKSPACE_BYTES,
    CompressedKVState,
    WorkspacePool,
    plan_scale_layout,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("KV_COMPRESSED_KV_GPU_TESTS") != "1",
    reason="requires an explicitly leased GPU and KV_COMPRESSED_KV_GPU_TESTS=1",
)


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("block_outer", [False, True])
def test_gpu_paired_physical_block_copy_zero_alias_and_reuse(fmt, block_outer):
    config, names = cache_config(fmt, block_outer=block_outer)
    context = {name: SimpleNamespace(compressed_kv_format=fmt) for name in names}
    before = torch.cuda.memory_allocated()
    state = CompressedKVState(plan_scale_layout(config, fmt), fmt, "cuda", context)
    raw = torch.full(
        (config.kv_cache_tensors[0].size,), 0xB7, dtype=torch.uint8, device="cuda"
    )
    tensor_config = config.kv_cache_tensors[0]
    caches = []
    for index, name in enumerate(names[:2]):
        cache = torch.as_strided(
            raw,
            (config.num_blocks, fmt.rows_per_page(512), fmt.payload_bytes(512)),
            (tensor_config.block_stride, fmt.payload_bytes(512), 1),
            index * tensor_config.layer_stride,
        )
        cache[0].fill_(0x35 + index)
        state.scales[name][0].fill_(2 + index)
        caches.append(cache)
    alias = state.scales[names[2]]
    assert alias is state.scales[names[0]]
    copy_kv_cache_blocks_inplace(
        [*caches, *state.scales.values()], config.num_blocks, [(0, 1)]
    )
    for index, name in enumerate(names[:2]):
        assert torch.equal(caches[index][0], caches[index][1])
        assert torch.equal(state.scales[name][0], state.scales[name][1])
        assert (caches[index][2] == 0xB7).all()
        assert (state.scales[name][2] == 0).all()
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = torch.device("cuda")
    block_stride = tensor_config.block_stride // 4
    payload_page = fmt.rows_per_page(512) * fmt.payload_bytes(512) // 4
    zeroer._meta = (
        torch.tensor(
            [cache.data_ptr() for cache in caches], dtype=torch.uint64, device="cuda"
        ),
        torch.tensor([block_stride] * 2, dtype=torch.int64, device="cuda"),
        torch.tensor([payload_page] * 2, dtype=torch.int64, device="cuda"),
        payload_page // 1024,
        1024,
        2,
    )
    state.bind_zeroer(zeroer)
    zeroer.zero_block_ids([1])
    for index, name in enumerate(names[:2]):
        assert (caches[index][1] == 0).all()
        assert (state.scales[name][1] == 0).all()
        assert (caches[index][0] == 0x35 + index).all()
    caches[0][1].fill_(0x79)
    state.scales[names[0]][1].fill_(11)
    copy_kv_cache_blocks_inplace(
        [*caches, *state.scales.values()], config.num_blocks, [(1, 2)]
    )
    assert (caches[0][2] == 0x79).all()
    assert (alias[2] == 11).all()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - before < WORKSPACE_BYTES + 16 * 1024**2
    state.release()


def make_state():
    config, names = cache_config(I8G64)
    context = {name: SimpleNamespace(compressed_kv_format=I8G64) for name in names}
    return CompressedKVState(plan_scale_layout(config, I8G64), I8G64, "cuda", context)


def test_workspace_orders_two_streams_without_new_allocations():
    state = make_state()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = torch.empty(2, dtype=torch.uint8, device="cuda")
    streams[0].wait_stream(torch.cuda.current_stream())
    streams[1].wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(streams[0]), state.workspace.acquire() as workspace:
        workspace.fill_(3)
        outputs[0].copy_(workspace[0])
    with torch.cuda.stream(streams[1]), state.workspace.acquire() as workspace:
        workspace.add_(4)
        outputs[1].copy_(workspace[0])
    torch.cuda.synchronize()
    assert outputs.cpu().tolist() == [3, 7]
    assert (state.workspace.tensor == 7).all()
    state.release()


def test_changed_input_graph_replay_with_external_worker_leases():
    state = make_state()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    inputs = [
        torch.full((1024,), 3, dtype=torch.uint8, device="cuda"),
        torch.full((1024,), 5, dtype=torch.uint8, device="cuda"),
    ]
    outputs = [torch.empty_like(value) for value in inputs]
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    last_recorded = state.workspace._last_recorded_stream
    for index, graph in enumerate(graphs):
        with (
            torch.cuda.graph(graph, stream=streams[index]),
            state.workspace.acquire(nested=True) as workspace,
        ):
            workspace[:1024].copy_(inputs[index])
            workspace[:1024].add_(index + 1)
            outputs[index].copy_(workspace[:1024])
        assert state.workspace._last_recorded_stream is last_recorded
    for seed in (13, 71, 129):
        for index, graph in enumerate(graphs):
            inputs[index].fill_(seed + index)
            streams[index].wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(streams[index]), state.workspace.acquire():
                graph.replay()
            assert state.workspace._last_recorded_stream == streams[index]
        torch.cuda.synchronize()
        assert (outputs[0] == seed + 1).all()
        assert (outputs[1] == seed + 3).all()
    state.release()


def test_workspace_same_stream_fifo_and_exception_handoff_preserve_values():
    pool = WorkspacePool(torch.zeros(4096, dtype=torch.uint8, device="cuda"))
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = torch.empty((3, 1024), dtype=torch.uint8, device="cuda")
    with torch.cuda.stream(streams[0]), pool.acquire() as workspace:
        workspace[:1024].fill_(1)
    with torch.cuda.stream(streams[0]), pool.acquire() as workspace:
        with pool.acquire(nested=True):
            workspace[:1024].add_(2)
        outputs[0].copy_(workspace[:1024])
    assert pool._last_recorded_stream == streams[0]
    with (
        pytest.raises(RuntimeError, match="expected lease failure"),
        torch.cuda.stream(streams[1]),
        pool.acquire() as workspace,
    ):
        workspace[:1024].mul_(3)
        outputs[1].copy_(workspace[:1024])
        raise RuntimeError("expected lease failure")
    assert pool._last_recorded_stream == streams[1] and not pool._leased
    with torch.cuda.stream(streams[0]), pool.acquire() as workspace:
        workspace[:1024].add_(4)
        outputs[2].copy_(workspace[:1024])
    torch.cuda.synchronize()
    assert pool._last_recorded_stream == streams[0]
    for output, expected in zip(outputs.cpu(), (3, 9, 13)):
        assert (output == expected).all()
