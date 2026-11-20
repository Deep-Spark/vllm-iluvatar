# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from conftest import requires_corex30

from vllm_iluvatar.model_executor.layers.fused_moe.f4f8_layout import sort_group_layout
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_f4f8_experts import (
    _prepare_group_gemm,
)

pytestmark = requires_corex30


@pytest.mark.parametrize("experts", [7, 128, 257, 288, 512])
@pytest.mark.parametrize("distribution", ["empty", "one_hot", "dense", "skewed"])
def test_split_layout_covers_every_local_row_once(cuda_device, experts, distribution):
    rng = torch.Generator().manual_seed(132)
    counts = torch.randint(0, 192, (experts,), generator=rng, dtype=torch.int32)
    if distribution == "empty":
        counts.zero_()
    elif distribution == "one_hot":
        counts.zero_()
        counts[-1] = 2048
    elif distribution == "dense":
        counts.add_(1)
        counts[0] = 2304
    else:
        counts[::2] = 0
        counts[1::7] = 1200
    ends = counts.cumsum(0, dtype=torch.int32)
    layout = torch.stack([ends - counts, ends], dim=1).to(cuda_device)
    planned = sort_group_layout(layout, split_large=True).cpu().tolist()
    assert len(planned) == experts
    assert all(b - a >= d - c for (a, b, _), (c, d, _) in zip(planned, planned[1:]))
    intervals = [[] for _ in range(experts)]
    for begin, end, expert in planned:
        assert 0 <= expert < experts and begin <= end
        if end > begin:
            intervals[expert].append((begin, end))
    for expert, spans in enumerate(intervals):
        cursor = int(ends[expert] - counts[expert])
        for begin, end in sorted(spans):
            assert cursor == begin
            cursor = end
        assert cursor == int(ends[expert])


def test_split_layout_reads_current_routes_on_graph_replay(cuda_device):
    original = torch.tensor(
        [[0, 1024], [1024, 1024], [1024, 1024], [1024, 1024]],
        device=cuda_device,
        dtype=torch.int32,
    )
    alternate = torch.tensor(
        [[0, 0], [0, 0], [0, 0], [0, 1024]], device=cuda_device, dtype=torch.int32
    )
    sort_group_layout(original, split_large=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = sort_group_layout(original, split_large=True)
    original.copy_(alternate)
    graph.replay()
    assert torch.equal(output, sort_group_layout(alternate, split_large=True))
    assert output[:, 2].cpu().tolist() == [3] * 4


@pytest.mark.parametrize("experts", [1, 3, 16, 36, 64, 72, 128])
def test_f4f8_expert_layout_sort(cuda_device, experts):
    torch.manual_seed(79)
    counts = torch.randint(0, 129, (experts,), dtype=torch.int32)
    counts[::3] = 0
    edges = torch.cat((torch.zeros(1, dtype=torch.int32), counts)).cumsum(0).int()
    layout = torch.stack((edges[:-1], edges[1:]), dim=1)
    order = torch.argsort(counts, descending=True, stable=True)
    expected = torch.cat((layout[order], order[:, None].int()), dim=1)
    actual = sort_group_layout(layout.to(cuda_device)).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("experts", "tokens", "topk", "expected"),
    [
        (36, 64, 8, "original"),
        (36, 65, 8, "sorted"),
        (36, 512, 8, "sorted"),
        (36, 513, 8, "original"),
        (128, 255, 8, "sorted"),
        (128, 257, 8, "split"),
        (288, 575, 8, "original"),
        (288, 576, 8, "split"),
        (288, 576, 4, "original"),
        (512, 1024, 8, "split"),
        (513, 1024, 8, "original"),
    ],
)
def test_projection_layout_dispatch(cuda_device, experts, tokens, topk, expected):
    # Only the last local expert receives a route for each token.
    ranges = torch.zeros((experts, 2), dtype=torch.int32, device=cuda_device)
    ranges[-1, 1] = tokens
    mode, tasks = _prepare_group_gemm(ranges, tokens, tokens * topk)
    assert mode == (1 if tokens <= 64 else 0)
    if expected == "original":
        assert tasks is ranges
        return

    active = tasks[tasks[:, 1] > tasks[:, 0]]
    assert torch.all(active[:, 2] == experts - 1)
    assert len(active) == ((tokens + 255) // 256 if expected == "split" else 1)


@pytest.mark.parametrize("first", [1, 2**31 - 256])
def test_last_split_uses_representable_exclusive_end(cuda_device, first):
    stop = torch.iinfo(torch.int32).max
    layout = torch.tensor(
        [[0, first], [first, stop], [stop, stop], [stop, stop]],
        device=cuda_device,
        dtype=torch.int32,
    )
    tasks = sort_group_layout(layout, split_large=True).cpu().tolist()
    assert all(0 <= begin <= end <= stop for begin, end, _ in tasks)
    for expert, start, end in [(0, 0, first), (1, first, stop)]:
        cursor = start
        for lo, hi in sorted((a, b) for a, b, e in tasks if e == expert and a < b):
            assert lo == cursor
            assert (lo - start) % 256 == 0
            cursor = hi
        assert cursor == end
