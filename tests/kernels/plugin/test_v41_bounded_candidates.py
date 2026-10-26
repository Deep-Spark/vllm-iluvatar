# SPDX-License-Identifier: Apache-2.0
"""Valid-prefix scores must survive ragged bounds and graph buffer reuse."""
import pytest
import torch
import vllm_iluvatar._C  # noqa: F401

from vllm_iluvatar.model_executor.kernels.attention.dsa.ilu_candidate_blocks import select_candidate_blocks


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('row_repeat', [1, 3])
@pytest.mark.parametrize('block_size', [8, 64])
def test_bounded_scores_leave_exterior_untouched(dtype, row_repeat, block_size):
    rows, width = 7, 4099
    groups = (rows + row_repeat - 1) // row_repeat
    source = torch.randn(rows, width * 2, device='cuda', dtype=dtype)
    logits = source[:, ::2]
    starts = torch.tensor([0, 17, 128, 9, 0, 11, 0][:groups], device='cuda', dtype=torch.int32)
    ends = torch.tensor([0, 513, 129, 4099, 8, 2048, 97][:groups], device='cuda', dtype=torch.int32)
    logits[1, 33] = float('nan')
    nblocks = (width + block_size - 1) // block_size
    storage = torch.full((rows + 2, nblocks + 16), -777, device='cuda', dtype=dtype)
    scores = storage[1:-1, :nblocks]
    scores.fill_(float('nan'))
    bounds = torch.empty((2, rows), device='cuda', dtype=torch.int32)
    torch.ops._C.v41_candidate_block_scores_bounded(logits, starts, ends, block_size, row_repeat, scores, bounds)
    expected = torch.ops._C.v41_candidate_block_scores(logits, starts, ends, block_size, row_repeat)
    limits = ((ends - starts).clamp(min=0) + block_size - 1) // block_size
    limits = limits.repeat_interleave(row_repeat)[:rows]
    assert torch.equal(bounds[0], torch.zeros_like(limits))
    assert torch.equal(bounds[1], limits)
    for row, limit in enumerate(limits.cpu().tolist()):
        torch.testing.assert_close(scores[row, :limit], expected[row, :limit], rtol=0, atol=0, equal_nan=True)
        assert torch.isnan(scores[row, limit:]).all()
    assert (storage[[0, -1]] == -777).all()
    assert (storage[:, nblocks:] == -777).all()


def test_finalize_never_selects_uninitialized_empty_row():
    scores = torch.full((3, 16), float('nan'), device='cuda', dtype=torch.bfloat16)
    scores[1, :2] = torch.tensor([1., -float('inf')], device='cuda', dtype=torch.bfloat16)
    bounds = torch.tensor([[0, 0, 0], [0, 2, 16]], device='cuda', dtype=torch.int32)
    out = torch.tensor([[0, -1, 15, 100], [0, 1, 2, -1], [0, -1, 15, 16]], device='cuda', dtype=torch.int32)
    torch.ops._C.v41_finalize_bounded_candidates(scores, bounds, out)
    expected = torch.tensor([[-1, -1, -1, -1], [0, -1, -1, -1], [0, -1, 15, -1]], device='cuda', dtype=torch.int32)
    assert torch.equal(out, expected)


@pytest.mark.parametrize('rows', [1, 7, 24, 129])
def test_nan_warmup_rows_have_complete_in_range_candidates(rows):
    width = 32768
    logits = torch.full((rows, width), float('nan'), device='cuda', dtype=torch.bfloat16)
    ends = torch.full((rows,), width, device='cuda', dtype=torch.int32)
    out = torch.full((rows, 2048), 123456, device='cuda', dtype=torch.int32)
    select_candidate_blocks(logits, None, ends, 2048, 8, out)
    ids = out.cpu()
    for row in ids:
        assert (row >= 0).all() and (row < width // 8).all()
        assert len(row.unique()) == 2048


@pytest.mark.parametrize('rows,width', [(1, 131081), (7, 131072), (18, 1048576), (129, 131072)])
def test_selection_changed_lengths_under_one_graph(rows, width):
    logits = torch.randn(rows, width, device='cuda', dtype=torch.bfloat16)
    ends = torch.full((rows,), width, device='cuda', dtype=torch.int32)
    out = torch.empty((rows, 2048), device='cuda', dtype=torch.int32)
    for _ in range(3):
        select_candidate_blocks(logits, None, ends, 2048, 8, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select_candidate_blocks(logits, None, ends, 2048, 8, out)
    for length in (width, 16385, 4096, 0, width - 7):
        ends.fill_(length)
        ends[::3] = max(0, length - 8)
        logits.uniform_(-1, 1)
        graph.replay()
        scores = torch.ops._C.v41_candidate_block_scores(logits, None, ends, 8, 1)
        expected = scores.topk(2048, dim=1).values
        ids = out.cpu()
        limits = ((ends.cpu() + 7) // 8)
        for row in range(rows):
            valid = ids[row][ids[row] >= 0]
            assert len(valid) == min(2048, int(limits[row]))
            assert len(valid.unique()) == len(valid)
            assert not len(valid) or int(valid.max()) < int(limits[row])
            if len(valid):
                assert int(limits[row]) - 1 in valid.tolist()
        # Equal scores at the selection boundary may choose different ids.
        selected = scores.gather(1, out.clamp(min=0).long())
        selected.masked_fill_(out < 0, -float('inf'))
        torch.testing.assert_close(selected.sort(dim=1, descending=True).values, expected,
                                   rtol=0, atol=0, equal_nan=True)
