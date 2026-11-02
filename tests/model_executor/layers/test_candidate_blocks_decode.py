"""Candidate membership is unchanged by decode batching and unsorted top-k."""
import pytest
import torch

from conftest import requires_corex30

from vllm_iluvatar.model_executor.kernels.attention.dsa.ilu_candidate_blocks import (
    select_candidate_blocks,
)


pytestmark = requires_corex30


def reference(logits, ends, k=2048, block=8):
    rows, width = logits.shape
    valid = torch.arange(width, device=logits.device)[None, :] < ends[:, None]
    scores = logits.masked_fill(~valid, -float('inf')).view(rows, -1, block).amax(-1)
    live = ends > 0
    indices = ((ends.long() - 1) // block).clamp_min(0)
    scores[torch.arange(rows, device=logits.device)[live], indices[live]] = float('inf')
    # The previous implementation bounded each top-k to seven rows at this width.
    result = []
    for start in range(0, rows, 7):
        values, ids = scores[start:start + 7].topk(k, dim=-1)
        result.append(torch.where(values != -float('inf'), ids, -1).int())
    return torch.cat(result).sort(-1).values


@pytest.mark.parametrize('rows', [6, 18, 24, 48, 66, 96, 97])
@pytest.mark.parametrize('kind', ['random', 'ties', 'nan'])
def test_candidate_membership_and_live_replay(rows, kind):
    width = 278528
    storage = torch.empty(rows, width + 16, device='cuda', dtype=torch.bfloat16)
    logits = storage[:, :width]
    logits.normal_()
    if kind == 'ties':
        logits.zero_()
        logits[:, 512:1024] = 1
    elif kind == 'nan':
        logits[:, 19] = float('nan')
        logits[:, 311] = float('nan')
        logits[:, 4096] = float('inf')
    ends = torch.full((rows,), 65536, device='cuda', dtype=torch.int32)
    output = torch.empty(rows, 2048, device='cuda', dtype=torch.int32)
    select_candidate_blocks(logits, None, ends, 2048, 8, output)
    assert torch.equal(output.sort(-1).values, reference(logits, ends))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select_candidate_blocks(logits, None, ends, 2048, 8, output)
    edge = torch.tensor([0, 1, 7, 8, 16383, width - 1], device='cuda', dtype=torch.int32)
    ends.copy_(edge.repeat((rows + 5) // 6)[:rows])
    graph.replay()
    assert torch.equal(output.sort(-1).values, reference(logits, ends))
    graph.reset()
