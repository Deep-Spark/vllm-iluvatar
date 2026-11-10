"""Candidate top-k scores are unchanged by decode batching and unsorted top-k.

Index identity at tied bf16 scores is backend-defined (radix vs torch.topk).
The selected score multiset, including the pinned newest block, must match.
"""

import pytest
import torch
from conftest import requires_corex30

from vllm_iluvatar.model_executor.kernels.attention.dsa.ilu_candidate_blocks import (
    select_candidate_blocks,
)

pytestmark = requires_corex30


def block_scores(logits, ends, block=8):
    rows, width = logits.shape
    valid = torch.arange(width, device=logits.device)[None, :] < ends[:, None]
    scores = logits.masked_fill(~valid, -float("inf")).view(rows, -1, block).amax(-1)
    live = ends > 0
    indices = ((ends.long() - 1) // block).clamp_min(0)
    scores[torch.arange(rows, device=logits.device)[live], indices[live]] = float("inf")
    return scores


def sorted_selected_scores(scores, ids):
    safe = ids.clamp(min=0).long()
    got = scores.gather(1, safe)
    got = torch.where(ids >= 0, got, got.new_full((), float("-inf")))
    return got.float().sort(-1).values


def assert_topk_scores(logits, ends, output, k=2048, block=8):
    scores = block_scores(logits, ends, block)
    got = sorted_selected_scores(scores, output)
    exp = scores.topk(k, dim=-1).values.float().sort(-1).values
    torch.testing.assert_close(got, exp, rtol=0.0, atol=0.0, equal_nan=True)


@pytest.mark.parametrize("rows", [6, 18, 24, 48, 66, 96, 97])
@pytest.mark.parametrize("kind", ["random", "ties", "nan"])
def test_candidate_membership_and_live_replay(rows, kind):
    width = 278528
    storage = torch.empty(rows, width + 16, device="cuda", dtype=torch.bfloat16)
    logits = storage[:, :width]
    logits.normal_()
    if kind == "ties":
        logits.zero_()
        logits[:, 512:1024] = 1
    elif kind == "nan":
        logits[:, 19] = float("nan")
        logits[:, 311] = float("nan")
        logits[:, 4096] = float("inf")
    ends = torch.full((rows,), 65536, device="cuda", dtype=torch.int32)
    output = torch.empty(rows, 2048, device="cuda", dtype=torch.int32)
    select_candidate_blocks(logits, None, ends, 2048, 8, output)
    assert_topk_scores(logits, ends, output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select_candidate_blocks(logits, None, ends, 2048, 8, output)
    edge = torch.tensor(
        [0, 1, 7, 8, 16383, width - 1], device="cuda", dtype=torch.int32
    )
    ends.copy_(edge.repeat((rows + 5) // 6)[:rows])
    graph.replay()
    assert_topk_scores(logits, ends, output)
    graph.reset()
