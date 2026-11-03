"""Unsorted prefill top-k preserves causal membership, including tied scores."""
import pytest
import torch

from conftest import requires_corex30
from vllm_iluvatar.model_executor.kernels.attention.dsa import ilu_candidate_blocks

pytestmark = requires_corex30


@pytest.mark.parametrize("rows,width", [(129, 24576), (4096, 32768)])
@pytest.mark.parametrize("ties", [False, True])
def test_prefill_membership(monkeypatch, rows, width, ties):
    torch.manual_seed(17)
    logits = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    if ties:
        logits.zero_()
        logits[:, 512:1024] = 1
    edge = torch.tensor([0, 1, 7, 8, 16383, width - 1, width],
                        device="cuda", dtype=torch.int32)
    ends = edge.repeat((rows + 6) // 7)[:rows]
    expected = torch.empty(rows, 2048, device="cuda", dtype=torch.int32)
    actual = torch.empty_like(expected)
    monkeypatch.setattr(ilu_candidate_blocks, "_PREFILL_UNSORTED", False)
    ilu_candidate_blocks.select_candidate_blocks(logits, None, ends, 2048, 8, expected)
    monkeypatch.setattr(ilu_candidate_blocks, "_PREFILL_UNSORTED", True)
    ilu_candidate_blocks.select_candidate_blocks(logits, None, ends, 2048, 8, actual)
    assert torch.equal(expected.sort(-1).values, actual.sort(-1).values)
