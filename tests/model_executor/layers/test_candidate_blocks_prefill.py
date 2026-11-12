"""Unsorted prefill top-k preserves causal membership, including tied scores."""

import pytest
import torch
from conftest import requires_corex30

from vllm_iluvatar import envs as ilu_envs
from vllm_iluvatar.model_executor.kernels.attention.dsa import ilu_candidate_blocks


@requires_corex30
@pytest.mark.parametrize("rows,width", [(129, 24576), (4096, 32768)])
@pytest.mark.parametrize("ties", [False, True])
def test_prefill_membership(monkeypatch, rows, width, ties):
    ilu_envs.apply()
    monkeypatch.setenv("VLLM_ILU_V41_PREFILL_UNSORTED", "0")
    torch.manual_seed(17)
    logits = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    if ties:
        logits.zero_()
        logits[:, 512:1024] = 1
    edge = torch.tensor(
        [0, 1, 7, 8, 16383, width - 1, width], device="cuda", dtype=torch.int32
    )
    ends = edge.repeat((rows + 6) // 7)[:rows]
    expected = torch.empty(rows, 2048, device="cuda", dtype=torch.int32)
    actual = torch.empty_like(expected)
    monkeypatch.setenv("VLLM_ILU_V41_PREFILL_UNSORTED", "0")
    ilu_candidate_blocks.select_candidate_blocks(logits, None, ends, 2048, 8, expected)
    monkeypatch.setenv("VLLM_ILU_V41_PREFILL_UNSORTED", "1")
    ilu_candidate_blocks.select_candidate_blocks(logits, None, ends, 2048, 8, actual)
    assert torch.equal(expected.sort(-1).values, actual.sort(-1).values)


@pytest.mark.parametrize("enabled", [False, True])
def test_prefill_fallback_reads_runtime_flag(monkeypatch, enabled):
    ilu_envs.apply()
    monkeypatch.setenv("VLLM_ILU_V41_PREFILL_UNSORTED", str(int(enabled)))
    monkeypatch.setattr(ilu_candidate_blocks, "cuda_arch", "ivcore30")
    rows, width = 129, 16392
    logits = torch.zeros((rows, width), dtype=torch.bfloat16)
    ends = torch.full((rows,), width, dtype=torch.int32)
    # A strided output forces the fallback instead of the native selector.
    out = torch.empty((rows, 4096), dtype=torch.int32)[:, ::2]
    scores = torch.arange(width // 8, dtype=torch.float32).expand(rows, -1)
    calls = []
    original_topk = torch.Tensor.topk

    def topk(self, k, *, dim, sorted):
        calls.append(sorted)
        return original_topk(self, k, dim=dim, sorted=sorted)

    def write(result, values, indices, k):
        result[:, :k].copy_(indices)

    monkeypatch.setattr(torch.Tensor, "topk", topk)
    monkeypatch.setattr(
        ilu_candidate_blocks.ixfops,
        "candidate_block_scores",
        lambda chunk, *args: scores[: len(chunk)],
    )
    monkeypatch.setattr(ilu_candidate_blocks.ixfops, "write_candidate_topk", write)
    ilu_candidate_blocks.select_candidate_blocks(logits, None, ends, 2048, 8, out)
    assert calls and all(sorted == (not enabled) for sorted in calls)
    assert torch.equal(out.sort(-1).values, torch.arange(1, 2049).expand_as(out))
