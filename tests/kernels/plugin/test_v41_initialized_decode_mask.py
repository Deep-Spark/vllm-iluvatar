# SPDX-License-Identifier: Apache-2.0
"""Initialized decode masking with reused CUDA graph buffers and live bounds."""
import pytest
import torch
import vllm_iluvatar._C  # noqa: F401

from vllm_iluvatar.model_executor.kernels.attention.dsa.ilu_candidate_blocks import (
    apply_initialized_decode_candidate_mask,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


@pytest.mark.parametrize("rows", [6, 12])
@pytest.mark.parametrize("width", [31, 1048576])
def test_initialized_mask_reuses_graph_without_reading_tail_flags(rows, width):
    device = "cuda"
    ends = torch.zeros(rows, dtype=torch.int32, device=device)
    candidates = torch.full((rows, 2048), -1, dtype=torch.int32, device=device)
    raw = torch.full((rows * width + 256,), -12345, dtype=torch.bfloat16, device=device)
    logits = raw[128:-128].view(rows, width)
    expected = torch.empty_like(logits)
    initial = torch.randn_like(logits)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply_initialized_decode_candidate_mask(logits, ends, candidates)

    # Grow, shrink and empty the same graph. Exterior logits are initialized
    # by the caller, while only the current live flag prefix is initialized.
    for length in (width, min(width, 32768), 7, 0, width):
        limits = [max(0, length - i) for i in range(rows)]
        ends.copy_(torch.tensor(limits, dtype=torch.int32, device=device))
        source = initial.clone()
        shortlist = torch.full_like(candidates, -1)
        for row, limit in enumerate(limits):
            source[row, limit:] = -float("inf")
            if limit:
                shortlist[row, 0] = 0
                shortlist[row, 1] = (limit - 1) // 8
            # Duplicate edge candidates retain the last valid column exactly
            # as the general mask does; negative candidates remain padding.
            shortlist[row, -2:] = (width + 7) // 8 + 1
        candidates.copy_(shortlist)
        logits.copy_(source)
        expected.copy_(source)
        graph.replay()
        torch.ops._C.v41_apply_candidate_mask(expected, None, ends, candidates, 8, 1)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        assert bool((raw[:128] == -12345).all())
        assert bool((raw[-128:] == -12345).all())
