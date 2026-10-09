# SPDX-License-Identifier: Apache-2.0
"""The fused write-back behind ``select_candidate_blocks``.

The op folds ``torch.topk``'s (values, indices) into the shortlist row the
selector returns: an index in the first ``k`` columns, ``-1`` where the slot
scored ``-inf`` or past ``k``. What it must match is the exact torch expression
it replaced --

    keep = values != -inf
    out[:, :k] = where(keep, indices.int(), -1)
    out[:, k:] = -1

-- including a kept NaN (``NaN != -inf`` is true), which is the reference's tie
behaviour. The end-to-end test then runs the whole selector, whose only change
is this op, against that same expression over real scores.
"""

from __future__ import annotations

import pytest
import torch
import vllm_iluvatar._C  # noqa: F401  (registers torch.ops._C)

from vllm_iluvatar.model_executor.kernels.attention.dsa.ilu_candidate_blocks import (  # noqa: E402
    select_candidate_blocks,
)

DEVICE = "cuda"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _reference(
    values: torch.Tensor, indices: torch.Tensor, k: int, out_width: int
) -> torch.Tensor:
    rows = values.size(0)
    out = torch.full((rows, out_width), -1, dtype=torch.int32, device=DEVICE)
    keep = values[:, :k] != -float("inf")
    out[:, :k] = torch.where(keep, indices[:, :k].to(torch.int32), -1)
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("k,out_width", [(1, 1), (32, 32), (16, 48), (7, 64)])
def test_write_candidate_topk_matches_torch(dtype, k: int, out_width: int):
    rows, cols = 70, 256
    torch.manual_seed(k * 100 + out_width)
    scores = torch.randn(rows, cols, device=DEVICE, dtype=dtype)
    # Force some -inf and NaN survivors into the top-k the way real scores do:
    # a padding block scores -inf, a warmup row can carry NaN.
    scores[::5, :] = -float("inf")
    scores[3, :4] = float("nan")
    top = scores.topk(k, dim=-1)

    out = torch.empty(rows, out_width, dtype=torch.int32, device=DEVICE)
    torch.ops._C.v41_write_candidate_topk(out, top.values, top.indices, k)

    expected = _reference(top.values, top.indices, k, out_width)
    assert torch.equal(out, expected)


def test_write_candidate_topk_noncontiguous_row_slice():
    """The selector hands in ``out[r0:r1]``, a row-sliced view."""
    rows, out_width, k = 40, 32, 16
    torch.manual_seed(0)
    full = torch.empty(rows, out_width, dtype=torch.int32, device=DEVICE)
    scores = torch.randn(rows, 128, device=DEVICE, dtype=torch.bfloat16)
    top = scores.topk(k, dim=-1)

    torch.ops._C.v41_write_candidate_topk(
        full[8:24], top.values[8:24], top.indices[8:24], k
    )
    expected = _reference(top.values[8:24], top.indices[8:24], k, out_width)
    assert torch.equal(full[8:24], expected)


def _selector_reference(scores_rows, logits, row_ke, k, out_width):
    out = torch.full((logits.size(0), out_width), -1, dtype=torch.int32, device=DEVICE)
    for r0, (vals, idxs) in scores_rows:
        keep = vals != -float("inf")
        out[r0 : r0 + vals.size(0), :k] = torch.where(keep, idxs.to(torch.int32), -1)
    return out


@pytest.mark.parametrize("rows", [96, 300])
def test_select_candidate_blocks_end_to_end(rows: int):
    """The whole selector, across the 128-row slicing boundary.

    Compares against a straight ``v41_candidate_block_scores`` + torch top-k +
    the old write expression, so a slice boundary or the tail fill going wrong
    would show.
    """
    block_size, topk_blocks, width = 64, 24, 4096
    torch.manual_seed(rows)
    logits = torch.randn(rows, width, device=DEVICE, dtype=torch.bfloat16)
    row_ke = torch.randint(1, width + 1, (rows,), dtype=torch.int32, device=DEVICE)
    out = torch.empty(rows, topk_blocks, dtype=torch.int32, device=DEVICE)

    select_candidate_blocks(logits, None, row_ke, topk_blocks, block_size, out)

    # Reference: same scores kernel, torch top-k, old write, in one shot.
    nblocks = -(-width // block_size)
    k = min(topk_blocks, nblocks)
    scores = torch.ops._C.v41_candidate_block_scores(
        logits, None, row_ke, block_size, 1
    )
    top = scores.topk(k, dim=-1)
    expected = torch.full((rows, topk_blocks), -1, dtype=torch.int32, device=DEVICE)
    keep = top.values != -float("inf")
    expected[:, :k] = torch.where(keep, top.indices.to(torch.int32), -1)

    assert torch.equal(out, expected)
