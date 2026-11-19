# SPDX-License-Identifier: Apache-2.0
"""Paged IndexPool scoring bounds and causal recall on eager and capture shapes."""

from types import SimpleNamespace

import pytest
import torch

from tests.conftest import requires_cuda
from vllm_iluvatar.models.hy_v4_flash import indexpool


@requires_cuda
@pytest.mark.parametrize("context_limit", [4096, 32768])
@torch.inference_mode()
def test_batch_bounded_group_recall(context_limit, monkeypatch):
    page = 64
    positions = [2047, 2048, 2061, 2062]
    n = len(positions)
    device = "cuda"
    # The last request schedules two tokens within the batch.
    seq_lens = torch.tensor([2048, 2049, 2063], dtype=torch.int32, device=device)
    starts = torch.tensor([0, 1, 2, 4], dtype=torch.int32, device=device)
    owners = torch.tensor([0, 1, 2, 2], dtype=torch.int32, device=device)
    pages = 33
    table = (
        torch.arange(3 * pages, dtype=torch.int32, device=device)
        .view(3, pages)
        .flip(1)
        .contiguous()
    )
    pooled = torch.empty(
        (3 * pages * page // 4, 128), dtype=torch.bfloat16, device=device
    )
    # Monotone unique scores for complete groups, independent of their physical page.
    for req in range(3):
        group = torch.arange(pages * page // 4, device=device)
        slots = table[req, group // (page // 4)].long() * (page // 4) + group % (
            page // 4
        )
        pooled[slots] = 0
        # Two exact BF16 components encode integer scores above 256 without ties.
        pooled[slots, 0] = (group // 16).to(torch.bfloat16)
        pooled[slots, 1] = (group % 16).to(torch.bfloat16)
    q = torch.zeros((n, 16, 128), dtype=torch.bfloat16, device=device)
    q[:, :, 0] = 16
    q[:, :, 1] = 1
    weights = torch.ones((n, 16), dtype=torch.bfloat16, device=device)
    meta = SimpleNamespace(
        block_size=page,
        block_table=table,
        req_id_per_token=owners,
        query_start_loc=starts,
        seq_lens=seq_lens,
        num_actual_tokens=n,
    )
    idx = torch.empty((n, 1, 2048), dtype=torch.int32, device=device)
    lens = torch.empty(n, dtype=torch.int32, device=device)
    pool = SimpleNamespace(
        max_groups=context_limit // 4, fuse_topk=context_limit <= 4096
    )
    grids = []
    scores = indexpool._scores

    class RecordGrid:
        def __getitem__(self, grid):
            grids.append(grid)
            return scores[grid]

    monkeypatch.setattr(indexpool, "_scores", RecordGrid())
    for limit in (2063, context_limit):
        meta.max_seq_len = limit
        indexpool.FlashIndexPool._select_groups(
            pool, q, weights, pooled, meta, idx, lens
        )
        assert grids[-1] == ((limit + page - 1) // page * (page // 4), n)
        for i, p in enumerate(positions):
            group, offset = divmod(p, 4)
            first = max(0, group - 511)
            tokens = torch.arange(first * 4, p + 1, device=device)
            request = min(i, 2)
            expected = table[request, tokens // page] * page + tokens % page
            count = min(511, group) * 4 + offset + 1
            assert lens[i].item() == count
            torch.testing.assert_close(
                idx[i, 0, :count].long().sort().values,
                expected.sort().values,
                atol=0,
                rtol=0,
            )
            assert (idx[i, 0, count:] == -1).all()
