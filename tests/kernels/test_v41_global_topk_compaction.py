# SPDX-License-Identifier: Apache-2.0
"""Reference test for ``global_topk_indices_and_lens`` compaction.

Feat 0.31 already ships a warp-ballot block scan in ixformer
``global_topk.cu``. This file keeps the documented stable-compaction
contract so a GPU host can still check the native op. Local py_compile
does not execute it.
"""
from __future__ import annotations

import pytest
import torch
from ixformer.inference.functions.models import deepseek_v41 as ixfops

DEVICE = "cuda"


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not hasattr(ixfops, "global_topk_indices_and_lens"),
    reason="requires CUDA and global_topk_indices_and_lens",
)


def reference(
    topk_indices,
    token_to_req,
    block_table,
    block_size,
    is_valid_token,
    positions,
    compress_ratio,
    slot_limit,
    pad_walk_tail,
):
    num_tokens, topk = topk_indices.shape
    out = torch.empty_like(topk_indices)
    lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    bt_width = block_table.shape[1]

    for t in range(num_tokens):
        if not bool(is_valid_token[t]):
            out[t] = 0 if pad_walk_tail else -1
            lens[t] = 0
            continue
        eff = topk
        if positions is not None:
            eff = max(0, min(topk, (int(positions[t]) + 1) // compress_ratio))
        req = int(token_to_req[t])
        kept = []
        for k in range(eff):
            local = int(topk_indices[t, k])
            if local < 0:
                continue
            blk, off = divmod(local, block_size)
            if blk >= bt_width:
                continue
            page = int(block_table[req, blk])
            if page <= 0:
                continue
            slot = page * block_size + off
            if slot_limit > 0 and slot >= slot_limit:
                continue
            kept.append(slot)
        fill = (kept[-1] if kept else 0) if pad_walk_tail else -1
        row = kept + [fill] * (topk - len(kept))
        out[t] = torch.tensor(row, dtype=torch.int32, device=topk_indices.device)
        lens[t] = len(kept)
    return out, lens


def make(num_tokens, topk, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    num_reqs, bt_width, block_size = 4, 8, 16
    block_table = torch.randint(
        0, 40, (num_reqs, bt_width), dtype=torch.int32, generator=g, device=DEVICE
    )
    topk_indices = torch.randint(
        -2,
        bt_width * block_size + 8,
        (num_tokens, topk),
        dtype=torch.int32,
        generator=g,
        device=DEVICE,
    )
    token_to_req = torch.randint(
        0, num_reqs, (num_tokens,), dtype=torch.int32, generator=g, device=DEVICE
    )
    is_valid = torch.ones(num_tokens, dtype=torch.bool, device=DEVICE)
    is_valid[::7] = False
    return topk_indices, token_to_req, block_table, block_size, is_valid


@pytest.mark.parametrize("topk", [1, 16, 129, 300, 512])
@pytest.mark.parametrize("num_tokens", [1, 24, 64])
@pytest.mark.parametrize("pad_walk_tail", [False, True])
def test_matches_reference(topk, num_tokens, pad_walk_tail):
    ti, t2r, bt, bs, valid = make(num_tokens, topk, seed=topk * 1000 + num_tokens)
    got_idx, got_len = ixfops.global_topk_indices_and_lens(
        ti, t2r, bt, bs, valid, None, 1, 0, pad_walk_tail
    )
    want_idx, want_len = reference(ti, t2r, bt, bs, valid, None, 1, 0, pad_walk_tail)
    assert torch.equal(got_len, want_len)
    assert torch.equal(got_idx, want_idx)


def test_compaction_is_stable():
    topk, num_tokens = 512, 8
    ti, t2r, bt, bs, valid = make(num_tokens, topk, seed=17)
    got_idx, got_len = ixfops.global_topk_indices_and_lens(
        ti, t2r, bt, bs, valid, None, 1, 0, False
    )
    want_idx, _ = reference(ti, t2r, bt, bs, valid, None, 1, 0, False)
    for t in range(num_tokens):
        n = int(got_len[t])
        assert torch.equal(got_idx[t, :n], want_idx[t, :n])
