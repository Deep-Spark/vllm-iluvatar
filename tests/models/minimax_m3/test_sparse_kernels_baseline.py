# SPDX-License-Identifier: Apache-2.0
"""Shape tests against ixformer MiniMax-M3 torch backends (no local golden)."""

from __future__ import annotations

import ixformer.inference.functions as ixf
import torch
from ixformer.inference.functions.minimax_m3.indexer import SPARSE_BLOCK_SIZE


def test_index_score_topk_and_sparse_attn_shapes():
    torch.manual_seed(0)
    q_len, n_heads, head_dim = 4, 2, 16
    blk = SPARSE_BLOCK_SIZE
    idx_q = torch.randn(q_len, n_heads, head_dim)
    cache = torch.randn(2, blk, head_dim)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    cu = torch.tensor([0, q_len], dtype=torch.int32)
    seq = torch.tensor([q_len], dtype=torch.int32)
    pref = torch.tensor([0], dtype=torch.int32)

    score = ixf.minimax_m3_index_score(
        idx_q,
        cache,
        block_table,
        cu,
        seq,
        pref,
        q_len,
        q_len,
        n_heads,
        backend="torch",
    )
    assert score.shape[0] == n_heads and score.shape[1] == q_len
    assert score.dtype == torch.float32

    topk = ixf.minimax_m3_index_topk(
        score,
        cu,
        pref,
        q_len,
        topk=2,
        init_blocks=0,
        local_blocks=1,
        backend="torch",
    )
    assert topk.shape == (q_len, n_heads, 2)
    assert (topk[..., 0] == 0).all()  # local block forced

    n_q = 4
    q = torch.randn(q_len, n_q, head_dim)
    kv = torch.randn(2, 2, blk, n_heads, head_dim)
    out = torch.zeros_like(q)
    ixf.minimax_m3_sparse_attn(
        q,
        kv,
        topk,
        block_table,
        cu,
        seq,
        pref,
        q_len,
        n_heads,
        head_dim**-0.5,
        out,
        backend="torch",
    )
    assert torch.isfinite(out).all()
    assert out.norm() > 0


def test_decode_path():
    torch.manual_seed(1)
    n_heads, head_dim = 2, 16
    blk = SPARSE_BLOCK_SIZE
    seq_len = 5
    idx_q = torch.randn(1, n_heads, head_dim)
    cache = torch.randn(2, blk, head_dim)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    seq = torch.tensor([seq_len], dtype=torch.int32)
    topk = ixf.minimax_m3_index_decode(
        idx_q,
        cache,
        block_table,
        seq,
        seq_len,
        2,
        0,
        1,
        n_heads,
        1,
        1,
        backend="torch",
    )
    assert topk.shape == (1, n_heads, 2)

    q = torch.randn(1, 4, head_dim)
    kv = torch.randn(2, 2, blk, n_heads, head_dim)
    out = torch.zeros_like(q)
    ixf.minimax_m3_sparse_attn_decode(
        q,
        kv,
        topk,
        block_table,
        seq,
        n_heads,
        head_dim**-0.5,
        out,
        1,
        backend="torch",
    )
    assert torch.isfinite(out).all()
