# SPDX-License-Identifier: Apache-2.0
"""Fused QK-norm/RoPE/KV-insert tests via ixformer torch backend."""

from __future__ import annotations

import ixformer.inference.functions as ixf
import torch


def _make_cos_sin(max_pos: int, rotary_dim: int) -> torch.Tensor:
    cs = torch.zeros(max_pos, rotary_dim)
    cs[:, : rotary_dim // 2] = 1.0
    return cs


def test_sparse_qkv_layout_index_k_is_one_head():
    """Sparse fused qkv last-dim must be nq+2*nkv+niq+1 heads."""
    torch.manual_seed(0)
    nq, nkv, niq, d = 4, 2, 2, 8
    t = 3
    block_size = 4
    q = torch.randn(t, nq * d)
    k = torch.arange(t * nkv * d, dtype=torch.float32).reshape(t, nkv, d)
    v = torch.arange(100, 100 + t * nkv * d, dtype=torch.float32).reshape(t, nkv, d)
    iq = torch.randn(t, niq * d)
    ik = torch.arange(1000, 1000 + t * d, dtype=torch.float32).reshape(t, d)
    qkv = torch.cat(
        [q, k.reshape(t, -1), v.reshape(t, -1), iq, ik],
        dim=-1,
    ).contiguous()

    q_norm = torch.zeros(d)
    k_norm = torch.zeros(d)
    iq_norm = torch.zeros(d)
    ik_norm = torch.zeros(d)
    cos_sin = _make_cos_sin(16, rotary_dim=4)
    positions = torch.arange(t, dtype=torch.long)
    slots = torch.arange(t, dtype=torch.long)

    # ixformer's scatter_kv_nhd packs K/V in the last dim of a 4-D
    # [pages, nkv, block_size, 2*d] cache (K = [..., :d], V = [..., d:]).
    kv_cache = torch.zeros(2, nkv, block_size, 2 * d)
    index_cache = torch.zeros(2, block_size, d)
    q_out = torch.empty(t, nq * d)
    index_q_out = torch.empty(t, niq * d)

    bad = torch.cat([qkv, torch.zeros(t, (niq - 1) * d)], dim=-1)
    try:
        ixf.fused_minimax_m3_qknorm_rope_kv_insert(
            bad,
            q_norm,
            k_norm,
            cos_sin,
            positions,
            nq,
            nkv,
            4,
            1e-6,
            iq_norm,
            ik_norm,
            niq,
            slots,
            slots,
            kv_cache,
            index_cache,
            block_size,
            q_out,
            index_q_out,
            backend="torch",
        )
        raised = False
    except ValueError:
        raised = True
    assert raised, "expected ValueError for wrong qkv last dim"

    ixf.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_norm,
        k_norm,
        cos_sin,
        positions,
        nq,
        nkv,
        4,
        1e-6,
        iq_norm,
        ik_norm,
        niq,
        slots,
        slots,
        kv_cache,
        index_cache,
        block_size,
        q_out,
        index_q_out,
        backend="torch",
    )

    for i in range(t):
        assert kv_cache[0, :, i, :d].abs().sum() > 0
        assert kv_cache[0, :, i, d:].abs().sum() > 0
        assert index_cache[0, i].abs().sum() > 0

    assert index_cache.shape[-1] == d


def test_scatter_nhd():
    torch.manual_seed(1)
    nq, nkv, d = 2, 2, 8
    t = 2
    block_size = 4
    qkv = torch.randn(t, (nq + 2 * nkv) * d)
    q_norm = torch.zeros(d)
    k_norm = torch.zeros(d)
    cos_sin = _make_cos_sin(8, 4)
    positions = torch.arange(t, dtype=torch.long)
    slots = torch.tensor([1, 2], dtype=torch.long)

    # >>> ILU MODIFICATION >>>
    # ixformer's scatter_kv_nhd now packs K/V in the last dim of a 4-D
    # [pages, nkv, block_size, 2*d] cache (K = [..., :d], V = [..., d:]).
    # The legacy 5-D HND layout is no longer supported by ixformer, so the
    # HND portion of this test has been dropped.
    # <<< ILU MODIFICATION <<<
    kv_nhd = torch.zeros(1, nkv, block_size, 2 * d)
    ixf.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv.clone(),
        q_norm,
        k_norm,
        cos_sin,
        positions,
        nq,
        nkv,
        4,
        1e-6,
        slot_mapping=slots,
        kv_cache=kv_nhd,
        block_size=block_size,
        backend="torch",
    )
    assert kv_nhd[0, :, 1, :d].abs().sum() > 0
    assert kv_nhd[0, :, 2, :d].abs().sum() > 0
    assert kv_nhd[0, :, 1, d:].abs().sum() > 0
    assert kv_nhd[0, :, 2, d:].abs().sum() > 0


def test_negative_slots_skipped():
    nq, nkv, d = 2, 1, 8
    t = 2
    block_size = 4
    qkv = torch.randn(t, (nq + 2 * nkv) * d)
    # >>> ILU MODIFICATION >>>
    # 4-D packed NHD layout: [pages, nkv, block_size, 2*d] (K=[...,:d], V=[...,d:]).
    # <<< ILU MODIFICATION <<<
    kv = torch.zeros(1, nkv, block_size, 2 * d)
    ixf.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        torch.zeros(d),
        torch.zeros(d),
        _make_cos_sin(8, 4),
        torch.arange(t),
        nq,
        nkv,
        4,
        1e-6,
        slot_mapping=torch.tensor([-1, 0]),
        kv_cache=kv,
        block_size=block_size,
        backend="torch",
    )
    assert kv[0, :, 0, :d].abs().sum() > 0
    assert kv[0, :, 1, :d].abs().sum() == 0
