# SPDX-License-Identifier: Apache-2.0
"""CPU tests for MiniMax-M3 sparse config validation and NHD layout."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.attention.backends.utils import get_kv_cache_layout

from vllm_iluvatar.models.minimax_m3.w4a8.sparse_attention import (
    IluMiniMaxM3SparseBackend,
    resolve_sparse_attention_cfg,
)


def _cfg(**overrides):
    base = {
        "sparse_topk_blocks": 16,
        "sparse_init_block": 1,
        "sparse_local_block": 2,
        "sparse_block_size": 128,
        "sparse_num_index_heads": 4,
        "sparse_index_dim": 128,
    }
    base.update(overrides)
    return SimpleNamespace(sparse_attention_config=base)


def test_resolve_sparse_attention_cfg_ok():
    got = resolve_sparse_attention_cfg(_cfg())
    assert got["topk_blocks"] == 16
    assert got["init_blocks"] == 1
    assert got["local_blocks"] == 2


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"sparse_init_block": -1}, "must be >= 0"),
        ({"sparse_local_block": -2}, "must be >= 0"),
        ({"sparse_init_block": 8, "sparse_local_block": 9}, "exceeds"),
        ({"sparse_topk_blocks": 0}, "must be > 0"),
        ({"sparse_block_size": 64}, "must be 128"),
    ],
)
def test_resolve_sparse_attention_cfg_rejects_invalid(overrides, match):
    with pytest.raises(ValueError, match=match):
        resolve_sparse_attention_cfg(_cfg(**overrides))


def test_hnd_layout_env_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    get_kv_cache_layout.cache_clear()
    try:
        with pytest.raises(ValueError, match="NHD"):
            IluMiniMaxM3SparseBackend.get_kv_cache_stride_order()
    finally:
        get_kv_cache_layout.cache_clear()


def test_hnd_resolved_layout_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "vllm_iluvatar.models.minimax_m3.w4a8.sparse_attention.get_kv_cache_layout",
        lambda: "HND",
    )
    with pytest.raises(ValueError, match="NHD"):
        IluMiniMaxM3SparseBackend.get_kv_cache_stride_order()


def test_nhd_layout_keeps_identity_stride(monkeypatch):
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")
    get_kv_cache_layout.cache_clear()
    try:
        assert IluMiniMaxM3SparseBackend.get_kv_cache_stride_order() == (
            0,
            1,
            2,
            3,
            4,
        )
    finally:
        get_kv_cache_layout.cache_clear()
