# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MiniMax-M3 token-major top-k workspace."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_iluvatar.models.minimax_m3.w4a8 import indexer as m3_indexer


def _device() -> torch.device:
    return (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def _make_indexer(
    *,
    heads: int = 4,
    topk: int = 8,
    max_tokens: int = 64,
) -> m3_indexer.IluMiniMaxM3Indexer:
    idx = m3_indexer.IluMiniMaxM3Indexer.__new__(m3_indexer.IluMiniMaxM3Indexer)
    idx.num_index_heads = heads
    idx.topk_blocks = topk
    idx.topk_indices_buffer = torch.empty(
        (max_tokens, heads, topk), dtype=torch.int32, device=_device()
    )
    return idx


def test_dense_topk_out_is_token_major_prefix():
    idx = _make_indexer()
    out = idx._dense_topk_out(3, token_offset=0)
    assert out is not None
    assert out.shape == (3, 4, 8)
    assert out.is_contiguous()
    assert out.stride() == (32, 8, 1)
    assert out.data_ptr() == idx.topk_indices_buffer.data_ptr()


def test_decode_and_prefill_share_storage():
    idx = _make_indexer()
    decode = idx._dense_topk_out(2, token_offset=0)
    prefill = idx._dense_topk_out(17, token_offset=2)
    assert decode is not None and prefill is not None
    assert decode.shape == (2, 4, 8)
    assert prefill.shape == (17, 4, 8)
    assert decode.is_contiguous() and prefill.is_contiguous()
    assert (
        prefill.data_ptr() == decode.data_ptr() + decode.numel() * decode.element_size()
    )


def test_dense_topk_out_returns_none_for_empty_split():
    idx = _make_indexer()
    assert idx._dense_topk_out(0, token_offset=0) is None


def test_dense_topk_out_raises_when_buffer_is_too_small():
    idx = _make_indexer(max_tokens=8)
    with pytest.raises(RuntimeError, match="too small"):
        idx._dense_topk_out(5, token_offset=6)


def test_missing_buffer_raises():
    idx = _make_indexer()
    idx.topk_indices_buffer = None
    with pytest.raises(RuntimeError, match="requires a shared topk_indices_buffer"):
        idx._dense_topk_out(2, token_offset=0)


def test_indexer_forward_uses_dense_out_for_multi_head(monkeypatch):
    idx = _make_indexer()
    idx.index_head_dim = 128
    idx.num_kv_heads = 4
    idx.init_blocks = 0
    idx.local_blocks = 0
    idx.index_cache = SimpleNamespace(
        prefix="layer.attn.index_cache", kv_cache=torch.empty(1)
    )

    def fake_decode(iq, kv, *args, out=None):
        assert out is not None
        assert out.is_contiguous()
        assert out.shape == (2, 4, 8)
        out.fill_(7)
        return out

    fake_ixfops = MagicMock()
    fake_ixfops.minimax_m3_index_decode.side_effect = fake_decode
    monkeypatch.setattr(m3_indexer, "ixfops", fake_ixfops)

    md = m3_indexer.IluMiniMaxM3IndexerMetadata(
        seq_lens=torch.ones(1, dtype=torch.int32),
        max_seq_len=128,
        slot_mapping=torch.zeros(2, dtype=torch.int64),
        num_actual_tokens=2,
        num_decodes=1,
        num_decode_tokens=2,
        num_prefills=0,
        num_prefill_tokens=0,
        prefill=None,
        decode=m3_indexer.IluMiniMaxM3IndexerDecodeMetadata(
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            seq_lens=torch.ones(1, dtype=torch.int32),
            max_seq_len=128,
            decode_query_len=2,
            max_decode_query_len=2,
        ),
    )

    class _Ctx:
        attn_metadata = {"layer.attn.index_cache": md}

    monkeypatch.setattr(m3_indexer, "get_forward_context", lambda: _Ctx())

    iq = torch.zeros(2, 4 * 128)
    decode_topk, prefill_topk = idx.forward(iq)
    assert prefill_topk is None
    assert decode_topk is not None
    assert decode_topk.data_ptr() == idx.topk_indices_buffer.data_ptr()
    assert torch.all(decode_topk == 7)
