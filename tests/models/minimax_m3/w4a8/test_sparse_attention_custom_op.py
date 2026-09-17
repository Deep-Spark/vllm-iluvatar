# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MiniMax-M3 sparse attention custom-op + fake (V4 pattern)."""

from __future__ import annotations

import torch

from vllm_iluvatar.models.minimax_m3.w4a8 import attention as m3_attn
from vllm_iluvatar.models.minimax_m3.w4a8 import (
    sparse_attention as m3_sparse,
)


def test_custom_op_registered():
    assert hasattr(torch.ops.vllm, "ilu_minimax_m3_sparse_attention")


def test_fake_impl_is_noop():
    qkv = torch.empty(2, 16)
    positions = torch.zeros(2, dtype=torch.long)
    out = torch.zeros(2, 8)
    assert (
        m3_attn.ilu_minimax_m3_sparse_attention_fake(qkv, positions, out, "layer.attn")
        is None
    )
    assert torch.equal(out, torch.zeros(2, 8))


def test_trampoline_dispatches_to_attention_impl(monkeypatch):
    calls: list[tuple] = []

    class _Layer:
        layer_name = "test.attn"

        def attention_impl(self, qkv, positions, out):
            calls.append((qkv, positions, out))
            out.fill_(1.0)

    layer = _Layer()

    class _Ctx:
        no_compile_layers = {layer.layer_name: layer}

    monkeypatch.setattr(m3_attn, "get_forward_context", lambda: _Ctx())

    qkv = torch.zeros(2, 16)
    positions = torch.zeros(2, dtype=torch.long)
    out = torch.zeros(2, 8)
    m3_attn.ilu_minimax_m3_sparse_attention(qkv, positions, out, layer.layer_name)

    assert len(calls) == 1
    assert calls[0][0] is qkv and calls[0][2] is out
    assert torch.all(out == 1.0)


def test_tcu_workspace_reservation_uses_ixformer_size_query(monkeypatch):
    real_empty = torch.empty
    sizes: list[tuple] = []
    queries: list[tuple[int, int, int]] = []

    def _record(*args, **kwargs):
        sizes.append((args, kwargs))
        return real_empty(0)

    def _workspace_size(total_q, num_kv_heads, topk):
        queries.append((total_q, num_kv_heads, topk))
        return 12345

    impl = m3_sparse.IluMiniMaxM3SparseImpl.__new__(m3_sparse.IluMiniMaxM3SparseImpl)
    impl.num_kv_heads = 2
    impl.topk_blocks = 32
    ref = real_empty(8, 16)

    monkeypatch.setattr(m3_sparse.torch, "empty", _record)
    monkeypatch.setattr(
        m3_sparse.ixfops,
        "minimax_m3_sparse_attn_tcu_workspace_size",
        _workspace_size,
    )
    impl.reserve_tcu_workspace(ref)
    monkeypatch.undo()

    assert queries == [(8, 2, 32)]
    assert len(sizes) == 1
    assert sizes[0][0] == (12345,)
    assert sizes[0][1]["dtype"] is torch.uint8


def test_profile_forward_still_calls_custom_op_and_o_proj(monkeypatch):
    op_calls: list[str] = []

    class _Projection:
        def __call__(self, hidden_states):
            return hidden_states, None

    class _OProj:
        def __call__(self, attn_out):
            return attn_out.new_zeros((attn_out.shape[0], 16)), None

    class _Layer:
        layer_name = "layer.attn"
        qkv_proj = _Projection()
        o_proj = _OProj()
        q_size = 16
        kv_cache = torch.empty(0)

    def _op(qkv, positions, out, layer_name):
        op_calls.append(layer_name)
        out.zero_()

    monkeypatch.setattr(torch.ops.vllm, "ilu_minimax_m3_sparse_attention", _op)

    hidden_states = torch.randn(8, 16)
    output = m3_attn.IluMiniMaxM3SparseAttention.forward(
        _Layer(),
        positions=torch.arange(8),
        hidden_states=hidden_states,
    )

    assert op_calls == ["layer.attn"]
    assert output.shape == hidden_states.shape
    assert torch.count_nonzero(output) == 0


def test_profile_attention_impl_allocates_row_temps_and_reserves_tcu(
    monkeypatch,
):
    refs: list[torch.Tensor] = []

    class _Impl:
        def reserve_tcu_workspace(self, ref):
            refs.append(ref)

    class _Layer:
        impl = _Impl()
        kv_cache = torch.empty(0)
        q_size = 12
        index_q_size = 4

    class _Ctx:
        attn_metadata = None

    monkeypatch.setattr(m3_attn, "get_forward_context", lambda: _Ctx())

    qkv = torch.randn(8, 16)
    out = torch.ones(8, 12)
    m3_attn.IluMiniMaxM3SparseAttention.attention_impl(
        _Layer(),
        qkv,
        positions=torch.arange(8),
        out=out,
    )

    assert refs == [qkv]
    assert torch.count_nonzero(out) == 0
