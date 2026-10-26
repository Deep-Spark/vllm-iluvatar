# SPDX-License-Identifier: Apache-2.0
"""Indexer wrapper contracts and owned-buffer CUDA graph lifetime tests."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.utils.torch_utils import _encode_layer_name

import vllm_iluvatar.ilu_register as ilu_register


def _install_fake_ixformer(monkeypatch):
    ixformer = ModuleType("ixformer")
    ixformer.__path__ = []
    inference = ModuleType("ixformer.inference")
    inference.__path__ = []
    functions = ModuleType("ixformer.inference.functions")
    functions.dsa_indexer_mqa_logits_with_blocks = object()
    functions.dsa_update_topk_indices = object()
    monkeypatch.setitem(sys.modules, "ixformer", ixformer)
    monkeypatch.setitem(sys.modules, "ixformer.inference", inference)
    monkeypatch.setitem(sys.modules, "ixformer.inference.functions", functions)


def _load_sparse_attn_indexer(monkeypatch):
    _install_fake_ixformer(monkeypatch)
    return importlib.import_module(
        "vllm_iluvatar.model_executor.layers.sparse_attn_indexer"
    )


def test_sparse_attn_indexer_package_is_auto_discovered():
    assert "vllm_iluvatar.model_executor.layers" in ilu_register._OOT_OP_PACKAGES


def test_ilu_sparse_attn_indexer_registers_oot(monkeypatch):
    sparse_attn_indexer = _load_sparse_attn_indexer(monkeypatch)

    assert issubclass(sparse_attn_indexer.IluSparseAttnIndexer, SparseAttnIndexer)
    assert (
        op_registry_oot["SparseAttnIndexer"] is sparse_attn_indexer.IluSparseAttnIndexer
    )


def test_ilu_sparse_attn_indexer_fake_returns_mutated_buffer(monkeypatch):
    sparse_attn_indexer = _load_sparse_attn_indexer(monkeypatch)
    buffer = torch.empty((2, 4), dtype=torch.int32)

    result = sparse_attn_indexer.ilu_sparse_attn_indexer_fake(
        hidden_states=torch.empty((2, 8)),
        k_cache_prefix="layer.0.indexer",
        kv_cache=torch.empty((1, 1, 8)),
        q=torch.empty((2, 1, 8)),
        k=torch.empty((2, 8)),
        weights=torch.empty((2,)),
        topk_tokens=4,
        head_dim=8,
        max_model_len=16,
        total_seq_lens=16,
        topk_indices_buffer=buffer,
    )

    assert result is buffer


def test_ilu_sparse_attn_indexer_fake_requires_output_buffer(monkeypatch):
    sparse_attn_indexer = _load_sparse_attn_indexer(monkeypatch)

    with pytest.raises(AssertionError):
        sparse_attn_indexer.ilu_sparse_attn_indexer_fake(
            hidden_states=torch.empty((2, 8)),
            k_cache_prefix="layer.0.indexer",
            kv_cache=torch.empty((1, 1, 8)),
            q=torch.empty((2, 1, 8)),
            k=torch.empty((2, 8)),
            weights=torch.empty((2,)),
            topk_tokens=4,
            head_dim=8,
            max_model_len=16,
            total_seq_lens=16,
            topk_indices_buffer=None,
        )


def test_ilu_sparse_attn_indexer_forward_cuda_calls_registered_op(
    monkeypatch,
    default_vllm_config,
):
    sparse_attn_indexer = _load_sparse_attn_indexer(monkeypatch)
    calls = []
    output = torch.empty((2, 4), dtype=torch.int32)

    def fake_op(*args):
        calls.append(args)
        return output

    k_cache = SimpleNamespace(
        prefix="model.layers.0.self_attn.indexer",
        kv_cache=object(),
    )
    topk_indices_buffer = torch.empty((2, 4), dtype=torch.int32)
    layer = sparse_attn_indexer.IluSparseAttnIndexer(
        k_cache=k_cache,
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=4,
        head_dim=8,
        max_model_len=16,
        max_total_seq_len=32,
        topk_indices_buffer=topk_indices_buffer,
    )

    hidden_states = torch.empty((2, 8))
    q = torch.empty((2, 1, 8))
    k = torch.empty((2, 8))
    weights = torch.empty((2,))

    with patch.object(torch.ops.vllm, "ilu_sparse_attn_indexer", fake_op, create=True):
        result = layer.forward_cuda(hidden_states, q, k, weights)

    assert result is output
    assert len(calls) == 1
    args = calls[0]
    assert args[0] is hidden_states
    assert args[1] == _encode_layer_name(k_cache.prefix)
    assert args[2] is k_cache.kv_cache
    assert args[3] is q
    assert args[4] is k
    assert args[5] is weights
    assert args[6:10] == (4, 8, 16, 32)
    assert args[10] is topk_indices_buffer
    assert args[-2:] == (None, None)


def _owned_test_layer(sparse_attn_indexer, *, enabled, max_seqs=2):
    return sparse_attn_indexer.IluSparseAttnIndexer(
        k_cache=SimpleNamespace(prefix="layer.0.indexer", kv_cache=object()),
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=4,
        head_dim=128,
        max_model_len=33,
        max_total_seq_len=66,
        topk_indices_buffer=torch.empty((max(1, max_seqs) * 6, 4), dtype=torch.int32),
        owned_decode_state=enabled,
        max_num_decode_seqs=max_seqs,
    )


def _owned_test_metadata(tokens, *, prefill=False, padded=False):
    from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
        IluDeepSeekV32IndexerDecodeMetadata,
        IluDeepseekV32IndexerMetadata,
    )

    batch = max(1, tokens // 6)
    lens = torch.ones(tokens, dtype=torch.int32)
    prefix = torch.arange(batch + 1, dtype=torch.int32) * (tokens // batch)
    decode = IluDeepSeekV32IndexerDecodeMetadata(
        block_table=torch.zeros((batch, 1), dtype=torch.int32),
        seq_lens=lens,
        decode_lens=torch.full((batch,), tokens // batch, dtype=torch.int32),
        requires_padding=padded,
        cu_seqlen_ks=torch.zeros_like(lens),
        cu_seqlen_ke=lens,
        valid_lens=lens,
        cu_seqlens_kv=torch.arange(batch + 1, dtype=torch.int32),
        cu_seqlens_q=prefix,
        max_context_len=33,
        max_q_len=tokens // batch,
        max_kv_len=1,
    )
    return IluDeepseekV32IndexerMetadata(
        seq_lens=torch.ones(batch, dtype=torch.int32),
        num_reqs=batch,
        max_query_len=tokens // batch,
        max_seq_len=1,
        num_actual_tokens=tokens,
        query_start_loc=prefix,
        slot_mapping=torch.zeros(tokens, dtype=torch.int32),
        head_dim=128,
        num_decodes=batch,
        num_decode_tokens=tokens,
        num_prefills=int(prefill),
        num_prefill_tokens=int(prefill),
        decode=decode,
    )


def test_disabled_owner_never_reads_forward_context(monkeypatch, default_vllm_config):
    sparse = _load_sparse_attn_indexer(monkeypatch)
    layer = _owned_test_layer(sparse, enabled=False)
    calls = []

    def unexpected_context():
        raise AssertionError("disabled owner must delegate without metadata lookup")

    monkeypatch.setattr(sparse, "get_forward_context", unexpected_context)
    with patch.object(torch.ops.vllm, "ilu_sparse_attn_indexer",
                      lambda *args: calls.append(args) or args[10], create=True):
        output = layer.forward_cuda(torch.empty(6, 8), torch.empty(6, 32, 128),
                                    torch.empty(6, 128), torch.empty(6, 32))
    assert output is layer.topk_indices_buffer
    assert len(calls) == 1 and calls[0][-2:] == (None, None)


@pytest.mark.parametrize("kind", ["full6", "full12", "full18", "full24", "prefill", "padded",
                                 "unsupported", "none", "missing", "list"])
def test_owned_buffer_selection_in_forward_context(monkeypatch, default_vllm_config, kind):
    from vllm.forward_context import set_forward_context

    sparse = _load_sparse_attn_indexer(monkeypatch)
    layer = _owned_test_layer(sparse, enabled=True, max_seqs=4)
    tokens = int(kind[4:]) if kind.startswith("full") else 8 if kind == "unsupported" else 6
    md = _owned_test_metadata(tokens, prefill=kind == "prefill", padded=kind == "padded")
    metadata = {layer.k_cache.prefix: md}
    if kind == "none":
        metadata = None
    elif kind == "missing":
        metadata = {}
    elif kind == "list":
        metadata = [metadata]
    calls = []
    with patch.object(torch.ops.vllm, "ilu_sparse_attn_indexer",
                      lambda *args: calls.append(args) or args[10], create=True):
        with set_forward_context(metadata, default_vllm_config):
            output = layer.forward_cuda(torch.empty(tokens, 8), torch.empty(tokens, 32, 128),
                                        torch.empty(tokens, 128), torch.empty(tokens, 32))
    assert output is layer.topk_indices_buffer and len(calls) == 1
    if kind.startswith("full"):
        logits, extent = calls[0][-2:]
        assert logits.data_ptr() == layer._owned_decode_logits.data_ptr()
        assert extent.data_ptr() == layer._owned_decode_extent.data_ptr()
        assert logits.shape == (tokens, 33) and logits.stride(0) == 64
        assert logits.dtype == torch.bfloat16 and extent.dtype == torch.int32
        assert torch.all(extent == -1)
        assert layer._owned_decode_logits.shape == (24, 64)
        assert not any("owned_decode" in name for name in layer.state_dict())
    else:
        assert calls[0][-2:] == (None, None)


@pytest.mark.parametrize("max_seqs", [0, 1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("enabled", [False, True])
def test_owned_allocation_matches_configured_sequence_limit(
    monkeypatch, default_vllm_config, max_seqs, enabled
):
    sparse = _load_sparse_attn_indexer(monkeypatch)
    layer = _owned_test_layer(sparse, enabled=enabled, max_seqs=max_seqs)
    expected = tuple(range(6, max_seqs * 6 + 1, 6)) if enabled and 1 <= max_seqs <= 4 else ()
    assert layer._owned_decode_tokens == expected
    buffers = dict(layer.named_buffers())
    assert set(buffers) == ({"_owned_decode_logits", "_owned_decode_extent"} if expected else set())
    assert sum(t.numel() * t.element_size() for t in buffers.values()) == (
        (expected[-1] if expected else 0) * (64 * 2 + 4))
    ranges = sorted((t.untyped_storage().data_ptr(),
                     t.untyped_storage().data_ptr() + t.untyped_storage().nbytes())
                    for t in buffers.values())
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
    assert not layer.state_dict()


def test_shared_owner_views_follow_module_storage_transform(monkeypatch, default_vllm_config):
    from vllm.forward_context import set_forward_context

    sparse = _load_sparse_attn_indexer(monkeypatch)
    layer = _owned_test_layer(sparse, enabled=True, max_seqs=3)
    original = layer._owned_decode_logits
    layer._apply(lambda tensor: tensor.clone())
    assert original.data_ptr() != layer._owned_decode_logits.data_ptr()
    calls = []
    with patch.object(
        torch.ops.vllm,
        "ilu_sparse_attn_indexer",
        lambda *args: calls.append(args) or args[10],
        create=True,
    ):
        for tokens in (6, 18, 12, 6):
            metadata = {layer.k_cache.prefix: _owned_test_metadata(tokens)}
            with set_forward_context(metadata, default_vllm_config):
                layer.forward_cuda(
                    torch.empty(tokens, 8),
                    torch.empty(tokens, 32, 128),
                    torch.empty(tokens, 128),
                    torch.empty(tokens, 32),
                )
            logits, extent = calls[-1][-2:]
            assert logits.data_ptr() == layer._owned_decode_logits.data_ptr()
            assert extent.data_ptr() == layer._owned_decode_extent.data_ptr()
            assert logits.stride(0) == 64


@pytest.mark.parametrize("max_seqs", [1, 2, 3, 4])
def test_owned_graph_shapes_share_physical_extent(
    monkeypatch, default_vllm_config, max_seqs
):
    """Shrink/grow and slot reuse must clear the last physical row contents."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    from ixformer.inference.functions.deepseek.int8_split_cache import (
        dsa_indexer_mqa_logits_split,
    )

    sparse = _load_sparse_attn_indexer(monkeypatch)
    width, length, bs = 8193, 4097, 64
    tokens = max_seqs * 6
    layer = sparse.IluSparseAttnIndexer(
        k_cache=SimpleNamespace(prefix="layer.0.indexer", kv_cache=object()),
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=512,
        head_dim=128,
        max_model_len=width,
        max_total_seq_len=width * max_seqs,
        topk_indices_buffer=torch.empty(
            (tokens, 512), device="cuda", dtype=torch.int32
        ),
        owned_decode_state=True,
        max_num_decode_seqs=max_seqs,
    )
    torch.manual_seed(1026)
    pages = (length + bs - 1) // bs
    values = torch.randint(
        0, 256, (max_seqs * pages, bs, 128), device="cuda", dtype=torch.uint8
    )
    scales = torch.rand(max_seqs * pages, bs, device="cuda") * 0.02 + 0.01
    q = torch.randint(-127, 128, (tokens, 32, 128), device="cuda", dtype=torch.int8)
    weights = torch.randn(tokens, 32, device="cuda") * 0.001
    table = torch.randperm(max_seqs * pages, device="cuda").int().view(max_seqs, pages)
    ends = torch.full((tokens,), length, device="cuda", dtype=torch.int32)
    cq = torch.arange(max_seqs + 1, device="cuda", dtype=torch.int32) * 6
    ck = torch.arange(max_seqs + 1, device="cuda", dtype=torch.int32) * length
    graphs = {}

    def run(batch, owned=True):
        n = batch * 6
        return dsa_indexer_mqa_logits_split(
            q[:n],
            cq[: batch + 1],
            ck[: batch + 1],
            values,
            scales,
            table[:batch],
            weights[:n],
            max_q_len=6,
            max_kv_len=length,
            max_context_len=width,
            valid_lens=ends[:n],
            logits=layer._owned_decode_logits[:n, :width] if owned else None,
            previous_extent=layer._owned_decode_extent[:n] if owned else None,
        )

    for batch in range(1, max_seqs + 1):
        run(batch)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run(batch)
        graphs[batch] = g
    for step, (batch, end) in enumerate(
        [
            (1, 17),
            (max_seqs, length),
            (max(1, max_seqs - 1), 65),
            (1, 0),
            (max_seqs, 1),
            (1, length),
            (max_seqs, 0),
            (max_seqs, length),
        ]
    ):
        ends.fill_(end)
        if step % 2:
            table.copy_(table.roll(1, 1))
            scales.mul_(1.01)
        untouched = layer._owned_decode_logits[batch * 6 :].clone()
        untouched_extent = layer._owned_decode_extent[batch * 6 :].clone()
        graphs[batch].replay()
        expected = run(batch, False)
        torch.cuda.synchronize()
        assert torch.equal(layer._owned_decode_logits[: batch * 6, :width], expected)
        # Padding is intentionally uninitialized and may contain NaN bits;
        # guard untouched storage bytewise, including that padding.
        assert torch.equal(
            layer._owned_decode_logits[batch * 6 :].view(torch.uint8),
            untouched.view(torch.uint8),
        )
        assert torch.equal(layer._owned_decode_extent[batch * 6 :], untouched_extent)


@pytest.mark.parametrize("max_seqs", [1, 2, 3, 5])
def test_unallocated_decode_shape_delegates_original_path(
    monkeypatch, default_vllm_config, max_seqs
):
    from vllm.forward_context import set_forward_context

    sparse = _load_sparse_attn_indexer(monkeypatch)
    layer = _owned_test_layer(sparse, enabled=True, max_seqs=max_seqs)
    tokens = 6 * (max_seqs + 1) if max_seqs < 4 else 6
    metadata = {layer.k_cache.prefix: _owned_test_metadata(tokens)}
    calls = []
    with patch.object(torch.ops.vllm, "ilu_sparse_attn_indexer",
                      lambda *args: calls.append(args) or args[10], create=True):
        with set_forward_context(metadata, default_vllm_config):
            layer.forward_cuda(torch.empty(tokens, 8), torch.empty(tokens, 32, 128),
                               torch.empty(tokens, 128), torch.empty(tokens, 32))
    assert len(calls) == 1 and calls[0][-2:] == (None, None)


def test_active_owner_requires_forward_context(monkeypatch, default_vllm_config):
    from vllm.forward_context import override_forward_context

    sparse = _load_sparse_attn_indexer(monkeypatch)
    layer = _owned_test_layer(sparse, enabled=True)
    with override_forward_context(None):
        with pytest.raises(AssertionError, match="Forward context is not set"):
            layer.forward_cuda(torch.empty(6, 8), torch.empty(6, 32, 128),
                               torch.empty(6, 128), torch.empty(6, 32))
