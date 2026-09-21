# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar sparse attention indexer CustomOp wrapper."""

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
    functions.dsa_indexer_mqa_logits = object()
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
