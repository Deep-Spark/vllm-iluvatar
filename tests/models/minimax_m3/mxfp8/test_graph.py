# SPDX-License-Identifier: Apache-2.0
"""mxfp8 MiniMax-M3 graph and attention contracts."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest
import torch

PACKAGE = "vllm_iluvatar.models.minimax_m3.mxfp8"
_MODEL_PY = (
    Path(__file__).resolve().parents[4]
    / "vllm_iluvatar"
    / "models"
    / "minimax_m3"
    / "mxfp8"
    / "model.py"
)


def test_routed_experts_receive_the_config_swiglu_parameters():
    tree = ast.parse(_MODEL_PY.read_text(), filename=str(_MODEL_PY))
    moe_cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxM3MoE"
    )
    factory = next(
        node
        for node in ast.walk(moe_cls)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FusedMoEFactory"
    )
    passed = {
        kw.arg: ast.unparse(kw.value) for kw in factory.keywords if kw.arg is not None
    }
    for name in ("swiglu_limit", "swiglu_alpha", "swiglu_beta"):
        assert passed.get(name) == f"config.{name}", (
            f"MiniMaxM3MoE must forward {name} to the routed experts"
        )
    assert passed.get("activation") == "'swigluoai_uninterleave'"


def test_dense_mlp_uses_community_act_then_linear():
    tree = ast.parse(_MODEL_PY.read_text(), filename=str(_MODEL_PY))
    mlp_cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxM3MLP"
    )
    init_fn = next(
        node
        for node in mlp_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    forward_fn = next(
        node
        for node in mlp_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    assigned = {
        ast.unparse(stmt.targets[0]): ast.unparse(stmt.value)
        for stmt in init_fn.body
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
    }
    assert assigned.get("self.act_fn", "").startswith("SiluAndMulWithClamp(")
    source = ast.unparse(forward_fn)
    assert "self.act_fn(gate_up)" in source
    assert "self.down_proj(x)" in source
    assert "enable_fuse_quant" not in ast.unparse(mlp_cls)
    assert "swigluoai_uninterleave_dynamic_scaled_fp8" not in ast.unparse(mlp_cls)


def test_reuses_community_common_graph():
    """Indexer / sparse-attention metadata live in community ``common``."""
    indexer = importlib.import_module(f"{PACKAGE}.indexer")
    sparse = importlib.import_module(f"{PACKAGE}.sparse_attention")
    common_indexer = importlib.import_module("vllm.models.minimax_m3.common.indexer")
    common_sparse = importlib.import_module(
        "vllm.models.minimax_m3.common.sparse_attention"
    )

    assert indexer.MiniMaxM3IndexerCache is common_indexer.MiniMaxM3IndexerCache
    assert indexer.MiniMaxM3IndexerMetadata is common_indexer.MiniMaxM3IndexerMetadata
    assert (
        indexer.MiniMaxM3IndexerTritonMetadataBuilder
        is common_indexer.MiniMaxM3IndexerTritonMetadataBuilder
    )
    assert sparse.MiniMaxM3SparseMetadata is common_sparse.MiniMaxM3SparseMetadata
    assert (
        sparse.MiniMaxM3SparseMetadataBuilder
        is common_sparse.MiniMaxM3SparseMetadataBuilder
    )
    assert issubclass(
        indexer.MiniMaxM3IndexerBackend, common_indexer.MiniMaxM3IndexerBackend
    )
    assert issubclass(
        sparse.MiniMaxM3SparseBackend, common_sparse.MiniMaxM3SparseBackend
    )


def test_sparse_decode_is_not_eager_broken():
    """Indexer + sparse attend must stay inside FULL_DECODE_ONLY capture."""
    model = importlib.import_module(f"{PACKAGE}.attention")
    sparse_source = inspect.getsource(model.MiniMaxM3SparseAttention._run_attention)
    assert "eager_break_during_capture" not in sparse_source


def test_require_unquantized_m3_caches_rejects_fp8():
    from vllm_iluvatar.models.minimax_m3.mxfp8.attention import (
        _require_unquantized_m3_caches,
    )

    _require_unquantized_m3_caches("auto", torch.bfloat16, "bf16")
    _require_unquantized_m3_caches("bfloat16", torch.bfloat16, "bf16")
    with pytest.raises(NotImplementedError, match="kv_cache_dtype"):
        _require_unquantized_m3_caches("fp8", torch.float8_e4m3fn, "bf16")
    with pytest.raises(NotImplementedError, match="indexer_kv_dtype"):
        _require_unquantized_m3_caches("auto", torch.bfloat16, "fp8")
    with pytest.raises(NotImplementedError, match="resolved to"):
        _require_unquantized_m3_caches("auto", torch.float16, "bf16")
