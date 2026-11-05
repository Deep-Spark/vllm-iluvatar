# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""ixaite MiniMax sparse backend / gather / fail-closed gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from vllm_iluvatar.backends.ixaite.models.minimax_m3 import attention as attn_mod
from vllm_iluvatar.backends.ixaite.models.minimax_m3.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseIxaiteImpl,
    gather_paged_kv_nhd,
    minimax_m3_sparse_attn_decode,
)

_IXAITE_MINIMAX = (
    Path(__file__).resolve().parents[4]
    / "vllm_iluvatar"
    / "backends"
    / "ixaite"
    / "models"
    / "minimax_m3"
)


def test_sparse_modules_do_not_import_nvidia_msa_or_ixformer():
    forbidden = ("ixformer", "flash_attn", "flashinfer", "triton")
    for path in _IXAITE_MINIMAX.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(
                        alias.name == f or alias.name.startswith(f + ".")
                        for f in forbidden
                    ), f"{path.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not any(
                    node.module == f or node.module.startswith(f + ".")
                    for f in forbidden
                ), f"{path.name} imports from {node.module}"


def test_backend_name_and_kv_shape_contract():
    assert "IXAITE" in MiniMaxM3SparseBackend.get_name()
    assert MiniMaxM3SparseBackend.get_impl_cls() is MiniMaxM3SparseIxaiteImpl
    shape = MiniMaxM3SparseBackend.get_kv_cache_shape(8, 128, 4, 128, "bfloat16")
    assert shape == (8, 2, 128, 4, 128)
    with pytest.raises(NotImplementedError, match="fp8"):
        MiniMaxM3SparseBackend.get_kv_cache_shape(8, 128, 4, 128, "fp8")


def test_ixaite_indexer_owns_side_cache_module():
    # Construction needs a live vllm config for MiniMaxM3IndexerCache registration.
    # Shape/API smoke only: helper decode path stays device-free.
    from vllm_iluvatar.backends.ixaite.models.minimax_m3.indexer import (
        IxaiteMiniMaxM3Indexer,
        decode_indexer_topk,
    )

    q = torch.randn(1, 2, 16, dtype=torch.bfloat16)
    k = torch.randn(1, 64, 1, 16, dtype=torch.bfloat16)
    seq = torch.tensor([64], dtype=torch.int32)
    topk = decode_indexer_topk(
        q, k, seq, block_size=16, topk=4, init_blocks=0, local_blocks=1
    )
    assert topk.shape == (2, 1, 4)
    assert IxaiteMiniMaxM3Indexer is not None


def test_gather_paged_kv_nhd_roundtrip():
    blocks, page, hkv, d = 4, 16, 2, 8
    kv = torch.randn(blocks, 2, page, hkv, d, dtype=torch.bfloat16)
    # One request using blocks 1 then 0 for 24 tokens.
    block_table = torch.tensor([[1, 0, -1, -1]], dtype=torch.int32)
    seq_lens = torch.tensor([24], dtype=torch.int32)
    k, v = gather_paged_kv_nhd(
        kv, block_table, seq_lens, num_kv_heads=hkv, block_size=None
    )
    assert k.shape == (1, 24, hkv, d)
    assert torch.equal(k[0, :16], kv[1, 0])
    assert torch.equal(k[0, 16:24], kv[0, 0, :8])
    assert torch.equal(v[0, :16], kv[1, 1])


def test_sparse_decode_reference_smoke():
    batch, hq, hkv, d, page, topk, n_blocks = 1, 4, 2, 16, 16, 2, 4
    seq = page * n_blocks
    kv = torch.randn(n_blocks, 2, page, hkv, d, dtype=torch.bfloat16)
    block_table = torch.arange(n_blocks, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([seq], dtype=torch.int32)
    q = torch.randn(batch, hq, d, dtype=torch.bfloat16)
    # token-major topk [Q, Hkv, topk]
    topk_idx = (
        torch.arange(topk, dtype=torch.int32)
        .view(1, 1, topk)
        .expand(batch, hkv, topk)
        .contiguous()
    )
    out = torch.empty_like(q)
    minimax_m3_sparse_attn_decode(
        q,
        kv,
        topk_idx,
        block_table,
        seq_lens,
        hkv,
        d**-0.5,
        out,
        1,
        block_size=page,
        reference=True,
    )
    assert out.shape == q.shape
    assert torch.isfinite(out.float()).all()


def test_fused_qk_insert_requires_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA present; CPU fail-closed path not exercised")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        attn_mod.fused_minimax_m3_qknorm_rope_kv_insert(
            torch.zeros(1, 48),
            torch.ones(16),
            torch.ones(16),
            torch.zeros(8, 32),
            torch.zeros(1, dtype=torch.long),
            2,
            1,
            16,
            1e-6,
        )


def test_indexer_reference_topk_and_guards():
    from vllm_iluvatar.backends.ixaite.models.minimax_m3.indexer import (
        decode_indexer_topk,
        require_supported_indexer,
    )

    require_supported_indexer(score_type="max", indexer_kv_dtype="bf16")
    with pytest.raises(NotImplementedError, match="score_type"):
        require_supported_indexer(score_type="mean")

    b, hq, hkv, d, s, bs, topk = 1, 4, 2, 16, 64, 16, 4
    q = torch.randn(b, hq, d, dtype=torch.bfloat16)
    k = torch.randn(b, s, hkv, d, dtype=torch.bfloat16)
    seq = torch.tensor([s], dtype=torch.int32)
    topk_idx = decode_indexer_topk(q, k, seq, block_size=bs, topk=topk, reference=True)
    assert topk_idx.shape == (hq, b, topk)


def test_sparse_attention_construct_binds_ixaite_impl(monkeypatch):
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    fake_cfg = type(
        "C",
        (),
        {
            "model_config": type("M", (), {"dtype": torch.bfloat16})(),
            "attention_config": type("A", (), {"indexer_kv_dtype": "auto"})(),
            "compilation_config": type("Comp", (), {"static_forward_context": {}})(),
        },
    )()
    monkeypatch.setattr(attn_mod, "get_current_vllm_config", lambda: fake_cfg)
    # MiniMaxM3IndexerCache reads config from vllm.config directly.
    import vllm.config as vllm_config_mod
    import vllm.models.minimax_m3.common.indexer as community_indexer

    monkeypatch.setattr(vllm_config_mod, "get_current_vllm_config", lambda: fake_cfg)
    monkeypatch.setattr(community_indexer, "get_current_vllm_config", lambda: fake_cfg)
    monkeypatch.setattr(
        attn_mod,
        "kv_cache_dtype_str_to_dtype",
        lambda *_: torch.bfloat16,
    )
    monkeypatch.setattr(attn_mod, "set_default_quant_scales", lambda *a, **k: None)
    monkeypatch.setattr(
        attn_mod,
        "MinimaxM3QKVParallelLinearWithIndexer",
        lambda *a, **k: torch.nn.Linear(8, 8),
    )
    monkeypatch.setattr(
        attn_mod, "RowParallelLinear", lambda *a, **k: torch.nn.Linear(8, 8)
    )
    monkeypatch.setattr(
        attn_mod,
        "MiniMAXGemmaRMSNorm",
        lambda *a, **k: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        attn_mod,
        "get_rope",
        lambda *a, **k: type(
            "R", (), {"cos_sin_cache": torch.zeros(1), "rotary_dim": 16}
        )(),
    )

    config = type(
        "Cfg",
        (),
        {
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "sparse_attention_config": {
                "sparse_num_index_heads": 2,
                "sparse_index_dim": 16,
                "sparse_topk_blocks": 4,
                "sparse_block_size": 16,
            },
        },
    )()
    layer = attn_mod.MiniMaxM3SparseAttention(
        config, layer_id=0, quant_config=None, prefix="layers.0.self_attn"
    )
    assert layer.attn_backend is MiniMaxM3SparseBackend
    assert isinstance(layer.impl, MiniMaxM3SparseIxaiteImpl)
    assert layer.impl.topk_blocks == 4
    assert layer.impl.block_size == 16
