# SPDX-License-Identifier: Apache-2.0
"""CPU tests for DeepSeek-V4.1 INT8 indexer record and page split."""

from __future__ import annotations

import pytest
import torch


def test_indexer_kv_dtype_accepts_int8(monkeypatch):
    from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import indexer_kv_dtype

    monkeypatch.setenv("VLLM_ILU_V41_INDEXER_KV_DTYPE", "int8")
    assert indexer_kv_dtype() == "int8"


def test_indexer_kv_dtype_rejects_unknown(monkeypatch):
    from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import indexer_kv_dtype

    monkeypatch.setenv("VLLM_ILU_V41_INDEXER_KV_DTYPE", "fp4")
    with pytest.raises(ValueError, match="bf16, fp8, or int8"):
        indexer_kv_dtype()


def test_indexer_k_store_int8_record(monkeypatch):
    from vllm_iluvatar.models.deepseek_v41.common.ops import indexer_k_store as store

    captured = {}

    def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(store.ixfops, "dsv41_indexer_k_norm_rope_store", fake)
    k_pre = torch.zeros(2, 128, dtype=torch.bfloat16)
    positions = torch.zeros(2, dtype=torch.int64)
    cos_sin = torch.zeros(8, 64)
    weight = torch.zeros(128, dtype=torch.bfloat16)
    cache = torch.zeros(1, 64, 132, dtype=torch.uint8)
    slots = torch.arange(2, dtype=torch.int64)
    store.indexer_k_norm_rope_store(
        k_pre, positions, cos_sin, weight, 1e-6, cache, slots, 1, "int8"
    )
    assert captured["args"][8] == "int8"


def test_indexer_k_store_fp8_record(monkeypatch):
    from vllm_iluvatar.models.deepseek_v41.common.ops import indexer_k_store as store

    captured = {}

    def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(store.ixfops, "dsv41_indexer_k_norm_rope_store", fake)
    k_pre = torch.zeros(2, 128, dtype=torch.bfloat16)
    positions = torch.zeros(2, dtype=torch.int64)
    cos_sin = torch.zeros(8, 64)
    weight = torch.zeros(128, dtype=torch.bfloat16)
    cache = torch.zeros(1, 64, 132, dtype=torch.uint8)
    slots = torch.arange(2, dtype=torch.int64)
    store.indexer_k_norm_rope_store(
        k_pre, positions, cos_sin, weight, 1e-6, cache, slots, 1, "fp8"
    )
    assert captured["args"][8] == "fp8"


def test_indexer_k_store_rejects_unknown_record():
    from vllm_iluvatar.models.deepseek_v41.common.ops.indexer_k_store import (
        indexer_k_norm_rope_store,
    )

    k_pre = torch.zeros(1, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="indexer record"):
        indexer_k_norm_rope_store(
            k_pre,
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(4, 64),
            torch.zeros(128, dtype=torch.bfloat16),
            1e-6,
            torch.zeros(1, 64, 128, dtype=torch.bfloat16),
            torch.zeros(1, dtype=torch.int64),
            1,
            "fp4",
        )
