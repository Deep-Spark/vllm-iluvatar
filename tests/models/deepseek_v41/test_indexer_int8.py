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

    monkeypatch.setattr(store.ixfops, "indexer_k_norm_rope_store", fake)
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

    monkeypatch.setattr(store.ixfops, "indexer_k_norm_rope_store", fake)
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


@pytest.mark.parametrize("record", ["bf16", "fp8"])
def test_indexer_prequantized_w8a8_bypasses_tensor_projection(record, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm_iluvatar.models.deepseek_v41 import attention as mod

    indexer = mod.IluDeepseekV41Indexer.__new__(mod.IluDeepseekV41Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.owns_k = False
    indexer.record = record
    indexer.n_head, indexer.head_dim = 2, 4
    indexer.softmax_scale = 0.5
    query = (torch.ones(3, 8, dtype=torch.int8), torch.ones(3, 1), torch.bfloat16)
    projected = torch.ones(3, 8, dtype=torch.bfloat16)
    indexer.wq_b = Mock(return_value=(projected, None))
    indexer.wq_b.weight = torch.ones(8, 8, dtype=torch.int8)
    indexer.q_bf16_buffer = torch.empty(3, 8, dtype=torch.bfloat16)
    indexer.q_fp8_buffer = torch.empty(3, 2, 4, dtype=torch.float8_e4m3fn)
    indexer.weights_fp32_buffer = torch.empty(3, 2)
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
    )
    monkeypatch.setattr(
        mod.ixfops,
        "fused_wqb_rope_fp8",
        Mock(side_effect=AssertionError("tuple entered fused projection")),
    )

    def prepare(q, positions, cache, weights, factor, output, folded):
        output.copy_(q)
        folded.copy_(weights * factor)

    monkeypatch.setattr(mod.ixfops, "indexer_q_rope_fp8_out", prepare)
    monkeypatch.setattr(mod.ixfops, "indexer_q_rope", lambda *args: None)
    weights = torch.ones(3, 2)
    result = indexer(
        query,
        None,
        weights,
        torch.arange(3),
        SimpleNamespace(cos_sin_cache=torch.ones(3, 4)),
    )
    indexer.wq_b.assert_called_once()
    assert indexer.wq_b.call_args.args[0] is query
    assert result[0].shape == (3, 2, 4)
    assert torch.equal(result[0].float(), projected.reshape(3, 2, 4).float())
