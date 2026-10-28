# SPDX-License-Identifier: Apache-2.0
"""Check default decode dispatch and index representation without a model."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "enabled,compact,prefills,heads,expected",
    [
        (None, True, 0, 32, "pool4"),
        (False, True, 0, 32, "paged"),
        (True, True, 0, 32, "pool4"),
        (False, True, 1, 32, "pool4"),
        (True, True, 1, 32, "pool4"),
        (True, False, 0, 32, "paged"),
        (True, False, 1, 32, "token"),
        (True, True, 0, 16, "paged"),
        (True, True, 1, 16, "pool4"),
    ],
)
def test_pool4_decode_dispatch(
    monkeypatch, enabled, compact, prefills, heads, expected
):
    from vllm_iluvatar.envs import _ILU_CANONICAL_ONLY_ENV_VARS
    from vllm_iluvatar.v1.attention.backends.mla import ilu_flashmla_sparse as mod

    if enabled is None:
        key = "VLLM_ILU_GLM5NEXT_MLA_POOL4_DECODE"
        monkeypatch.delenv(key, raising=False)
        enabled = _ILU_CANONICAL_ONLY_ENV_VARS[key]()
    impl = mod.IluFlashMLASparseImpl.__new__(mod.IluFlashMLASparseImpl)
    impl._indexer = SimpleNamespace(compact_pool_indices=compact)
    impl.use_pool4_decode = enabled
    impl.softmax_scale = 0.0625
    q = torch.zeros(3, heads, 512, dtype=torch.bfloat16)
    kv = torch.zeros(128, 1, 512, dtype=torch.bfloat16)
    selection = torch.zeros(3, 512 if compact else 2048, dtype=torch.int32)
    lengths = torch.tensor([1, 2, 3], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_prefills=prefills,
        num_decodes=1,
        topk_tokens=2048,
        req_id_per_token=None,
        block_table=None,
        seq_lens=None,
        query_start_loc=None,
        block_size=128,
    )
    calls = []

    def global_pool_indices(*args, expand):
        calls.append(("expand", expand))
        return torch.zeros(3, 2176 if expand else 576, dtype=torch.int32), lengths

    def paged(query, cache, indices, valid_lengths, scale):
        calls.append(("paged", indices.shape[-1]))
        assert valid_lengths is lengths
        return torch.zeros_like(query)

    def sparse(query, cache, indices, valid_lengths, kpool):
        calls.append(("sparse", kpool))
        assert valid_lengths is lengths
        return torch.zeros_like(query)

    monkeypatch.setattr(mod, "global_pool_indices", global_pool_indices)
    monkeypatch.setattr(
        mod.ixf_ops,
        "global_topk_indices_and_lens",
        lambda *args: (torch.zeros(3, 2176, dtype=torch.int32), lengths),
    )
    monkeypatch.setattr(mod.ixfops, "flash_mla_sparse_with_kvcache", paged)
    monkeypatch.setattr(impl, "_bf16_flash_mla_kernel", sparse)
    result = impl._forward_bf16_kv(q, kv, selection, metadata)
    assert result.shape == q.shape
    if compact:
        assert calls[0] == ("expand", expected == "paged")
    assert calls[-1] == (
        ("paged", 2176)
        if expected == "paged"
        else ("sparse", 4 if expected == "pool4" else 1)
    )


def test_pool4_decode_default_and_override(monkeypatch):
    from vllm_iluvatar.envs import _ILU_CANONICAL_ONLY_ENV_VARS

    key = "VLLM_ILU_GLM5NEXT_MLA_POOL4_DECODE"
    monkeypatch.delenv(key, raising=False)
    assert _ILU_CANONICAL_ONLY_ENV_VARS[key]() is True
    monkeypatch.setenv(key, "0")
    assert _ILU_CANONICAL_ONLY_ENV_VARS[key]() is False
    monkeypatch.setenv(key, "1")
    assert _ILU_CANONICAL_ONLY_ENV_VARS[key]() is True


@pytest.mark.parametrize(
    "arch,cache,kpool,width",
    [
        ("ivcore11", "bfloat16", 4, 512),
        ("ivcore11", "auto", 4, 512),
        ("ivcore11", "fp8", 4, 2176),
        ("ivcore30", "bfloat16", 4, 2176),
        ("ivcore11", "bfloat16", 1, 2048),
    ],
)
def test_logical_topk_width_and_mtp_row_compaction(
    monkeypatch, arch, cache, kpool, width
):
    from vllm.models.glm5next.common.mtp import Glm5NextMultiTokenPredictor

    from vllm_iluvatar.models.glm5next.attention import glm5next_topk_buffer_width

    monkeypatch.setattr(torch, "corex_arch", arch, raising=False)
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    assert glm5next_topk_buffer_width(cfg, 2048, kpool) == width
    indices = torch.arange(6 * width, dtype=torch.int32).view(6, width)
    selected = torch.tensor([5, 0, 3])
    expected = indices[selected].clone()
    model = SimpleNamespace(
        _mtp_mla_attns=[SimpleNamespace(topk_indices_buffer=indices)]
    )
    Glm5NextMultiTokenPredictor.compact_topk_indices(model, selected)
    torch.testing.assert_close(indices[:3], expected)
