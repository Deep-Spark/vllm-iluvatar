# SPDX-License-Identifier: Apache-2.0
"""Check default decode dispatch and index representation without a model."""

from types import SimpleNamespace

import pytest
import torch


def _config(cache="bfloat16", **shape):
    text = dict(
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        num_attention_heads=64,
    )
    text.update(shape)
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, hf_text_config=SimpleNamespace(**text)
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )


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
        (True, True, 0, 8, "paged"),
        (True, True, 1, 8, "token"),
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
    impl.use_ixattn_kpool4 = False
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
        assert calls[0] == ("expand", expected != "pool4")
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
        ("ivcore30", "bfloat16", 4, 512),
        ("ivcore11", "bfloat16", 1, 2048),
    ],
)
def test_logical_topk_width_and_mtp_row_compaction(
    monkeypatch, arch, cache, kpool, width
):
    from vllm.models.glm5next.common.mtp import Glm5NextMultiTokenPredictor

    from vllm_iluvatar.models.glm5next.attention import glm5next_topk_buffer_width

    monkeypatch.setattr(torch, "corex_arch", arch, raising=False)
    monkeypatch.setattr(
        "ixformer.inference.ixattn_backend.ixattn_available", lambda: True
    )
    monkeypatch.setenv("VLLM_ILU_MLA_ATTN_QUANT_TYPE", "BF16")
    cfg = _config(cache)
    assert glm5next_topk_buffer_width(cfg, 2048, kpool) == width
    indices = torch.arange(6 * width, dtype=torch.int32).view(6, width)
    selected = torch.tensor([5, 0, 3])
    expected = indices[selected].clone()
    model = SimpleNamespace(
        _mtp_mla_attns=[SimpleNamespace(topk_indices_buffer=indices)]
    )
    Glm5NextMultiTokenPredictor.compact_topk_indices(model, selected)
    torch.testing.assert_close(indices[:3], expected)


@pytest.mark.parametrize(
    "cache_bytes,prefills,decodes,heads,explicit_backend,expected_kpool,expected_backend",
    [
        ((1 << 31) + 1024, 1, 0, 64, None, 4, None),
        ((1 << 31) + 1024, 1, 1, 64, None, 4, None),
        (4096, 0, 1, 64, None, 4, None),
        (4096, 1, 0, 64, "ixinfer", 4, "ixinfer"),
        (4096, 1, 0, 8, None, 1, None),
    ],
)
def test_compact_kpool4_dispatch_without_env(
    monkeypatch,
    cache_bytes,
    prefills,
    decodes,
    heads,
    explicit_backend,
    expected_kpool,
    expected_backend,
):
    from vllm_iluvatar.v1.attention.backends.mla import ilu_flashmla_sparse as mod

    impl = mod.IluFlashMLASparseImpl.__new__(mod.IluFlashMLASparseImpl)
    impl._indexer = SimpleNamespace(compact_pool_indices=True)
    impl.use_ixattn_kpool4 = False
    impl.use_pool4_decode = True
    impl.softmax_scale = 0.0625
    impl.num_heads = heads
    q = torch.empty(512, heads, 512, dtype=torch.bfloat16, device="meta")
    kv = torch.empty(cache_bytes // 1024, 1, 512, dtype=torch.bfloat16, device="meta")
    selection = torch.empty(512, 512, dtype=torch.int32, device="meta")
    lengths = torch.empty(512, dtype=torch.int32, device="meta")
    metadata = SimpleNamespace(
        num_prefills=prefills,
        num_decodes=decodes,
        topk_tokens=2048,
        req_id_per_token=None,
        block_table=None,
        seq_lens=None,
        query_start_loc=None,
        block_size=128,
    )
    calls = []

    def producer(*args, expand):
        calls.append(("expand", expand))
        return selection, lengths

    def sparse_fwd(**kwargs):
        calls.append(("fwd", kwargs["kpool"], kwargs.get("backend")))
        assert kwargs["topk_length"] is lengths
        return torch.empty_like(q)

    monkeypatch.setattr(mod, "global_pool_indices", producer)
    monkeypatch.setattr(mod.ixfops, "flash_mla_sparse_fwd", sparse_fwd)
    out = impl._forward_bf16_kv(q, kv, selection, metadata, backend=explicit_backend)
    assert out.shape == q.shape
    assert calls[0] == ("expand", expected_kpool != 4)
    assert calls[-1] == ("fwd", expected_kpool, expected_backend)


@pytest.mark.parametrize("ixattn,int8", [(False, False), (True, True)])
def test_cq_compact_pool_gate_preserves_other_backends(monkeypatch, ixattn, int8):
    from vllm_iluvatar.models.glm5next.attention import glm5next_topk_buffer_width

    monkeypatch.setattr(torch, "corex_arch", "ivcore30", raising=False)
    monkeypatch.setattr(
        "ixformer.inference.ixattn_backend.ixattn_available", lambda: ixattn
    )
    monkeypatch.setenv("VLLM_ILU_MLA_ATTN_QUANT_TYPE", "INT8" if int8 else "BF16")
    cfg = _config()
    assert glm5next_topk_buffer_width(cfg, 2048, 4) == 2176


@pytest.mark.parametrize(
    "shape",
    [
        {"qk_rope_head_dim": 64},
        {"kv_lora_rank": 256},
        {"qk_nope_head_dim": 128},
        {"num_attention_heads": 16},
        {"num_attention_heads": 65},
    ],
)
def test_cq_compact_pool_requires_supported_attention(monkeypatch, shape):
    from vllm_iluvatar.models.glm5next.attention import glm5next_topk_buffer_width

    monkeypatch.setattr(torch, "corex_arch", "ivcore30", raising=False)
    monkeypatch.setattr(
        "ixformer.inference.ixattn_backend.ixattn_available", lambda: True
    )
    monkeypatch.setenv("VLLM_ILU_MLA_ATTN_QUANT_TYPE", "BF16")
    assert glm5next_topk_buffer_width(_config(**shape), 2048, 4) == 2176


def test_mtp_reuses_physical_rows_and_frozen_tail(monkeypatch):
    from ixformer.inference.functions import (
        ref_gather_kpool4_rows,
        ref_pool_indices_to_ixattn_kpool4,
    )

    from vllm_iluvatar.models.glm5next.mtp import IluGlm5NextMultiTokenPredictor
    from vllm_iluvatar.v1.attention.backends.mla import ilu_flashmla_sparse as mod

    impl = mod.IluFlashMLASparseImpl.__new__(mod.IluFlashMLASparseImpl)
    impl.use_ixattn_kpool4 = impl.use_ixattn_compact_pools = True
    impl._reuse_ixattn_pool_indices = False
    impl._ixattn_pool_starts = torch.empty((6, 1, 64), dtype=torch.int32)
    impl._ixattn_token_counts = torch.empty(6, dtype=torch.int32)
    impl._ixattn_decode_plans = [(0, 1)] * 7
    impl._ixattn_decode_workspace = None
    impl.softmax_scale = 1 / 16
    assert impl.configure_mtp_pool_reuse(3)
    indices = torch.arange(4, dtype=torch.int32).repeat(6, 1)
    positions = torch.tensor([8, 9, 10, 11, 12, 13])
    metadata = SimpleNamespace(
        req_id_per_token=torch.zeros(6, dtype=torch.int32),
        block_table=torch.tensor([[2]], dtype=torch.int32),
        block_size=128,
        slot_mapping=torch.arange(6),
    )
    q = torch.zeros(6, 16, 512, dtype=torch.bfloat16)
    kv = torch.zeros(384, 1, 512, dtype=torch.bfloat16)
    calls = []
    observed = []

    def mapping(*args, **kwargs):
        calls.append("map")
        ref_pool_indices_to_ixattn_kpool4(*args, **kwargs)

    def attention(**kwargs):
        observed.append((kwargs["indices"].clone(), kwargs["topk_length"].clone()))
        return torch.zeros_like(kwargs["q"])

    monkeypatch.setattr(mod.ixfops, "pool_indices_to_ixattn_kpool4", mapping)
    monkeypatch.setattr(mod.ixfops, "gather_kpool4_rows", ref_gather_kpool4_rows)
    monkeypatch.setattr(mod.ixfops, "flash_mla_sparse_fwd", attention)
    model = IluGlm5NextMultiTokenPredictor.__new__(IluGlm5NextMultiTokenPredictor)
    torch.nn.Module.__init__(model)
    model._mtp_mla_attns = [
        SimpleNamespace(skip_topk=False, topk_indices_buffer=indices)
    ]
    model._mtp_pool_impls = [impl]
    model.set_skip_topk(False)
    impl._forward_bf16_kv(q, kv, indices, metadata, positions=positions)
    row_ids = torch.tensor([4, 0, 2])
    expected = tuple(t[row_ids].clone() for t in observed[-1])
    model.compact_topk_indices(row_ids)
    model.set_skip_topk(True)
    # Change all inputs that a remapping would read, including crossing a pool
    # boundary. The frozen selection must survive all four draft iterations.
    positions.add_(7)
    indices.fill_(777)
    metadata.block_table.fill_(0)
    impl._ixattn_pool_starts.fill_(-99)
    for _ in range(4):
        impl._forward_bf16_kv(q[:3], kv, indices[:3], metadata, positions=positions[:3])
        for got, want in zip(observed[-1], expected):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert calls == ["map"]
    model.set_skip_topk(False)
    indices.copy_(torch.arange(4, dtype=torch.int32))
    impl._forward_bf16_kv(q, kv, indices, metadata, positions=positions)
    assert calls == ["map", "map"]
    model.compact_topk_indices(row_ids[:1])
    assert torch.all(impl._ixattn_reuse_starts[1:] == -1)
    assert torch.all(impl._ixattn_reuse_counts[1:] == 0)


def test_mtp_legacy_logical_compaction_is_unchanged():
    from vllm_iluvatar.models.glm5next.mtp import IluGlm5NextMultiTokenPredictor

    model = IluGlm5NextMultiTokenPredictor.__new__(IluGlm5NextMultiTokenPredictor)
    torch.nn.Module.__init__(model)
    indices = torch.arange(6 * 512, dtype=torch.int32).view(6, 512)
    rows = torch.tensor([5, 0, 3])
    expected = indices[rows].clone()
    model._mtp_mla_attns = [
        SimpleNamespace(skip_topk=False, topk_indices_buffer=indices)
    ]
    model._mtp_pool_impls = [None]
    model.set_skip_topk(True)
    model.compact_topk_indices(rows)
    assert model._mtp_mla_attns[0].skip_topk
    torch.testing.assert_close(indices[:3], expected, atol=0, rtol=0)
