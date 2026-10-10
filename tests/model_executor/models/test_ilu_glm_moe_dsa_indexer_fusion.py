# SPDX-License-Identifier: Apache-2.0
"""Exercise GLM indexer preparation through the real plugin call chain."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from vllm_iluvatar.model_executor.layers import sparse_attn_indexer as sparse
from vllm_iluvatar.models.glm_moe_dsa import attention as attn
from vllm_iluvatar.v1.attention.backends.mla import ilu_indexer as backend


def rotate(x, positions, trig):
    angles = trig[positions].float()
    c, s = angles[:, :32], angles[:, 32:]
    if x.ndim == 3:
        c, s = c[:, None], s[:, None]
    even, odd = x[..., 0:64:2].float(), x[..., 1:64:2].float()
    a, b = even * c - odd * s, even * s + odd * c
    x[..., 0:64:2] = a.to(x.dtype)
    x[..., 1:64:2] = b.to(x.dtype)


def cache_insert(k, cache, slots):
    valid = slots >= 0
    cache[slots[valid] // cache.shape[1], slots[valid] % cache.shape[1]] = k[valid]


def reference_ops(monkeypatch):
    def q_op(q, weights, positions, trig, softmax_scale, head_scale):
        rotate(q, positions, trig)
        return (weights * softmax_scale) * head_scale

    def k_op(k, weight, bias, positions, trig, cache, slots, eps):
        assert weight.dtype == bias.dtype == torch.float32
        norm = F.layer_norm(k.float(), (128,), weight, bias, eps).to(k.dtype)
        rotate(norm, positions, trig)
        cache_insert(norm, cache, slots)

    for name, func in [
        ("glm_dsa_indexer_q_rope_weights", q_op),
        ("glm_dsa_indexer_k_norm_rope_cache", k_op),
    ]:
        monkeypatch.setattr(attn.ixf_ops, name, func, raising=False)
    monkeypatch.setattr(
        torch.ops,
        "_ixformer_C",
        SimpleNamespace(
            glm_dsa_indexer_q_rope_weights=q_op, glm_dsa_indexer_k_norm_rope_cache=k_op
        ),
    )


def make_indexer(monkeypatch, *, tokens=8, enabled=True, device="cpu", pcp_size=1):
    monkeypatch.setattr(attn.envs, "VLLM_ILU_GLM_DSA_INDEXER_PREPARE_FUSION", enabled)
    generator = torch.Generator().manual_seed(81)
    q = torch.randn(tokens, 32 * 128, generator=generator).bfloat16()
    kw = torch.randn(tokens, 160, generator=generator).bfloat16()

    class Projection(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.register_buffer("value", value)

        def forward(self, x):
            return self.value.clone(), None

    monkeypatch.setattr(attn, "ReplicatedLinear", lambda *a, **k: Projection(q))
    monkeypatch.setattr(
        attn, "MergedColumnParallelLinear", lambda *a, **k: Projection(kw)
    )
    monkeypatch.setattr(backend, "get_max_prefill_buffer_size", lambda _: 64)
    config = SimpleNamespace(
        index_topk=4, index_n_heads=32, index_head_dim=128, qk_rope_head_dim=64
    )
    vc = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=64),
        compilation_config=SimpleNamespace(static_forward_context={}),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp_size),
    )
    monkeypatch.setattr(attn, "get_current_vllm_config", lambda: vc)
    layer = attn.IluGlmMoeDsaIndexer(
        vc,
        config,
        6144,
        2048,
        None,
        SimpleNamespace(),
        torch.full((tokens, 4), -7, dtype=torch.int32, device=device),
        prefix="model.layers.0.self_attn.indexer",
    ).to(device)
    layer.k_cache.kv_cache = torch.full(
        (16, 4, 128), 17, dtype=torch.bfloat16, device=device
    )
    if device == "cpu":

        def forward(h, q, k, w, skip_k_cache_insert=False):
            return sparse.ilu_sparse_attn_indexer(
                h,
                layer.k_cache.prefix,
                layer.k_cache.kv_cache,
                q,
                k,
                w,
                4,
                128,
                64,
                64,
                layer.topk_indices_buffer,
                skip_k_cache_insert=skip_k_cache_insert,
            )

        layer.indexer_op._forward_method = forward
    else:
        layer.indexer_op._forward_method = layer.indexer_op.forward_cuda
    return layer


class Rotary(torch.nn.Module):
    is_neox_style = False

    def __init__(self, device):
        super().__init__()
        angles = torch.arange(64 * 32).reshape(64, 32).float() / 100
        self.trig = torch.cat((angles.cos(), angles.sin()), -1).bfloat16().to(device)

    def _match_cos_sin_cache_dtype(self, q):
        return self.trig

    def forward(self, positions, q, k):
        rotate(q, positions, self.trig)
        rotate(k, positions, self.trig)
        return q, k


def install_metadata(monkeypatch, layer, slots, mode):
    n = slots.numel()
    nd = n if mode == "decode" else n // 2 if mode == "mixed" else 0
    chunk = SimpleNamespace(
        token_start=nd,
        token_end=n,
        max_context_len=16,
        max_q_len=1,
        max_kv_len=16,
        valid_lens=None,
        cu_seqlens_q=None,
        cu_seq_lens=None,
        block_table=None,
        cu_seqlen_ks=None,
        cu_seqlen_ke=None,
    )
    decode = SimpleNamespace(
        requires_padding=False,
        max_context_len=16,
        max_q_len=1,
        max_kv_len=16,
        valid_lens=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        block_table=None,
        cu_seqlen_ks=None,
        cu_seqlen_ke=None,
    )
    meta = backend.IluDeepseekV32IndexerMetadata(
        seq_lens=None,
        num_reqs=1,
        max_query_len=1,
        max_seq_len=16,
        num_actual_tokens=n,
        query_start_loc=None,
        slot_mapping=slots,
        head_dim=128,
        num_decodes=int(nd > 0),
        num_decode_tokens=nd,
        num_prefills=int(nd < n),
        num_prefill_tokens=n - nd,
        decode=decode,
        prefill=SimpleNamespace(chunks=[chunk]),
    )
    context = SimpleNamespace(attn_metadata={layer.k_cache.prefix: meta})
    monkeypatch.setattr(attn, "get_forward_context", lambda: context)
    monkeypatch.setattr(sparse, "get_forward_context", lambda: context)
    return context


@pytest.mark.parametrize("mode", ["decode", "prefill", "mixed"])
@pytest.mark.parametrize("short_slots", [False, True])
def test_cpu_wiring_matches_split_and_never_reinserts_raw_k(
    monkeypatch, default_vllm_config, mode, short_slots
):
    reference_ops(monkeypatch)
    _compare_paths(monkeypatch, mode, short_slots, "cpu", graph=False)


@torch.inference_mode()
def _compare_paths(monkeypatch, mode, short_slots, device, graph):
    tokens = 8
    layer = make_indexer(monkeypatch, tokens=tokens, device=device)
    slots = torch.tensor([3, 7, -1, 10, 15, 17, -1, 25], device=device)
    if short_slots:
        slots = slots[:6]
    install_metadata(monkeypatch, layer, slots, mode)
    rotary = Rotary(device)
    positions = torch.arange(tokens, device=device)
    hidden = torch.zeros(tokens, 6144, dtype=torch.bfloat16, device=device)
    records = []

    def logits(q, cuq, cuk, cache, table, weights, **kwargs):
        records.append((q.clone(), weights.clone(), cache.clone()))
        return torch.zeros(q.shape[0], 16, device=device)

    monkeypatch.setattr(sparse.ixfops, "dsa_indexer_mqa_logits", logits)
    monkeypatch.setattr(
        sparse.ixfops,
        "dsa_update_topk_indices",
        lambda logits, ks, ke, topk, out: out.fill_(3),
    )
    insert = Mock(side_effect=cache_insert)
    monkeypatch.setattr(sparse.ops, "indexer_k_cache", insert)
    layer.use_prepare_fusion = False
    expected = layer(hidden, hidden, positions, rotary).clone()
    expected_records = list(records)
    assert insert.call_count == 1
    insert.side_effect = AssertionError("old cache insert overwrites fused K")
    records.clear()
    layer.k_cache.kv_cache.fill_(17)
    layer.use_prepare_fusion = True
    actual = layer(hidden, hidden, positions, rotary)
    assert torch.equal(actual, expected)

    def check_records():
        assert len(records) == len(expected_records)
        for actual, expected in zip(records, expected_records):
            for i, (a, b) in enumerate(zip(actual, expected)):
                if i < 2:
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(a, b, rtol=0.01, atol=0.01)

    check_records()
    if graph:
        records.clear()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            layer(hidden, hidden, positions, rotary)
        # Inputs are cloned by the projection stubs on every replay, so the
        # in-place Q rotation must not compound across decode steps.
        for _ in range(2):
            layer.k_cache.kv_cache.fill_(17)
            g.replay()
            check_records()


@pytest.mark.parametrize("mode", ["decode", "prefill", "mixed"])
@pytest.mark.parametrize("short_slots", [False, True])
def test_native_indexer_prepare_and_graph(
    monkeypatch, default_vllm_config, cuda_device, mode, short_slots
):
    _compare_paths(monkeypatch, mode, short_slots, cuda_device, graph=True)


@torch.inference_mode()
def test_dummy_run_does_not_call_fused_cache_ops(monkeypatch, default_vllm_config):
    reference_ops(monkeypatch)
    layer = make_indexer(monkeypatch)
    context = SimpleNamespace(attn_metadata=None)
    monkeypatch.setattr(attn, "get_forward_context", lambda: context)
    monkeypatch.setattr(sparse, "get_forward_context", lambda: context)
    for name in ["glm_dsa_indexer_q_rope_weights", "glm_dsa_indexer_k_norm_rope_cache"]:
        monkeypatch.setattr(attn.ixf_ops, name, Mock(side_effect=AssertionError(name)))
    h = torch.zeros(8, 6144, dtype=torch.bfloat16)
    out = layer(h, h, torch.arange(8), Rotary("cpu"))
    assert torch.equal(out, torch.full_like(out, -7))


def test_opt_in_fails_early_when_new_native_ops_are_missing(
    monkeypatch, default_vllm_config
):
    monkeypatch.setattr(torch.ops, "_ixformer_C", SimpleNamespace())
    with pytest.raises(RuntimeError, match="requires ixformer"):
        make_indexer(monkeypatch)
    assert not make_indexer(monkeypatch, enabled=False).use_prepare_fusion


def test_meta_registered_fused_path(monkeypatch, default_vllm_config):
    """Check native schemas and the custom-op fake path without a GPU launch."""
    from ixformer.core import config

    layer = make_indexer(monkeypatch, device="meta")
    slots = torch.empty(8, dtype=torch.int64, device="meta")
    install_metadata(monkeypatch, layer, slots, "mixed")
    with config.enable_torch_ops(True):
        out = layer(
            torch.empty(8, 6144, dtype=torch.bfloat16, device="meta"),
            torch.empty(8, 2048, dtype=torch.bfloat16, device="meta"),
            torch.empty(8, dtype=torch.int64, device="meta"),
            Rotary("meta"),
        )
    assert out.device.type == "meta"
    assert out.shape == (8, 4)


@torch.inference_mode()
def test_pcp_uses_split_preparation_without_local_fusion_ops(
    monkeypatch, default_vllm_config
):
    monkeypatch.setattr(torch.ops, "_ixformer_C", SimpleNamespace())
    layer = make_indexer(monkeypatch, pcp_size=2)
    assert not layer.use_prepare_fusion
    q = torch.zeros(8, 32, 128, dtype=torch.bfloat16)
    layer.indexer_op._forward_method = Mock(return_value=q)
    hidden = torch.zeros(8, 6144, dtype=torch.bfloat16)
    layer(hidden, hidden, torch.arange(8), Rotary("cpu"))
    args, kwargs = layer.indexer_op._forward_method.call_args
    assert not kwargs.get("skip_k_cache_insert", False)
    # K must have passed through normalization/RoPE before PCP gathers it.
    raw_k = layer.wk_weights_proj.value[:, :128]
    expected_k = layer.k_norm(raw_k)
    rotate(expected_k, torch.arange(8), Rotary("cpu").trig)
    torch.testing.assert_close(args[2], expected_k)
