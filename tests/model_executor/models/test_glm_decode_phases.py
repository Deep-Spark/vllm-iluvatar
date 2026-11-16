# SPDX-License-Identifier: Apache-2.0
"""DSA integration preserves mutation/cache/packed-Q contracts across replay."""

from types import SimpleNamespace

import ixformer.inference.functions as F
import pytest
import torch

from vllm_iluvatar.ilu_util import arch
from vllm_iluvatar.models.glm_moe_dsa import attention as mod

# 这组用例走真实 packed sparse MLA / Marlin，仅支持 ivcore30（CQ 机器）。
# 在 ivcore11 机器上跳过，避免全量回归时以硬断言形式红掉。
pytestmark = pytest.mark.skipif(
    not arch.is_ivcore30(),
    reason="requires ivcore30 (CQ); glm DSA real-backend suite",
)


def make_attention(enabled, trig, tokens, vup_weights, heads=32):
    a = mod.IluGlmMoeDsaSparseMLA.__new__(mod.IluGlmMoeDsaSparseMLA)
    torch.nn.Module.__init__(a)
    a.num_heads = heads
    a.kv_lora_rank = 512
    a.qk_nope_head_dim = 192
    a.qk_rope_head_dim = 64
    a.v_head_dim = 256
    a.prefix = "test"
    a._decode_prepare_enabled = enabled
    a._fp8_kv_scale_frozen = True
    a.use_rms_rope_fp8_kv_fusion = True
    a.use_fp8_attn = True
    a.kv_cache = torch.zeros(2, 64, 640, device="cuda", dtype=torch.uint8)
    a._k_scale = torch.tensor([0.02], device="cuda")
    a.rotary_emb = SimpleNamespace(_match_cos_sin_cache_dtype=lambda q: trig)
    a._k_up_proj = lambda x: torch.nn.functional.pad(x, (0, 320)).contiguous()

    def sparse(nope, rope, cache, meta, prepared=None):
        if prepared is None:
            packed = torch.empty(tokens, 32, 640, device="cuda", dtype=torch.uint8)
            scales = torch.empty(tokens, 32, device="cuda")
            F.glm_dsa_fp8_q_pack(nope, rope, packed, scales)
            prepared = (packed, scales)
        a.last_attn = nope
        a.last_packed, a.last_scales = prepared
        return nope

    a._sparse_mla_kernel_fp8 = sparse
    a.kv_b_proj = SimpleNamespace(W_UV=vup_weights[0], W_UV_S=vup_weights[1])
    real_vup = a._v_up_proj

    def vup(x, outputs=None):
        a.last_vup = real_vup(x, outputs=outputs)
        return a.last_vup

    a._v_up_proj = vup
    return a


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("tokens", [16, 17])
def test_dsa_prepare_graph_and_vup_output(monkeypatch, padded, tokens):
    capacity = tokens + 1 if padded else tokens
    meta = SimpleNamespace(
        num_actual_tokens=tokens,
        block_size=64,
        pcp_world_size=1,
        slot_mapping=torch.arange(tokens, device="cuda"),
    )
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(attn_metadata=meta)
    )
    trig = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(tokens, device="cuda")
    quant_weight = torch.randint(
        -127, 128, (32, 512, 256), device="cuda", dtype=torch.int8
    )
    vup_scales = torch.rand(32, 256, 1, device="cuda") / 1024
    dense_weight = (quant_weight.float() * vup_scales.transpose(1, 2)).bfloat16()
    packed_weight = F.marlin_w8_weight_repack(quant_weight, reformat="k16n16")
    vup_weights = (packed_weight, vup_scales)
    baseline = make_attention(False, trig, tokens, vup_weights)
    fused = make_attention(True, trig, tokens, vup_weights)
    q = torch.randn(capacity, 32, 256, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(capacity, 576, device="cuda", dtype=torch.bfloat16)
    q1 = q.clone()
    kv1 = kv.clone()

    def run(a, q, kv):
        return a(q, kv[:, :512], kv[:, 512:], positions, weight, 1e-05)

    run(baseline, q, kv)
    run(fused, q1, kv1)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out1 = run(fused, q1, kv1)
    for seed in (64, 218):
        torch.manual_seed(seed)
        q.normal_()
        kv.normal_()
        q1.copy_(q)
        kv1.copy_(kv)
        out0 = run(baseline, q, kv)
        g.replay()
        assert torch.equal(q, q1)
        assert torch.equal(kv, kv1)
        assert torch.equal(baseline.kv_cache, fused.kv_cache)
        assert torch.equal(baseline.last_packed, fused.last_packed)
        assert torch.equal(baseline.last_scales, fused.last_scales)
        assert torch.equal(out0[:tokens], out1[:tokens])
        assert out1.data_ptr() == fused.last_vup.data_ptr()
        expected = torch.einsum(
            "mhk,hkn->mhn", baseline.last_attn.cpu().float(), dense_weight.cpu().float()
        ).bfloat16()
        torch.testing.assert_close(
            out1[:tokens].view_as(expected).cpu(), expected, atol=0.02, rtol=0.01
        )


@pytest.mark.parametrize(
    "phase,tokens,decode_tokens",
    [
        ("decode", 1, 1),
        ("prefill", 17, 0),
        ("mixed", 17, 1),
    ],
)
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("heads", [16, 32, 64])
@torch.inference_mode()
def test_real_sparse_backend_prepare_ab_and_graph(
    monkeypatch, phase, tokens, decode_tokens, padded, heads
):
    """R3: real packed sparse MLA and Marlin, identical off/on inputs.

    Metadata labels exercise the current phase-independent preparation
    contract. This fixture supplies known global sparse indices; runner
    scheduling and PCP collectives require Ruler's model/distributed suite.
    """
    from vllm_iluvatar.ilu_util import arch

    assert arch.is_ivcore30(), "Ruler must load paired CQ ixformer"
    capacity = tokens + int(padded)
    meta = SimpleNamespace(
        num_actual_tokens=tokens,
        num_decode_tokens=decode_tokens,
        num_prefills=int(phase != "decode"),
        block_size=64,
        pcp_world_size=1,
        slot_mapping=torch.arange(tokens, device="cuda"),
    )
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(attn_metadata=meta)
    )
    rng = torch.Generator().manual_seed(912)
    angles = torch.randn(64, 32, generator=rng)
    trig = torch.cat((angles.cos(), angles.sin()), -1).bfloat16().cuda()
    weight = torch.randn(512, generator=rng).bfloat16().cuda()
    positions = torch.arange(tokens, device="cuda")
    qw = torch.randint(
        -127, 128, (heads, 512, 256), generator=rng, dtype=torch.int8
    ).cuda()
    scales = torch.rand(heads, 256, 1, generator=rng).cuda() / 1024
    weights = (F.marlin_w8_weight_repack(qw, reformat="k16n16"), scales)
    baseline = make_attention(False, trig, tokens, weights, heads)
    fused = make_attention(True, trig, tokens, weights, heads)
    indices = torch.arange(64, device="cuda", dtype=torch.int32)
    indices = indices.expand(tokens, -1).contiguous()
    indices = indices.masked_fill(indices > positions[:, None], -1).unsqueeze(1)
    lengths = (positions + 1).to(torch.int32)
    for a in (baseline, fused):
        # Remove the old test double: dispatch the actual class method, which
        # calls dsa_flash_mla_with_fp8_kvcache and consumes its latent output.
        del a._sparse_mla_kernel_fp8
        a._fp8_q_heads, a.softmax_scale = heads, 1 / 16
        a._global_topk = lambda *args: (indices, lengths)
    if phase == "prefill" and padded and heads == 32:
        # One existing case also exercises shifted cache slots and a negative
        # slot. Supply compact valid global indices, avoiding attention to the
        # padding slot while retaining its Q/K mutation checks.
        slots = torch.arange(tokens, device="cuda") * 2
        slots[5] = -1
        meta.slot_mapping = slots
        indices = torch.full((tokens, 1, 64), -1, device="cuda", dtype=torch.int32)
        lengths = torch.empty(tokens, device="cuda", dtype=torch.int32)
        for row in range(tokens):
            valid = slots[: row + 1]
            valid = valid[valid >= 0]
            indices[row, 0, : valid.numel()] = valid.to(torch.int32)
            lengths[row] = valid.numel()
    native_prepare = F.glm_dsa_prepare
    native_sparse = F.dsa_flash_mla_with_fp8_kvcache
    calls = {"prepare": 0, "sparse": 0}

    def checked_prepare(*args, **kwargs):
        calls["prepare"] += 1
        return native_prepare(*args, **kwargs)

    def checked_sparse(*args, **kwargs):
        calls["sparse"] += 1
        return native_sparse(*args, **kwargs)

    monkeypatch.setattr(F, "glm_dsa_prepare", checked_prepare)
    monkeypatch.setattr(
        F, "dsa_flash_mla_sparse_with_fp8_kvcache_diff_kv", checked_sparse
    )
    q = torch.randn(capacity, heads, 256, generator=rng).bfloat16().cuda()
    kv = torch.randn(capacity, 576, generator=rng).bfloat16().cuda()
    q_f, kv_f = q.clone(), kv.clone()

    def run(a, q, kv):
        return a(q, kv[:, :512], kv[:, 512:], positions, weight, 1e-5)

    run(baseline, q, kv)
    assert calls == {"prepare": 0, "sparse": 1}
    run(fused, q_f, kv_f)
    assert calls == {"prepare": 1, "sparse": 2}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(fused, q_f, kv_f)
    for _ in range(2):
        q.normal_()
        kv.normal_()
        q_f.copy_(q)
        kv_f.copy_(kv)
        baseline.kv_cache.zero_()
        fused.kv_cache.zero_()
        expected = run(baseline, q, kv)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(baseline.kv_cache, fused.kv_cache)
        assert torch.equal(q, q_f)
        assert torch.equal(kv, kv_f)
        assert torch.isfinite(captured[:tokens]).all()
        # Identical packed-Q/cache imply identical real attention and V-up.
        torch.testing.assert_close(captured[:tokens], expected[:tokens], atol=0, rtol=0)


@torch.inference_mode()
def test_zero_actual_tokens_skips_prepare_attention_and_vup(monkeypatch):
    """Graph padding capacity is nonzero; actual request count is zero."""
    from vllm_iluvatar.ilu_util import arch

    assert arch.is_ivcore30(), "Ruler must load paired CQ ixformer"
    meta = SimpleNamespace(
        num_actual_tokens=0,
        num_decode_tokens=0,
        block_size=64,
        pcp_world_size=1,
        slot_mapping=torch.empty(0, dtype=torch.int64, device="cuda"),
    )
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(attn_metadata=meta)
    )
    a = make_attention(
        True, torch.zeros(1, 64, device="cuda", dtype=torch.bfloat16), 0, (None, None)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("zero actual tokens launched prepare/attention/V-up")

    monkeypatch.setattr(F, "glm_dsa_prepare", forbidden)
    del a._sparse_mla_kernel_fp8
    a._sparse_mla_kernel_fp8 = forbidden
    a._v_up_proj = forbidden
    q = torch.zeros(1, 32, 256, device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros(1, 576, device="cuda", dtype=torch.bfloat16)
    positions = torch.empty(0, device="cuda", dtype=torch.int64)
    weight = torch.ones(512, device="cuda", dtype=torch.bfloat16)

    def run():
        return a(q, kv[:, :512], kv[:, 512:], positions, weight, 1e-5)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    graph.replay()
    assert output.shape == (1, 32 * 256)
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("case", ["online_calibration", "unsupported_heads"])
@torch.inference_mode()
def test_real_forward_prepare_fallback(monkeypatch, case, default_vllm_config):
    """First online calibration and unsupported heads use the real old path."""
    from vllm_iluvatar.ilu_util import arch

    assert arch.is_ivcore30(), "requires paired CQ candidate"
    heads, tokens = (8 if case == "unsupported_heads" else 32), 4
    positions = torch.arange(tokens, device="cuda")
    metadata = SimpleNamespace(
        num_actual_tokens=tokens,
        num_decode_tokens=tokens,
        block_size=64,
        pcp_world_size=1,
        slot_mapping=positions,
    )
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
    )
    qw = torch.randint(-127, 128, (heads, 512, 256), device="cuda", dtype=torch.int8)
    scales = torch.rand(heads, 256, 1, device="cuda") / 1024
    weights = (F.marlin_w8_weight_repack(qw, reformat="k16n16"), scales)
    layers = []
    indices = torch.arange(64, device="cuda", dtype=torch.int32).expand(tokens, -1)
    indices = indices.masked_fill(indices > positions[:, None], -1).unsqueeze(1)
    lengths = (positions + 1).to(torch.int32)
    for enabled in (False, True):
        a = make_attention(
            enabled and heads in (16, 32, 64), None, tokens, weights, heads
        )
        a.use_rms_rope_fp8_kv_fusion = False
        a._rms_rope_fp8_kv_fusion_available = False
        a._fp8_kv_scale_frozen = case != "online_calibration"
        a._fp8_q_heads, a.softmax_scale = max(16, heads), 1 / 16
        a.rotary_emb = mod.get_rope(
            64,
            max_position=64,
            rope_parameters={"rope_type": "default", "rope_theta": 10000},
            is_neox_style=False,
        ).cuda()
        a._global_topk = lambda *args: (indices, lengths)
        del a._sparse_mla_kernel_fp8
        layers.append(a)

    def forbidden(*args, **kwargs):
        raise AssertionError("ineligible forward selected glm_dsa_prepare")

    monkeypatch.setattr(F, "glm_dsa_prepare", forbidden)
    q = torch.randn(tokens, heads, 256, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(tokens, 576, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(512, device="cuda", dtype=torch.bfloat16)
    outputs = []
    for a in layers:
        qc, kvc = q.clone(), kv.clone()
        output = a(qc, kvc[:, :512], kvc[:, 512:], positions, weight, 1e-5)
        assert a._fp8_kv_scale_frozen
        assert torch.isfinite(output).all()
        outputs.append(output.clone())
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    assert torch.equal(layers[0].kv_cache, layers[1].kv_cache)
    assert torch.equal(layers[0]._k_scale, layers[1]._k_scale)
