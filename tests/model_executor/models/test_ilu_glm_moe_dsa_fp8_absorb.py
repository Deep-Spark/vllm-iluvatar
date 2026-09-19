# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 official-FP8 absorbed up-proj: load-time W_UK / W_UV build + forward.

The build (layouts, scale directions, read-safe strides, hook ordering, weight
release) is CPU-only. The two GEMM paths are CUDA-gated and compared against a
BF16 einsum on the dequantised weight.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.models.glm_moe_dsa import fp8_attention as fp8_mod

_FP8 = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8).max)

# GLM-5.2 absorbed MLA geometry at TP4 (num_heads is per-rank).
_HEADS, _NOPE, _V, _LORA = 16, 192, 256, 512


def _make_kv_b_proj(device="cpu", heads=_HEADS, nope=_NOPE, v=_V, lora=_LORA):
    """A stand-in for ``kv_b_proj`` after IluFp8LinearMethod's per-channel requant."""
    g = torch.Generator().manual_seed(1234)
    n = heads * (nope + v)
    w = torch.randn((n, lora), generator=g).clamp(-3.0, 3.0).to(_FP8).to(device)
    scale = (torch.rand((n,), generator=g) * 0.4 + 0.05).to(device)
    return SimpleNamespace(
        weight=w,
        weight_scale=scale,
        bmm_geom=SimpleNamespace(
            num_heads=heads,
            qk_nope_head_dim=nope,
            v_head_dim=v,
            kv_lora_rank=lora,
        ),
        quant_method=SimpleNamespace(process_weights_after_loading=lambda layer: None),
    )


def _make_mla(kv_b_proj, *, fp8_absorb=True, prefix="model.layers.0.self_attn.attn"):
    mla = object.__new__(fp8_mod.IluGlmMoeDsaFp8SparseMLA)
    mla.kv_b_proj = kv_b_proj
    mla.prefix = prefix
    mla.use_fp8_absorb_up = fp8_absorb
    return mla


def _dequant_ref(kv_b_proj):
    """fp32 ``[H, nope+v, lora]`` gold from the checkpoint bytes + scale."""
    geom = kv_b_proj.bmm_geom
    heads = geom.num_heads
    w = kv_b_proj.weight.float().view(heads, -1, geom.kv_lora_rank)
    s = kv_b_proj.weight_scale.float().view(heads, -1, 1)
    return w * s


# --------------------------------------------------------------------------- #
# load-time build                                                             #
# --------------------------------------------------------------------------- #


def test_fp8_absorb_layout_and_strides():
    kv_b = _make_kv_b_proj()
    mla = _make_mla(kv_b)
    mla._build_fp8_absorb()

    # einsum operand shapes: b[G, D, R] with G = num_heads.
    assert tuple(mla.W_UK.shape) == (_HEADS, _LORA, _NOPE)  # k-up: R=192, D=512
    assert tuple(mla.W_UV.shape) == (_HEADS, _V, _LORA)  # v-up: R=512, D=256
    assert mla.W_UK.dtype == _FP8 and mla.W_UV.dtype == _FP8
    assert mla.W_UK.is_contiguous() and mla.W_UV.is_contiguous()

    # Scales are [G, D] fp32 with the read-safe row stride the native fp8
    # einsum epilogue needs (>= ceil(D/128)*128).
    for scale, d in ((mla.W_UK_S, _LORA), (mla.W_UV_S, _V)):
        assert tuple(scale.shape) == (_HEADS, d)
        assert scale.dtype == torch.float32
        assert scale.stride(1) == 1
        assert scale.stride(0) >= ((d + 127) // 128) * 128


def test_v_up_reuses_checkpoint_bytes_exactly():
    """v-up contracts over ``lora``, so the checkpoint's per-output-row scale is
    already on the output axis: no requant, no added error."""
    kv_b = _make_kv_b_proj()
    w_src = kv_b.weight.clone()
    s_src = kv_b.weight_scale.clone()
    mla = _make_mla(kv_b)
    mla._build_fp8_absorb()

    w_v = w_src.view(_HEADS, _NOPE + _V, _LORA)[:, _NOPE:, :]
    s_v = s_src.view(_HEADS, _NOPE + _V)[:, _NOPE:]
    assert torch.equal(mla.W_UV.view(torch.uint8), w_v.contiguous().view(torch.uint8))
    assert torch.equal(mla.W_UV_S, s_v)


def test_k_up_requant_is_transposed_and_output_scaled():
    """k-up contracts over ``nope``, so the checkpoint scale would land on the
    contraction axis; the build transposes and rescales per ``lora`` channel."""
    kv_b = _make_kv_b_proj()
    ref = _dequant_ref(kv_b)[:, :_NOPE, :].transpose(1, 2)  # [H, lora, nope]
    mla = _make_mla(kv_b)
    mla._build_fp8_absorb()

    # The scale is a per-output-channel amax/448 of the dequantised weight.
    expected_scale = (ref.abs().amax(dim=-1) / _FP8_MAX).clamp_min(1e-12)
    torch.testing.assert_close(mla.W_UK_S, expected_scale, rtol=1e-6, atol=0.0)

    # Round-trip stays within one e4m3 step (~1/16 relative for normals).
    deq = mla.W_UK.float() * mla.W_UK_S.unsqueeze(-1)
    rel = (deq - ref).norm() / ref.norm()
    assert rel < 0.04, f"k-up requant relative error {rel:.4f} too large"
    assert mla.W_UK.float().abs().max() <= _FP8_MAX


def test_bf16_fallback_keeps_einsum_layout():
    kv_b = _make_kv_b_proj()
    ref = _dequant_ref(kv_b)
    mla = _make_mla(kv_b, fp8_absorb=False)
    mla._build_bf16_absorb()

    assert tuple(mla.W_UK.shape) == (_HEADS, _NOPE, _LORA)  # "thp,hpl->thl"
    assert tuple(mla.W_UV.shape) == (_HEADS, _V, _LORA)  # "thl,hvl->thv"
    assert mla.W_UK.dtype == torch.bfloat16
    assert mla.W_UK_S is None and mla.W_UV_S is None
    torch.testing.assert_close(
        mla.W_UK.float(), ref[:, :_NOPE, :], rtol=1e-2, atol=1e-2
    )


@pytest.mark.parametrize("fp8_absorb", [True, False])
def test_build_absorb_releases_packed_weight(fp8_absorb):
    kv_b = _make_kv_b_proj()
    mla = _make_mla(kv_b, fp8_absorb=fp8_absorb)
    mla._build_absorb()

    assert not hasattr(kv_b, "weight")
    assert mla.W_UK is not None and mla.W_UV is not None
    assert (mla.W_UK_S is not None) is fp8_absorb


def test_absorb_hook_runs_after_process_weights():
    """The build must observe the per-channel requant, so it chains onto the
    tail of kv_b_proj's own process_weights_after_loading."""
    order = []
    kv_b = _make_kv_b_proj()

    def original(layer):
        order.append("process")
        assert hasattr(layer, "weight")

    kv_b.quant_method.process_weights_after_loading = original
    mla = _make_mla(kv_b)
    mla._install_absorb_hook()
    assert kv_b._ilu_fp8_absorb_hooked is True

    # Installing twice must not double-wrap (the hook is per-layer, and the
    # second wrap would run the build against an already-freed weight).
    hooked_once = kv_b.quant_method.process_weights_after_loading
    mla._install_absorb_hook()
    assert kv_b.quant_method.process_weights_after_loading is hooked_once

    kv_b.quant_method.process_weights_after_loading(kv_b)
    assert order == ["process"]
    assert mla.W_UK is not None
    assert not hasattr(kv_b, "weight")

    # A different layer sharing the quant_method must not trigger this MLA's build.
    other = _make_kv_b_proj()
    kv_b.quant_method.process_weights_after_loading(other)
    assert hasattr(other, "weight")


def test_missing_weight_and_bad_scale_are_reported():
    kv_b = _make_kv_b_proj()
    del kv_b.weight
    with pytest.raises(RuntimeError, match="no 'weight'"):
        _make_mla(kv_b)._build_fp8_absorb()

    kv_b = _make_kv_b_proj()
    del kv_b.weight_scale
    with pytest.raises(RuntimeError, match="neither weight_scale"):
        _make_mla(kv_b)._build_fp8_absorb()

    # A 2-D (block) scale means process_weights_after_loading has not run.
    kv_b = _make_kv_b_proj()
    kv_b.weight_scale = torch.ones(8, 4)
    with pytest.raises(ValueError, match="per-channel kv_b_proj scale"):
        _make_mla(kv_b)._build_fp8_absorb()

    kv_b = _make_kv_b_proj()
    kv_b.weight = kv_b.weight[:-1]
    with pytest.raises(ValueError, match="weight shape"):
        _make_mla(kv_b)._build_fp8_absorb()

    kv_b = _make_kv_b_proj()
    kv_b.quant_method = None
    with pytest.raises(RuntimeError, match="no quant_method"):
        _make_mla(kv_b)._install_absorb_hook()


# --------------------------------------------------------------------------- #
# forward                                                                     #
# --------------------------------------------------------------------------- #

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@cuda_only
@pytest.mark.parametrize("num_tokens", [1, 7, 128])
@torch.inference_mode()
def test_fp8_up_proj_matches_bf16_reference(num_tokens):
    """fp8 k-up / v-up against a BF16 einsum on the dequantised weight.

    Both halves quantise the activation per (head, token) to e4m3 and the
    weight per output channel, so the dot product carries ~3 mantissa bits of
    input error over R=192 / R=512 terms. Elementwise tolerances are
    meaningless for outputs that cross zero; compare in norm.
    """
    kv_b = _make_kv_b_proj(device="cuda")
    ref = _dequant_ref(kv_b)  # [H, nope+v, lora] fp32
    w_uk_ref = ref[:, :_NOPE, :]
    w_uv_ref = ref[:, _NOPE:, :]

    mla = _make_mla(kv_b)
    mla._build_fp8_absorb()

    g = torch.Generator(device="cuda").manual_seed(7)
    # k-up input is a split view of q ([T, H, nope+rope]) -- keep it strided so
    # the quant kernel's non-contiguous head stride is exercised.
    q = torch.randn(
        (num_tokens, _HEADS, _NOPE + 64), generator=g, device="cuda"
    ).bfloat16()
    q_nope = q[..., :_NOPE]
    x = torch.randn((num_tokens, _HEADS, _LORA), generator=g, device="cuda").bfloat16()

    k_out = mla._k_up_proj(q_nope)
    v_out = mla._v_up_proj(x)
    assert tuple(k_out.shape) == (num_tokens, _HEADS, _LORA)
    assert tuple(v_out.shape) == (num_tokens, _HEADS, _V)
    assert k_out.dtype == torch.bfloat16 and v_out.dtype == torch.bfloat16

    k_ref = torch.einsum("thp,hpl->thl", q_nope.float(), w_uk_ref)
    v_ref = torch.einsum("thl,hvl->thv", x.float(), w_uv_ref)
    for got, want, name in ((k_out, k_ref, "k-up"), (v_out, v_ref, "v-up")):
        rel = (got.float() - want).norm() / want.norm()
        assert rel < 0.06, f"{name} norm-relative error {rel:.4f} exceeds fp8 budget"


@cuda_only
@torch.inference_mode()
def test_fp8_and_bf16_paths_agree():
    """The env=0 fallback and the fp8 path must be the same GEMM, not just the
    same shape -- they are A/B'd against each other in accuracy runs."""
    kv_b_fp8 = _make_kv_b_proj(device="cuda")
    kv_b_bf16 = _make_kv_b_proj(device="cuda")

    mla_fp8 = _make_mla(kv_b_fp8)
    mla_fp8._build_fp8_absorb()
    mla_bf16 = _make_mla(kv_b_bf16, fp8_absorb=False)
    mla_bf16._build_bf16_absorb()

    g = torch.Generator(device="cuda").manual_seed(3)
    q_nope = torch.randn((32, _HEADS, _NOPE), generator=g, device="cuda").bfloat16()
    x = torch.randn((32, _HEADS, _LORA), generator=g, device="cuda").bfloat16()

    for got, want, name in (
        (mla_fp8._k_up_proj(q_nope), mla_bf16._k_up_proj(q_nope), "k-up"),
        (mla_fp8._v_up_proj(x), mla_bf16._v_up_proj(x), "v-up"),
    ):
        rel = (got.float() - want.float()).norm() / want.float().norm()
        assert rel < 0.06, f"{name} fp8-vs-bf16 relative error {rel:.4f}"


@cuda_only
@torch.inference_mode()
def test_fp8_up_proj_is_cuda_graph_capturable():
    """vLLM decode runs under a captured graph; the quant + einsum pair must
    capture (no host syncs, no shape-dependent control flow on device values)."""
    kv_b = _make_kv_b_proj(device="cuda")
    mla = _make_mla(kv_b)
    mla._build_fp8_absorb()

    q_nope = torch.randn((8, _HEADS, _NOPE), device="cuda").bfloat16()
    pool = torch.cuda.graphs.graph_pool_handle()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            mla._k_up_proj(q_nope)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        out = mla._k_up_proj(q_nope)
    eager = mla._k_up_proj(q_nope)

    q_nope.copy_(torch.randn_like(q_nope))
    graph.replay()
    torch.cuda.synchronize()
    replayed = out.clone()
    assert torch.equal(replayed, mla._k_up_proj(q_nope))
    assert eager.shape == out.shape
