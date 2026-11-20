# SPDX-License-Identifier: Apache-2.0
"""Tests for the Iluvatar FP8 block-scaled dense kernel (per-channel requant).

Covers the runtime path introduced on top of the torch-equivalent stage:

* ``process_weights_after_loading`` requantizes the block-scaled fp8 weight into
  a per-output-channel layout via the ixformer op
  ``ixfops.fp8_weight_block_to_channel_requant`` (mocked here, so these tests are
  device-independent and assert the call contract + parameter wiring): the raw
  block scale (fp32 or e8m0) is passed straight through, the requant runs in
  place on ``weight.data``, and only the scale parameter is replaced. The requant
  numerics (incl. the e8m0 decode) live in the ixformer op's own test suite;
* a non-conforming ``K % 64 != 0`` raises at load time;
* ``apply_block_scaled_mm`` drives either the fused ``nt_v2`` quant +
  ``fp8_per_channel_gemv(TN_V2)`` decode path (unquantized + shape gate +
  capability probe) or the plain per-token quant + ``fp8_per_channel_mm`` path
  (prequant / gate miss / missing nt_v2 API / missing ixmm gemv); mocks keep
  the tests device-independent and assert call contracts. nt_v2 returns exact
  ``M`` (no pad-to-4; ixmm RaggedM finishes ragged rows);
* the on-device ``is_supported`` gate (arch forced to ivcore30).

The pure-torch block reference ``_apply_block_scaled_mm_torch`` (no longer on the
runtime path) is covered by ``test_ilu_fp8_block_scaled_registry.py``.
"""

from __future__ import annotations

import types

import pytest
import torch
from conftest import requires_cuda
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)

from vllm_iluvatar.model_executor.kernels.linear.scaled_mm import (
    ilu_fp8_block_scaled_mm as mod,
)
from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (  # noqa: E501
    IluFp8BlockScaledMMKernel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kernel(gn: int, gk: int, out_dtype: torch.dtype) -> IluFp8BlockScaledMMKernel:
    k = IluFp8BlockScaledMMKernel.__new__(IluFp8BlockScaledMMKernel)
    k.weight_group_shape = (gn, gk)
    k.config = types.SimpleNamespace(out_dtype=out_dtype)
    return k


def _make_layer(weight: torch.Tensor, block_scale: torch.Tensor) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(block_scale, requires_grad=False)
    return layer


# ---------------------------------------------------------------------------
# process_weights_after_loading: block -> per-channel requant (op call contract)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float8_e8m0fnu])
def test_process_weights_after_loading_calls_requant_op(monkeypatch, scale_dtype):
    import ixformer.inference.functions as ixfops

    N, K, gn, gk = 256, 256, 128, 128
    nb, kb = (N + gn - 1) // gn, (K + gk - 1) // gk
    weight = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
    if scale_dtype == torch.float8_e8m0fnu:
        block_scale = torch.full((nb, kb), 127, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
    else:
        block_scale = torch.ones(nb, kb, dtype=torch.float32)

    monkeypatch.setattr(
        Fp8BlockScaledMMLinearKernel,
        "process_weights_after_loading",
        lambda self, layer: None,
    )

    captured: dict = {}

    def fake_requant(
        w, bs, group_n, group_k, fp8_output=None, output_scales=None, backend=None
    ):
        captured["weight"] = w
        captured["block_scale"] = bs
        captured["groups"] = (group_n, group_k)
        captured["fp8_output"] = fp8_output
        out_scale = torch.arange(w.shape[0], dtype=torch.float32, device=w.device)
        return fp8_output, out_scale

    monkeypatch.setattr(ixfops, "fp8_weight_block_to_channel_requant", fake_requant)

    layer = _make_layer(weight, block_scale)
    orig_weight = layer.weight
    orig_scale = layer.weight_scale_inv
    kernel = _make_kernel(gn, gk, torch.bfloat16)
    kernel.process_weights_after_loading(layer)

    # Groups come from weight_group_shape.
    assert captured["groups"] == (gn, gk)
    # In-place requant: input and fp8_output both alias the weight buffer.
    assert captured["weight"].data_ptr() == orig_weight.data_ptr()
    assert captured["fp8_output"].data_ptr() == orig_weight.data_ptr()
    # Raw block scale (fp32 or e8m0) is passed straight through (no Python decode).
    assert captured["block_scale"].dtype == scale_dtype
    assert captured["block_scale"].data_ptr() == orig_scale.data_ptr()
    # Weight parameter is updated in place (not replaced); only the scale param is
    # swapped for the returned [N] per-channel scale.
    assert layer.weight is orig_weight
    assert layer.weight_scale_inv.dtype == torch.float32
    assert tuple(layer.weight_scale_inv.shape) == (N,)
    torch.testing.assert_close(
        layer.weight_scale_inv.detach().float(),
        torch.arange(N, dtype=torch.float32),
    )


def test_process_weights_after_loading_skips_bmm_layer(monkeypatch):
    # ``is_bmm`` layers *without* ``bmm_geom`` have no fp8-einsum consumer, so
    # there is nothing to reshape them into: the per-channel requant is skipped
    # and the 2-D block scale is left intact.
    monkeypatch.setattr(
        Fp8BlockScaledMMLinearKernel,
        "process_weights_after_loading",
        lambda self, layer: None,
    )

    import ixformer.inference.functions as ixfops

    def boom(*args, **kwargs):
        raise AssertionError("requant op must not be called for is_bmm layers")

    monkeypatch.setattr(ixfops, "fp8_weight_block_to_channel_requant", boom)

    N, K, gn, gk = 256, 256, 128, 128
    nb, kb = (N + gn - 1) // gn, (K + gk - 1) // gk
    weight = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
    block_scale = torch.ones(nb, kb, dtype=torch.float32)
    layer = _make_layer(weight, block_scale)
    layer.is_bmm = True
    orig_weight = layer.weight
    orig_scale = layer.weight_scale_inv
    kernel = _make_kernel(gn, gk, torch.bfloat16)

    kernel.process_weights_after_loading(layer)

    # Block layout preserved untouched: weight param and 2-D block scale kept.
    assert layer.weight is orig_weight
    assert layer.weight_scale_inv is orig_scale
    assert tuple(layer.weight_scale_inv.shape) == (nb, kb)
    # No grouped einsum views are built when there is no geometry.
    assert not hasattr(layer, "bmm_b_fp8")
    assert not hasattr(layer, "bmm_bs")


@pytest.mark.parametrize("o_lora_rank", [256, 192], ids=["D_128aligned", "D_unaligned"])
def test_process_weights_after_loading_bmm_layer_builds_grouped_views(
    monkeypatch, o_lora_rank
):
    # ``wo_a`` (is_bmm) *with* ``bmm_geom``: requant to per-channel fp8 in place,
    # then attach the grouped views the fp8 einsum consumes -- ``bmm_b_fp8``
    # [G, D, R] (aliases the weight buffer) + read-safe ``bmm_bs`` [G, D]. The
    # unaligned-D flavour exercises the copy into a padded read-safe scale.
    import ixformer.inference.functions as ixfops

    G, D = 2, o_lora_rank
    hpg, head_dim = 2, 128
    R = hpg * head_dim  # 256
    N, K = G * D, R
    gn, gk = 128, 128
    nb, kb = (N + gn - 1) // gn, (K + gk - 1) // gk

    weight = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
    block_scale = torch.ones(nb, kb, dtype=torch.float32)

    monkeypatch.setattr(
        Fp8BlockScaledMMLinearKernel,
        "process_weights_after_loading",
        lambda self, layer: None,
    )

    captured: dict = {}

    def fake_requant(
        w, bs, group_n, group_k, fp8_output=None, output_scales=None, backend=None
    ):
        captured["fp8_output"] = fp8_output
        out_scale = torch.arange(w.shape[0], dtype=torch.float32, device=w.device)
        return fp8_output, out_scale

    monkeypatch.setattr(ixfops, "fp8_weight_block_to_channel_requant", fake_requant)

    layer = _make_layer(weight, block_scale)
    layer.is_bmm = True
    layer.bmm_geom = types.SimpleNamespace(
        n_local_groups=G,
        o_lora_rank=D,
        heads_per_group=hpg,
        head_dim=head_dim,
    )
    orig_weight = layer.weight
    kernel = _make_kernel(gn, gk, torch.bfloat16)
    kernel.process_weights_after_loading(layer)

    # In-place requant + per-channel scale replaces the 2-D block scale.
    assert captured["fp8_output"].data_ptr() == orig_weight.data_ptr()
    assert tuple(layer.weight_scale_inv.shape) == (N,)

    # Grouped fp8 weight view: [G, D, R], contiguous, aliases the weight buffer
    # (zero-copy -- no forward weight work).
    assert tuple(layer.bmm_b_fp8.shape) == (G, D, R)
    assert layer.bmm_b_fp8.dtype == torch.float8_e4m3fn
    assert layer.bmm_b_fp8.is_contiguous()
    assert layer.bmm_b_fp8.data_ptr() == orig_weight.data_ptr()

    # Read-safe Bs for fp8_per_channel_einsum: [G, D], last dim contiguous, row
    # stride padded to >= ceil(D/128)*128.
    safe_stride = ((D + 127) // 128) * 128
    assert tuple(layer.bmm_bs.shape) == (G, D)
    assert layer.bmm_bs.dtype == torch.float32
    assert layer.bmm_bs.stride(1) == 1
    assert layer.bmm_bs.stride(0) >= safe_stride


def test_process_weights_after_loading_k_not_multiple_of_64_raises(monkeypatch):
    # K guard fires before the CUDA op is reached, so this needs no device.
    monkeypatch.setattr(
        Fp8BlockScaledMMLinearKernel,
        "process_weights_after_loading",
        lambda self, layer: None,
    )
    N, K, gn, gk = 128, 100, 128, 128  # K=100 is not a multiple of 64
    weight = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
    block_scale = torch.ones(
        (N + gn - 1) // gn, (K + gk - 1) // gk, dtype=torch.float32
    )
    layer = _make_layer(weight, block_scale)
    kernel = _make_kernel(gn, gk, torch.bfloat16)

    with pytest.raises(ValueError, match="K % 64"):
        kernel.process_weights_after_loading(layer)


# ---------------------------------------------------------------------------
# apply_block_scaled_mm: ixformer per-channel path (mocked)
# ---------------------------------------------------------------------------


def _boom(name: str):
    def _fn(*args, **kwargs):
        raise AssertionError(f"{name} must not be called on this path")

    return _fn


def test_apply_block_scaled_mm_per_channel_path(monkeypatch):
    """Gate miss (N % 16 != 0): plain quant + mm, not TN_V2 gemv."""
    import ixformer.inference.functions as ixfops

    captured: dict = {}

    def fake_quant(x, fp8_output=None, output_scales=None, backend=None):
        captured["quant_in"] = x
        m, k = x.shape
        a_fp8 = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device=x.device)
        a_scale = torch.ones(m, dtype=torch.float32, device=x.device)
        return a_fp8, a_scale

    def fake_mm(A, B, As, Bs, *args, **kwargs):
        captured["mm_args"] = (A, B, As, Bs)
        m, n = A.shape[0], B.shape[0]
        return torch.full((m, n), 2.0, dtype=torch.bfloat16, device=A.device)

    monkeypatch.setattr(ixfops, "fp8_per_token_dynamic_scaled", fake_quant)
    monkeypatch.setattr(ixfops, "fp8_per_channel_mm", fake_mm)
    monkeypatch.setattr(
        ixfops,
        "fp8_per_token_dynamic_scaled_nt_v2",
        _boom("nt_v2 quant"),
        raising=False,
    )
    monkeypatch.setattr(ixfops, "fp8_per_channel_gemv", _boom("gemv"), raising=False)

    M, N, K = 4, 8, 128  # N % 16 != 0 → mm fallback
    kernel = _make_kernel(128, 128, torch.bfloat16)
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
    As = A.new_empty(1)  # placeholder (apply_input_quant=False)
    Bs = torch.ones(N, dtype=torch.float32)

    out = kernel.apply_block_scaled_mm(A, B, As, Bs)

    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    # per-channel contract: the raw bf16 activation is fed to the per-token quant,
    # and the per-channel GEMM receives the fp8 weight + [M] act scale + [N] wt
    # scale unchanged.
    assert captured["quant_in"].shape == (M, K)
    mm_a, mm_b, mm_as, mm_bs = captured["mm_args"]
    assert mm_b is B
    assert mm_bs is Bs
    assert mm_as.shape == (M,)
    assert mm_a.dtype == torch.float8_e4m3fn


def test_apply_block_scaled_mm_gemv_nt_v2_unquantized(monkeypatch):
    """No prequant + gemv gate + capability: fused nt_v2 quant + TN_V2 gemv."""
    import ixformer.inference.functions as ixfops

    captured: dict = {}

    def fake_nt_v2(x, *, backend=None):
        captured["nt_v2_in"] = x
        m, k = x.shape
        a_fp8 = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device=x.device)
        a_scale = torch.ones(m, dtype=torch.float32, device=x.device)
        return a_fp8, a_scale, m

    def fake_gemv(A, B, As, Bs, output=None, *args, **kwargs):
        captured["gemv"] = {
            "A": A,
            "B": B,
            "As": As,
            "Bs": Bs,
            "format": kwargs.get("format"),
            "backend": kwargs.get("backend"),
            "output": output,
        }
        m, n = A.shape[0], B.shape[0]
        vals = torch.full((m, n), 3.0, dtype=torch.bfloat16, device=A.device)
        if output is not None:
            output.copy_(vals)
            return output
        return vals

    # Force the capability stand-in so the contract test does not depend on the
    # concrete ixformer/ixmm build installed in the test env.
    monkeypatch.setattr(mod, "_nt_v2_gemv_ready", lambda ixfops: True)
    monkeypatch.setattr(
        ixfops, "fp8_per_token_dynamic_scaled_nt_v2", fake_nt_v2, raising=False
    )
    monkeypatch.setattr(ixfops, "fp8_per_channel_gemv", fake_gemv, raising=False)
    monkeypatch.setattr(ixfops, "fp8_per_token_dynamic_scaled", _boom("plain quant"))
    monkeypatch.setattr(ixfops, "fp8_per_channel_mm", _boom("mm"))
    monkeypatch.setattr(
        ixfops, "pack_act_fp8_gemv_nt_v2", _boom("pack_act"), raising=False
    )

    M, N, K = 1, 64, 128  # decode shape; contiguous B → stride0 % 64 == 0
    kernel = _make_kernel(128, 128, torch.bfloat16)
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
    As = A.new_empty(1)
    Bs = torch.ones(N, dtype=torch.float32)

    out = kernel.apply_block_scaled_mm(A, B, As, Bs)

    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert captured["nt_v2_in"] is A
    gemv = captured["gemv"]
    assert gemv["B"] is B
    assert gemv["Bs"] is Bs
    assert gemv["format"] == "TN_V2"
    assert gemv["backend"] == "ixmm"
    assert gemv["A"].shape[0] == M
    assert gemv["As"].shape == (M,)
    assert gemv["output"] is not None


def test_apply_block_scaled_mm_falls_back_when_nt_v2_api_missing(monkeypatch):
    """Shape gate passes but nt_v2 helper absent → plain quant + mm."""
    import ixformer.inference.functions as ixfops

    captured: dict = {}

    def fake_quant(x, fp8_output=None, output_scales=None, backend=None):
        captured["quant_in"] = x
        m, k = x.shape
        a_fp8 = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device=x.device)
        a_scale = torch.ones(m, dtype=torch.float32, device=x.device)
        return a_fp8, a_scale

    def fake_mm(A, B, As, Bs, *args, **kwargs):
        captured["mm_args"] = (A, B, As, Bs)
        return torch.zeros(
            A.shape[0], B.shape[0], dtype=torch.bfloat16, device=A.device
        )

    # Stand-in for an older ixformer without the fused nt_v2 API: delattr and
    # leave the real capability probe so hasattr fails before quant+pack.
    if hasattr(ixfops, "fp8_per_token_dynamic_scaled_nt_v2"):
        monkeypatch.delattr(ixfops, "fp8_per_token_dynamic_scaled_nt_v2")
    monkeypatch.setattr(ixfops, "fp8_per_token_dynamic_scaled", fake_quant)
    monkeypatch.setattr(ixfops, "fp8_per_channel_mm", fake_mm)
    monkeypatch.setattr(ixfops, "fp8_per_channel_gemv", _boom("gemv"), raising=False)

    M, N, K = 1, 64, 128  # would pass shape gate if capability were present
    kernel = _make_kernel(128, 128, torch.bfloat16)
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
    As = A.new_empty(1)
    Bs = torch.ones(N, dtype=torch.float32)

    out = kernel.apply_block_scaled_mm(A, B, As, Bs)

    assert out.shape == (M, N)
    assert captured["quant_in"] is A
    assert captured["mm_args"][1] is B


def test_apply_block_scaled_mm_falls_back_when_ixmm_gemv_unavailable(monkeypatch):
    """nt_v2 API present but ixmm gemv backend missing → plain quant + mm."""
    import ixformer.inference.functions as ixfops
    from ixformer.inference import backend_registry as br

    captured: dict = {}

    def fake_quant(x, fp8_output=None, output_scales=None, backend=None):
        captured["quant_in"] = x
        m, k = x.shape
        a_fp8 = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device=x.device)
        a_scale = torch.ones(m, dtype=torch.float32, device=x.device)
        return a_fp8, a_scale

    def fake_mm(A, B, As, Bs, *args, **kwargs):
        captured["mm_args"] = (A, B, As, Bs)
        return torch.zeros(
            A.shape[0], B.shape[0], dtype=torch.bfloat16, device=A.device
        )

    def fake_nt_v2(*args, **kwargs):
        raise AssertionError("nt_v2 quant must not run when ixmm gemv is absent")

    # Metadata stand-in: API exists, but available_backends omits ixmm — the
    # real ``_nt_v2_gemv_ready`` probe must refuse before quant+pack.
    monkeypatch.setattr(br.registry, "available_backends", lambda op: ["torch"])
    monkeypatch.setattr(
        ixfops, "fp8_per_token_dynamic_scaled_nt_v2", fake_nt_v2, raising=False
    )
    monkeypatch.setattr(ixfops, "fp8_per_token_dynamic_scaled", fake_quant)
    monkeypatch.setattr(ixfops, "fp8_per_channel_mm", fake_mm)
    monkeypatch.setattr(ixfops, "fp8_per_channel_gemv", _boom("gemv"), raising=False)

    M, N, K = 1, 64, 128
    kernel = _make_kernel(128, 128, torch.bfloat16)
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
    As = A.new_empty(1)
    Bs = torch.ones(N, dtype=torch.float32)

    assert mod._nt_v2_gemv_ready(ixfops) is False
    out = kernel.apply_block_scaled_mm(A, B, As, Bs)

    assert out.shape == (M, N)
    assert captured["quant_in"] is A
    assert captured["mm_args"][1] is B


def test_nt_v2_gemv_ready_probes_api_and_ixmm_backend(monkeypatch):
    """Direct metadata stand-in check for the capability probe itself."""
    from ixformer.inference import backend_registry as br

    # Missing API → False regardless of backend list.
    class _NoApi:
        pass

    assert mod._nt_v2_gemv_ready(_NoApi()) is False

    # API present, backend list without ixmm → False.
    class _WithApi:
        fp8_per_token_dynamic_scaled_nt_v2 = object()

    monkeypatch.setattr(br.registry, "available_backends", lambda op: ["torch"])
    assert mod._nt_v2_gemv_ready(_WithApi()) is False

    # Matching API + ixmm backend → True.
    monkeypatch.setattr(br.registry, "available_backends", lambda op: ["torch", "ixmm"])
    assert mod._nt_v2_gemv_ready(_WithApi()) is True


def test_apply_block_scaled_mm_prequant_uses_mm(monkeypatch):
    """Prefetched plain FP8 always stays on mm; no pack / gemv / nt_v2."""
    import ixformer.inference.functions as ixfops

    captured: dict = {}

    def fake_mm(A, B, As, Bs, *args, **kwargs):
        captured["mm_args"] = (A, B, As, Bs)
        m, n = A.shape[0], B.shape[0]
        return torch.full((m, n), 2.0, dtype=torch.bfloat16, device=A.device)

    monkeypatch.setattr(ixfops, "fp8_per_channel_mm", fake_mm)
    monkeypatch.setattr(ixfops, "fp8_per_token_dynamic_scaled", _boom("plain quant"))
    monkeypatch.setattr(
        ixfops,
        "fp8_per_token_dynamic_scaled_nt_v2",
        _boom("nt_v2 quant"),
        raising=False,
    )
    monkeypatch.setattr(ixfops, "fp8_per_channel_gemv", _boom("gemv"), raising=False)
    monkeypatch.setattr(
        ixfops, "pack_act_fp8_gemv_nt_v2", _boom("pack_act"), raising=False
    )

    M, N, K = 1, 64, 128  # would pass gemv gate if unquantized
    kernel = _make_kernel(128, 128, torch.bfloat16)
    A = torch.randn(M, K, dtype=torch.bfloat16)
    a_fp8 = torch.zeros(M, K, dtype=torch.float8_e4m3fn)
    a_scale = torch.ones(M, dtype=torch.float32)
    B = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
    As = A.new_empty(1)
    Bs = torch.ones(N, dtype=torch.float32)

    mod.clear_activation_prequant()
    try:
        mod.register_activation_prequant(A, a_fp8, a_scale)
        out = kernel.apply_block_scaled_mm(A, B, As, Bs)
    finally:
        mod.clear_activation_prequant()

    assert out.shape == (M, N)
    mm_a, mm_b, mm_as, mm_bs = captured["mm_args"]
    assert mm_a is a_fp8
    assert mm_as is a_scale
    assert mm_b is B
    assert mm_bs is Bs


def test_apply_block_scaled_mm_falls_back_when_m_too_large(monkeypatch):
    """Unquantized but M > 16: plain quant + mm."""
    import ixformer.inference.functions as ixfops

    captured: dict = {}

    def fake_quant(x, fp8_output=None, output_scales=None, backend=None):
        captured["quant_in"] = x
        m, k = x.shape
        a_fp8 = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device=x.device)
        a_scale = torch.ones(m, dtype=torch.float32, device=x.device)
        return a_fp8, a_scale

    def fake_mm(A, B, As, Bs, *args, **kwargs):
        captured["mm_args"] = (A, B, As, Bs)
        return torch.zeros(
            A.shape[0], B.shape[0], dtype=torch.bfloat16, device=A.device
        )

    monkeypatch.setattr(ixfops, "fp8_per_token_dynamic_scaled", fake_quant)
    monkeypatch.setattr(ixfops, "fp8_per_channel_mm", fake_mm)
    monkeypatch.setattr(
        ixfops,
        "fp8_per_token_dynamic_scaled_nt_v2",
        _boom("nt_v2 quant"),
        raising=False,
    )
    monkeypatch.setattr(ixfops, "fp8_per_channel_gemv", _boom("gemv"), raising=False)

    M, N, K = 32, 64, 128  # M > 16
    kernel = _make_kernel(128, 128, torch.bfloat16)
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
    As = A.new_empty(1)
    Bs = torch.ones(N, dtype=torch.float32)

    out = kernel.apply_block_scaled_mm(A, B, As, Bs)

    assert out.shape == (M, N)
    assert captured["quant_in"] is A
    assert captured["mm_args"][1] is B


# ---------------------------------------------------------------------------
# is_supported arch gate (on-device)
# ---------------------------------------------------------------------------


@requires_cuda
def test_is_supported_true_on_device(cuda_device, monkeypatch):
    # On a real Iluvatar device is_cuda_alike() is True; force the arch predicate
    # so the assertion does not depend on the concrete ixformer build.
    monkeypatch.setattr(mod, "_is_ivcore30_fn", lambda: True)
    ok, reason = IluFp8BlockScaledMMKernel.is_supported()
    assert ok is True
    assert reason is None
