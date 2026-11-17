# SPDX-License-Identifier: Apache-2.0
"""Exercise the linear entry points through the native ixformer fused API."""

import pytest
import torch
from conftest import requires_corex30

from vllm_iluvatar.model_executor.layers.quantization.backends.fp8 import (
    linear as fp8_linear,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.linear import (
    native_to_fp8 as fp8_online,
)


def _impl(online):
    return fp8_online if online else fp8_linear


def _method(online):
    impl = _impl(online)
    cls = impl.IluFp8PerChannelOnlineLinearMethod if online else impl.IluFp8LinearMethod
    method = object.__new__(cls)
    if not online:
        method.int8_scheme = None
    method.gemm_format = "TN"
    method.block_quant = False
    method._runtime_context = None
    return method


@requires_corex30
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("m", range(1, 7))
def test_fused_linear_uses_bf16_input_without_quant_workspace(monkeypatch, online, m):
    torch.manual_seed(17 + m)
    layer = torch.nn.Module()
    layer.output_size_per_partition = 96
    layer._can_use_gemm = True
    layer.weight = torch.randn(96, 128, device="cuda").to(torch.float8_e4m3fn)
    layer.weight_scale = torch.rand(96, device="cuda") * 0.02 + 0.002
    x = torch.randn(1, m, 128, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(96, device="cuda", dtype=torch.bfloat16)
    cpu = x.cpu().float().reshape(m, 128)
    scale = (cpu.abs().amax(-1, keepdim=True) * (1.0 / 448.0)).clamp_min(1e-12)
    q = (cpu / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    expected = (
        (
            (q @ layer.weight.cpu().float().T) * (scale * layer.weight_scale.cpu())
            + bias.cpu().float()
        )
        .to(torch.bfloat16)
        .float()
    )
    impl = _impl(online)
    monkeypatch.setattr(
        impl,
        "fp8_per_token_dynamic_scaled",
        lambda *a, **kw: pytest.fail("fused linear allocated a quantization workspace"),
    )
    result = _method(online).apply(layer, x, bias)
    assert result.shape == (1, m, 96) and result.dtype == x.dtype
    diff = (result.cpu().float().reshape(m, 96) - expected).abs()
    assert (
        diff <= 0.02 * expected.abs() + 0.01 * expected.square().mean().sqrt()
    ).all()


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("m", [7, 16, 17])
def test_unoptimized_bs_retains_separate_quantization(monkeypatch, online, m):
    layer = torch.nn.Module()
    layer.output_size_per_partition = 96
    layer._can_use_gemm = True
    layer.weight = torch.empty(96, 128, dtype=torch.float8_e4m3fn)
    layer.weight_scale = torch.ones(96)
    x = torch.ones(m, 128, dtype=torch.bfloat16)
    calls = []

    def quant(x):
        calls.append("quant")
        return x.to(torch.float8_e4m3fn), torch.ones(m)

    def gemv(a, b, sa, sb, out, *args, **kw):
        calls.append("gemv")
        out.fill_(7)

    def mm(a, b, sa, sb, out, *args, **kw):
        calls.append("mm")
        out.fill_(7)

    impl = _impl(online)
    monkeypatch.setattr(impl, "fp8_per_token_dynamic_scaled", quant)
    monkeypatch.setattr(impl.ixfops, "fp8_per_channel_gemv", gemv)
    monkeypatch.setattr(impl.ixfops, "fp8_per_channel_mm", mm)
    result = _method(online).apply(layer, x)
    assert calls == ["quant", "gemv" if m <= 16 else "mm"]
    assert result.dtype == x.dtype and result.shape == (m, 96)
    assert bool((result == 7).all())


@requires_corex30
@pytest.mark.parametrize("online", [False, True])
def test_fused_linear_traces_as_a_full_graph(monkeypatch, online):
    from ixformer.core import config

    torch.manual_seed(29)
    monkeypatch.setattr(config, "IXFORMER_USE_TORCH_OPS", True)
    method = _method(online)
    layer = torch.nn.Module()
    layer.output_size_per_partition = 96
    layer._can_use_gemm = True
    layer.weight = torch.randn(96, 128, device="cuda").to(torch.float8_e4m3fn)
    layer.weight_scale = torch.ones(96, device="cuda") * 0.01
    x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(
        _impl(online),
        "fp8_per_token_dynamic_scaled",
        lambda *a, **kw: pytest.fail("compiled fusion used separate quantization"),
    )
    expected = method.apply(layer, x)
    compiled = torch.compile(
        lambda value: method.apply(layer, value), backend="eager", fullgraph=True
    )
    assert torch.equal(compiled(x), expected)
