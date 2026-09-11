# SPDX-License-Identifier: Apache-2.0
"""Tests for Iluvatar activation OOT replacements.

CPU-only except for the ``SwigluStepAndMul`` numeric case at the bottom, which
needs the real ixformer kernel and is gated by ``requires_cuda``.
"""

from __future__ import annotations

import inspect
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
import torch
import vllm._custom_ops as ops
from conftest import requires_cuda
from vllm.model_executor.custom_op import CustomOp, op_registry_oot
from vllm.model_executor.layers.activation import (
    FastGELU,
    FatreluAndMul,
    GeluAndMul,
    MulAndSilu,
    NewGELU,
    QuickGELU,
    SiluAndMul,
    SiluAndMulWithClamp,
    SwigluStepAndMul,
)

from vllm_iluvatar.model_executor.layers.activation import (
    IluFastGELU,
    IluFatreluAndMul,
    IluGeluAndMul,
    IluMulAndSilu,
    IluNewGELU,
    IluQuickGELU,
    IluSiluAndMul,
    IluSiluAndMulWithClamp,
    IluSwigluStepAndMul,
    swiglu_clamp,
)


class _CudaLikePlatform:
    def is_cuda_alike(self) -> bool:
        return True

    def is_xpu(self) -> bool:
        return False

    def is_rocm(self) -> bool:
        return False

    def is_cpu(self) -> bool:
        return False


def test_activation_oot_registration_uses_upstream_class_names():
    assert op_registry_oot["FatreluAndMul"] is IluFatreluAndMul
    assert op_registry_oot["SiluAndMul"] is IluSiluAndMul
    assert op_registry_oot["SiluAndMulWithClamp"] is IluSiluAndMulWithClamp
    assert op_registry_oot["MulAndSilu"] is IluMulAndSilu
    assert op_registry_oot["GeluAndMul"] is IluGeluAndMul
    assert op_registry_oot["NewGELU"] is IluNewGELU
    assert op_registry_oot["FastGELU"] is IluFastGELU
    assert op_registry_oot["QuickGELU"] is IluQuickGELU
    assert op_registry_oot["SwigluStepAndMul"] is IluSwigluStepAndMul


def test_activation_classes_bind_plugin_custom_ops(monkeypatch):
    sentinels = {
        "fatrelu_and_mul": object(),
        "silu_and_mul": object(),
        "silu_and_mul_with_clamp": object(),
        "mul_and_silu": object(),
        "gelu_and_mul": object(),
        "gelu_tanh_and_mul": object(),
        "gelu_new": object(),
        "gelu_fast": object(),
        "gelu_quick": object(),
    }
    for name, sentinel in sentinels.items():
        monkeypatch.setattr(ops, name, sentinel, raising=False)

    with (
        patch(
            "vllm_iluvatar.model_executor.layers.activation.current_platform",
            _CudaLikePlatform(),
        ),
        patch.object(CustomOp, "__init__", return_value=None),
    ):
        assert FatreluAndMul().op is sentinels["fatrelu_and_mul"]
        assert SiluAndMul().op is sentinels["silu_and_mul"]
        assert SiluAndMulWithClamp(7.0).op is sentinels["silu_and_mul_with_clamp"]
        assert MulAndSilu().op is sentinels["mul_and_silu"]
        assert GeluAndMul().op is sentinels["gelu_and_mul"]
        assert GeluAndMul(approximate="tanh").op is sentinels["gelu_tanh_and_mul"]
        assert NewGELU().op is sentinels["gelu_new"]
        assert FastGELU().op is sentinels["gelu_fast"]
        assert QuickGELU().op is sentinels["gelu_quick"]


def test_silu_and_mul_with_clamp_supports_non_default_alpha_beta():
    x = torch.tensor([[2.0, -3.0, 4.0, -5.0]], dtype=torch.float32)
    out = torch.empty((1, 2), dtype=torch.float32)

    ops.silu_and_mul_with_clamp(out, x, limit=3.0, alpha=1.5, beta=2.0)

    gate = torch.clamp(x[..., :2], max=3.0)
    up = torch.clamp(x[..., 2:], min=-3.0, max=3.0)
    expected = gate * torch.sigmoid(1.5 * gate) * (up + 2.0)
    torch.testing.assert_close(out, expected)


def test_silu_and_mul_with_clamp_forwards_alpha_beta_to_ixformer(monkeypatch):
    """alpha/beta selection lives in ixformer, not in a plugin-side branch."""
    from vllm_iluvatar import _custom_ops as ilu_ops

    seen = {}

    def _fake(input, output, limit, alpha, beta):
        seen["kwargs"] = (limit, alpha, beta)
        output.zero_()
        return output

    monkeypatch.setattr(ilu_ops.ixf_ops, "silu_and_mul_with_clamp", _fake)
    ilu_ops.silu_and_mul_with_clamp(
        torch.empty(1, 2), torch.zeros(1, 4), limit=7.0, alpha=1.702, beta=1.0
    )

    assert seen["kwargs"] == (7.0, 1.702, 1.0)


def test_silu_and_mul_with_clamp_binds_plugin_custom_op():
    with (
        patch(
            "vllm_iluvatar.model_executor.layers.activation.current_platform",
            _CudaLikePlatform(),
        ),
        patch.object(CustomOp, "__init__", return_value=None),
    ):
        act = SiluAndMulWithClamp(7.0, alpha=1.5, beta=2.0)

    assert act.op is ops.silu_and_mul_with_clamp


# ---------------------------------------------------------------------------
# SwigluStepAndMul (Step-3.5 shared expert, layers 43/44)
#
# The delta vs community is forward_cuda: community calls the triton kernel
# ``swiglustep_and_mul_triton`` directly (not through ``self.op``), the plugin
# routes to the ixformer facade ``ixf_ops.swiglustep_and_mul``.  The CPU tests
# below fake that facade to pin the call contract; the device test at the end
# runs the real kernel against ``forward_native``.
# ---------------------------------------------------------------------------


def _install_fake_ixformer(monkeypatch, swiglustep_and_mul):
    ixformer = ModuleType("ixformer")
    inference = ModuleType("ixformer.inference")
    functions = ModuleType("ixformer.inference.functions")
    functions.swiglustep_and_mul = swiglustep_and_mul
    inference.functions = functions
    ixformer.inference = inference
    monkeypatch.setitem(sys.modules, "ixformer", ixformer)
    monkeypatch.setitem(sys.modules, "ixformer.inference", inference)
    monkeypatch.setitem(sys.modules, "ixformer.inference.functions", functions)


def _forbid_community_triton_kernel(monkeypatch):
    """Fail loudly if the OOT forward_cuda falls back to the community path."""
    import vllm.model_executor.layers.activation as community_activation

    def _boom(*args, **kwargs):
        raise AssertionError("community triton kernel must not be reached")

    monkeypatch.setattr(
        community_activation, "swiglustep_and_mul_triton", _boom, raising=True
    )


def test_swiglustep_and_mul_instantiates_oot_subclass(default_vllm_config):
    """``SwigluStepAndMul(limit=...)`` (step3p5.py:122) yields the OOT class."""
    layer = SwigluStepAndMul(limit=7)

    assert isinstance(layer, IluSwigluStepAndMul)
    assert layer.limit == 7.0
    assert isinstance(layer.limit, float)
    # Only forward_cuda is overridden; forward_native stays the community
    # clamp-after-silu baseline the device test below compares against.
    assert IluSwigluStepAndMul.forward_native is SwigluStepAndMul.forward_native


def test_swiglustep_and_mul_default_limit_matches_community(default_vllm_config):
    """step3p5.py always passes ``limit=`` explicitly, so a diverging default
    would be silent. ``SwigluStepAndMul()`` instantiates the OOT subclass, so
    this compares the plugin default against the community declaration."""
    community_default = (
        inspect.signature(SwigluStepAndMul.__init__).parameters["limit"].default
    )

    assert community_default == 7.0
    assert SwigluStepAndMul().limit == community_default


def test_swiglustep_and_mul_rejects_none_limit(default_vllm_config):
    with pytest.raises(ValueError):
        SwigluStepAndMul(limit=None)


def test_swiglustep_forward_cuda_wires_ixformer_facade(
    monkeypatch, default_vllm_config
):
    """Whole ``x`` in, plugin-allocated ``out`` buffer, numeric limit, and the
    facade's return value passed straight back to the caller."""
    calls = []

    def fake_swiglustep_and_mul(*, input, output, limit):
        calls.append((input, output, limit))
        output.fill_(1.0)
        return output

    _install_fake_ixformer(monkeypatch, fake_swiglustep_and_mul)
    _forbid_community_triton_kernel(monkeypatch)

    layer = SwigluStepAndMul(limit=7.0)
    x = torch.randn(5, 2 * 64, dtype=torch.bfloat16)
    x_before = x.clone()

    out = layer.forward_cuda(x)

    assert len(calls) == 1
    got_input, got_output, got_limit = calls[0]
    assert got_input is x
    assert got_limit == 7.0
    assert got_output.shape == (5, 64)
    assert got_output.dtype == x.dtype
    assert got_output.device == x.device
    assert out is got_output
    assert torch.equal(x, x_before)


def test_swiglustep_forward_cuda_keeps_leading_dims(monkeypatch, default_vllm_config):
    """3-D activations (batch, seq, 2*d) keep their leading dims: only the last
    dim is halved."""
    _install_fake_ixformer(monkeypatch, lambda *, input, output, limit: output)

    layer = SwigluStepAndMul(limit=16.0)
    out = layer.forward_cuda(torch.randn(2, 3, 2 * 128))

    assert out.shape == (2, 3, 128)


def test_swiglustep_forward_cuda_matches_native(monkeypatch, default_vllm_config):
    """With a facade that implements the documented clamp-after-silu contract,
    forward_cuda must reproduce ``forward_native`` — this catches wiring bugs
    such as a wrong ``d`` split or a dropped ``limit``."""

    def ref_swiglustep_and_mul(*, input, output, limit):
        gate, up = input.chunk(2, dim=-1)
        gate = torch.nn.functional.silu(gate).clamp(max=limit)
        output.copy_(gate * up.clamp(min=-limit, max=limit))
        return output

    _install_fake_ixformer(monkeypatch, ref_swiglustep_and_mul)

    layer = SwigluStepAndMul(limit=7.0)
    x = torch.randn(17, 2 * 32, dtype=torch.float32) * 10.0

    torch.testing.assert_close(layer.forward_cuda(x), layer.forward_native(x))


def test_swiglustep_is_not_the_dsv4_clamp_before_silu_variant(default_vllm_config):
    """Guard against re-routing this op to ``silu_and_mul_with_clamp``: with
    ``silu(gate) > limit`` the two clamp orders disagree, so the Step-3.5 layer
    must stay on the clamp-after-silu semantic."""
    limit = 7.0
    gate = torch.full((4, 8), 20.0)
    up = torch.ones((4, 8))
    x = torch.cat([gate, up], dim=-1)

    layer = SwigluStepAndMul(limit=limit)
    native = layer.forward_native(x)

    # clamp-after-silu: silu(20) ~= 20 -> capped at limit, times up = 1.
    torch.testing.assert_close(native, torch.full((4, 8), limit))
    # clamp-before-silu would instead give silu(7) ~= 6.994.
    assert (native - swiglu_clamp(x, limit)).abs().max() > 1e-3


_SWIGLUSTEP_TOL = {
    torch.float32: dict(atol=1e-4, rtol=1e-5),
    torch.float16: dict(atol=5e-3, rtol=5e-3),
    torch.bfloat16: dict(atol=5e-2, rtol=3e-2),
}


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("limit", [7.0, 16.0])
@pytest.mark.parametrize("shape", [(7, 2 * 512), (2, 3, 2 * 128), (2048, 2 * 4096)])
@torch.inference_mode()
def test_swiglustep_forward_cuda_matches_native_on_device(
    cuda_device, default_vllm_config, dtype, limit, shape
):
    torch.manual_seed(0)
    # Span [-3*limit, 3*limit] so both clamp branches are exercised.
    x = (torch.rand(shape, device=cuda_device) * 6.0 - 3.0) * limit
    x = x.to(dtype)

    layer = SwigluStepAndMul(limit=limit)
    out = layer.forward_cuda(x)
    ref = layer.forward_native(x)

    assert out.shape == ref.shape
    torch.testing.assert_close(out.float(), ref.float(), **_SWIGLUSTEP_TOL[dtype])
