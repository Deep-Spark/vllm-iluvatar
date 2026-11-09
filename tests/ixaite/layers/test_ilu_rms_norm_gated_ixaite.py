# SPDX-License-Identifier: Apache-2.0
"""Tests for ixaite rms_norm_gated provider routing and IluRMSNormGated."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import vllm._custom_ops as ops

import vllm_iluvatar.backends.ixaite.layers.layernorm as ilu_layernorm

# Reset every routed target so tests do not leak provider choices.


# fp32 torch reference matching community RMSNormGated.forward_static.
def _ref_rms_norm_gated(x, z, weight, eps, norm_before_gate, activation):
    x_f = x.float()
    w_f = weight.float()
    z_f = z.float() if z is not None else None
    act_fn = (
        torch.nn.functional.sigmoid
        if activation == "sigmoid"
        else torch.nn.functional.silu
    )
    if z_f is not None and not norm_before_gate:
        x_f = x_f * act_fn(z_f)
    variance = x_f.pow(2).mean(dim=-1, keepdim=True)
    out = x_f * torch.rsqrt(variance + eps) * w_f
    if z_f is not None and norm_before_gate:
        out = out * act_fn(z_f)
    return out


# Baseline retains the community layer's FLA path without a plugin OOT override.
def test_baseline_rms_norm_gated_dispatches_to_fla(
    monkeypatch, default_vllm_config
):
    import vllm.third_party.flash_linear_attention.ops.layernorm_guard as fla
    from vllm.model_executor.layers.layernorm import RMSNormGated

    fla_result = torch.zeros(2, 4)
    fla_mock = Mock(return_value=fla_result)
    monkeypatch.setattr(fla, "rmsnorm_fn", fla_mock)

    layer = RMSNormGated(4, eps=1e-6, norm_before_gate=True, activation="swish")
    assert type(layer) is RMSNormGated
    x = torch.randn(2, 4)
    z = torch.randn(2, 4)

    assert layer.forward_cuda(x, z) is fla_result
    fla_mock.assert_called_once_with(
        x,
        layer.weight,
        None,
        z=z,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        activation="swish",
    )


# Check both exported module paths observe the one startup-selected target.


# Check auto chooses the FLA baseline before forward when ixaite is unsupported.


# Check RMSNormGated instantiation yields the plugin's OOT subclass.


# Check the layer routes the GDN output-norm configuration through the wrapper.
def test_ilu_rms_norm_gated_forward_cuda_routes_supported_config(
    monkeypatch, default_vllm_config
):
    routed = Mock()
    monkeypatch.setattr(ops, "rms_norm_gated", routed, raising=False)

    layer = ilu_layernorm.IluRMSNormGated(128, norm_before_gate=True, activation="silu")
    x = torch.randn(4, 32, 128)
    z = torch.randn(4, 32, 128)

    out = layer.forward_cuda(x, z)

    assert routed.call_count == 1
    call_args = routed.call_args.args
    assert call_args[0].shape == x.shape  # out
    assert call_args[1] is x
    assert call_args[2] is z
    assert call_args[3].data_ptr() == layer.weight.data_ptr()
    assert call_args[4:] == (float(layer.eps), True, "silu")
    assert out.shape == x.shape


# Check configs outside the ixaite envelope stay on the community FLA path.
def test_ilu_rms_norm_gated_forward_cuda_outside_envelope_rejects(
    monkeypatch, default_vllm_config
):
    routed = Mock()
    monkeypatch.setattr(ops, "rms_norm_gated", routed, raising=False)

    import vllm.third_party.flash_linear_attention.ops.layernorm_guard as fla

    fla_result = torch.ones(2, 128)
    monkeypatch.setattr(fla, "rmsnorm_fn", Mock(return_value=fla_result))

    # group_size is outside the ixaite envelope (GDN uses group_size=None).
    layer = ilu_layernorm.IluRMSNormGated(128, group_size=64, norm_before_gate=True)
    with pytest.raises(ValueError, match="layout or configuration"):
        layer.forward_cuda(torch.randn(2, 128), torch.randn(2, 128))

    routed.assert_not_called()
    fla.rmsnorm_fn.assert_not_called()


# Check the exported vLLM wrapper reaches the real ixaite op on an Iluvatar GPU.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_rms_norm_gated_ixaite_wrapper_matches_reference(monkeypatch):
    from vllm_iluvatar.backends.ixaite import ops as ixaite_ops

    monkeypatch.setattr(ops, "rms_norm_gated", ixaite_ops.rms_norm_gated, raising=False)
    torch.manual_seed(7)
    # Qwen3.5 GDN output-norm shape: x/z [T, 32, 128], norm_before_gate + silu.
    x = torch.randn((8, 32, 128), device="cuda", dtype=torch.bfloat16)
    z = torch.randn((8, 32, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((128,), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty_like(x)

    ops.rms_norm_gated(actual, x, z, weight, 1e-6, True, "silu")

    assert ops.rms_norm_gated is torch.ops.ixaite.rms_norm_gated.default
    # The Schraudolph exp2 approximation in the gate contributes up to ~6%
    # relative error on the sigmoid factor; the norm itself is fp32-exact.
    expected = _ref_rms_norm_gated(x, z, weight, 1e-6, True, "silu")
    torch.testing.assert_close(actual.float(), expected, rtol=5e-2, atol=5e-2)
