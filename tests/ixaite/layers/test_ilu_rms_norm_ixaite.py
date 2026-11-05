# SPDX-License-Identifier: Apache-2.0
"""Tests for ixaite rms_norm provider routing and IluRMSNorm (no residual)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import vllm._custom_ops as ops

import vllm_iluvatar._custom_ops as ilu_ops
import vllm_iluvatar.backends.ixaite.layers.layernorm as ilu_layernorm


def _ref_rms_norm(x, weight, epsilon):
    y = x.float()
    variance = y.pow(2).mean(dim=-1, keepdim=True)
    out = y * torch.rsqrt(variance + epsilon)
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)


def test_rms_norm_custom_op_dispatches_to_ixformer(monkeypatch):
    ixformer = Mock()
    monkeypatch.setattr(ilu_ops.ixf_ops, "rms_norm", ixformer)

    out, x, weight = object(), object(), object()
    ilu_ops.rms_norm(out, x, weight, 1e-6)
    ixformer.assert_called_once_with(x, weight, 1e-6, out)


def test_baseline_rms_norm_forwards_missing_weight_unchanged(monkeypatch):
    ixformer = Mock()
    monkeypatch.setattr(ilu_ops.ixf_ops, "rms_norm", ixformer)

    x = torch.randn(4, 128)
    out = torch.empty_like(x)
    ilu_ops.rms_norm(out, x, None, 1e-6)

    assert ixformer.call_args.args[1] is None


def test_ilu_rms_norm_plain_routes_rms_norm(monkeypatch, default_vllm_config):
    routed = Mock(side_effect=lambda out, x, weight, eps: out.copy_(x))
    monkeypatch.setattr(ops, "rms_norm", routed, raising=False)

    layer = ilu_layernorm.IluRMSNorm(128)
    x = torch.randn(4, 128)
    out = layer.forward_native(x)

    routed.assert_called_once()
    call_args = routed.call_args.args
    assert call_args[1] is x
    assert call_args[3] == layer.variance_epsilon
    assert out.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_rms_norm_ixaite_wrapper_matches_reference(monkeypatch):
    from vllm_iluvatar.backends.ixaite import ops as ixaite_ops

    monkeypatch.setattr(ops, "rms_norm", ixaite_ops.rms_norm, raising=False)
    torch.manual_seed(11)
    x = torch.randn((8, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((4096,), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)

    ops.rms_norm(out, x, weight, 1e-6)
    selected = ops.rms_norm
    assert selected.__name__ == "rms_norm"
    assert selected.__module__ == "ixaite.ops.rms_norm"

    expected = _ref_rms_norm(x, weight, 1e-6)
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_ilu_rms_norm_forward_native_h4096(monkeypatch, default_vllm_config):
    from vllm_iluvatar.backends.ixaite import ops as ixaite_ops

    monkeypatch.setattr(ops, "rms_norm", ixaite_ops.rms_norm, raising=False)

    layer = ilu_layernorm.IluRMSNorm(4096)
    assert isinstance(layer, ilu_layernorm.IluRMSNorm)
    torch.manual_seed(3)
    x = torch.randn((8, 4096), device="cuda", dtype=torch.bfloat16)
    layer = layer.to(device="cuda", dtype=torch.bfloat16)
    actual = layer.forward_native(x)
    expected = _ref_rms_norm(x, layer.weight.data, layer.variance_epsilon)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


# Exercise ixaite's optional-weight path independently of the process-wide backend.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_ilu_rms_norm_without_weight_on_ixaite(monkeypatch, default_vllm_config):
    from vllm_iluvatar.backends.ixaite import ops as ixaite_ops

    monkeypatch.setattr(ops, "rms_norm", ixaite_ops.rms_norm, raising=False)

    layer = ilu_layernorm.IluRMSNorm(128, has_weight=False)
    assert not layer.pass_weight
    layer = layer.to(device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(7)
    x = torch.randn((4, 128), device="cuda", dtype=torch.bfloat16)

    actual = layer.forward_native(x)

    expected = _ref_rms_norm(x, None, layer.variance_epsilon)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
