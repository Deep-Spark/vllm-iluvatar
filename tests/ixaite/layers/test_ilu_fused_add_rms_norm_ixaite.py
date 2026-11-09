# SPDX-License-Identifier:  Apache-2.0
"""Tests for ixaite fused_add_rms_norm provider routing and IluRMSNorm."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import vllm._custom_ops as ops

import vllm_iluvatar._custom_ops as ilu_ops
import vllm_iluvatar.backends.ixaite.layers.layernorm as ilu_layernorm


def _ref_fused_add_rms_norm(x, residual, weight, epsilon):
    y = x.float() + residual.float()
    residual_out = y.to(x.dtype)
    variance = y.pow(2).mean(dim=-1, keepdim=True)
    out = y * torch.rsqrt(variance + epsilon)
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype), residual_out


def test_fused_add_rms_norm_custom_op_dispatches_to_ixformer(monkeypatch):
    ixformer = Mock(return_value=(object(), object()))
    monkeypatch.setattr(ilu_ops.ixf_ops, "residual_rms_norm", ixformer)

    x = object()
    residual = object()
    weight = object()

    result = ilu_ops.fused_add_rms_norm(x, residual, weight, 1e-6)
    ixformer.assert_called_once_with(x, weight, 1e-6, 1.0, residual)
    assert result == ixformer.return_value


def test_ilu_rms_norm_residual_routes_fused_add_rms_norm(
    monkeypatch, default_vllm_config
):
    routed = Mock(return_value=(torch.zeros(4, 128), torch.ones(4, 128)))
    monkeypatch.setattr(ops, "fused_add_rms_norm", routed, raising=False)

    layer = ilu_layernorm.IluRMSNorm(128)
    x = torch.randn(4, 128)
    residual = torch.randn(4, 128)
    out, residual_out = layer.forward_native(x, residual)

    routed.assert_called_once()
    call_args = routed.call_args.args
    assert call_args[0] is x
    assert call_args[1] is residual
    assert call_args[3] == layer.variance_epsilon
    assert out.shape == x.shape
    assert residual_out.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_fused_add_rms_norm_ixaite_wrapper_matches_reference(monkeypatch):
    from vllm_iluvatar.backends.ixaite import ops as ixaite_ops

    monkeypatch.setattr(
        ops, "fused_add_rms_norm", ixaite_ops.fused_add_rms_norm, raising=False
    )
    torch.manual_seed(7)
    x = torch.randn((8, 4096), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((8, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((4096,), device="cuda", dtype=torch.bfloat16)

    actual_out, actual_res = ops.fused_add_rms_norm(x, residual, weight, 1e-6)
    selected = ops.fused_add_rms_norm
    assert selected.__name__ == "fused_add_rms_norm"
    assert selected.__module__ == "ixaite.ops.fused_add_rms_norm"

    expected_out, expected_res = _ref_fused_add_rms_norm(x, residual, weight, 1e-6)
    torch.testing.assert_close(
        actual_out.float(), expected_out.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        actual_res.float(), expected_res.float(), rtol=2e-2, atol=2e-2
    )
