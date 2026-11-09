# SPDX-License-Identifier: Apache-2.0
"""Tests for ixaite gemma_rms_norm provider routing and IluGemmaRMSNorm."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import vllm._custom_ops as ops

import vllm_iluvatar.backends.ixaite.layers.layernorm as ilu_layernorm


def _ref_gemma_rms_norm(x, residual, weight, epsilon):
    y = x.float() + residual.float()
    residual_out = y.to(x.dtype)
    variance = y.pow(2).mean(dim=-1, keepdim=True)
    out = y * torch.rsqrt(variance + epsilon) * (1.0 + weight.float())
    return out.to(x.dtype), residual_out


# Baseline's original layer calls ixformer directly, without an added public wrapper.
def test_baseline_gemma_rms_norm_dispatches_to_ixformer(
    monkeypatch, default_vllm_config
):
    from vllm_iluvatar.model_executor.layers import layernorm as baseline_layernorm

    ixformer = Mock(return_value=(object(), object()))
    monkeypatch.setattr(baseline_layernorm.ixf_ops, "gemma_rms_norm", ixformer)

    layer = baseline_layernorm.IluGemmaRMSNorm(128)
    x = torch.randn(4, 128)
    residual = torch.randn_like(x)

    result = layer.forward_cuda(x, residual)
    ixformer.assert_called_once()
    arguments = ixformer.call_args.kwargs
    assert arguments["input"] is x
    assert arguments["weight"].data_ptr() == layer.weight.data_ptr()
    assert arguments["eps"] == layer.variance_epsilon
    assert arguments["residual"] is residual
    assert result == ixformer.return_value


def test_ilu_gemma_rms_norm_cuda_routes_gemma_rms_norm(
    monkeypatch, default_vllm_config
):
    routed = Mock(return_value=(torch.zeros(4, 128), torch.ones(4, 128)))
    monkeypatch.setattr(ops, "gemma_rms_norm", routed, raising=False)

    layer = ilu_layernorm.IluGemmaRMSNorm(128)
    x = torch.randn(4, 128)
    residual = torch.randn(4, 128)
    out, residual_out = layer.forward_cuda(x, residual)

    routed.assert_called_once()
    call_args = routed.call_args.args
    assert call_args[0] is x
    # The raw Gemma weight is passed through; the kernel applies (1 + w).
    assert call_args[1].data_ptr() == layer.weight.data_ptr()
    assert call_args[2] == layer.variance_epsilon
    assert call_args[3] is residual
    assert out.shape == x.shape
    assert residual_out.shape == x.shape


def test_ilu_gemma_rms_norm_cuda_routes_plain_norm(monkeypatch, default_vllm_config):
    routed = Mock(return_value=torch.zeros(4, 128))
    monkeypatch.setattr(ops, "gemma_rms_norm", routed, raising=False)

    layer = ilu_layernorm.IluGemmaRMSNorm(128)
    x = torch.randn(4, 128)
    out = layer.forward_cuda(x)

    routed.assert_called_once()
    call_args = routed.call_args.args
    assert call_args[0] is x
    assert call_args[1].data_ptr() == layer.weight.data_ptr()
    assert call_args[3] is None
    assert out.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_gemma_rms_norm_ixaite_wrapper_matches_reference(monkeypatch):
    from vllm_iluvatar.backends.ixaite import ops as ixaite_ops

    monkeypatch.setattr(ops, "gemma_rms_norm", ixaite_ops.gemma_rms_norm, raising=False)
    torch.manual_seed(7)
    x = torch.randn((8, 4096), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((8, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((4096,), device="cuda", dtype=torch.bfloat16)

    actual_out, actual_res = ops.gemma_rms_norm(x, weight, 1e-6, residual)
    selected = ops.gemma_rms_norm
    assert selected.__name__ == "gemma_rms_norm"
    assert selected.__module__ == "ixaite.ops.gemma_rms_norm"

    expected_out, expected_res = _ref_gemma_rms_norm(x, residual, weight, 1e-6)
    torch.testing.assert_close(
        actual_out.float(), expected_out.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        actual_res.float(), expected_res.float(), rtol=2e-2, atol=2e-2
    )
