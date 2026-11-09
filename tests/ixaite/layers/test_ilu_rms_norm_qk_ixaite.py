# SPDX-License-Identifier: Apache-2.0
"""Tests for ixaite rms_norm_qk provider routing and D=128 geometry."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import vllm._custom_ops as ops

import vllm_iluvatar._custom_ops as ilu_ops
from vllm_iluvatar.model_executor.layers.layernorm import RMSNormQK


def _ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float()
    var = y.pow(2).mean(dim=-1, keepdim=True)
    return (y * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def test_rms_norm_qk_custom_op_dispatches_to_ixformer(monkeypatch):
    ixformer = Mock()
    monkeypatch.setattr(ilu_ops.ixf_ops, "rms_norm_qk", ixformer)

    args = (object(), object(), object(), object(), object(), object(), 1e-6)
    ilu_ops.rms_norm_qk(*args)
    ixformer.assert_called_once_with(
        args[2], args[3], args[4], args[5], args[6], args[0], args[1]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize(
    ("tokens", "hq", "hk"),
    (
        (1, 32, 8),
        (7, 32, 8),
        (4, 16, 4),
    ),
)
def test_rms_norm_qk_ixaite_d128_matches_reference(
    monkeypatch, tokens: int, hq: int, hk: int
):
    from vllm_iluvatar.backends.ixaite import ops as ixaite_ops

    monkeypatch.setattr(ops, "rms_norm_qk", ixaite_ops.rms_norm_qk, raising=False)

    torch.manual_seed(tokens * 17 + hq)
    d = 128
    q = torch.randn(tokens, hq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(tokens, hk, d, device="cuda", dtype=torch.bfloat16)
    wq = torch.randn(d, device="cuda", dtype=torch.bfloat16)
    wk = torch.randn(d, device="cuda", dtype=torch.bfloat16)
    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)

    ops.rms_norm_qk(out_q, out_k, q, k, wq, wk, 1e-6)
    selected = ops.rms_norm_qk
    assert selected.__name__ == "rms_norm_qk"
    assert selected.__module__ == "ixaite.ops.rms_norm_qk"

    torch.testing.assert_close(
        out_q.float(), _ref(q, wq, 1e-6).float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        out_k.float(), _ref(k, wk, 1e-6).float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_rms_norm_qk_layer_forward_cuda(monkeypatch, default_vllm_config):
    from vllm_iluvatar.backends.ixaite import ops as ixaite_ops

    monkeypatch.setattr(ops, "rms_norm_qk", ixaite_ops.rms_norm_qk, raising=False)

    layer = RMSNormQK(128, 128, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(9)
    q = torch.randn(4, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(4, 8, 128, device="cuda", dtype=torch.bfloat16)
    wq = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    wk = torch.randn(128, device="cuda", dtype=torch.bfloat16)

    out_q, out_k = layer.forward_cuda(q, k, wq, wk)
    torch.testing.assert_close(
        out_q.float(), _ref(q, wq, 1e-6).float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        out_k.float(), _ref(k, wk, 1e-6).float(), rtol=2e-2, atol=2e-2
    )
