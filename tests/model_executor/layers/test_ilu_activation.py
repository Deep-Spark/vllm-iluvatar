# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar activation OOT replacements."""

from __future__ import annotations

from unittest.mock import patch

import torch
from vllm.model_executor.custom_op import CustomOp, op_registry_oot
from vllm.model_executor.layers.activation import (
    FatreluAndMul,
    FastGELU,
    GeluAndMul,
    MulAndSilu,
    NewGELU,
    QuickGELU,
    SiluAndMul,
    SiluAndMulWithClamp,
)

import vllm._custom_ops as ops
from vllm_iluvatar.model_executor.layers.activation import (
    IluFatreluAndMul,
    IluFastGELU,
    IluGeluAndMul,
    IluMulAndSilu,
    IluNewGELU,
    IluQuickGELU,
    IluSiluAndMul,
    IluSiluAndMulWithClamp,
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

    with patch(
        "vllm_iluvatar.model_executor.layers.activation.current_platform",
        _CudaLikePlatform(),
    ):
        with patch.object(CustomOp, "__init__", return_value=None):
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


def test_silu_and_mul_with_clamp_binds_plugin_custom_op():
    with patch(
        "vllm_iluvatar.model_executor.layers.activation.current_platform",
        _CudaLikePlatform(),
    ):
        with patch.object(CustomOp, "__init__", return_value=None):
            act = SiluAndMulWithClamp(7.0, alpha=1.5, beta=2.0)

    assert act.op is ops.silu_and_mul_with_clamp
