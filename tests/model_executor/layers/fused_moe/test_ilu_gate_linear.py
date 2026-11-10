# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar GateLinear OOT replacement."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_iluvatar.model_executor.layers.fused_moe.router.ilu_gate_linear import (
    IluGateLinear,
)


class _CudaLikePlatform:
    def is_device_capability(self, capability: tuple[int, int]) -> bool:
        return capability == (9, 0)

    def is_device_capability_family(self, family: int) -> bool:
        return family == 100

    def is_cuda(self) -> bool:
        return True


def test_gate_linear_oot_registration_uses_upstream_class_name():
    assert issubclass(IluGateLinear, GateLinear)
    assert op_registry_oot["GateLinear"] is IluGateLinear


def test_gate_linear_router_package_is_auto_discovered():
    from vllm_iluvatar.ilu_register import _OOT_OP_PACKAGES

    assert "vllm_iluvatar.model_executor.layers.fused_moe.router" in _OOT_OP_PACKAGES


def _fake_community_gate_init(self, *args, **kwargs) -> None:
    torch.nn.Module.__init__(self)
    self.out_dtype = kwargs.get("out_dtype")
    self.weight = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float32))
    self._dsv3_max_batch = 16
    self.allow_specialized_router_gemm = True
    self.allow_dsv3_router_gemm = True
    self.allow_fp32_router_gemm = True
    self.allow_cublas_router_gemm = True


def test_gate_linear_keeps_community_dispatch_flags_after_init():
    with patch(
        "vllm.model_executor.layers.fused_moe.router.gate_linear.current_platform",
        _CudaLikePlatform(),
    ):
        with patch.object(GateLinear, "__init__", _fake_community_gate_init):
            gate = GateLinear(
                input_size=3072,
                output_size=256,
                bias=False,
                out_dtype=torch.float32,
                params_dtype=torch.float32,
            )

    assert isinstance(gate, IluGateLinear)
    assert gate.allow_specialized_router_gemm is True
    assert gate.allow_dsv3_router_gemm is True
    assert gate.allow_fp32_router_gemm is True
    assert gate.allow_cublas_router_gemm is True


def test_gate_linear_set_out_dtype_keeps_community_flag_behavior():
    with patch(
        "vllm.model_executor.layers.fused_moe.router.gate_linear.current_platform",
        _CudaLikePlatform(),
    ):
        with patch.object(GateLinear, "__init__", _fake_community_gate_init):
            gate = GateLinear(
                input_size=3072,
                output_size=256,
                bias=False,
                params_dtype=torch.bfloat16,
            )

    gate.set_out_dtype(torch.float32)

    assert isinstance(gate, IluGateLinear)
    assert gate.allow_specialized_router_gemm is True
    assert gate.allow_cublas_router_gemm is True


def _fallback_gate(out_dtype):
    gate = IluGateLinear.__new__(IluGateLinear)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(torch.eye(2, dtype=torch.float32))
    gate.input_size = 2
    gate.output_size = 2
    gate.out_dtype = out_dtype
    gate.is_unquantized = True
    gate.return_bias = True
    gate.allow_ll_bf16_gemm = False
    gate.allow_fp32_router_gemm = False
    gate.allow_bf16x3_router_gemm = False
    gate.allow_cublas_router_gemm = False
    return gate


def test_gate_linear_fallback_delegates_to_community():
    gate = _fallback_gate(None)
    x = torch.tensor([[1.0, 2.0]])
    calls = []

    def fake_replicated_forward(self, input_):
        calls.append((self, input_))
        return "fallback-output", "fallback-bias"

    with patch.object(ReplicatedLinear, "forward", fake_replicated_forward):
        assert gate.forward(x) == ("fallback-output", "fallback-bias")

    assert calls == [(gate, x)]


def test_gate_linear_forward_casts_input_and_output_dtype_for_fallback():
    gate = _fallback_gate(torch.bfloat16)
    x = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)

    def fake_replicated_forward(self, input_):
        assert self is gate
        assert input_.dtype == torch.float32
        return input_, None

    with patch.object(ReplicatedLinear, "forward", fake_replicated_forward):
        output, output_bias = gate.forward(x)

    assert output_bias is None
    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize("enabled,n,k,expected", [
    (False, 384, 5120, "stock"),
    (True, 384, 5120, "split"),
    (True, 256, 5120, "stock"),
    (True, 384, 3072, "stock"),
])
@pytest.mark.parametrize("return_bias", [False, True])
def test_gate_linear_splitk_opt_in_and_shape_guard(enabled, n, k, expected, return_bias):
    import ixformer.inference.functions as ixfops
    from vllm_iluvatar.model_executor.layers.fused_moe.router import ilu_gate_linear

    gate = _fallback_gate(torch.float32)
    gate.weight = torch.nn.Parameter(torch.empty(n, k, dtype=torch.bfloat16))
    gate.return_bias = return_bias
    x = torch.empty(3, k, dtype=torch.bfloat16)
    value = torch.zeros(3, n, dtype=torch.float32)
    with patch.object(ilu_gate_linear, "_DS41_ROUTER_SPLITK", enabled):
        with patch.object(ixfops, "gemm_bf16_bf16_fp32", return_value=value) as stock:
            with patch.object(ixfops, "ds4_router_gate_gemm", return_value=value) as split:
                result = gate(x)
    if expected == "split":
        split.assert_called_once_with(x, gate.weight)
        stock.assert_not_called()
    else:
        stock.assert_called_once_with(x, gate.weight, format="TN")
        split.assert_not_called()
    if return_bias:
        assert result[0] is value and result[1] is None
    else:
        assert result is value


@pytest.mark.parametrize("quantized,bias,out_dtype", [
    (True, False, torch.float32),
    (False, True, torch.float32),
    (False, False, torch.bfloat16),
])
def test_gate_linear_splitk_preserves_community_exclusions(quantized, bias, out_dtype):
    import ixformer.inference.functions as ixfops
    from vllm_iluvatar.model_executor.layers.fused_moe.router import ilu_gate_linear

    gate = _fallback_gate(out_dtype)
    gate.is_unquantized = not quantized
    if quantized:
        del gate.weight  # Quantized gates need not expose a plain weight.
    else:
        gate.weight = torch.nn.Parameter(torch.empty(384, 5120, dtype=torch.bfloat16))
    gate.bias = torch.nn.Parameter(torch.zeros(384)) if bias else None
    x = torch.empty(3, 5120, dtype=torch.bfloat16)
    sentinel = object()
    with patch.object(ilu_gate_linear, "_DS41_ROUTER_SPLITK", True):
        with patch.object(ixfops, "ds4_router_gate_gemm") as split:
            with patch.object(GateLinear, "forward", return_value=sentinel) as parent:
                assert gate(x) is sentinel
    split.assert_not_called()
    parent.assert_called_once_with(x)
