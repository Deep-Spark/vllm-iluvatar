# SPDX-License-Identifier: Apache-2.0
"""Iluvatar GateLinear registration, architecture dispatch and dtype contracts."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from conftest import requires_cuda
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
    with (
        patch(
            "vllm.model_executor.layers.fused_moe.router.gate_linear.current_platform",
            _CudaLikePlatform(),
        ),
        patch.object(GateLinear, "__init__", _fake_community_gate_init),
    ):
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
    with (
        patch(
            "vllm.model_executor.layers.fused_moe.router.gate_linear.current_platform",
            _CudaLikePlatform(),
        ),
        patch.object(GateLinear, "__init__", _fake_community_gate_init),
    ):
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


def test_gate_linear_fallback_calls_replicated_linear_directly():
    gate = IluGateLinear.__new__(IluGateLinear)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(torch.eye(2, dtype=torch.float32))
    gate.input_size = 2
    gate.output_size = 2
    gate.out_dtype = None
    gate.allow_dsv3_router_gemm = False
    gate.allow_fp32_router_gemm = False
    gate.allow_cublas_router_gemm = False

    x = torch.tensor([[1.0, 2.0]])
    calls = []

    def fake_replicated_forward(self, input_):
        calls.append((self, input_))
        return "fallback-output", "fallback-bias"

    with (
        patch.object(
            GateLinear,
            "forward",
            side_effect=AssertionError("GateLinear.forward should not be re-entered"),
        ),
        patch.object(ReplicatedLinear, "forward", fake_replicated_forward),
    ):
        assert gate.forward(x) == ("fallback-output", "fallback-bias")

    assert calls == [(gate, x)]


def test_gate_linear_forward_casts_input_and_output_dtype_for_fallback():
    gate = IluGateLinear.__new__(IluGateLinear)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(torch.eye(2, dtype=torch.float32))
    gate.input_size = 2
    gate.output_size = 2
    gate.out_dtype = torch.bfloat16
    x = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)

    def fake_replicated_forward(self, input_):
        assert self is gate
        assert input_.dtype == torch.float32
        return input_, None

    with patch.object(ReplicatedLinear, "forward", fake_replicated_forward):
        output, output_bias = gate.forward(x)

    assert output_bias is None
    assert output.dtype == torch.bfloat16


@requires_cuda
@pytest.mark.parametrize(
    "m,arch,n,dtype,expected",
    [
        (1, "ivcore30", 128, torch.bfloat16, "native"),
        (8, "ivcore30", 128, torch.bfloat16, "native"),
        (32, "ivcore30", 128, torch.bfloat16, "native"),
        (33, "ivcore30", 128, torch.bfloat16, "fallback"),
        (96, "ivcore30", 128, torch.bfloat16, "fallback"),
        (127, "ivcore30", 128, torch.bfloat16, "fallback"),
        (128, "ivcore30", 128, torch.bfloat16, "mixed"),
        (2048, "ivcore30", 128, torch.bfloat16, "mixed"),
        (4096, "ivcore30", 128, torch.bfloat16, "mixed"),
        (4097, "ivcore30", 128, torch.bfloat16, "fallback"),
        (0, "ivcore11", 128, torch.bfloat16, "native"),
        (1, "ivcore11", 128, torch.bfloat16, "native"),
        (6, "ivcore11", 128, torch.bfloat16, "native"),
        (7, "ivcore11", 128, torch.bfloat16, "torch"),
        (8, "ivcore11", 128, torch.bfloat16, "torch"),
        (32, "ivcore11", 128, torch.bfloat16, "torch"),
        (33, "ivcore11", 128, torch.bfloat16, "fallback"),
        (128, "ivcore11", 128, torch.bfloat16, "fallback"),
        (2048, "ivcore11", 128, torch.bfloat16, "fallback"),
        (4096, "ivcore11", 128, torch.bfloat16, "fallback"),
        (8, "ivcore20", 128, torch.bfloat16, "fallback"),
        (2048, "ivcore20", 128, torch.bfloat16, "fallback"),
        (8, None, 128, torch.bfloat16, "fallback"),
        (8, "ivcore30", 64, torch.bfloat16, "fallback"),
        (8, "ivcore30", 128, torch.float32, "fallback"),
        (6, "ivcore11", 64, torch.bfloat16, "fallback"),
        (8, "ivcore11", 128, torch.float32, "fallback"),
    ],
)
def test_fp32_router_dispatch_preserves_weight_and_arch_contract(
    monkeypatch, m, arch, n, dtype, expected
):
    import ixformer.inference.functions as ops

    import vllm_iluvatar.model_executor.layers.fused_moe.router.ilu_gate_linear as mod

    gate = IluGateLinear.__new__(IluGateLinear)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(
        torch.ones(n, 6144, device="cuda", dtype=torch.float32)
    )
    gate.bias = None
    gate.out_dtype = torch.float32
    x = torch.zeros(m, 6144, device="cuda", dtype=dtype)
    calls = []
    monkeypatch.setattr(mod, "cuda_arch", lambda: arch)

    def router(input, weight, *, backend):
        assert input is x and weight is gate.weight
        assert backend in ("ixformer_native", "torch")
        calls.append("native" if backend == "ixformer_native" else "torch")
        return torch.zeros(m, n, device="cuda", dtype=torch.float32)

    def mixed(input, weight):
        assert input is x and weight is gate.weight
        calls.append("mixed")
        return torch.zeros(m, n, device="cuda", dtype=torch.float32)

    def fallback(self, input):
        assert input.dtype == torch.float32
        calls.append("fallback")
        return torch.zeros(m, n, device="cuda", dtype=torch.float32), None

    monkeypatch.setattr(ops, "router_gemm_bf16_fp32_fp32", router)
    monkeypatch.setattr(ops, "mixed_type_linear", mixed)
    monkeypatch.setattr(ReplicatedLinear, "forward", fallback)
    ptr = gate.weight.data_ptr()
    output, bias = gate.forward(x)
    assert calls == [expected] and bias is None
    assert output.shape == (m, n) and output.dtype == torch.float32
    assert gate.weight.data_ptr() == ptr and gate.weight.dtype == torch.float32
