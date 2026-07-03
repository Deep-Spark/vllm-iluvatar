# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar GateLinear OOT replacement."""

from __future__ import annotations

from unittest.mock import patch

import torch
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

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
    self.out_dtype = kwargs.get("out_dtype")
    self.allow_specialized_router_gemm = True
    self.allow_dsv3_router_gemm = True
    self.allow_fp32_router_gemm = True
    self.allow_cublas_router_gemm = True


def test_gate_linear_disables_specialized_router_gemm_after_community_init():
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
    assert gate.allow_specialized_router_gemm is False
    assert gate.allow_dsv3_router_gemm is False
    assert gate.allow_fp32_router_gemm is False
    assert gate.allow_cublas_router_gemm is False


def test_gate_linear_set_out_dtype_does_not_reenable_cublas_router_gemm():
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
    assert gate.allow_specialized_router_gemm is False
    assert gate.allow_cublas_router_gemm is False
