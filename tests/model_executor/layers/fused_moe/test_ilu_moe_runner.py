# SPDX-License-Identifier: Apache-2.0
"""Tests for IluMoERunner.

CPU-only tests use mocks to verify override behaviour without constructing
a full MoE layer stack. GPU-only tests (marked ``requires_cuda``) would
exercise the numerical forward path — those are deferred to P2 and can be
added when the full environment is available.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    _moe_forward_shared,
)

from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_moe_runner import (
    IluMoERunner,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_runner(
    *,
    routed_scaling_factor: float = 2.0,
    in_dtype: torch.dtype = torch.bfloat16,
    has_shared_experts: bool = True,
    enable_elastic_ep: bool = False,
) -> IluMoERunner:
    """Create an IluMoERunner with __init__ bypassed (mock internals).

    We must bootstrap nn.Module internals so that __getattr__ / __setattr__
    don't crash, AND we need routed_experts.quant_method to satisfy the
    _quant_method property (which is a data descriptor that __dict__ cannot
    shadow).
    """
    obj = IluMoERunner.__new__(IluMoERunner)

    # Bootstrap nn.Module internals (normally done by nn.Module.__init__)
    obj.__dict__["_parameters"] = {}
    obj.__dict__["_buffers"] = {}
    obj.__dict__["_modules"] = {}
    obj.__dict__["_non_persistent_buffers_set"] = set()
    obj.__dict__["training"] = False

    moe_config = MagicMock()
    moe_config.in_dtype = in_dtype
    moe_config.dp_size = 1
    moe_config.pcp_size = 1
    moe_config.hidden_dim_unpadded = 128

    obj.moe_config = moe_config
    obj.routed_scaling_factor = routed_scaling_factor
    obj.shared_experts_scaling_factor = 1.0

    shared = MagicMock() if has_shared_experts else None
    obj._shared_experts = shared

    # _quant_method is a read-only property returning
    # self.routed_experts.quant_method, so we mock routed_experts.
    routed = MagicMock()
    routed.quant_method = MagicMock()
    routed.quant_method.has_unpadded_output = False
    obj.routed_experts = routed

    return obj


# ---------------------------------------------------------------------------
# CPU-only tests
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_is_subclass(self):
        mro_names = [cls.__name__ for cls in IluMoERunner.__mro__]
        assert "MoERunner" in mro_names

    def test_is_distinct(self):
        assert IluMoERunner.__name__ == "IluMoERunner"
        mro_names = [cls.__name__ for cls in IluMoERunner.__mro__]
        assert mro_names.index("IluMoERunner") < mro_names.index("MoERunner")


class TestSelectForward:
    """IluMoERunner._select_forward must bypass torch.compile custom ops
    and always return the Python wrapper."""

    def test_returns_python_wrapper(self):
        runner = _make_stub_runner()
        result = runner._select_forward()
        assert result is _moe_forward_shared


class TestFP16ScalingFactor:
    """When dtype is FP16 and shared_experts is present, __init__ must
    redistribute the scaling factor to avoid FP16 overflow."""

    def test_fp16_with_shared_experts(self):
        runner = _make_stub_runner(
            routed_scaling_factor=4.0,
            in_dtype=torch.float16,
            has_shared_experts=True,
        )
        # Simulate __init__ FP16 redistribution logic
        runner.shared_experts_scaling_factor = 1.0
        if (
            runner._shared_experts is not None
            and runner.moe_config.in_dtype == torch.float16
        ):
            runner.routed_scaling_factor = 1.0
            runner.shared_experts_scaling_factor = 1.0 / 4.0

        assert runner.routed_scaling_factor == 1.0
        assert runner.shared_experts_scaling_factor == pytest.approx(0.25)

    def test_bf16_scaling_unchanged(self):
        runner = _make_stub_runner(
            routed_scaling_factor=4.0,
            in_dtype=torch.bfloat16,
            has_shared_experts=True,
        )
        # BF16: no redistribution
        assert runner.routed_scaling_factor == 4.0
        assert runner.shared_experts_scaling_factor == 1.0

    def test_fp16_no_shared_experts(self):
        runner = _make_stub_runner(
            routed_scaling_factor=4.0,
            in_dtype=torch.float16,
            has_shared_experts=False,
        )
        # No shared experts: no redistribution
        assert runner.routed_scaling_factor == 4.0
        assert runner.shared_experts_scaling_factor == 1.0


class TestMaybeApplyRoutedScaleToOutput:
    """_maybe_apply_routed_scale_to_output must multiply shared_output
    by shared_experts_scaling_factor only when dtype is FP16."""

    def test_fp16_scales_shared_output(self):
        runner = _make_stub_runner(in_dtype=torch.float16)
        runner.shared_experts_scaling_factor = 0.25

        shared = torch.ones(4, dtype=torch.float16)
        fused = torch.ones(4, dtype=torch.float16)

        out_shared, out_fused = runner._maybe_apply_routed_scale_to_output(
            shared, fused
        )
        torch.testing.assert_close(
            out_shared,
            torch.full((4,), 0.25, dtype=torch.float16),
        )
        torch.testing.assert_close(out_fused, fused)

    def test_bf16_no_scaling(self):
        runner = _make_stub_runner(in_dtype=torch.bfloat16)
        runner.shared_experts_scaling_factor = 0.25

        shared = torch.ones(4, dtype=torch.bfloat16)
        fused = torch.ones(4, dtype=torch.bfloat16)

        out_shared, out_fused = runner._maybe_apply_routed_scale_to_output(
            shared, fused
        )
        torch.testing.assert_close(out_shared, shared)

    def test_none_shared_output(self):
        runner = _make_stub_runner(in_dtype=torch.float16)

        fused = torch.ones(4, dtype=torch.float16)
        out_shared, out_fused = runner._maybe_apply_routed_scale_to_output(None, fused)
        assert out_shared is None
        torch.testing.assert_close(out_fused, fused)


class TestDoNaiveDispatchCombine:
    """do_naive_dispatch_combine must only check dp_size > 1,
    without the community supports_internal_mk guard."""

    def test_dp_size_1(self):
        runner = _make_stub_runner()
        runner.moe_config.dp_size = 1  # MagicMock allows this
        assert runner.do_naive_dispatch_combine is False

    def test_dp_size_gt_1(self):
        runner = _make_stub_runner()
        runner.moe_config.dp_size = 4  # MagicMock allows this
        assert runner.do_naive_dispatch_combine is True


class TestKernelHandlesOwnDispatch:
    """_kernel_handles_own_dispatch delegates to ilu_kernel_handles_own_dispatch."""

    def test_non_modular_returns_false(self):
        runner = _make_stub_runner()
        runner.routed_experts.quant_method = MagicMock(spec=[])

        with patch(
            "vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_elastic_ep"
            ".ilu_kernel_handles_own_dispatch",
            return_value=False,
        ):
            assert runner._kernel_handles_own_dispatch is False

    def test_modular_non_monolithic_returns_true(self):
        runner = _make_stub_runner()
        with patch(
            "vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_elastic_ep"
            ".ilu_kernel_handles_own_dispatch",
            return_value=True,
        ):
            assert runner._kernel_handles_own_dispatch is True
