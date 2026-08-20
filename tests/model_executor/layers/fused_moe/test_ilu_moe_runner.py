# SPDX-License-Identifier: Apache-2.0
"""Tests for IluMoERunner.

CPU-only tests use mocks to verify override behaviour without constructing
a full MoE layer stack. GPU-only tests (marked ``requires_cuda``) would
exercise the numerical forward path — those are deferred to P2 and can be
added when the full environment is available.
"""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _moe_forward_shared,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod as CommunityUnquantizedFusedMoEMethod,
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


class UnquantizedFusedMoEMethod(CommunityUnquantizedFusedMoEMethod):
    def __init__(self):
        pass


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

    def test_init_sets_runner_aligned_routed_experts_scale(self):
        routed = MagicMock()
        moe_config = MagicMock()
        moe_config.in_dtype = torch.bfloat16

        def fake_super_init(self, *args, routed_scaling_factor, **kwargs):
            self.routed_scaling_factor = routed_scaling_factor
            self.routed_experts = routed
            self.routed_experts.routed_scaling_factor = 9.0
            self._shared_experts = MagicMock()
            self.moe_config = moe_config

        vllm_config = MagicMock()
        vllm_config.parallel_config.enable_elastic_ep = False
        with patch.object(MoERunner, "__init__", fake_super_init), patch(
            "vllm_iluvatar.model_executor.layers.fused_moe.runner"
            ".ilu_moe_runner.get_current_vllm_config",
            return_value=vllm_config,
        ):
            runner = IluMoERunner(routed_scaling_factor=4.0)

        assert runner.routed_scaling_factor == 4.0
        assert runner.shared_experts_scaling_factor == 1.0
        assert routed.routed_scaling_factor == 9.0
        assert routed._ilu_output_scale == 4.0

    def test_init_fp16_shared_resets_routed_experts_scale(self):
        routed = MagicMock()
        moe_config = MagicMock()
        moe_config.in_dtype = torch.float16

        def fake_super_init(self, *args, routed_scaling_factor, **kwargs):
            self.routed_scaling_factor = routed_scaling_factor
            self.routed_experts = routed
            self.routed_experts.routed_scaling_factor = 9.0
            self._shared_experts = MagicMock()
            self.moe_config = moe_config

        vllm_config = MagicMock()
        vllm_config.parallel_config.enable_elastic_ep = False
        with patch.object(MoERunner, "__init__", fake_super_init), patch(
            "vllm_iluvatar.model_executor.layers.fused_moe.runner"
            ".ilu_moe_runner.get_current_vllm_config",
            return_value=vllm_config,
        ):
            runner = IluMoERunner(routed_scaling_factor=4.0)

        assert runner.routed_scaling_factor == 1.0
        assert runner.shared_experts_scaling_factor == pytest.approx(0.25)
        assert routed.routed_scaling_factor == 9.0
        assert routed._ilu_output_scale == 1.0

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

    def test_unquantized_bf16_scales_fused_output_with_runner_scale(self):
        runner = _make_stub_runner(in_dtype=torch.bfloat16)
        runner.routed_experts.quant_method = UnquantizedFusedMoEMethod()
        runner.routed_experts._ilu_output_scale = 3.0
        runner.shared_experts_scaling_factor = 0.5
        shared = torch.ones(4, dtype=torch.bfloat16)
        fused = torch.ones(4, dtype=torch.bfloat16)

        out_shared, out_fused = runner._maybe_apply_routed_scale_to_output(
            shared, fused
        )

        torch.testing.assert_close(out_shared, shared)
        torch.testing.assert_close(
            out_fused,
            torch.full((4,), 3.0, dtype=torch.bfloat16),
        )

    def test_fp16_shared_output_uses_init_reset_unquantized_scale(self):
        runner = _make_stub_runner(in_dtype=torch.float16)
        runner.routed_experts.quant_method = UnquantizedFusedMoEMethod()
        runner.routed_experts._ilu_output_scale = 1.0
        runner.shared_experts_scaling_factor = 0.5
        shared = torch.ones(4, dtype=torch.float16)
        fused = torch.ones(4, dtype=torch.float16)

        out_shared, out_fused = runner._maybe_apply_routed_scale_to_output(
            shared, fused
        )

        torch.testing.assert_close(
            out_shared,
            torch.full((4,), 0.5, dtype=torch.float16),
        )
        torch.testing.assert_close(out_fused, fused)

    def test_none_shared_output(self):
        runner = _make_stub_runner(in_dtype=torch.float16)

        fused = torch.ones(4, dtype=torch.float16)
        out_shared, out_fused = runner._maybe_apply_routed_scale_to_output(None, fused)
        assert out_shared is None
        torch.testing.assert_close(out_fused, fused)


class TestApplyQuantMethod:
    def test_mk_internal_overlap_runs_and_returns_shared_experts(self):
        runner = _make_stub_runner()
        shared_output = torch.full((2, 3), 2.0)
        fused_output = torch.full((2, 3), 3.0)
        shared_input = torch.full((2, 3), 1.0)
        topk_weights = torch.full((2, 1), 0.5)
        topk_ids = torch.zeros((2, 1), dtype=torch.int32)

        runner._shared_experts._determine_shared_experts_order.return_value = (
            SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
        )
        runner._shared_experts.output = shared_output
        runner._maybe_apply_shared_experts = MagicMock()
        runner.router = MagicMock()
        runner.router.select_experts.return_value = (topk_weights, topk_ids)
        runner.routed_experts.quant_method.is_monolithic = False
        runner.routed_experts.forward_modular.return_value = fused_output

        returned_shared, returned_fused = runner._apply_quant_method(
            hidden_states=shared_input,
            router_logits=torch.zeros((2, 4)),
            shared_experts_input=shared_input,
        )

        call_kwargs = runner.routed_experts.forward_modular.call_args.kwargs
        assert call_kwargs["shared_experts"] is runner._shared_experts
        assert call_kwargs["shared_experts_input"] is shared_input
        assert returned_shared is shared_output
        assert returned_fused is fused_output


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


class TestForward:
    def test_non_tuple_hidden_states_delegate_to_community_forward(self):
        runner = _make_stub_runner()
        hidden_states = torch.ones((2, 3))
        router_logits = torch.ones((2, 4))
        input_ids = torch.arange(2)
        expected = torch.full((2, 3), 5.0)

        with patch.object(MoERunner, "forward", return_value=expected) as forward:
            result = runner.forward(hidden_states, router_logits, input_ids)

        assert result is expected
        forward.assert_called_once_with(hidden_states, router_logits, input_ids)


class TestForwardImpl:
    def test_fused_gate_uses_combined_gate_weight(self):
        runner = _make_stub_runner(has_shared_experts=False)
        runner.gate = MagicMock(return_value=(torch.full((2, 3), -1.0), None))
        runner._fse_fuse_gate = True
        runner._combined_gate_weight = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )
        runner._maybe_fuse_gate_weights = MagicMock()
        runner._sequence_parallel_context = nullcontext

        hidden_states = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        original_router_logits = torch.zeros(2, 2)
        expected_router_logits = torch.tensor(
            [
                [1.0, 4.0],
                [4.0, 10.0],
            ]
        )

        runner._maybe_sync_shared_experts_stream = MagicMock()
        runner._maybe_dispatch = MagicMock(
            side_effect=lambda hidden, logits, input_ids: (
                hidden,
                logits,
                input_ids,
            )
        )
        runner._apply_quant_method = MagicMock(
            return_value=(None, torch.full((2, 3), 7.0))
        )
        runner._maybe_combine = MagicMock(return_value=torch.full((2, 3), 7.0))

        result = runner._forward_impl(
            hidden_states=hidden_states,
            router_logits=original_router_logits,
            shared_experts_input=None,
            input_ids=None,
        )

        runner._maybe_fuse_gate_weights.assert_called_once_with()
        runner.gate.assert_not_called()
        torch.testing.assert_close(
            runner._apply_quant_method.call_args.kwargs["router_logits"],
            expected_router_logits,
        )
        torch.testing.assert_close(result, torch.full((2, 3), 7.0))
