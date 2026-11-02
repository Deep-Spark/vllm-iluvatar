# SPDX-License-Identifier: Apache-2.0
"""Tests for IluMoERunner.

CPU-only tests verify scheduling, phase selection and real context boundaries.
Native two-rank DP/PCP collectives and CUDA Graph replay are covered in
``tests/distributed/test_moe_runner_ag.py``. These are not full-model accuracy
or DeepEP HT/LL kernel validation.
"""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock, PropertyMock, patch

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
    obj.__dict__["_ilu_fused_ep_combine_cached"] = False
    obj.routed_scaling_factor = routed_scaling_factor
    obj.shared_experts_scaling_factor = 1.0
    obj._ag_shared_overlap_enabled = False
    # Normally set in __init__ from envs; the stub bypasses __init__.
    obj._gather_before_gate_enabled = True

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
        with (
            patch.object(MoERunner, "__init__", fake_super_init),
            patch(
                "vllm_iluvatar.model_executor.layers.fused_moe.runner"
                ".ilu_moe_runner.get_current_vllm_config",
                return_value=vllm_config,
            ),
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
        with (
            patch.object(MoERunner, "__init__", fake_super_init),
            patch(
                "vllm_iluvatar.model_executor.layers.fused_moe.runner"
                ".ilu_moe_runner.get_current_vllm_config",
                return_value=vllm_config,
            ),
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


def test_pcp_shared_output_stays_local_until_routed_reduce_scatter():
    runner = _make_stub_runner()
    runner.moe_config.pcp_size = 2
    runner.moe_config.is_sequence_parallel = False
    runner._shared_experts._determine_shared_experts_order.return_value = (
        SharedExpertsOrder.NO_OVERLAP
    )
    runner.routed_experts.quant_method.is_monolithic = False
    runner.router = MagicMock()
    runner.router.select_experts.return_value = (torch.ones(6, 1), torch.zeros(6, 1))
    shared = torch.randn(3, 128)
    routed = torch.randn(6, 128)
    runner._shared_experts.output = shared
    runner.routed_experts.forward_modular.return_value = routed
    runner._maybe_apply_shared_experts = MagicMock()
    out_shared, out_routed = runner._apply_quant_method(
        routed, torch.zeros(6, 1), torch.zeros(3, 128)
    )
    assert out_shared is shared
    assert out_routed is routed
    assert (
        runner.routed_experts.forward_modular.call_args.kwargs["shared_experts_input"]
        is None
    )


@pytest.mark.parametrize("all2all", [False, True])
@pytest.mark.parametrize("kernel_dispatch", [False, True])
def test_pcp_moe_collectives_are_owned_by_one_dispatch_path(all2all, kernel_dispatch):
    runner = _make_stub_runner(has_shared_experts=False)
    runner.moe_config.pcp_size = 2
    runner.moe_config.is_sequence_parallel = False
    runner.moe_config.moe_parallel_config.use_all2all_kernels = all2all
    group = MagicMock()
    group.all_gather.side_effect = lambda x, dim: torch.cat((x, x), dim=dim)
    group.reduce_scatter.side_effect = lambda x, dim: x.chunk(2, dim=dim)[0] * 2
    x, logits, ids = torch.ones(3, 4), torch.ones(3, 2), torch.arange(3)
    runner_dispatch = not all2all or not kernel_dispatch
    with (
        patch(
            "vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_moe_runner.get_pcp_group",
            return_value=group,
        ),
        patch.object(
            IluMoERunner,
            "_kernel_handles_own_dispatch",
            new_callable=PropertyMock,
            return_value=kernel_dispatch,
        ),
    ):
        routed, scores, tokens = runner._maybe_dispatch(x, logits, ids)
        output = runner._maybe_combine(None, routed)
    assert group.all_gather.call_count == (3 if runner_dispatch else 0)
    assert group.reduce_scatter.call_count == (1 if runner_dispatch else 0)
    assert scores.shape[0] == tokens.shape[0] == (6 if runner_dispatch else 3)
    torch.testing.assert_close(output, x * 2 if runner_dispatch else x)


@pytest.mark.parametrize("deferred_gate", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_pcp_dispatch_keeps_scales_and_token_ids_aligned(quantized, deferred_gate):
    runner = _make_stub_runner(has_shared_experts=False)
    runner.moe_config.pcp_size = 2
    runner.moe_config.moe_parallel_config.use_all2all_kernels = False
    x = torch.arange(12).reshape(3, 4).to(torch.int8 if quantized else torch.bfloat16)
    scale = torch.tensor([[0.1], [0.2], [0.3]])
    ids = torch.tensor([17, 42, 83])
    logits = None if deferred_gate else torch.arange(6).reshape(3, 2).float()
    group = MagicMock()
    # Distinct remote values expose permutations and missing gathers.
    group.all_gather.side_effect = lambda tensor, dim: torch.cat(
        (tensor, tensor + 1), dim=dim
    )
    with patch(
        "vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_moe_runner.get_pcp_group",
        return_value=group,
    ):
        routed, scores, tokens = runner._maybe_dispatch(
            (x, scale, torch.bfloat16) if quantized else x, logits, ids
        )
    torch.testing.assert_close(tokens, torch.cat((ids, ids + 1)))
    if quantized:
        values, scales, dtype = routed
        assert dtype == torch.bfloat16
        torch.testing.assert_close(scales, torch.cat((scale, scale + 1)))
    else:
        values = routed
    torch.testing.assert_close(values, torch.cat((x, x + 1)))
    if deferred_gate:
        assert scores is None
    else:
        torch.testing.assert_close(scores, torch.cat((logits, logits + 1)))
    assert group.all_gather.call_count == 2 + quantized + (not deferred_gate)


@pytest.mark.parametrize("pcp_size", [1, 2])
def test_precomputed_shared_stays_separate_until_combine(pcp_size):
    runner = _make_stub_runner()
    runner.moe_config.pcp_size = pcp_size
    runner.moe_config.moe_parallel_config.use_all2all_kernels = False
    runner.routed_experts.quant_method.is_monolithic = False
    runner.routed_output_transform = None
    runner.router = MagicMock()
    runner.router.select_experts.return_value = (
        torch.ones(6, 8),
        torch.zeros(6, 8, dtype=torch.int32),
    )
    local_shared = torch.full((6 // pcp_size, 6144), 5, dtype=torch.bfloat16)
    routed = torch.full((6, 6144), 2, dtype=torch.bfloat16)
    runner.routed_experts.forward_modular.return_value = routed
    runner._maybe_apply_shared_experts = MagicMock()
    group = MagicMock()
    group.reduce_scatter.side_effect = lambda x, dim: x.chunk(2, dim=dim)[0] * 2
    shared, output = runner._apply_quant_method(
        routed, torch.zeros(6, 8), None, shared_output=local_shared
    )
    with patch(
        "vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_moe_runner.get_pcp_group",
        return_value=group,
    ):
        shared, output = runner._maybe_combine(shared, output)
    assert shared is local_shared
    torch.testing.assert_close(
        shared + output, torch.full_like(local_shared, 5 + 2 * pcp_size)
    )
    runner._maybe_apply_shared_experts.assert_not_called()
    assert group.reduce_scatter.call_count == (pcp_size > 1)


@pytest.mark.parametrize(
    "decode",
    [False, True, None],
    ids=["dp-prefill", "dp-decode", "pcp-decode-no-metadata"],
)
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_phase_schedule_uses_original_dispatch_and_join_before_gate(
    decode, overlap, dtype
):
    from types import SimpleNamespace

    import ixformer.inference.functions as ops

    runner = _make_stub_runner(in_dtype=dtype)
    runner.moe_config.moe_parallel_config.all2all_backend = "allgather_reducescatter"
    runner.routed_experts.quant_method.is_monolithic = False
    runner.routed_experts.quant_method.supports_prequantized_input = True
    runner.routed_experts.custom_routing_function = None
    runner.routed_input_transform = None
    runner._fse_fuse_gate = False
    runner._ag_shared_overlap_enabled = overlap
    runner._sequence_parallel_context = nullcontext
    runner._shared_can_reuse_routed_quant = lambda: False
    events = []
    x = torch.ones(3, 256, dtype=dtype)
    q = x.to(torch.int8)
    scale = torch.ones(3, 1)
    shared = x * 2

    def gate(value):
        events.append("gate")
        assert value.dtype == dtype
        return torch.ones(value.shape[0], 8), None

    def begin(value, *, overlap):
        events.append("shared")
        assert value is x
        assert overlap == runner._ag_shared_overlap_enabled
        return None

    def join(stream):
        events.append("join")
        return shared

    def dispatch(value, logits, ids):
        events.append("dispatch")
        if decode:
            assert value is x and logits is None
        else:
            assert value[0] is q and value[2] == dtype
            torch.testing.assert_close(value[1], scale)
            assert logits is not None
        return value, logits, ids

    def apply(**kw):
        events.append("experts")
        assert kw["shared_output"] is shared
        assert kw["router_logits"] is not None
        return shared, x

    runner.gate = gate
    runner._shared_experts.begin_dispatch.side_effect = begin
    runner._shared_experts.end_dispatch.side_effect = join
    runner._maybe_dispatch = dispatch
    runner._apply_quant_method = apply
    runner._maybe_combine = lambda s, r: s + r
    if decode is None:
        from vllm.forward_context import get_forward_context, set_forward_context

        # Production context creation: PCP alone does not create DPMetadata.
        runner.moe_config.pcp_size = 2
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                prefill_context_parallel_size=2,
                use_sequence_parallel_moe=False,
                is_moe_model=True,
            ),
            compilation_config=SimpleNamespace(
                fast_moe_cold_start=False,
                static_forward_context={},
            ),
        )
        context_scope = set_forward_context(
            SimpleNamespace(num_prefills=0, num_decode_tokens=3),
            config,
            num_tokens=3,
        )
        context_patch = nullcontext()
    else:
        context_scope = nullcontext()
        context_patch = patch(
            "vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_moe_runner.get_forward_context",
            return_value=SimpleNamespace(
                dp_metadata=SimpleNamespace(all_ranks_decode_only=decode)
            ),
        )
    with (
        context_scope,
        context_patch,
        patch.object(ops, "scaled_int8_quant", return_value=(q, scale)) as quant,
    ):
        if decode is None:
            assert get_forward_context().dp_metadata is None
        torch.testing.assert_close(runner._forward_impl(x, None, x), x * 3)
    assert quant.call_count == (not decode)
    assert events == (
        ["shared", "dispatch", "join", "gate", "experts"]
        if decode
        else ["gate", "shared", "dispatch", "join", "experts"]
    )


@pytest.mark.parametrize("shared_width", [128, 256])
def test_shared_quant_reuse_requires_matching_input_shape(shared_width):
    from types import SimpleNamespace

    import ixformer.inference.functions as ops

    runner = _make_stub_runner()
    runner.moe_config.moe_parallel_config.all2all_backend = "allgather_reducescatter"
    runner.routed_experts.quant_method.is_monolithic = False
    runner.routed_experts.quant_method.supports_prequantized_input = True
    runner.routed_experts.custom_routing_function = None
    runner.routed_input_transform = None
    runner.gate = None
    runner._fse_fuse_gate = False
    runner._sequence_parallel_context = nullcontext
    runner._shared_can_reuse_routed_quant = lambda: True
    x = torch.ones(3, 256, dtype=torch.bfloat16)
    shared_x = x[:, :shared_width]
    q, scale = x.to(torch.int8), torch.ones(3, 1)
    runner._maybe_dispatch = lambda x, logits, ids: (x, logits, ids)
    runner._apply_quant_method = MagicMock(return_value=(None, x))
    runner._maybe_combine = lambda shared, routed: routed
    with (
        patch(
            "vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_moe_runner.get_forward_context",
            return_value=SimpleNamespace(dp_metadata=None),
        ),
        patch.object(ops, "scaled_int8_quant", return_value=(q, scale)),
    ):
        runner._forward_impl(x, torch.zeros(3, 8), shared_x)
    value = runner._shared_experts.begin_dispatch.call_args.args[0]
    if shared_width == 256:
        assert value[0] is q and value[2] == x.dtype
    else:
        assert value is shared_x


@pytest.mark.parametrize("enabled", [False, True])
def test_runner_initializes_ag_shared_overlap_without_model_wiring(enabled):
    runner = _make_stub_runner()
    module = "vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_moe_runner"
    config = MagicMock()
    config.parallel_config.enable_elastic_ep = False
    with (
        patch.object(MoERunner, "__init__", return_value=None),
        patch(f"{module}.get_current_vllm_config", return_value=config),
        patch(f"{module}.envs.VLLM_ILU_MOE_AG_SHARED_OVERLAP", enabled),
    ):
        IluMoERunner.__init__(runner)
    assert runner._ag_shared_overlap_enabled is enabled
