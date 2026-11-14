# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed import communication_op
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.layers.fused_moe.config import (
    int8_w8a8_moe_quant_config,
    mxfp4_w4a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_iluvatar.model_executor.layers.fused_moe import ilu_fused_moe_ep as ep_impl
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
    IluFusedMoEEP,
)
from vllm_iluvatar.model_executor.layers.fused_moe.prepare_finalize import (
    ilu_deepep_ll,
)


def test_nvsm_dispatch_quant_kwargs_follow_moe_quant_config(monkeypatch):
    monkeypatch.setattr(
        ilu_deepep_ll.current_platform,
        "fp8_dtype",
        lambda: torch.float8_e4m3fn,
    )

    int8_config = int8_w8a8_moe_quant_config(
        w1_scale=torch.ones(1),
        w2_scale=torch.ones(1),
        a1_scale=None,
        a2_scale=None,
        per_act_token_quant=True,
    )
    fp4_fp8_config = mxfp4_w4a8_moe_quant_config(
        w1_scale=torch.ones(1),
        w2_scale=torch.ones(1),
    )

    assert ilu_deepep_ll.nvsm_dispatch_quant_kwargs(int8_config) == {
        "use_int8": True,
        "use_fp8": False,
    }
    assert ilu_deepep_ll.nvsm_dispatch_quant_kwargs(fp4_fp8_config) == {
        "use_int8": False,
        "use_fp8": True,
        "fp8_format": torch.float8_e4m3fn,
        "fp8_group_size": None,
    }

    ep = IluFusedMoEEP()
    ep.experts = SimpleNamespace(
        _quant_method=SimpleNamespace(moe_quant_config=fp4_fp8_config)
    )
    assert ep._nvsm_dispatch_quant_kwargs() == {
        "use_int8": False,
        "use_fp8": True,
        "fp8_format": torch.float8_e4m3fn,
        "fp8_group_size": None,
    }

    ep.experts._quant_method.moe_quant_config = int8_config
    assert ep._nvsm_dispatch_quant_kwargs() == {
        "use_int8": True,
        "use_fp8": False,
    }


def test_nvsm_dispatch_quant_kwargs_cover_quant_dtype_families(monkeypatch):
    fp8_dtype = torch.float8_e4m3fn
    monkeypatch.setattr(
        ilu_deepep_ll.current_platform,
        "fp8_dtype",
        lambda: fp8_dtype,
    )
    cases = (
        (None, None, False, False),
        (None, "int4", True, False),
        (None, torch.int8, True, False),
        ("mxint8", "int4", True, False),
        (fp8_dtype, fp8_dtype, False, True),
        ("fp4", "fp4", False, True),
        ("fp8", "mxfp4", False, True),
        ("mxfp4", "mxfp4", False, True),
        ("mxfp8", "mxfp4", False, True),
        ("mxfp8_e4m3", "mxfp4", False, True),
        ("mxfp8_e5m2", "mxfp4", False, True),
        ("nvfp4", "nvfp4", False, True),
        (fp8_dtype, "int4", False, True),
    )

    for quant_dtype, weight_dtype, use_int8, use_fp8 in cases:
        config = SimpleNamespace(
            quant_dtype=quant_dtype,
            weight_quant_dtype=weight_dtype,
        )
        kwargs = ilu_deepep_ll.nvsm_dispatch_quant_kwargs(config)

        assert kwargs["use_int8"] is use_int8
        assert kwargs["use_fp8"] is use_fp8
        if use_fp8:
            assert kwargs["fp8_format"] == fp8_dtype
            assert kwargs["fp8_group_size"] is None
        else:
            assert "fp8_format" not in kwargs
            assert "fp8_group_size" not in kwargs


def test_unique_topk_receive_capacity_is_explicit(monkeypatch):
    import pytest

    config = SimpleNamespace(quant_dtype=torch.int8, weight_quant_dtype=None)
    monkeypatch.delenv("IXFORMER_NVSM_COMBINE_FP8", raising=False)
    monkeypatch.delenv("IXFORMER_NVSM_UNIQUE_TOPK", raising=False)
    assert "unique_topk" not in ilu_deepep_ll.nvsm_dispatch_quant_kwargs(config)
    monkeypatch.setenv("IXFORMER_NVSM_UNIQUE_TOPK", "1")
    assert ilu_deepep_ll.nvsm_dispatch_quant_kwargs(config)["unique_topk"] is True
    monkeypatch.setenv("IXFORMER_NVSM_UNIQUE_TOPK", "2")
    with pytest.raises(ValueError, match="IXFORMER_NVSM_UNIQUE_TOPK"):
        ilu_deepep_ll.nvsm_dispatch_quant_kwargs(config)


def test_nvsm_reuses_initialized_ep_transport_only_when_requested(monkeypatch):
    import pytest

    config = SimpleNamespace(quant_dtype=torch.int8, weight_quant_dtype=None)
    group = object()
    monkeypatch.setattr(
        ilu_deepep_ll, "get_ep_group", lambda: SimpleNamespace(device_group=group)
    )
    monkeypatch.delenv("IXFORMER_NVSM_REUSE_EP_COMM", raising=False)
    assert "ipc_group" not in ilu_deepep_ll.nvsm_dispatch_quant_kwargs(config)
    monkeypatch.setenv("IXFORMER_NVSM_REUSE_EP_COMM", "1")
    assert ilu_deepep_ll.nvsm_dispatch_quant_kwargs(config)["ipc_group"] is group
    monkeypatch.setenv("IXFORMER_NVSM_REUSE_EP_COMM", "invalid")
    with pytest.raises(ValueError, match="IXFORMER_NVSM_REUSE_EP_COMM"):
        ilu_deepep_ll.nvsm_dispatch_quant_kwargs(config)


class _CPUSharedRunner(SimpleNamespace):
    # Exercise the actual stale-kernel property and the public core helper.
    _fused_output_is_reduced = MoERunner._fused_output_is_reduced
    _maybe_reduce_shared_expert_output = MoERunner._maybe_reduce_shared_expert_output


def _cpu_flatten_case(
    monkeypatch,
    dtype,
    *,
    tp_size=2,
    actual_reduced=True,
    with_shared=True,
    sequence_parallel=False,
    old_reduced=False,
    fused=False,
):
    """Fake only transport/expert arithmetic; run the real flatten and helpers."""
    monkeypatch.setattr(
        ep_impl, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(ep_impl, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(ep_impl, "VLLM_ILU_DEEPEP_USE_SCATTER", False)
    monkeypatch.setattr(ep_impl, "VLLM_ILU_LL_FUSED_AG_AR", fused)
    hidden = torch.arange(24, dtype=dtype).view(6, 4)
    local_shared = hidden + 1
    peer_shared = hidden + 7
    routed = hidden + 32
    scale = 0.25
    applied_scale = scale if dtype == torch.float16 else 1.0
    events, reductions, reduced_queries = [], [], []

    def shared_layer(x):
        assert x is hidden
        events.append("shared")
        return local_shared.clone()

    def prepare_async(x, *args, **kwargs):
        events.append("prepare")

        def hook():
            events.append("prepare_hook")

        def receive():
            events.append("prepare_receive")
            return x.clone(), None, object(), None, None

        return hook, receive

    def apply(*, output, **kwargs):
        events.append("apply")
        output.copy_(routed[: len(output)])

    def finalize_async(output, fused_output, *args):
        events.append("finalize")

        def hook():
            events.append("finalize_hook")

        def receive():
            events.append("finalize_receive")
            output.copy_(fused_output)

        return hook, receive

    def output_is_reduced():
        reduced_queries.append(actual_reduced)
        return actual_reduced

    routed_experts = SimpleNamespace(
        w13_weight=torch.empty(0),
        w2_weight=torch.empty(0),
        global_num_experts=16,
        expert_map=None,
        apply_router_weight_on_input=False,
        activation="silu",
    )
    old_kernel = (
        SimpleNamespace(output_is_reduced=lambda: True) if old_reduced else None
    )
    runner = _CPUSharedRunner(
        _quant_method=SimpleNamespace(moe_kernel=old_kernel),
        routed_experts=routed_experts,
        shared_experts=SimpleNamespace(_layer=shared_layer) if with_shared else None,
        moe_config=SimpleNamespace(
            in_dtype=dtype, is_sequence_parallel=sequence_parallel
        ),
        shared_experts_scaling_factor=scale,
    )
    assert runner._fused_output_is_reduced is old_reduced
    kernel = SimpleNamespace(
        prepare_finalize=SimpleNamespace(
            supports_async=lambda: True,
            prepare_async=prepare_async,
            finalize_async=finalize_async,
            buffer=object(),
        ),
        fused_experts=SimpleNamespace(
            quant_config=object(),
            expects_unquantized_inputs=False,
            workspace_dtype=lambda d: d,
            apply=apply,
            a2_scale=None,
            finalize_weight_and_reduce_impl=lambda: object(),
        ),
        output_is_reduced=output_is_reduced,
    )

    def device_sum(x):
        # Known contribution from the other TP rank, after the FP16 seam scale.
        torch.testing.assert_close(x, local_shared * applied_scale, rtol=0, atol=0)
        reductions.append(x.clone())
        return x + peer_shared * applied_scale

    # Keep the real public all_reduce TP1 short circuit as well as the core
    # MoERunner helper; replace only the device collective for this CPU test.
    tp_group = SimpleNamespace(
        world_size=tp_size,
        use_custom_op_call=False,
        _all_reduce_out_place=device_sum,
    )
    tp_group.all_reduce = lambda x: GroupCoordinator.all_reduce(tp_group, x)
    monkeypatch.setattr(communication_op, "get_tp_group", lambda: tp_group)
    ep = IluFusedMoEEP()
    ep.experts = runner
    return SimpleNamespace(
        ep=ep,
        kernel=kernel,
        hidden=hidden,
        local_shared=local_shared,
        peer_shared=peer_shared,
        routed=routed,
        applied_scale=applied_scale,
        events=events,
        reductions=reductions,
        reduced_queries=reduced_queries,
        shared_input=hidden if with_shared else None,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tp_size,actual_reduced,with_shared,sequence_parallel,old_reduced",
    [
        (2, True, True, False, False),  # Actual LL reduced, old quant kernel None.
        (2, True, False, False, False),
        (2, False, True, False, False),
        (2, False, True, False, True),  # Old reduction flag must not force an AR.
        (1, True, True, False, False),
        (2, True, True, True, False),  # Core SP contract owns its gather elsewhere.
    ],
)
def test_flatten_shared_reduction_follows_the_actual_kernel(
    monkeypatch,
    dtype,
    tp_size,
    actual_reduced,
    with_shared,
    sequence_parallel,
    old_reduced,
):
    case = _cpu_flatten_case(
        monkeypatch,
        dtype,
        tp_size=tp_size,
        actual_reduced=actual_reduced,
        with_shared=with_shared,
        sequence_parallel=sequence_parallel,
        old_reduced=old_reduced,
    )
    output = case.ep._forward_dp_ep_flatten(
        case.hidden,
        torch.ones(6, 2),
        torch.zeros(6, 2, dtype=torch.int64),
        case.shared_input,
        moe_kernel=case.kernel,
    )
    should_reduce = (
        tp_size > 1 and actual_reduced and with_shared and not sequence_parallel
    )
    assert len(case.reductions) == int(should_reduce)
    expected = case.routed
    if with_shared:
        shared = case.local_shared * case.applied_scale
        if should_reduce:
            shared = shared + case.peer_shared * case.applied_scale
        expected = expected + shared
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    expected_events = ["prepare", "prepare_hook"]
    if with_shared:
        expected_events.append("shared")
    expected_events += [
        "prepare_receive",
        "apply",
        "finalize",
        "finalize_hook",
        "finalize_receive",
    ]
    assert case.events == expected_events


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_combine_keeps_its_early_return_and_shared_scaling(monkeypatch, dtype):
    from ixformer.distributed import nvsm_ep

    case = _cpu_flatten_case(monkeypatch, dtype, fused=True)
    sentinel = torch.full_like(case.hidden, 123)
    native_calls = []

    def fused_combine(**kwargs):
        native_calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(nvsm_ep, "nvsm_low_latency_fused_combine_ag_ar", fused_combine)
    output = case.ep._forward_dp_ep_flatten(
        case.hidden,
        torch.ones(6, 2),
        torch.zeros(6, 2, dtype=torch.int64),
        case.shared_input,
        moe_kernel=case.kernel,
    )
    assert output is sentinel and len(native_calls) == 1
    call = native_calls[0]
    assert call["tp_size"] == 2 and call["tp_num_tokens_per_rank"] == [3, 3]
    assert call["x"].shape == (3, 4)
    assert call["shared_experts_scaling_factor"] == case.applied_scale
    torch.testing.assert_close(call["shared_x"], case.local_shared, rtol=0, atol=0)
    assert not case.reductions and not case.reduced_queries
    assert case.events == [
        "prepare",
        "prepare_hook",
        "shared",
        "prepare_receive",
        "apply",
    ]
