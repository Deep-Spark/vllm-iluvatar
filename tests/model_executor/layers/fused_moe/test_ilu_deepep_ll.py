# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe import FusedMoEActivationFormat
from vllm.model_executor.layers.fused_moe.config import (
    int8_w8a8_moe_quant_config,
    mxfp4_w4a8_moe_quant_config,
)

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


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("num_tokens", [0, 4])
@pytest.mark.parametrize(
    "dtype,packed",
    [(torch.bfloat16, False), (torch.float16, False), (torch.bfloat16, True)],
)
def test_glm_ll_shared_expert_overlaps_dispatch_when_enabled(
    monkeypatch, enabled, tp_size, num_tokens, dtype, packed
):
    from vllm_iluvatar.model_executor.layers.fused_moe import ilu_fused_moe_ep as ep

    events = []
    x = torch.zeros((num_tokens, 8), dtype=dtype)
    shared_input = (x.to(torch.int8), torch.ones(num_tokens), dtype) if packed else x
    main_stream = SimpleNamespace(
        wait_stream=lambda other: events.append("main_wait_shared")
    )
    shared_stream = SimpleNamespace(
        wait_stream=lambda other: events.append("shared_wait_main")
    )
    monkeypatch.setattr(ep, "current_stream", lambda: main_stream)
    monkeypatch.setattr(ep, "aux_stream", lambda: shared_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda self, stream: events.append(
            "record_shared_input" if stream is shared_stream else "record_shared_output"
        ),
    )
    monkeypatch.setattr(ep, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(ep, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        ep,
        "get_tp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(ixfd_group=object())
        ),
    )

    def all_gather(gathered, local, **kwargs):
        events.append("all_gather")
        gathered[:2].copy_(local)
        gathered[2:].copy_(local)

    monkeypatch.setattr(ep.ixfd, "all_gather_into_tensor", all_gather)

    def shared_layer(inp):
        events.append("shared")
        assert inp is shared_input
        return torch.full_like(x, 5)

    def prepare(hidden_states, *args, **kwargs):
        events.append("dispatch")

        def receiver():
            events.append("dispatch_receiver")
            return hidden_states, None, None, None, None

        return lambda: events.append("dispatch_hook"), receiver

    def routed_apply(**kwargs):
        events.append("routed")
        kwargs["output"].fill_(2)

    def finalize(output, *args):
        events.append("finalize")
        return lambda: output.fill_(2)

    routed = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        activation=None,
    )
    fused = SimpleNamespace(
        quant_config=None,
        expects_unquantized_inputs=False,
        a2_scale=None,
        workspace_dtype=lambda dtype: dtype,
        apply=routed_apply,
        finalize_weight_and_reduce_impl=lambda: None,
    )
    runner = SimpleNamespace(
        routed_experts=routed,
        shared_experts=SimpleNamespace(_layer=shared_layer),
        moe_config=SimpleNamespace(in_dtype=x.dtype),
        shared_experts_scaling_factor=0.4,
        _fused_output_is_reduced=False,
    )
    model = SimpleNamespace(
        experts=runner,
        _ll_shared_dispatch_overlap_enabled=enabled,
        _ll_expert_output_buffer=lambda a, dtype: torch.empty_like(a),
    )
    pf = SimpleNamespace(
        activation_format=FusedMoEActivationFormat.BatchedExperts,
        supports_async=lambda: True,
        prepare_async=prepare,
        finalize_async=finalize,
    )

    result = ep.IluFusedMoEEP._forward_dp_ep_flatten(
        model,
        x,
        torch.ones((num_tokens, 1)),
        torch.zeros((num_tokens, 1), dtype=torch.int32),
        x,
        shared_gate_up_input=shared_input if packed else None,
        enable_fused_ag_ar=False,
        moe_kernel=SimpleNamespace(prepare_finalize=pf, fused_experts=fused),
    )
    expected = 7 if dtype == torch.bfloat16 else 4
    torch.testing.assert_close(result, torch.full_like(x, expected))
    if enabled and num_tokens > 0:
        expected_events = [
            "shared_wait_main",
            *(["record_shared_input"] * (2 if packed else 1)),
            "shared",
            "dispatch",
            "dispatch_hook",
            "dispatch_receiver",
            "main_wait_shared",
            "record_shared_output",
            "routed",
            "finalize",
        ]
    else:
        expected_events = [
            "dispatch",
            "dispatch_hook",
            "shared",
            "dispatch_receiver",
            "routed",
            "finalize",
        ]
    if tp_size > 1 and num_tokens > 0:
        expected_events.append("all_gather")
    assert events == expected_events


@pytest.mark.parametrize("enabled", [False, True])
def test_glm_ll_dispatch_overlap_preserves_fused_shared_reduction(monkeypatch, enabled):
    from ixformer.distributed import nvsm_ep

    from vllm_iluvatar.model_executor.layers.fused_moe import ilu_fused_moe_ep as ep

    events = []
    x = torch.zeros((4, 8), dtype=torch.bfloat16)
    main_stream = SimpleNamespace(
        wait_stream=lambda other: events.append("main_wait_shared")
    )
    shared_stream = SimpleNamespace(
        wait_stream=lambda other: events.append("shared_wait_main")
    )
    monkeypatch.setattr(ep, "current_stream", lambda: main_stream)
    monkeypatch.setattr(ep, "aux_stream", lambda: shared_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda self, stream: events.append(
            "record_shared_input" if self is x else "record_shared_output"
        ),
    )
    monkeypatch.setattr(ep, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(ep, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(ep, "VLLM_ILU_LL_FUSED_AG_AR", True)

    def reduce_shared(shared):
        events.append("tp_all_reduce")
        return shared * 2

    monkeypatch.setattr(
        ep, "get_tp_group", lambda: SimpleNamespace(all_reduce=reduce_shared)
    )

    def fused_combine(**kwargs):
        events.append(
            "fused_routed_only" if kwargs["shared_x"] is None else "fused_all"
        )
        routed = torch.full_like(x, 2)
        if kwargs["shared_x"] is not None:
            routed += kwargs["shared_x"] * 2
        return routed

    monkeypatch.setattr(nvsm_ep, "nvsm_low_latency_fused_combine_ag_ar", fused_combine)

    def shared_layer(inp):
        events.append("shared")
        return torch.full_like(inp, 5)

    routed = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        activation=None,
    )
    fused = SimpleNamespace(
        quant_config=None,
        expects_unquantized_inputs=False,
        a2_scale=None,
        workspace_dtype=lambda dtype: dtype,
        apply=lambda **kwargs: kwargs["output"].fill_(2),
    )
    runner = SimpleNamespace(
        routed_experts=routed,
        shared_experts=SimpleNamespace(_layer=shared_layer),
        moe_config=SimpleNamespace(in_dtype=x.dtype),
        shared_experts_scaling_factor=1.0,
    )
    model = SimpleNamespace(
        experts=runner,
        _ll_shared_dispatch_overlap_enabled=enabled,
        _ll_expert_output_buffer=lambda a, dtype: torch.empty_like(a),
    )
    pf = SimpleNamespace(
        activation_format=FusedMoEActivationFormat.BatchedExperts,
        supports_async=lambda: True,
        prepare_async=lambda local, *args, **kwargs: lambda: (
            local,
            None,
            None,
            None,
            None,
        ),
        buffer=object(),
    )
    result = ep.IluFusedMoEEP._forward_dp_ep_flatten(
        model,
        x,
        torch.ones((4, 1)),
        torch.zeros((4, 1), dtype=torch.int32),
        x,
        moe_kernel=SimpleNamespace(prepare_finalize=pf, fused_experts=fused),
    )
    torch.testing.assert_close(result, torch.full_like(x, 12))
    if enabled:
        assert events == [
            "shared_wait_main",
            "record_shared_input",
            "shared",
            "main_wait_shared",
            "record_shared_output",
            "fused_all",
        ]
    else:
        assert events == ["shared", "fused_all"]


@pytest.mark.parametrize("use_fused", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_glm_ll_shared_dispatch_overlap_cuda_graph_replay(monkeypatch, use_fused):
    from vllm_iluvatar.model_executor.layers.fused_moe import ilu_fused_moe_ep as ep

    x = torch.zeros((4, 8), device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(ep, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(ep, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        ep,
        "get_tp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(ixfd_group=object()),
            all_reduce=lambda shared: shared * 2,
        ),
    )
    monkeypatch.setattr(ep, "VLLM_ILU_LL_FUSED_AG_AR", True)
    shared_stream = torch.cuda.Stream()
    monkeypatch.setattr(ep, "aux_stream", lambda: shared_stream)

    from ixformer.distributed import nvsm_ep

    def fused_combine(**kwargs):
        assert kwargs["shared_x"] is not None
        return torch.full_like(x, 2) + kwargs["shared_x"] * 2

    monkeypatch.setattr(nvsm_ep, "nvsm_low_latency_fused_combine_ag_ar", fused_combine)

    def all_gather(gathered, local, **kwargs):
        gathered[:2].copy_(local)
        gathered[2:].copy_(local)

    monkeypatch.setattr(ep.ixfd, "all_gather_into_tensor", all_gather)
    routed = SimpleNamespace(
        w13_weight=None,
        w2_weight=None,
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        activation=None,
    )
    fused = SimpleNamespace(
        quant_config=None,
        expects_unquantized_inputs=False,
        a2_scale=None,
        workspace_dtype=lambda dtype: dtype,
        apply=lambda **kwargs: kwargs["output"].fill_(2),
        finalize_weight_and_reduce_impl=lambda: None,
    )
    runner = SimpleNamespace(
        routed_experts=routed,
        shared_experts=SimpleNamespace(_layer=lambda inp: inp + 5),
        moe_config=SimpleNamespace(in_dtype=x.dtype),
        _fused_output_is_reduced=False,
    )
    model = SimpleNamespace(
        experts=runner,
        _ll_shared_dispatch_overlap_enabled=True,
        _ll_expert_output_buffer=lambda a, dtype: torch.empty_like(a),
    )
    pf = SimpleNamespace(
        activation_format=FusedMoEActivationFormat.BatchedExperts,
        supports_async=lambda: True,
        prepare_async=lambda local, *args, **kwargs: lambda: (
            local,
            None,
            None,
            None,
            None,
        ),
        finalize_async=lambda output, *args: lambda: output.fill_(2),
        buffer=object(),
    )
    moe_kernel = SimpleNamespace(prepare_finalize=pf, fused_experts=fused)

    def forward():
        return ep.IluFusedMoEEP._forward_dp_ep_flatten(
            model,
            x,
            torch.ones((4, 1), device=x.device),
            torch.zeros((4, 1), dtype=torch.int32, device=x.device),
            x,
            enable_fused_ag_ar=use_fused,
            moe_kernel=moe_kernel,
        )

    forward()  # Initialize the stream and allocator before capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = forward()
    graph.replay()
    torch.testing.assert_close(result, torch.full_like(x, 12 if use_fused else 7))
    x.fill_(3)
    graph.replay()
    torch.testing.assert_close(result, torch.full_like(x, 18 if use_fused else 10))


@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("1", True)])
def test_glm_ll_shared_dispatch_overlap_env(monkeypatch, value, expected):
    from vllm_iluvatar.envs import _ILU_CANONICAL_ONLY_ENV_VARS

    name = "VLLM_ILU_GLM_LL_SHARED_DISPATCH_OVERLAP"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert _ILU_CANONICAL_ONLY_ENV_VARS[name]() is expected


@pytest.mark.parametrize("value,expected", [(None, True), ("0", False), ("1", True)])
def test_ll_shared_dispatch_overlap_env(monkeypatch, value, expected):
    from vllm_iluvatar.envs import _ILU_CANONICAL_ONLY_ENV_VARS

    name = "VLLM_ILU_V41_LL_SHARED_DISPATCH_OVERLAP"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert _ILU_CANONICAL_ONLY_ENV_VARS[name]() is expected


def test_moe_enables_ll_shared_dispatch_overlap_by_default(monkeypatch):
    from vllm_iluvatar.models.deepseek_v41.model import (
        _ll_shared_dispatch_overlap_enabled,
    )

    monkeypatch.delenv("VLLM_ILU_V41_LL_SHARED_DISPATCH_OVERLAP", raising=False)
    assert _ll_shared_dispatch_overlap_enabled() is True
    monkeypatch.setenv("VLLM_ILU_V41_LL_SHARED_DISPATCH_OVERLAP", "0")
    assert _ll_shared_dispatch_overlap_enabled() is False
    monkeypatch.setenv("VLLM_ILU_V41_LL_SHARED_DISPATCH_OVERLAP", "1")
    assert _ll_shared_dispatch_overlap_enabled() is True
