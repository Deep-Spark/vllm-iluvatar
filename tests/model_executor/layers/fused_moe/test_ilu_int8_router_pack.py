# SPDX-License-Identifier: Apache-2.0
"""Native packet/expand integration and safe MoERunner dispatch selection."""

from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import ixformer.inference.functions as ops
import pytest
import torch
from conftest import requires_corex30

from tests.model_executor.layers.fused_moe.test_ilu_moe_runner import _make_stub_runner
from tests.model_executor.layers.quantization.moe_test_utils import make_nn_graph_method
from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_int8_router_pack import (
    pack_router_input,
    unpack_router_input,
)
from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_moe_runner import (
    IluMoERunner,
)


@pytest.mark.parametrize(
    "unsupported",
    [
        None,
        "tp",
        "attention_tp",
        "pcp",
        "elastic",
        "gate_policy",
        "router_dtype",
        "custom_router",
        "quant_method",
        "transform",
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_packed_dispatch_capability(unsupported, enabled, monkeypatch):
    monkeypatch.setenv("VLLM_ILU_V41_PACKED_INT8_DISPATCH", str(int(enabled)))
    runner = _make_stub_runner()
    runner.moe_config.moe_parallel_config.use_ep = True
    runner.moe_config.ep_size = runner.moe_config.dp_size = 2
    runner.moe_config.tp_size = runner.moe_config.pcp_size = 1
    runner.moe_config.is_sequence_parallel = False
    runner._ilu_elastic_ep = runner._fse_fuse_gate = False
    runner.gate = SimpleNamespace(out_dtype=torch.float32, gather_before_gate=False)
    runner.routed_experts.custom_routing_function = None
    runner.routed_experts.quant_method.supports_packed_int8_input = True
    runner.routed_input_transform = None
    if unsupported == "tp":
        runner.moe_config.tp_size = 2
    elif unsupported == "attention_tp":
        runner.moe_config.ep_size = 4
    elif unsupported == "pcp":
        runner.moe_config.pcp_size = 2
    elif unsupported == "elastic":
        runner._ilu_elastic_ep = True
    elif unsupported == "gate_policy":
        runner.gate.gather_before_gate = True
    elif unsupported == "router_dtype":
        runner.gate.out_dtype = torch.bfloat16
    elif unsupported == "custom_router":
        runner.routed_experts.custom_routing_function = lambda x: x
    elif unsupported == "quant_method":
        runner.routed_experts.quant_method.supports_packed_int8_input = False
    elif unsupported == "transform":
        runner.routed_input_transform = lambda x: x
    with patch.object(
        IluMoERunner,
        "do_naive_dispatch_combine",
        new_callable=PropertyMock,
        return_value=True,
    ):
        assert runner._can_pack_int8_dispatch() == (enabled and unsupported is None)


@requires_corex30
@pytest.mark.parametrize("ids_dtype", [None, torch.int32, torch.int64])
def test_packet_expand_roundtrip(ids_dtype):
    rows, hidden, experts = 17, 256, 8
    torch.manual_seed(43)
    q = torch.randint(-128, 128, (rows, hidden), device="cuda", dtype=torch.int8)
    scale = torch.randn(rows, 1, device="cuda")
    logits = torch.randn(rows, experts, device="cuda")
    ids = (
        None
        if ids_dtype is None
        else torch.arange(rows, device="cuda", dtype=ids_dtype) - 7
    )
    packet, layout = pack_router_input(q, scale, logits, ids)
    (packed, packed_scale, dtype), out_logits, out_ids = unpack_router_input(
        packet, layout, torch.bfloat16
    )
    assert dtype == torch.bfloat16
    assert torch.equal(logits, out_logits)
    assert out_ids is None if ids is None else torch.equal(out_ids, ids)
    mapping = torch.arange(rows * 2, device="cuda", dtype=torch.int32)
    output = torch.empty((rows * 2, hidden), device="cuda", dtype=torch.int8)
    output_scale = torch.empty(rows * 2, device="cuda")
    ops.moe_expand_input_ep(
        hidden_states=packed,
        src_to_expert_row=mapping,
        expand_tokens=rows * 2,
        topk=2,
        output=output,
        output_scale=output_scale,
        input_scale=None if isinstance(packed_scale, int) else packed_scale,
        packed_scale_offset=packed_scale if isinstance(packed_scale, int) else None,
    )
    # Each token contributes one row per top-k slot.
    torch.testing.assert_close(output, q.repeat_interleave(2, dim=0), atol=0, rtol=0)
    torch.testing.assert_close(
        output_scale, scale.flatten().repeat_interleave(2), atol=0, rtol=0
    )


@requires_corex30
@pytest.mark.parametrize("limit", [None, 7.0])
def test_packed_nn_moe_graph_and_changing_routes(limit, monkeypatch):
    method, layer, _, _ = make_nn_graph_method()
    layer.swiglu_limit = limit
    torch.manual_seed(43)
    x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
    q, scale = ops.scaled_int8_quant(x)
    logits = torch.randn(16, 8, device="cuda")
    token_ids = torch.arange(16, device="cuda", dtype=torch.int32)
    ids = torch.randint(4, 8, (16, 2), device="cuda", dtype=torch.int32)
    weights = torch.rand(16, 2, device="cuda")

    def call():
        packet, layout = pack_router_input(q, scale, logits, token_ids)
        packed, _, _ = unpack_router_input(packet, layout, x.dtype)
        return method._apply_group_gemm(layer, packed, weights, ids, None)

    def reference():
        return method._apply_group_gemm(layer, (q, scale, x.dtype), weights, ids, None)

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    q.neg_()
    scale.mul_(0.5)
    ids[:, 0] = 0  # Drop half the routes from this rank without changing capacity.
    graph.replay()
    torch.testing.assert_close(captured, reference(), atol=0, rtol=0)
    assert torch.isfinite(captured).all()
    ids.zero_()  # No local expert rows remain.
    graph.replay()
    assert torch.count_nonzero(captured) == 0


@pytest.mark.parametrize("overlap", [False, True])
def test_runner_reuses_norm_quant_and_gathers_one_packet(overlap, monkeypatch):
    monkeypatch.setenv("VLLM_ILU_V41_PACKED_INT8_DISPATCH", "1")
    from contextlib import nullcontext
    from functools import partial

    from vllm_iluvatar.model_executor.layers.fused_moe.runner import (
        ilu_moe_runner as mod,
    )

    runner = _make_stub_runner()
    runner.moe_config.moe_parallel_config.use_ep = True
    runner.moe_config.moe_parallel_config.all2all_backend = "allgather_reducescatter"
    runner.moe_config.ep_size = runner.moe_config.dp_size = 2
    runner.moe_config.tp_size = runner.moe_config.pcp_size = 1
    runner.moe_config.is_sequence_parallel = False
    runner._ilu_elastic_ep = runner._fse_fuse_gate = False
    runner._ag_shared_overlap_enabled = overlap
    runner._sequence_parallel_context = nullcontext
    runner.routed_input_transform = None
    runner.routed_experts.custom_routing_function = None
    runner.routed_experts.quant_method.is_monolithic = False
    runner.routed_experts.quant_method.supports_prequantized_input = True
    runner.routed_experts.quant_method.supports_packed_int8_input = True
    runner._shared_can_reuse_routed_quant = lambda: True
    x = torch.ones(3, 256, dtype=torch.bfloat16)
    q = torch.full_like(x, 7, dtype=torch.int8)
    scale = torch.ones(3, 1) / 7
    ids = torch.arange(3, dtype=torch.int32)
    events = []

    def gate(value):
        assert value is x
        events.append("gate")
        return torch.ones(3, 8), None

    gate.out_dtype = torch.float32
    gate.gather_before_gate = False
    runner.gate = gate

    def gather(packet, *, dim, sizes):
        assert packet.dtype == torch.uint8 and sizes == [3, 3]
        events.append("gather")
        return packet.repeat(2, 1)

    group = SimpleNamespace(world_size=2, rank_in_group=0, all_gatherv=gather)
    ctx = SimpleNamespace(
        is_padding=None,
        dp_metadata=SimpleNamespace(
            all_ranks_decode_only=True, get_chunk_sizes_across_dp_rank=lambda: [3, 3]
        ),
    )
    monkeypatch.setattr(mod, "get_dp_group", lambda: group)
    monkeypatch.setattr(mod, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(
        ops, "pack_router_packet", partial(ops.pack_router_packet, backend="torch")
    )
    monkeypatch.setattr(
        ops,
        "unpack_router_packet_metadata",
        partial(ops.unpack_router_packet_metadata, backend="torch"),
    )

    def requantize(*args, **kwargs):
        raise AssertionError("norm INT8 output was quantized again")

    monkeypatch.setattr(ops, "scaled_int8_quant", requantize)

    def begin(value, *, overlap):
        assert value[0] is q and overlap == runner._ag_shared_overlap_enabled
        events.append("shared")

    def end(stream):
        events.append("join")
        return x

    runner._shared_experts.begin_dispatch.side_effect = begin
    runner._shared_experts.end_dispatch.side_effect = end

    def apply(**kwargs):
        events.append("experts")
        packed, offset, dtype = kwargs["hidden_states"]
        assert offset == 256 and dtype == x.dtype
        assert torch.equal(packed[:, :256], q.repeat(2, 1))
        assert torch.equal(kwargs["input_ids"], ids.repeat(2))
        assert kwargs["router_logits"].shape == (6, 8)
        return kwargs["shared_output"], x

    runner._apply_quant_method = apply
    runner._maybe_combine = lambda shared, routed: shared + routed
    with patch.object(
        IluMoERunner,
        "do_naive_dispatch_combine",
        new_callable=PropertyMock,
        return_value=True,
    ):
        actual = runner._forward_impl(x, None, (q, scale, x.dtype), ids)
    assert torch.equal(actual, x * 2)
    assert events == ["gate", "shared", "gather", "join", "experts"]
