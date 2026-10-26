# SPDX-License-Identifier: Apache-2.0
"""Original TN execution: CPU dispatch checks and independent GPU numerics.

The GPU cases run on both CoreX11 and CoreX30; mock cases check only wiring.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from conftest import requires_cuda

from vllm_iluvatar.model_executor.layers.fused_moe import fused_moe as moe


@pytest.mark.parametrize("mode", ["decode", "prefill", "ep", "ep_empty"])
def test_existing_tn_dispatch_and_bias_arguments(mode):
    x = torch.ones(2, 96, dtype=torch.float16)
    w1 = torch.ones(2, 64, 96, dtype=x.dtype)
    w2 = torch.ones(2, 96, 32, dtype=x.dtype)
    weights = torch.ones(2, 1)
    ids = torch.zeros(2, 1, dtype=torch.int32)
    order = torch.arange(2, dtype=torch.int32)
    counts = torch.tensor([0, 0] if mode == "ep_empty" else [2, 0], dtype=torch.int32)
    up = torch.ones(2, 64, dtype=x.dtype)
    down = torch.ones_like(x)
    expected = torch.full_like(x, 5)
    config = SimpleNamespace(
        w1_bias=torch.ones(2, 64, dtype=x.dtype),
        w2_bias=torch.ones(2, 96, dtype=x.dtype),
    )
    metadata = SimpleNamespace(num_prefills=int(mode != "decode"), num_decodes=2)
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                moe,
                "get_forward_context",
                return_value=SimpleNamespace(attn_metadata=metadata),
            )
        )
        group = stack.enter_context(patch.object(moe, "get_ep_group"))
        group.return_value.device_group.rank.return_value = 1
        calls = {}
        returns = {
            "moe_compute_token_index_for_decode": (order, order, counts),
            "moe_compute_token_index": (order, order, counts, counts),
            "moe_compute_token_index_ep": (order, order, counts, counts, 2),
            "moe_expand_input": x,
            "silu_and_mul": torch.ones(2, 32, dtype=x.dtype),
            "moe_output_reduce_sum": expected,
        }
        for name, value in returns.items():
            calls[name] = stack.enter_context(
                patch.object(moe.ixfops, name, return_value=value)
            )
        for name in ("moe_w16a16_group_gemv", "moe_w16a16_group_gemm"):
            calls[name] = stack.enter_context(
                patch.object(moe.ixfops, name, side_effect=[up, down])
            )
        actual = moe._fused_experts_tn(
            x,
            w1,
            w2,
            weights,
            ids,
            "silu",
            global_num_experts=4,
            expert_map=torch.tensor([-1, -1, 0, 1]) if mode.startswith("ep") else None,
            quant_config=config,
        )
    if mode == "ep_empty":
        torch.testing.assert_close(actual, torch.zeros_like(x))
        calls["moe_expand_input"].assert_not_called()
        calls["moe_w16a16_group_gemm"].assert_not_called()
        return
    assert actual is expected
    selected, unused = ("gemv", "gemm") if mode == "decode" else ("gemm", "gemv")
    projections = calls[f"moe_w16a16_group_{selected}"].call_args_list
    calls[f"moe_w16a16_group_{unused}"].assert_not_called()
    assert len(projections) == 2
    for call, weight, bias in zip(
        projections, (w1, w2), (config.w1_bias, config.w2_bias)
    ):
        assert call.kwargs["weight"] is weight
        assert call.kwargs["bias"] is bias
        assert call.kwargs["format"] == "TN"
        assert call.kwargs["output_dtype"] == torch.float16
    if mode == "decode":
        calls["moe_compute_token_index_for_decode"].assert_called_once()
        calls["moe_compute_token_index"].assert_not_called()
    if mode == "ep":
        assert calls["moe_compute_token_index_ep"].call_args.kwargs == {
            "topk_ids": ids,
            "num_experts": 4,
            "start_expert_id": 2,
            "end_expert_id": 4,
        }
        assert (
            calls["moe_output_reduce_sum"].call_args.kwargs["mask"].shape == order.shape
        )


@pytest.mark.parametrize("activation", ["silu", "gelu", "swigluoai"])
def test_existing_activations_keep_their_operator(activation):
    x = torch.ones(2, 64, dtype=torch.float16)
    with patch.object(moe.ixfops, f"{activation}_and_mul") as op:
        assert moe._apply_tn_activation(x, activation, 7.0) is op.return_value
    op.assert_called_once_with(x)


@pytest.mark.parametrize("limit", [3.0, 10.0])
def test_swiglustep_passes_layer_limit(limit):
    x = torch.ones(2, 64, dtype=torch.float16)
    with patch("vllm.model_executor.layers.activation.swiglustep_and_mul_triton") as op:
        output = moe._apply_tn_activation(x, "swiglustep", limit)
    assert output.shape == (2, 32) and output.dtype == x.dtype
    assert op.call_args.args[0] is output and op.call_args.args[1] is x
    assert op.call_args.kwargs == {"limit": limit}


def _dense_reference(
    x, w1, w2, ids, weights, activation, expert_map=None, biases=None, limit=3.0
):
    # Independent per-token/per-route math; no ixformer routing or kernels.
    result = torch.zeros_like(x, dtype=torch.float32)
    for token in range(x.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            if expert_map is not None:
                expert = int(expert_map[expert])
            if expert < 0:
                continue
            gate_up = x[token].float() @ w1[expert].float().T
            if biases is not None:
                gate_up += biases.w1_bias[expert].float()
            gate_up = gate_up.to(x.dtype).float()
            if activation == "swigluoai":
                gate, up = gate_up[::2].clamp(max=7), gate_up[1::2].clamp(-7, 7)
                act = gate * torch.sigmoid(1.702 * gate) * (up + 1)
            else:
                gate, up = gate_up.chunk(2)
                if activation == "gelu":
                    gate = torch.nn.functional.gelu(gate)
                else:
                    gate = torch.nn.functional.silu(gate)
                if activation == "swiglustep":
                    gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
                act = gate * up
            projected = act.to(x.dtype).float() @ w2[expert].float().T
            if biases is not None:
                projected += biases.w2_bias[expert].float()
            result[token] += projected.to(x.dtype).float() * weights[token, route]
    return result.to(x.dtype)


@requires_cuda
@pytest.mark.parametrize(
    "dtype,activation,mode,bias",
    [
        (torch.float16, "silu", "decode", False),
        (torch.float16, "silu", "prefill", True),
        (torch.float16, "silu", "ep", True),
        (torch.bfloat16, "silu", "ep", False),
        (torch.bfloat16, "swiglustep", "prefill", False),
        (torch.float16, "gelu", "prefill", True),
        (torch.bfloat16, "swigluoai", "prefill", False),
        (torch.float16, "silu", "ep_empty", False),
    ],
)
@torch.inference_mode()
def test_tn_cuda_matches_independent_reference(dtype, activation, mode, bias):
    from ixformer.inference.backend_registry import registry

    from vllm_iluvatar.model_executor.layers.fused_moe.fused_moe import (
        ilu_w16a16_fused_experts,
    )

    # Keep the reducer on its compiled backend; never test reference fallback.
    assert registry.resolve("moe_output_reduce_sum", "ixformer_perf") == "ixformer_perf"
    torch.manual_seed(431)
    x = torch.randn(7, 128, device="cuda", dtype=dtype) / 8
    w1 = torch.randn(4, 192, 128, device="cuda", dtype=dtype) / 8
    w2 = torch.randn(4, 128, 96, device="cuda", dtype=dtype) / 8
    ids = (torch.arange(14, device="cuda").reshape(7, 2) % 4).int()
    expert_map = None
    if mode.startswith("ep"):
        expert_map = torch.tensor(
            [-1, -1, -1, -1, 0, 1, 2, 3], device="cuda", dtype=torch.int32
        )
        if mode != "ep_empty":
            ids[::2, 0] += 4
            ids[1::2] += 4
    weights = torch.tensor([[0.2, 0.8]], device="cuda").expand(7, 2).contiguous()
    biases = (
        SimpleNamespace(
            w1_bias=torch.randn(4, 192, device="cuda", dtype=dtype) / 8,
            w2_bias=torch.randn(4, 128, device="cuda", dtype=dtype) / 8,
        )
        if bias
        else None
    )
    metadata = SimpleNamespace(num_prefills=int(mode != "decode"), num_decodes=7)
    with (
        patch.object(
            moe,
            "get_forward_context",
            return_value=SimpleNamespace(attn_metadata=metadata),
        ),
        patch.object(moe, "get_ep_group") as group,
    ):
        group.return_value.device_group.rank.return_value = 1
        actual = ilu_w16a16_fused_experts(
            x,
            w1,
            w2,
            weights,
            ids,
            activation,
            expert_map=expert_map,
            global_num_experts=8 if expert_map is not None else 4,
            quant_config=biases,
            swiglu_limit=3.0,
            weight_format="TN",
        )
    torch.cuda.synchronize()
    expected = _dense_reference(x, w1, w2, ids, weights, activation, expert_map, biases)
    # Account for two rounded f16/bf16 projections and the final route reduction.
    tol = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    if mode != "ep_empty":
        unweighted = _dense_reference(
            x, w1, w2, ids, torch.ones_like(weights), activation, expert_map, biases
        )
        assert not torch.allclose(expected, unweighted, atol=tol, rtol=tol)


@requires_cuda
@torch.inference_mode()
def test_tn_decode_cuda_graph_replays_changed_routes():
    from vllm_iluvatar.model_executor.layers.fused_moe.fused_moe import (
        ilu_w16a16_fused_experts,
    )

    torch.manual_seed(433)
    x = torch.randn(7, 128, device="cuda", dtype=torch.float16) / 8
    w1 = torch.randn(4, 192, 128, device="cuda", dtype=x.dtype) / 8
    w2 = torch.randn(4, 128, 96, device="cuda", dtype=x.dtype) / 8
    ids = torch.zeros(7, 2, device="cuda", dtype=torch.int32)
    weights = torch.full((7, 2), 0.5, device="cuda")
    context = SimpleNamespace(
        attn_metadata=SimpleNamespace(num_prefills=0, num_decodes=7)
    )
    with patch.object(moe, "get_forward_context", return_value=context):

        def run():
            return ilu_w16a16_fused_experts(
                x, w1, w2, weights, ids, "silu", weight_format="TN"
            )

        for _ in range(3):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        ids[:, 0].fill_(2)
        ids[:, 1].fill_(3)
        x.add_(0.01)
        graph.replay()
        torch.cuda.synchronize()
    expected = _dense_reference(x, w1, w2, ids, weights, "silu")
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_tn_empty_tokens():
    x = torch.empty(0, 128, dtype=torch.float16)
    actual = moe._fused_experts_tn(
        x,
        torch.empty(4, 192, 128),
        torch.empty(4, 128, 96),
        torch.empty(0, 2),
        torch.empty(0, 2, dtype=torch.int32),
        "silu",
    )
    assert actual.shape == x.shape and actual.dtype == x.dtype


def test_tn_general_path_rejects_graph_capture():
    context = SimpleNamespace(
        attn_metadata=SimpleNamespace(num_prefills=1, num_decodes=0)
    )
    with (
        patch.object(moe, "get_forward_context", return_value=context),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.is_current_stream_capturing", return_value=True),
        pytest.raises(RuntimeError, match="not CUDA-graph safe"),
    ):
        moe._fused_experts_tn(
            torch.ones(1, 128),
            torch.ones(4, 192, 128),
            torch.ones(4, 128, 96),
            torch.ones(1, 2),
            torch.zeros(1, 2, dtype=torch.int32),
            "silu",
        )


@requires_cuda
@torch.inference_mode()
def test_tn_cuda_step5_ep8_weight_shape():
    # Step5 TP2/EP8 local weights: E=44, H=4096, I=1536.
    torch.manual_seed(443)
    x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16) / 64
    w1 = torch.randn(44, 3072, 4096, device="cuda", dtype=x.dtype) / 64
    w2 = torch.randn(44, 4096, 1536, device="cuda", dtype=x.dtype) / 40
    ids = torch.tensor([[0, 43]], device="cuda", dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]], device="cuda")
    context = SimpleNamespace(
        attn_metadata=SimpleNamespace(num_prefills=1, num_decodes=0)
    )
    with patch.object(moe, "get_forward_context", return_value=context):
        actual = moe._fused_experts_tn(
            x, w1, w2, weights, ids, "swiglustep", swiglu_limit=3.0
        )
    expected = _dense_reference(x, w1, w2, ids, weights, "swiglustep")
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
