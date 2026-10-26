# SPDX-License-Identifier: Apache-2.0
"""Layout, EP masking and graph replay coverage for CoreX30 NN MoE."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from conftest import requires_corex30

from vllm_iluvatar.model_executor.layers.fused_moe.fused_moe import (
    ilu_w16a16_fused_experts,
    normalize_nn_weights,
)
from vllm_iluvatar.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    IluUnquantizedFusedMoEMethod,
)


@pytest.mark.parametrize("formats", [("TN", "TN"), ("NN", "NN"), ("TN", "NN")])
def test_normalize_preserves_nn_and_loader_attributes(formats):
    layer = torch.nn.Module()
    expected = {}
    original = {}
    layer._ilu_w16a16_input_formats = dict(zip(("w13_weight", "w2_weight"), formats))
    for name, shape, fmt in zip(
        ("w13_weight", "w2_weight"),
        ((2, 64, 64), (2, 32, 64)),
        formats,
    ):
        nn = torch.randn(shape, dtype=torch.bfloat16)
        expected[name] = nn
        tensor = nn if fmt == "NN" else nn.transpose(-1, -2).contiguous()
        param = torch.nn.Parameter(tensor, requires_grad=False)
        param.weight_loader_numel = tensor.numel()
        param.weight_loader = lambda *args: None
        layer.register_parameter(name, param)
        original[name] = param
    normalize_nn_weights(layer)
    for name, fmt in zip(original, formats):
        actual = getattr(layer, name)
        torch.testing.assert_close(actual, expected[name])
        assert actual.weight_loader_numel == original[name].numel()
        assert actual.weight_loader is original[name].weight_loader
        if fmt == "NN":
            assert actual is original[name]
            assert actual.data_ptr() == original[name].data_ptr()
    pointers = [layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr()]
    normalize_nn_weights(layer)
    assert pointers == [layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr()]


def test_invalid_pair_does_not_partially_replace():
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.ones((2, 64, 64), dtype=torch.bfloat16))
    layer.w2_weight = torch.nn.Parameter(torch.ones((2, 64, 48), dtype=torch.bfloat16))
    before = layer.w13_weight
    with pytest.raises(ValueError):
        normalize_nn_weights(layer)
    assert layer.w13_weight is before
    assert "_ilu_w16a16_weight_format" not in layer.__dict__


def test_tn_reload_rejected_after_nn_conversion():
    method = object.__new__(IluUnquantizedFusedMoEMethod)
    layer = torch.nn.Module()

    def allocate(*args, **kwargs):
        for name in ("w13_weight", "w2_weight"):
            parameter = torch.nn.Parameter(torch.zeros(1))
            parameter.weight_loader = lambda *args: "loaded"
            layer.register_parameter(name, parameter)

    with (
        patch("vllm_iluvatar.ilu_util.arch.is_ivcore30", return_value=True),
        patch(
            "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method.UnquantizedFusedMoEMethod.create_weights",
            side_effect=allocate,
        ),
    ):
        method.create_weights(layer, 1, 64, 32, torch.bfloat16)
    assert layer.w13_weight.weight_loader(None) == "loaded"
    layer._ilu_w16a16_weight_format = "NN"
    with pytest.raises(RuntimeError, match="hot reload"):
        layer.w13_weight.weight_loader(None)


def _reference(x, w13, w2, weights, ids, activation, expert_map):
    output = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        local = ids[:, slot] if expert_map is None else expert_map[ids[:, slot].long()]
        for expert in range(w13.shape[0]):
            rows = (local == expert).nonzero().flatten()
            if rows.numel() == 0:
                continue
            gate, up = (
                (x[rows].float() @ w13[expert].float()).to(x.dtype).float().chunk(2, -1)
            )
            gate = torch.nn.functional.silu(gate)
            if activation == "swiglustep":
                gate, up = gate.clamp(max=7), up.clamp(-7, 7)
            y = ((gate * up).to(x.dtype).float() @ w2[expert].float()).to(x.dtype)
            output[rows] += y.float() * weights[rows, slot, None]
    return output.to(x.dtype)


@requires_corex30
@pytest.mark.parametrize(
    "activation,step5_native",
    [("silu", False), ("swiglustep", False), ("swiglustep", True)],
)
@pytest.mark.parametrize("tokens", [1, 19, 97])
@pytest.mark.parametrize("ep", [False, True])
def test_cuda_nn_eager_and_changed_route_graph(activation, tokens, ep, step5_native):
    torch.manual_seed(17)
    device = "cuda"
    x = torch.randn((tokens, 64), device=device, dtype=torch.bfloat16) / 8
    w13 = torch.randn((4, 64, 64), device=device, dtype=torch.bfloat16) / 8
    w2 = torch.randn((4, 32, 64), device=device, dtype=torch.bfloat16) / 8
    ids = torch.randint(8 if ep else 4, (tokens, 2), device=device, dtype=torch.int32)
    weights = torch.softmax(torch.randn((tokens, 2), device=device), -1)
    expert_map = (
        torch.tensor([2, -1, 0, -1, 3, -1, 1, -1], device=device, dtype=torch.int32)
        if ep
        else None
    )

    def run():
        return ilu_w16a16_fused_experts(
            x,
            w13,
            w2,
            weights,
            ids,
            activation,
            expert_map,
            step5_native_activation=step5_native,
        )

    for _ in range(3):
        output = run()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output,
        _reference(x, w13, w2, weights, ids, activation, expert_map),
        atol=2e-2,
        rtol=2e-2,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for all_remote in (False, True):
        ids.fill_(1 if ep and all_remote else 0)
        x.add_(0.01)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output,
            _reference(x, w13, w2, weights, ids, activation, expert_map),
            atol=2e-2,
            rtol=2e-2,
        )


@requires_corex30
@pytest.mark.parametrize("tokens", [1, 32, 97])
def test_cuda_step5_ep8_shape(tokens):
    torch.manual_seed(19)
    x = torch.randn((tokens, 4096), device="cuda", dtype=torch.bfloat16) / 64
    w13 = torch.randn((44, 4096, 3072), device="cuda", dtype=torch.bfloat16) / 64
    w2 = torch.randn((44, 1536, 4096), device="cuda", dtype=torch.bfloat16) / 40
    # An uneven compact range, including a >64-row expert at T=97.
    ids = torch.zeros((tokens, 2), device="cuda", dtype=torch.int32)
    ids[:, 1] = 43
    weights = torch.full((tokens, 2), 0.5, device="cuda")
    actual = ilu_w16a16_fused_experts(x, w13, w2, weights, ids, "swiglustep")
    expected = _reference(x, w13, w2, weights, ids, "swiglustep", None)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_swiglustep_limit(value):
    from vllm_iluvatar.model_executor.layers.fused_moe.activation import (
        swiglustep_limit,
    )

    with pytest.raises(ValueError, match="finite positive"):
        swiglustep_limit(value)


@requires_corex30
@pytest.mark.parametrize("limit", [3.0, 10.0])
@pytest.mark.parametrize("masked", [False, True])
def test_nondefault_limit_reaches_native_activation(limit, masked):
    # Saturating values deliberately distinguish limit=3/10 from the old fixed 7.
    x = torch.ones(2, 64, device="cuda", dtype=torch.bfloat16)
    w1 = torch.full((1, 64, 64), 0.125, device="cuda", dtype=torch.bfloat16)
    w2 = torch.zeros((1, 32, 64), device="cuda", dtype=torch.bfloat16)
    w2[0, :32, :32] = torch.eye(32, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(2, 1, device="cuda", dtype=torch.int32)
    weights = torch.ones(2, 1, device="cuda")
    result = ilu_w16a16_fused_experts(
        x,
        w1,
        w2,
        weights,
        ids,
        "swiglustep",
        step5_native_activation=masked,
        swiglu_limit=limit,
    )
    gate, up = (x.float() @ w1[0].float()).chunk(2, -1)
    ref = (
        torch.nn.functional.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
    ).bfloat16().float() @ w2[0].float()
    torch.testing.assert_close(result.float(), ref, rtol=0.01, atol=0.01)
    wrong = (
        torch.nn.functional.silu(gate).clamp(max=7) * up.clamp(-7, 7)
    ).bfloat16().float() @ w2[0].float()
    assert not torch.allclose(result.float(), wrong, rtol=0.01, atol=0.01)


@requires_corex30
@pytest.mark.parametrize(
    "dtype,expected_format", [(torch.float16, "TN"), (torch.bfloat16, "NN")]
)
@torch.inference_mode()
def test_cuda_weight_loading_selects_and_executes_matching_implementation(
    dtype, expected_format
):
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoeBackend,
    )

    from vllm_iluvatar.model_executor.layers.fused_moe import fused_moe as tn

    torch.manual_seed(439)
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.randn(4, 192, 128, device="cuda", dtype=dtype) / 8, requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.randn(4, 128, 96, device="cuda", dtype=dtype) / 8, requires_grad=False
    )
    original = (layer.w13_weight, layer.w2_weight)
    layer.activation = "silu"
    layer.apply_router_weight_on_input = False
    layer.expert_map = None
    layer.global_num_experts = 4
    method = object.__new__(IluUnquantizedFusedMoEMethod)
    method.unquantized_backend = UnquantizedMoeBackend.OOT
    method.moe_quant_config = None
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            enable_eplb=False, all2all_backend="allgather_reducescatter"
        ),
        has_bias=False,
        is_lora_enabled=False,
    )
    method.process_weights_after_loading(layer)
    assert layer._ilu_w16a16_weight_format == expected_format
    pointers = (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())
    method.process_weights_after_loading(layer)
    assert pointers == (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())
    x = torch.randn(3, 128, device="cuda", dtype=dtype) / 8
    ids = torch.tensor([[0, 1], [2, 3], [3, 0]], device="cuda", dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]], device="cuda").expand(3, 2).contiguous()
    context = SimpleNamespace(
        attn_metadata=SimpleNamespace(num_prefills=3, num_decodes=0)
    )
    with patch.object(tn, "get_forward_context", return_value=context):
        actual = method.forward_native(layer, x, weights, ids, None, None)
    expected = _reference(
        x,
        original[0].transpose(-1, -2),
        original[1].transpose(-1, -2),
        weights,
        ids,
        "silu",
        None,
    )
    tol = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


@requires_corex30
def test_cuda_nn_empty_tokens():
    x = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
    result = ilu_w16a16_fused_experts(
        x,
        torch.empty(4, 128, 192, device="cuda", dtype=x.dtype),
        torch.empty(4, 96, 128, device="cuda", dtype=x.dtype),
        torch.empty(0, 2, device="cuda"),
        torch.empty(0, 2, device="cuda", dtype=torch.int32),
        "silu",
    )
    assert result.shape == x.shape and result.dtype == x.dtype
