# SPDX-License-Identifier: Apache-2.0
"""Packed EP selection and actual TN-V2 expert graph regression."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import ixformer.inference.functions as ixf
import pytest
import torch

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    ilu_compressed_tensors_moe_w4a8_int8 as mod,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (
    W4A8GroupGemmInfo,
)


def make_method():
    obj = mod.IluCompressedTensorsW4A8Int8MoEMethod.__new__(
        mod.IluCompressedTensorsW4A8Int8MoEMethod
    )
    obj.version, obj.group_size = 2, -1
    obj.gemm1_info = W4A8GroupGemmInfo("TN", 4096, 4096)
    obj.gemm2_info = W4A8GroupGemmInfo("TN", 2048, 4096)
    return obj


def make_layer(rank=0):
    return SimpleNamespace(
        moe_config=SimpleNamespace(ep_rank=rank),
        global_num_experts=288,
        local_num_experts=18,
        top_k=8,
        activation=SimpleNamespace(value="silu"),
        swiglu_limit=10,
        w13_input_scale=None,
        w2_input_scale=None,
        w13_bias=None,
        w2_bias=None,
        _ilu_output_scale=2.5,
    )


@pytest.mark.parametrize(
    "rows,expected",
    [(0, False), (1, True), (17, True), (288, True), (512, True), (513, False)],
)
def test_packed_row_boundaries(rows, expected):
    assert (
        make_method()._use_ep_packed_decode(
            make_layer(), torch.empty(rows, 4096, dtype=torch.bfloat16), True
        )
        is expected
    )


@pytest.mark.parametrize(
    "case",
    [
        "prefill",
        "dtype",
        "hidden",
        "experts",
        "local_experts",
        "topk",
        "version",
        "group_size",
        "w13_nn",
        "w2_nn",
        "w13_smooth",
        "w2_smooth",
        "activation",
        "limit",
        "w13_padding",
        "w2_padding",
    ],
)
def test_packed_rejects_other_paths(case):
    method, layer = make_method(), make_layer()
    x = torch.empty(17, 4096, dtype=torch.bfloat16)
    decode = True
    if case == "prefill":
        decode = False
    elif case == "dtype":
        x = x.half()
    elif case == "hidden":
        x = x[:, :2048]
    elif case == "experts":
        layer.global_num_experts = 384
    elif case == "local_experts":
        layer.local_num_experts = 36
    elif case == "topk":
        layer.top_k = 6
    elif case == "version":
        method.version = 1
    elif case == "group_size":
        method.group_size = 64
    elif case == "w13_nn":
        method.gemm1_info.format_str = "NN"
    elif case == "w2_nn":
        method.gemm2_info.format_str = "NN"
    elif case == "w13_smooth":
        layer.w13_input_scale = torch.ones(4096)
    elif case == "w2_smooth":
        layer.w2_input_scale = torch.ones(2048)
    elif case == "activation":
        layer.activation.value = "gelu"
    elif case == "limit":
        layer.swiglu_limit = None
    elif case == "w13_padding":
        method.gemm1_info.n -= 2
    else:
        method.gemm2_info.k -= 2
    assert not method._use_ep_packed_decode(layer, x, decode)


def test_apply_ep_selects_packed_and_preserves_fallback(monkeypatch):
    method, layer = make_method(), make_layer()
    x = torch.empty(17, 4096, dtype=torch.bfloat16)
    ids = torch.zeros(17, 8, dtype=torch.int32)
    weights = torch.ones(17, 8)
    expected = object()
    method._apply_ep_packed_decode = Mock(return_value=expected)
    assert method._apply_ep(layer, x, weights, ids, None, True) is expected
    method._apply_ep_packed_decode.assert_called_once_with(layer, x, weights, ids, None)

    class LegacyReached(Exception):
        pass

    def legacy(**kwargs):
        raise LegacyReached

    monkeypatch.setattr(mod.ixfops, "moe_compute_token_index_ep", legacy)
    with pytest.raises(LegacyReached):
        method._apply_ep(layer, x, weights, ids, None, False)
    assert method._apply_ep_packed_decode.call_count == 1


@pytest.fixture(scope="module")
def native_weights():
    if not torch.cuda.is_available():
        pytest.skip("Iluvatar GPU required")
    torch.manual_seed(1731)
    result = {}
    for prefix, n, k in [("w13", 4096, 4096), ("w2", 4096, 2048)]:
        # Use the production TN V2 packer, including its secondary INT8 scales.
        w = torch.randint(-127, 128, (18, n, k), device="cuda", dtype=torch.int8)
        packed, scales, zeros = ixf.quant_repack_int4(w, -1, 2, "TN", False)
        result[prefix + "_weight"] = packed
        result[prefix + "_weight_scale"] = torch.full((18, n), 0.001, device="cuda")
        result[prefix + "_i8_weight_scale"] = scales
        result[prefix + "_i8_weight_zero"] = zeros
        del w
    return result


@pytest.mark.parametrize(
    "rows,rank,residual",
    [(1, 0, False), (17, 15, True), (288, 0, True), (512, 15, False)],
)
@torch.inference_mode()
def test_packed_matches_legacy_with_dynamic_graph(native_weights, rows, rank, residual):
    method, layer = make_method(), make_layer(rank)
    for name, value in native_weights.items():
        setattr(layer, name, value)
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * 0.1
    ids = torch.empty(rows, 8, device="cuda", dtype=torch.int32).random_(0, 288)
    weights = torch.rand(rows, 8, device="cuda")
    shared = torch.randn_like(x) if residual else None

    def new():
        return method._apply_ep(layer, x, weights, ids, shared, True)

    def old():
        with patch.object(method, "_use_ep_packed_decode", return_value=False):
            return method._apply_ep(layer, x, weights, ids, shared, True)

    new()
    old()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = new()
    for pattern in ("mixed", "empty", "one_expert", "all_local", "mixed"):
        x.normal_(std=10.0 if pattern == "one_expert" else 0.1)
        weights.uniform_()
        if shared is not None:
            shared.normal_()
        if pattern == "mixed":
            ids.random_(0, 288)
        elif pattern == "empty":
            ids.fill_((rank * 18 + 18) % 288)
        elif pattern == "one_expert":
            ids.fill_(rank * 18)
        else:
            ids.random_(rank * 18, (rank + 1) * 18)
        graph.replay()
        expected = old()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
        if pattern == "empty":
            torch.testing.assert_close(
                actual,
                torch.zeros_like(x) if shared is None else shared,
                rtol=0,
                atol=0,
            )
    graph.reset()
