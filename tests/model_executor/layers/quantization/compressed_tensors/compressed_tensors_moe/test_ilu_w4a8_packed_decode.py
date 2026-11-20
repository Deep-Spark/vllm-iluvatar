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
    obj.weight_symmetric, obj.has_bias = True, False
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


@pytest.mark.parametrize("rows", [1, 17, 288, 384, 512])
def test_prepared_nn_uses_packed_decode(rows):
    method, layer = make_method(), make_layer()
    for info in (method.gemm1_info, method.gemm2_info):
        info.format_str, info.format = "NN", 0
    method.version = 1
    layer._use_ixmm_decode = True
    assert method._use_ep_packed_decode(
        layer, torch.empty(rows, 4096, dtype=torch.bfloat16), True
    )
    assert not method._use_ep_packed_decode(
        layer, torch.empty(rows, 4096, dtype=torch.bfloat16), False
    )


@pytest.mark.parametrize(
    "rows,rank,residual",
    [(1, 0, False), (17, 15, True), (288, 0, True), (384, 15, True)],
)
@torch.inference_mode()
@pytest.mark.skipif(not mod.is_ivcore11(), reason="NN V1 EP decode is an MR path")
def test_ixmm_nn_v1_dynamic_graph_matches_eager(native_weights, rows, rank, residual):
    pytest.importorskip("ixmm.gemm.i4_i8_fp16")
    from ixformer.inference.functions.ops.moe.moe import _dequant_weight_int8

    reference_layer = make_layer(rank)
    method = make_method()
    method.gemm_format = "NN"
    method.version = 1
    # __new__ bypasses the production NN V1 constructor's fallback selection.
    method.decode_group_gemm = True
    for info in (method.gemm1_info, method.gemm2_info):
        info.format_str, info.format = "NN", 0
    layer = torch.nn.Module()
    layer.__dict__.update(vars(make_layer(rank)))
    for name, value in native_weights.items():
        setattr(reference_layer, name, value)
        setattr(layer, name, value)
    from ixmm.gemm.i4_i8_fp16 import preswizzle_w4a8_weight

    for name in ("w13", "w2"):
        logical = _dequant_weight_int8(
            getattr(layer, name + "_weight"), None, None, -1, 2, "TN"
        )
        raw = torch.div(logical, 18, rounding_mode="trunc").to(torch.int8)
        setattr(layer, name + "_weight", preswizzle_w4a8_weight(raw.transpose(-1, -2)))
        setattr(
            layer, name + "_weight_scale", getattr(layer, name + "_weight_scale") * 18
        )
    for name in (
        "w13_i8_weight_scale",
        "w2_i8_weight_scale",
        "w13_i8_weight_zero",
        "w2_i8_weight_zero",
    ):
        setattr(layer, name, None)
    assert method._supports_ixmm_decode(layer)
    layer._use_ixmm_decode = True
    assert method.gemm_format == "NN"
    for name in ("w13", "w2"):
        # Both kernel families must read exactly the same logical INT8 weight.
        tn = _dequant_weight_int8(
            getattr(reference_layer, name + "_weight"), None, None, -1, 2, "TN"
        )
        nn = _dequant_weight_int8(
            getattr(layer, name + "_weight"), None, None, -1, 1, "NN"
        )
        torch.testing.assert_close(nn * 18, tn, rtol=0, atol=0)
    del tn, nn
    # NN V1 weights and scales are used as-is; no runtime repacking/scaling.
    pointers = [getattr(layer, name + "_weight").data_ptr() for name in ("w13", "w2")]
    layer._use_ixmm_decode = True
    assert pointers == [
        getattr(layer, name + "_weight").data_ptr() for name in ("w13", "w2")
    ]

    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * 0.1
    ids = torch.empty(rows, 8, device="cuda", dtype=torch.int32).random_(0, 288)
    weights = torch.rand(rows, 8, device="cuda")
    shared = torch.randn_like(x) if residual else None

    def new():
        return method._apply_ep(layer, x, weights, ids, shared, True)

    new()
    with patch.object(method, "_use_ep_packed_decode", return_value=False):
        fallback = method._apply_ep(layer, x, weights, ids, shared, True)
        prefill = method._apply_ep(layer, x, weights, ids, shared, False)
    for result in (fallback, prefill):
        actual_eager = new()
        error = (result.float() - actual_eager.float()).norm()
        assert error / actual_eager.float().norm().clamp_min(1e-12) < 0.005
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
        # NN uses the mainline FP32 expand-quant contract, while TN's
        # format-1 path preserves BF16 intermediate rounding. Each input
        # quantizer is checked against its own CPU oracle in ixformer's
        # test_expand_quant_ep_decode.py; NN projections are checked below.
        # Dynamic routing must replay this same NN backend exactly.
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, new(), rtol=0, atol=0)
        if pattern == "empty":
            torch.testing.assert_close(
                actual,
                torch.zeros_like(x) if shared is None else shared,
                rtol=0,
                atol=0,
            )
    graph.reset()


@pytest.mark.parametrize("name", ["w13", "w2"])
@torch.inference_mode()
def test_ixmm_projection_matches_cpu_reference(native_weights, name):
    from ixformer.inference.functions.ops.moe.moe import _dequant_weight_int8

    ixmm = pytest.importorskip("ixmm.gemm.i4_i8_fp16")
    counts = torch.tensor(
        [0, 1, 3, 7, 33, 65, 2, 5, 0, 4, 10, 6, 8, 2, 4, 8, 1, 9],
        dtype=torch.int32,
    )
    ends = counts.cumsum(0, dtype=torch.int32)
    layout = torch.stack((ends - counts, ends), 1).cuda()
    logical = _dequant_weight_int8(
        native_weights[name + "_weight"].cpu(), None, None, -1, 2, "TN"
    )
    # Recover raw signed INT4 and pack independently of the plugin loader.
    raw = torch.div(logical, 18, rounding_mode="trunc").to(torch.int8).cuda()
    packed = ixmm.preswizzle_w4a8_weight(raw.transpose(-1, -2))
    experts, n, k = logical.shape
    m = int(counts.sum())
    a = torch.randint(-100, 101, (m, k), device="cuda", dtype=torch.int8)
    sa = torch.rand(m, device="cuda") * 0.002
    sw = native_weights[name + "_weight_scale"].reshape(experts, n)
    output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    ixf.moe_w4a8_group_gemm_ixmm(
        a,
        packed,
        sa,
        sw * 18,
        layout,
        output=output,
        group_size=0,
        alpha=1.0,
        backend="ixmm",
    )
    ac, sc, wc = a.cpu().float(), sa.cpu(), sw.cpu()
    reference = torch.empty((m, n), dtype=torch.bfloat16)
    start = 0
    for expert, count in enumerate(counts.tolist()):
        if count:
            reference[start : start + count] = (
                (ac[start : start + count] @ logical[expert].float().T)
                * sc[start : start + count, None]
                * wc[expert]
            ).bfloat16()
        start += count
    torch.testing.assert_close(output.cpu(), reference, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("fmt,expected", [("TN", False), ("NN", True)])
def test_ixmm_supports_nn_only(monkeypatch, fmt, expected):
    monkeypatch.setattr(mod, "is_ivcore11", lambda: True)
    monkeypatch.setattr("ixformer.inference.ixmm_backend.ixmm_available", lambda: True)
    method, layer = make_method(), make_layer()
    method.gemm_format = fmt
    method.version = 1
    method.gemm1_info = W4A8GroupGemmInfo(fmt, 4096, 4096)
    method.gemm2_info = W4A8GroupGemmInfo(fmt, 2048, 4096)
    for name in (
        "w13_i8_weight_scale",
        "w2_i8_weight_scale",
        "w13_i8_weight_zero",
        "w2_i8_weight_zero",
    ):
        setattr(layer, name, None)
    assert method._supports_ixmm_decode(layer) is expected


@pytest.mark.parametrize(
    "available,fmt,expected",
    [
        (False, "NN", False),
        (True, "NN", True),
        (True, "TN", False),
    ],
)
def test_loading_selects_available_ixmm_for_supported_layout(
    monkeypatch, available, fmt, expected
):
    monkeypatch.setattr(mod, "is_ivcore11", lambda: True)
    monkeypatch.setattr(
        "ixformer.inference.ixmm_backend.ixmm_available", lambda: available
    )
    method, layer = make_method(), make_layer()
    method.gemm_format = method.runtime_gemm_format = fmt
    method.version = 1
    method.gemm1_info = W4A8GroupGemmInfo(fmt, 4096, 4096)
    method.gemm2_info = W4A8GroupGemmInfo(fmt, 2048, 4096)
    method.weight_symmetric = True
    method.has_bias = method.static_input_scales = False
    for name in ("w13", "w2"):
        setattr(layer, name + "_weight", torch.empty(18, 1, 1, device="meta"))
        setattr(layer, name + "_weight_scale", torch.empty(18, 1, 4096, device="meta"))
        setattr(layer, name + "_i8_weight_scale", None)
        setattr(layer, name + "_i8_weight_zero", None)
    monkeypatch.setattr(mod, "replace_with_padded", Mock())
    method.process_weights_after_loading(layer)
    assert layer._use_ixmm_decode is expected
    assert not hasattr(layer, "_ixmm_w13_scale")
    assert not hasattr(layer, "_ixmm_w2_scale")


def test_ixmm_rejects_nn_v2(monkeypatch):
    method, layer = make_method(), make_layer()
    method.gemm_format = "NN"
    assert method.version == 2
    assert not method._supports_ixmm_decode(layer)


def test_ixmm_route_preserves_cq_mainline(monkeypatch):
    monkeypatch.setattr(mod, "is_ivcore11", lambda: False)
    method, layer = make_method(), make_layer()
    method.gemm_format = "NN"
    method.version = 1
    assert not method._supports_ixmm_decode(layer)


@pytest.mark.parametrize("unsupported", ["asymmetric", "bias", "runtime_layout"])
def test_ixmm_requires_original_nn_v1_contract(monkeypatch, unsupported):
    monkeypatch.setattr(mod, "is_ivcore11", lambda: True)
    method, layer = make_method(), make_layer()
    method.gemm_format, method.version = "NN", 1
    method.gemm1_info = W4A8GroupGemmInfo("NN", 4096, 4096)
    method.gemm2_info = W4A8GroupGemmInfo("NN", 2048, 4096)
    if unsupported == "asymmetric":
        method.weight_symmetric = False
    elif unsupported == "bias":
        method.has_bias = True
    else:
        method.gemm2_info.format_str = "TN"
    assert not method._supports_ixmm_decode(layer)
