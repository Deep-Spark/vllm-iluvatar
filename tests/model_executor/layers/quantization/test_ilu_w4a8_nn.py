# SPDX-License-Identifier: Apache-2.0
"""Routing, padding and NN grouped GEMM tests for Iluvatar W4A8 MoE."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import ixformer.inference.functions as ixf_ops
import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
)

from vllm_iluvatar.model_executor.layers.quantization import ilu_w4a8_nn as w4a8_moe
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    IluQuantConfig,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    ilu_compressed_tensors_moe_w4a8_int8 as legacy_w4a8,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_compressed_tensors_moe import (
    IluCompressedTensorsMoEMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (
    W4A8GroupGemmInfo,
)


def _make_nn_graph_method(experts=4, hidden=256, intermediate=256):
    rng = torch.Generator().manual_seed(71)
    cls = w4a8_moe.IluW4A8NNMoEMethod
    method = cls.__new__(cls)
    method.group_size = -1
    method.gemm_format = "NN"
    method.gemm1_info = W4A8GroupGemmInfo("NN", hidden, 2 * intermediate)
    method.gemm2_info = W4A8GroupGemmInfo("NN", intermediate, hidden)
    layer = torch.nn.Module()
    raw = []
    scales = []
    for prefix, n, k in (
        ("w13", 2 * intermediate, hidden),
        ("w2", hidden, intermediate),
    ):
        w = torch.randint(-7, 8, (experts, n, k), generator=rng, dtype=torch.int8)
        scale = torch.rand(experts, 1, n, generator=rng) * 0.002
        # Construct an NN checkpoint directly, independent of plugin packing.
        # Force every output channel's q8 amax=126, hence the NN scale is 18.
        w[..., 0] = 7
        q8 = (w.to(torch.int16) * 18).clamp(-127, 127).to(torch.int8).cuda()
        packed, extra, _ = ixf_ops.quant_repack_int4(q8, -1, 1, "NN", False)
        torch.testing.assert_close(extra, torch.full_like(extra, 18))
        layer.register_parameter(
            prefix + "_weight", torch.nn.Parameter(packed, requires_grad=False)
        )
        layer.register_parameter(
            prefix + "_weight_scale",
            torch.nn.Parameter((scale * 18).cuda(), requires_grad=False),
        )
        raw.append(w)
        scales.append(scale)
        for suffix in ("i8_weight_scale", "i8_weight_zero", "input_scale", "bias"):
            setattr(layer, prefix + "_" + suffix, None)
    layer.moe_config = SimpleNamespace(ep_rank=1)
    layer.local_num_experts = experts
    layer.global_num_experts = experts * 2
    layer.top_k = 2
    layer.expert_map = torch.arange(experts, device="cuda")
    layer.activation = SimpleNamespace(value="silu")
    layer._ilu_output_scale = 0.375
    return method, layer, raw, scales


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_static_nn_ixformer_against_cpu(dtype):
    method, layer, raw, scales = _make_nn_graph_method()
    rng = torch.Generator().manual_seed(8)
    a = torch.randint(-127, 128, (9, 256), dtype=torch.int8, generator=rng)
    sa = torch.rand(9, 1, generator=rng) * 0.01
    layout = torch.tensor([[0, 2], [2, 2], [2, 5], [5, 7]], dtype=torch.int32)
    layout = layout.cuda()
    for i, (prefix, info) in enumerate(
        (("w13", method.gemm1_info), ("w2", method.gemm2_info))
    ):
        out = torch.full(
            (9, info.get_pad_n()), float("nan"), device="cuda", dtype=dtype
        )
        method._group_gemm(
            info,
            a.cuda(),
            sa.cuda(),
            getattr(layer, prefix + "_weight"),
            getattr(layer, prefix + "_weight_scale"),
            layout,
            9,
            dtype,
            output=out,
        )
        for e, (lo, hi) in enumerate(layout.cpu().tolist()):
            expected = (
                (a[lo:hi].float() @ raw[i][e].float().T)
                * sa[lo:hi]
                * (scales[i][e] * 18)
            )
            torch.testing.assert_close(
                out[lo:hi].cpu(), expected.to(dtype), atol=0.008, rtol=0.008
            )
        assert out[7:].isnan().all().item()


# Both prefill and decode use the same NN ixmm compute surface.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["ep", "non_ep"])
@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_pipeline_replay(dtype, mode, topk, monkeypatch):
    method, layer, _, _ = _make_nn_graph_method()
    layer.top_k = topk
    ep = mode != "non_ep"
    if not ep:
        layer.global_num_experts = layer.local_num_experts
        layer.expert_map = None
    rng = torch.Generator().manual_seed(17)
    x = torch.randn(17, 256, generator=rng).to("cuda", dtype)
    weights = torch.rand(17, topk, generator=rng).cuda()
    ids = torch.randint(
        layer.global_num_experts, (17, topk), generator=rng, dtype=torch.int32
    ).cuda()
    residual = torch.randn(17, 256, generator=rng).to("cuda", dtype)

    def launch():
        return method.apply(layer, x, weights, ids, None, residual)

    # Both projections must use the ixformer ixmm bridge and device layout.
    original = ixf_ops.moe_w4a8_group_gemm_ixmm
    original_reduce = ixf_ops.moe_output_reduce_sum_ep

    def forbidden(*args, **kwargs):
        raise AssertionError("graph decode called GEMV/scatter")

    calls = []
    reductions = []

    def checked_gemm(*args, **kwargs):
        assert kwargs["backend"] == "ixmm"
        assert kwargs["group_layout"].is_cuda
        calls.append(kwargs["group_layout"])
        return original(*args, **kwargs)

    def checked_reduce(*args, **kwargs):
        assert kwargs["packed_to_expert_row"].is_cuda
        reductions.append(kwargs["packed_to_expert_row"])
        return original_reduce(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ixf_ops, "moe_w4a8_group_gemm_ixmm", checked_gemm)
        patch.setattr(ixf_ops, "moe_output_reduce_sum_ep", checked_reduce)
        patch.setattr(ixf_ops, "moe_w4a8_group_gemv", forbidden)
        patch.setattr(ixf_ops, "moe_expand_input", forbidden)
        launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = launch()
    assert len(calls) == 4 and calls[0] is calls[1] and calls[2] is calls[3]
    assert len(reductions) == 2
    assert ixf_ops.moe_w4a8_group_gemm_ixmm is original
    # Different valid rows and active experts at fixed addresses, including an
    # EP rank with no local work. Residual must still be written on empty rows.
    initial = ids.clone()
    for changed in (
        initial,
        torch.zeros_like(ids),
        torch.full_like(ids, layer.global_num_experts - 1),
    ):
        ids.copy_(changed)
        weights.mul_(0.9)
        graph.replay()
        expected = launch()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.01, rtol=0.01)


@pytest.mark.parametrize("ep", [False, True])
@pytest.mark.parametrize("num_tokens", [17, 1033])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nn_prefill_uses_ixformer_ixmm(ep, num_tokens, monkeypatch):
    method, layer, raw, scales = _make_nn_graph_method()
    if not ep:
        layer.global_num_experts = layer.local_num_experts
        layer.expert_map = None
    rng = torch.Generator().manual_seed(29)
    x = torch.randn(num_tokens, 256, generator=rng).to("cuda", torch.bfloat16)
    weights = torch.rand(num_tokens, 2, generator=rng).cuda()
    ids = torch.randint(
        layer.global_num_experts, (num_tokens, 2), generator=rng, dtype=torch.int64
    ).cuda()
    residual = torch.randn(num_tokens, 256, generator=rng).to("cuda", torch.bfloat16)
    monkeypatch.setattr(
        legacy_w4a8,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=object()),
    )
    monkeypatch.setattr(legacy_w4a8, "_is_only_decode", lambda metadata: False)
    native = ixf_ops.moe_w4a8_group_gemm_ixmm
    calls = []

    def checked(**kwargs):
        assert kwargs["backend"] == "ixmm"
        assert kwargs["group_layout"].is_cuda
        calls.append(kwargs["group_layout"])
        return native(**kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("NN prefill used legacy grouped GEMM")

    with monkeypatch.context() as patch:
        patch.setattr(ixf_ops, "moe_w4a8_group_gemm_ixmm", checked)
        patch.setattr(ixf_ops, "moe_w4a8_group_gemm", forbidden)
        actual = method.apply(layer, x, weights, ids, None, residual)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = method.apply(layer, x, weights, ids, None, residual)
    assert len(calls) == 4 and calls[0] is calls[1] and calls[2] is calls[3]
    graph.replay()
    torch.testing.assert_close(captured, actual, atol=0.01, rtol=0.01)

    quantized, input_scale = ixf_ops.scaled_int8_quant(x)
    qx = quantized.float().cpu()
    sx = input_scale.cpu()
    expected = torch.zeros((num_tokens, 256), dtype=torch.float32)
    ids_cpu, weights_cpu = ids.cpu(), weights.cpu()
    start = layer.moe_config.ep_rank * layer.local_num_experts if ep else 0
    for local in range(layer.local_num_experts):
        tokens, slots = torch.where(ids_cpu == start + local)
        # Accumulate integer products before applying the dequant scales,
        # matching the GEMM epilogue's BF16 rounding point.
        z = (
            (
                (qx[tokens] @ raw[0][local].float().T)
                * sx[tokens]
                * (scales[0][local] * 18)
            )
            .to(torch.bfloat16)
            .float()
        )
        gate, up = z.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
        act_scale = activated.abs().amax(dim=-1, keepdim=True) / 127
        inv = torch.where(act_scale > 0, act_scale.reciprocal(), 0)
        # activation_dynamic_scaled_int8 keeps amax in FP32 but stores each
        # activation in the input dtype before INT8 rounding (vals[i]).
        qa = (activated.to(torch.bfloat16).float() * inv).round().clamp(-128, 127)
        routed = (
            ((qa @ raw[1][local].float().T) * act_scale * (scales[1][local] * 18))
            .to(torch.bfloat16)
            .float()
        )
        expected.index_add_(0, tokens, routed * weights_cpu[tokens, slots, None])
    expected = (expected * layer._ilu_output_scale + residual.float().cpu()).to(
        torch.bfloat16
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=0.03, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nn_standard_experts_prefill_matches_monolithic(monkeypatch):
    from vllm.model_executor.layers.fused_moe import MoEActivation

    from vllm_iluvatar.model_executor.layers.quantization import (
        ilu_w4a8_nn_experts as std_w4a8,
    )

    method, layer, _, _ = _make_nn_graph_method()
    experts = std_w4a8.IluMoEStandardExpertsW4A8NN.__new__(
        std_w4a8.IluMoEStandardExpertsW4A8NN
    )
    experts.local_num_experts = layer.local_num_experts
    experts.format, experts.group_size = "NN", -1
    experts.gemm1_info, experts.gemm2_info = method.gemm1_info, method.gemm2_info
    experts.w13_input_scale = experts.w2_input_scale = experts.swiglu_limit = None
    experts.runner_scaling_factor = layer._ilu_output_scale
    experts.quant_config = SimpleNamespace(
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
    )
    monkeypatch.setattr(std_w4a8, "get_ep_group", lambda: SimpleNamespace(rank=1))
    rng = torch.Generator().manual_seed(41)
    x = torch.randn(33, 256, generator=rng).to("cuda", torch.bfloat16)
    ids = torch.randint(8, (33, 2), generator=rng, dtype=torch.int64).cuda()
    weights = torch.rand(33, 2, generator=rng).cuda()
    output = torch.empty((33, 256), device="cuda", dtype=torch.bfloat16)
    actual = experts.apply(
        output,
        x,
        layer.w13_weight,
        layer.w2_weight,
        weights,
        ids,
        MoEActivation.SILU,
        layer.global_num_experts,
        layer.expert_map,
        None,
        None,
        None,
        None,
    )
    expected = method._apply_group_gemm(layer, x, weights, ids.to(torch.int32), None)
    assert actual is output
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    ids.zero_()  # This EP rank has no local expert rows.
    output.fill_(float("nan"))
    experts.apply(
        output,
        x,
        layer.w13_weight,
        layer.w2_weight,
        weights,
        ids,
        MoEActivation.SILU,
        layer.global_num_experts,
        layer.expert_map,
        None,
        None,
        None,
        None,
    )
    torch.testing.assert_close(output, torch.zeros_like(output))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_glm_shape_decode():
    method, layer, _, _ = _make_nn_graph_method(
        experts=16, hidden=6144, intermediate=2048
    )
    layer.moe_config.ep_rank = 0
    layer.global_num_experts = 256
    layer.top_k = 8
    rng = torch.Generator().manual_seed(52)
    x = torch.randn(512, 6144, generator=rng).to("cuda", torch.bfloat16)
    ids = torch.randint(256, (512, 8), generator=rng, dtype=torch.int32).cuda()
    weights = torch.rand(512, 8, generator=rng).cuda()
    method.apply(layer, x, weights, ids, None, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = method.apply(layer, x, weights, ids, None, None)
    graph.replay()
    expected = method.apply(layer, x, weights, ids, None, None)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.01, rtol=0.01)


_STATIC_NN_PREFIX = "model.layers.3.mlp.experts"


def _static_nn_checkpoint_config():
    return {
        "format": "iluvatar-mixed",
        "config_groups": {
            "experts": {
                "targets": [r"re:.*\.experts"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "strategy": "channel",
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "token",
                    "symmetric": True,
                    "dynamic": True,
                },
                "iluvatar": {"gemm_format": "NN", "pack_version": 1},
            }
        },
        "ignore": [],
    }


@pytest.fixture
def _static_nn_dispatch(monkeypatch):
    # Same MoE selection hook installed by register_general(), without loading
    # unrelated model/worker patches into this CPU test process.
    monkeypatch.setattr(
        CompressedTensorsMoEMethod,
        "get_moe_method",
        staticmethod(IluCompressedTensorsMoEMethod.get_moe_method),
    )
    monkeypatch.setattr(
        legacy_w4a8.envs, "VLLM_ILU_MOE_EP_LL_MOE_ALGO", "", raising=False
    )


def _select_static_nn(*, prefix=_STATIC_NN_PREFIX, source=None):
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="glm_moe_dsa"),
        dtype=torch.bfloat16,
    )
    quant = IluQuantConfig.from_config(source or _static_nn_checkpoint_config())
    layer = MagicMock(spec=RoutedExperts)
    layer.moe_config = SimpleNamespace(has_bias=False)
    with set_current_vllm_config(config):
        method = quant.get_quant_method(layer, prefix)
    return method


@pytest.mark.usefixtures("_static_nn_dispatch")
@pytest.mark.parametrize("arch", ["ivcore11", "ivcore30"])
@pytest.mark.parametrize(
    "format,version", [("TN", 2), ("NN", 2), ("NN", 1), ("NN", None)]
)
def test_arch_and_layout_select_quant_class(monkeypatch, arch, format, version):
    from vllm_iluvatar.ilu_util import arch as arch_util

    monkeypatch.setattr(arch_util, "cuda_arch", lambda: arch)
    source = _static_nn_checkpoint_config()
    source["config_groups"]["experts"]["iluvatar"] = {
        "gemm_format": format,
    }
    if version is not None:
        source["config_groups"]["experts"]["iluvatar"]["pack_version"] = version
    new_path = arch == "ivcore30" and format == "NN"

    def backend_probe(op):
        assert new_path, "legacy method must not probe the CQ ixmm backend"
        assert op == "moe_w4a8_group_gemm_ixmm"
        return ["ixmm"]

    monkeypatch.setattr(w4a8_moe.registry, "available_backends", backend_probe)
    if arch == "ivcore11" and version != 2:
        # ee989a is deliberately deferred: keep the original v2 contract.
        with pytest.raises(AssertionError, match="only supports version 2"):
            _select_static_nn(source=source)
        return
    if new_path and version == 2:
        with pytest.raises(
            ValueError, match="Unsupported NN checkpoint pack_version=2"
        ):
            _select_static_nn(source=source)
        return
    method = _select_static_nn(source=source)
    expected = (
        w4a8_moe.IluW4A8NNMoEMethod
        if new_path
        else legacy_w4a8.IluCompressedTensorsW4A8Int8MoEMethod
    )
    assert type(method) is expected
    assert not hasattr(method, "decode_group_gemm")
    if new_path:
        assert not hasattr(method, "version")


@pytest.mark.parametrize("version", [2, 3, True, "1", None])
def test_nn_rejects_incompatible_explicit_pack_version(version):
    with pytest.raises(ValueError, match="Unsupported NN checkpoint pack_version"):
        w4a8_moe.validate_nn_pack_version(version)


def test_direct_nn_class_rejects_ivcore11(monkeypatch):
    from vllm_iluvatar.ilu_util import arch as arch_util

    monkeypatch.setattr(arch_util, "cuda_arch", lambda: "ivcore11")
    with pytest.raises(RuntimeError, match="requires ivcore30"):
        w4a8_moe.IluW4A8NNMoEMethod(None, None, None)


def test_nn_class_rejects_missing_ixmm(monkeypatch):
    from vllm_iluvatar.ilu_util import arch as arch_util

    monkeypatch.setattr(arch_util, "cuda_arch", lambda: "ivcore30")
    monkeypatch.setattr(w4a8_moe.registry, "available_backends", lambda op: ["torch"])
    with pytest.raises(RuntimeError, match="requires the ixformer ixmm backend"):
        w4a8_moe.IluW4A8NNMoEMethod(None, None, None)


@pytest.mark.parametrize("algorithm", ["GEMV", "GEMM", "FUSED"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_modular_factory_selects_dedicated_nn_experts(monkeypatch, algorithm, batched):
    from vllm.model_executor.layers.fused_moe import FusedMoEActivationFormat

    from vllm_iluvatar.model_executor.layers.quantization import ilu_w4a8_nn_experts
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        ilu_moe_batch_experts,
    )

    method, layer, _, _ = _make_nn_graph_method()
    method.moe = SimpleNamespace()
    method.moe_quant_config = method.get_fused_moe_quant_config(layer)
    monkeypatch.setattr(ilu_moe_batch_experts, "MOE_ALGO", algorithm)
    prepare = SimpleNamespace(
        activation_format=(
            FusedMoEActivationFormat.BatchedExperts
            if batched
            else FusedMoEActivationFormat.Standard
        ),
        max_num_tokens_per_rank=lambda: 16 if batched else None,
        num_dispatchers=lambda: 2,
    )
    expert = method.select_gemm_impl(prepare, layer)
    expected = (
        ilu_w4a8_nn_experts.IluMoEBatchedExpertsW4A8NN
        if batched
        else ilu_w4a8_nn_experts.IluMoEStandardExpertsW4A8NN
    )
    assert type(expert) is expected
    assert expert.format == "NN"
    assert not hasattr(expert, "version")
    assert expert.runner_scaling_factor == layer._ilu_output_scale


@pytest.mark.parametrize("prefix", [_STATIC_NN_PREFIX, "model.layers.78.mlp.experts"])
@pytest.mark.usefixtures("_static_nn_dispatch")
def test_static_nn_selects_dedicated_method(prefix):
    source = _static_nn_checkpoint_config()
    before = copy.deepcopy(source)
    method = _select_static_nn(prefix=prefix, source=source)
    assert isinstance(method, w4a8_moe.IluW4A8NNMoEMethod)
    assert method.gemm_format == "NN"
    assert not hasattr(method, "version")
    assert source == before


@pytest.mark.usefixtures("_static_nn_dispatch")
@pytest.mark.parametrize("algorithm", ["GEMV", "GEMM", "FUSED"])
def test_static_nn_post_load_preserves_weights_and_scales(monkeypatch, algorithm):
    monkeypatch.setattr(legacy_w4a8.envs, "VLLM_ILU_MOE_EP_LL_MOE_ALGO", algorithm)
    method = _select_static_nn()
    layer = torch.nn.Module()
    method.create_weights(
        layer, 2, 256, 256, torch.bfloat16, weight_loader=lambda *a, **k: None
    )
    assert layer.w13_weight.shape == (2, 256, 256)
    assert layer.w2_weight.shape == (2, 256, 128)
    originals = {}
    for prefix in ("w13", "w2"):
        weight = getattr(layer, prefix + "_weight")
        weight.data.random_(-128, 128)
        getattr(layer, prefix + "_weight_scale").data.fill_(0.125)
        for suffix in ("_weight", "_weight_scale"):
            name = prefix + suffix
            originals[name] = getattr(layer, name).clone()
    method.process_weights_after_loading(layer)
    assert method.gemm_format == "NN"
    assert not hasattr(method, "version")
    for name, expected in originals.items():
        torch.testing.assert_close(getattr(layer, name), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.usefixtures("_static_nn_dispatch")
def test_static_nn_uses_ixmm_under_graph_capture(monkeypatch):
    method = _select_static_nn()
    layer = torch.nn.Module()
    with torch.device("cuda"):
        method.create_weights(
            layer, 2, 256, 256, torch.bfloat16, weight_loader=lambda *a, **k: None
        )
    for name in ("w13", "w2"):
        getattr(layer, name + "_weight").data.fill_(0x11)
        getattr(layer, name + "_weight_scale").data.fill_(0.001)
    method.process_weights_after_loading(layer)
    layer.top_k = 1
    layer.local_num_experts = 2
    layer.global_num_experts = 2
    layer.expert_map = None
    layer.activation = SimpleNamespace(value="silu")
    x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(8, device="cuda", dtype=torch.int32).remainder(2)[:, None]
    weights = torch.ones(8, 1, device="cuda")

    original = w4a8_moe.ixfops.moe_w4a8_group_gemm_ixmm
    calls = []

    def graph_gemm(**kwargs):
        assert kwargs["backend"] == "ixmm"
        assert kwargs["group_layout"].is_cuda
        calls.append(kwargs["weight"].shape)
        return original(**kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("configured NN decode fell back to GEMV/old GEMM")

    monkeypatch.setattr(w4a8_moe.ixfops, "moe_w4a8_group_gemm_ixmm", graph_gemm)
    monkeypatch.setattr(w4a8_moe.ixfops, "moe_w4a8_group_gemv", forbidden)

    def launch():
        return method.apply(layer, x, weights, ids, None, None)

    launch()  # Warm up before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = launch()
    assert len(calls) == 4  # Two ixformer GEMMs per decode, eager and capture.
    x.mul_(0.5)
    ids.zero_()  # Change active experts at fixed captured addresses.
    graph.replay()
    expected = launch()
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)


def test_nn_fused_prefill_keeps_regular_nn_path(monkeypatch):
    method = w4a8_moe.IluW4A8NNMoEMethod.__new__(w4a8_moe.IluW4A8NNMoEMethod)
    regular = MagicMock(return_value="nn")
    monkeypatch.setattr(method, "apply", regular)
    assert method.apply_fused_moe("layer", "x", "weights", "ids") == "nn"
    regular.assert_called_once_with("layer", "x", "weights", "ids", None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("intermediate", [128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("limit", [None, 7.0])
@pytest.mark.parametrize("flattened", [False, True])
def test_ll_nn_fixed_capacity_and_graph(
    monkeypatch, intermediate, dtype, limit, flattened
):
    from vllm.model_executor.layers.fused_moe import MoEActivation

    from vllm_iluvatar.model_executor.layers.quantization import (
        ilu_w4a8_nn_experts as batch,
    )

    method, layer, raw, scales = _make_nn_graph_method(intermediate=intermediate)
    # TP16 uses K=128 for down_proj, padded to 256 after loading.
    layer.w2_weight.data = torch.nn.functional.pad(
        layer.w2_weight.data, (0, 0, 0, method.gemm2_info.pad_k - intermediate)
    )
    expert = batch.IluMoEBatchedExpertsW4A8NN.__new__(batch.IluMoEBatchedExpertsW4A8NN)
    expert.format, expert.group_size = "NN", -1
    expert.w13_input_scale = expert.w2_input_scale = expert.swiglu_limit = None
    expert.swiglu_limit = limit
    expert.gemm1_info, expert.gemm2_info = method.gemm1_info, method.gemm2_info
    expert.quant_config = method.get_fused_moe_quant_config(layer)
    capacity, experts, hidden = 5, 4, 256
    x_cpu = torch.randint(-32, 33, (experts, capacity, hidden), dtype=torch.int8)
    sa_cpu = torch.full((experts, capacity), 0.02)
    x, sa = x_cpu.cuda(), sa_cpu.cuda()
    if flattened:
        x = x.reshape(-1, hidden)
    counts = torch.tensor([2, 0, 5, 1], dtype=torch.int32, device="cuda")
    output = torch.empty((experts, capacity, hidden), device="cuda", dtype=dtype)
    ids = torch.zeros((1, 1), device="cuda", dtype=torch.int32)

    def launch():
        return expert.apply(
            output,
            x,
            layer.w13_weight,
            layer.w2_weight,
            torch.ones_like(ids, dtype=torch.float32),
            ids,
            MoEActivation.SILU,
            experts,
            None,
            sa,
            None,
            None,
            None,
            SimpleNamespace(expert_num_tokens=counts),
        )

    native = batch.ixff.moe_w4a8_group_gemm_ixmm
    layouts = []
    activation_counts = []

    def checked(*args, **kwargs):
        assert kwargs["backend"] == "ixmm"
        layouts.append(kwargs["group_layout"])
        return native(*args, **kwargs)

    activation_name = "moe_group_gemm_swiglu_quant"
    native_activation = getattr(batch.ixff, activation_name)

    def checked_activation(*args, **kwargs):
        activation_counts.append(kwargs["expert_num_tokens"])
        assert "output_format" not in kwargs
        assert kwargs["max_num_tokens_per_expert"] == capacity
        assert kwargs["limit"] == limit
        return native_activation(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(batch.ixff, "moe_w4a8_group_gemm_ixmm", checked)
        patch.setattr(batch.ixff, activation_name, checked_activation)
        launch()
    assert len(layouts) == 2 and layouts[0] is layouts[1]
    assert len(activation_counts) == 1
    assert torch.equal(activation_counts[0], counts)
    assert layouts[0].is_cuda
    assert layouts[0].cpu().tolist() == [[0, 2], [5, 5], [10, 15], [15, 16]]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for active in ([2, 0, 5, 1], [0, 0, 0, 0], [0, 3, 1, 5]):
        counts.copy_(torch.tensor(active, device="cuda", dtype=torch.int32))
        output.fill_(float("nan"))
        graph.replay()
        expected = torch.full_like(output, float("nan"), device="cpu")
        for e, count in enumerate(active):
            if count == 0:
                continue
            w13 = raw[0][e].float() * (scales[0][e] * 18).T
            w2 = raw[1][e].float() * (scales[1][e] * 18).T
            z = ((x_cpu[e, :count].float() @ w13.T) * sa_cpu[e, :count, None]).to(dtype)
            gate, up = z.float().chunk(2, -1)
            if limit is not None:
                gate = gate.clamp(max=limit)
                up = up.clamp(-limit, limit)
            act = torch.nn.functional.silu(gate) * up
            scale = act.abs().amax(-1, keepdim=True) / 127
            inv = torch.where(scale > 0, scale.reciprocal(), 0)
            q = (act * inv).round().clamp(-127, 127)
            expected[e, :count] = ((q @ w2.T) * scale).to(dtype)
        # Inactive rows deliberately remain poisoned: no full-buffer clear.
        torch.testing.assert_close(
            output.cpu(), expected, atol=0.01, rtol=0.02, equal_nan=True
        )
        for e, count in enumerate(active):
            assert torch.isfinite(output[e, :count]).all()
            assert torch.isnan(output[e, count:]).all()
