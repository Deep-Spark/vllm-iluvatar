# SPDX-License-Identifier: Apache-2.0
"""Projection routing/loading tests and optional GPU numerical regressions."""

import pytest
import torch
from torch import nn

from vllm_iluvatar import envs as ilu_envs
from vllm_iluvatar.ilu_util import arch
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (
    ilu_compressed_tensors_w8a8_int8 as pq,
)


@pytest.fixture(autouse=True)
def mr_arch(monkeypatch):
    monkeypatch.setattr(arch, "cuda_arch", lambda: "ivcore11")


class OriginalMethod:
    def create_weights(self, layer, weight):
        layer.weight = nn.Parameter(weight.clone(), requires_grad=False)

    def process_weights_after_loading(self, layer):
        layer.loaded = True


def make_layer(weight, rows=None, enabled=True):
    layer = nn.Module()
    original = OriginalMethod()
    layer.quant_method = original
    if enabled:
        layer.quant_method = pq.IluW8A8OnlineLinearMethod(original, quantized_rows=rows)
    layer.quant_method.create_weights(layer, weight)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.loaded
    return layer, original


@pytest.mark.parametrize(
    "mla,kda", [(False, False), (True, False), (False, True), (True, True)]
)
def test_independent_switches(monkeypatch, mla, kda):
    for name, enabled in [("MLA", mla), ("KDA", kda)]:
        key = f"VLLM_ILU_GLM5NEXT_{name}_PROJ_W8A8"
        monkeypatch.setenv(key, str(int(enabled)))
        active = ilu_envs._ILU_CANONICAL_ONLY_ENV_VARS[key]()
        layer, original = make_layer(
            torch.ones(64, 64, dtype=torch.bfloat16), enabled=active
        )
        assert active == enabled
        assert isinstance(layer.quant_method, pq.IluW8A8OnlineLinearMethod) == enabled
        if not enabled:
            assert layer.quant_method is original
            assert layer.weight.dtype == torch.bfloat16
            assert not list(layer.buffers())


def test_default_off(monkeypatch):
    for name in ("MLA", "KDA"):
        key = f"VLLM_ILU_GLM5NEXT_{name}_PROJ_W8A8"
        monkeypatch.delenv(key, raising=False)
        assert ilu_envs._ILU_CANONICAL_ONLY_ENV_VARS[key]() is False


@pytest.mark.parametrize("rows", [None, 64])
def test_weight_quantization_and_gate_preservation(rows):
    torch.manual_seed(18)
    weight = torch.randn(128, 64, dtype=torch.bfloat16)
    weight[0].zero_()
    saved = weight.clone()
    layer, _ = make_layer(weight, rows)
    n = rows or 128
    scales = saved[:n].float().abs().amax(1).clamp_min(1e-12) / 127
    expected = (saved[:n].float() / scales[:, None]).round().clamp(-127, 127)
    torch.testing.assert_close(layer.weight, expected.to(torch.int8).T.contiguous())
    torch.testing.assert_close(layer.weight_scale.flatten(), scales)
    assert layer.weight.is_contiguous() and not layer.weight.requires_grad
    assert torch.isfinite(scales).all()
    torch.testing.assert_close(weight, saved)
    if rows:
        torch.testing.assert_close(layer.bf16_tail_weight, saved[rows:])


@pytest.mark.parametrize("rows", [0, -1, 11])
def test_invalid_qkv_rows(rows):
    with pytest.raises(ValueError, match="Quantized row count"):
        make_layer(torch.ones(10, 8, dtype=torch.bfloat16), rows)


def test_reject_non_bf16():
    with pytest.raises(ValueError, match="2D BF16"):
        make_layer(torch.ones(10, 8))


@pytest.mark.parametrize("rows", [None, 64])
@pytest.mark.parametrize("use_perf", [False, True])
def test_forward_shape_and_gate(monkeypatch, rows, use_perf):
    def quantize(x, scale, azp, symmetric):
        assert scale is None and azp is None and symmetric
        scale = x.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
        q = (x.float() / scale).round().clamp(-127, 127).to(torch.int8)
        return q, scale, None

    def gemm(q, weight, sx, sw, bias, *, format, output, out_dtype):
        assert format == "NN" and bias is None
        prod = (q.float() @ weight.float()) * sx.reshape(-1, 1) * sw.reshape(1, -1)
        output.copy_(prod.to(out_dtype))
        return output

    monkeypatch.setattr(pq.ops, "scaled_int8_quant", quantize)
    monkeypatch.setattr(pq.ixf_ops, "w8a8", gemm)
    torch.manual_seed(18)
    weight = torch.randn(128, 64, dtype=torch.bfloat16)
    layer, _ = make_layer(weight, rows)
    x = torch.randn(2, 64, 3, dtype=torch.bfloat16).transpose(1, 2)
    dispatched = []
    if use_perf:

        def perf_gemm(*args, **kwargs):
            dispatched.append(True)
            return gemm(*args, **kwargs)

        # Exercise the real scheme's PERF lookup instead of bypassing it.
        layer.quant_method.scheme.perf_table["ivcore11"][(rows or 128, 64)] = {
            "format": "NN",
            "func": [(0, perf_gemm)],
        }
    output = layer.quant_method.apply(layer, x)
    assert bool(dispatched) == use_perf
    assert output.shape == (2, 3, 128)
    assert output.dtype == torch.bfloat16
    if rows:
        torch.testing.assert_close(
            output[..., rows:], torch.nn.functional.linear(x, weight[rows:])
        )
    with pytest.raises(ValueError, match="bias"):
        layer.quant_method.apply(layer, x, torch.zeros(10))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Iluvatar GPU")
@pytest.mark.parametrize("m", [4, 8, 24, 48])
@pytest.mark.parametrize(
    "n,k,rows",
    [
        (2048, 4096, None),
        (8192, 1536, None),
        (4096, 8192, None),
        (12576, 4096, 12288),
        (4096, 4096, None),
    ],
)
def test_gpu_projection_and_graph(m, n, k, rows):
    torch.manual_seed(18)
    weight = torch.randn(n, k, dtype=torch.bfloat16)
    layer, _ = make_layer(weight.cuda(), rows)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        layer.quant_method.apply(layer, x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer.quant_method.apply(layer, x)
    for factor in (1.0, -0.5):
        x.mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        # CPU GEMM uses the production activation quantizer's actual outputs.
        # PERF may keep TN checkpoint layout (N, K); NN stores the transpose.
        q = torch.empty_like(x, dtype=torch.int8)
        scale = torch.empty(m, dtype=torch.float32, device="cuda")
        pq.ixf_ops.dynamic_scaled_int8_quant(q, x, scale)
        quant_n = rows or n
        packed = layer.weight.detach().float().cpu()
        if layer.quant_method.scheme.format == "NN":
            gemm_w = packed[:, :quant_n]
        else:
            gemm_w = packed[:quant_n, :k].T
        w_scale = layer.weight_scale.detach().float().cpu().flatten()[:quant_n]
        reference = (q.cpu().float() @ gemm_w) * scale.cpu()[:, None] * w_scale
        if rows:
            gate = x.cpu().float() @ weight[rows:].float().T
            reference = torch.cat((reference, gate), dim=-1)
        reference = reference.bfloat16().float()
        actual = output.cpu().float()
        assert torch.isfinite(actual).all()
        assert ((actual - reference).norm() / reference.norm()).item() < 0.003
