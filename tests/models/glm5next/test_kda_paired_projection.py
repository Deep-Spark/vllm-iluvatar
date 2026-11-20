# SPDX-License-Identifier: Apache-2.0
"""Numerical and reload checks for paired KDA projections."""

import ixformer.inference.functions as ixfops
import pytest
import torch
from ixformer.cuda import cuda_arch
from torch import nn
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes.ilu_compressed_tensors_w8a8_int8 import (
    IluW8A8OnlineLinearMethod,
)
from vllm_iluvatar.models.glm5next.kda import IluGlm5NextLinearAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


class DenseProjection(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5,
            requires_grad=False,
        )
        self.bias = None
        self.gather_output = False
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, x):
        if isinstance(self.quant_method, IluW8A8OnlineLinearMethod):
            return self.quant_method.apply(self, x), None
        return ixfops.linear(x, self.weight), None


def make_attention(quantized, heads=32, hidden=256):
    layer = object.__new__(IluGlm5NextLinearAttention)
    nn.Module.__init__(layer)
    layer.head_dim = 128
    layer.local_num_heads = heads
    layer.local_projection_size = heads * layer.head_dim
    n = layer.local_projection_size
    layer.in_proj_qkvbfg_a = DenseProjection(3 * n + heads + 256, hidden)
    layer.f_b_proj = DenseProjection(n, 128)
    layer.g_b_proj = DenseProjection(n, 128)
    layer.register_buffer("_fg_b_weight", None, persistent=False)
    if quantized:
        proj = layer.in_proj_qkvbfg_a
        proj.quant_method = IluW8A8OnlineLinearMethod(proj.quant_method, 3 * n)
        proj.quant_method.process_weights_after_loading(proj)
    return layer


def reference(layer, x):
    projected = layer.in_proj_qkvbfg_a(x)[0]
    qkv, beta, f, g = projected.split(
        [3 * layer.local_projection_size, layer.local_num_heads, 128, 128], dim=-1
    )
    shape = (-1, layer.local_num_heads, 128)
    return qkv, beta, layer.f_b_proj(f)[0].view(shape), layer.g_b_proj(g)[0].view(shape)


@pytest.fixture(scope="module", params=[(False, 32), (True, 32), (True, 16)])
def attention(request):
    torch.manual_seed(81)
    layer = make_attention(*request.param)
    layer.process_weights_after_loading(torch.bfloat16)
    return layer


@pytest.mark.parametrize("m", [1, 16, 48, 96, 512, 2048])
def test_paired_projection_graph_matches_original(attention, m):
    x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
    attention._project_qkv_and_gates(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = attention._project_qkv_and_gates(x)
    for factor in (1.0, -0.5):
        x.mul_(factor)
        graph.replay()
        expected = reference(attention, x)
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert torch.isfinite(a).all()
            torch.testing.assert_close(
                a, b, rtol=0 if i < 2 else 0.02, atol=0 if i < 2 else 0.02
            )


@pytest.mark.parametrize(
    "replace_storage",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                cuda_arch != "ivcore30",
                reason="paired weight storage preservation requires CQ ixmm",
            ),
        ),
    ],
)
def test_parameter_reload_updates_graph(replace_storage):
    torch.manual_seed(82)
    layer = make_attention(True)
    layer.process_weights_after_loading(torch.bfloat16)
    if replace_storage:
        assert layer._fg_b_weight is not None
    x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
    layer._project_qkv_and_gates(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer._project_qkv_and_gates(x)
    # A weight reload writes through the existing named Parameters. The
    # captured batched operation must observe both updates, not a stale copy.
    with torch.no_grad():
        layer.f_b_proj.weight.mul_(0.5)
        layer.g_b_proj.weight.mul_(-0.25)
    graph.replay()
    for a, b in zip(actual, reference(layer, x)):
        torch.testing.assert_close(a, b, rtol=0.02, atol=0.02)
    if not replace_storage:
        return
    # Finalization must keep the captured addresses valid across later reloads,
    # including loaders that replace Parameter storage instead of copying it.
    layer.process_weights_after_loading(torch.bfloat16)
    with torch.no_grad():
        layer.f_b_proj.weight.data = layer.f_b_proj.weight.detach().clone() * -0.5
        layer.g_b_proj.weight.data = layer.g_b_proj.weight.detach().clone() * 0.25
    layer.process_weights_after_loading(torch.bfloat16)
    graph.replay()
    for a, b in zip(actual, reference(layer, x)):
        torch.testing.assert_close(a, b, rtol=0.02, atol=0.02)


def test_non_ixmm_backend_keeps_original_projection(monkeypatch):
    from vllm_iluvatar.models.glm5next import kda

    torch.manual_seed(83)
    layer = make_attention(True)
    with monkeypatch.context() as patch:
        patch.setattr(kda.ix_registry, "resolve", lambda op: "ixinfer")
        layer.process_weights_after_loading(torch.bfloat16)
    x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
    for a, b in zip(layer._project_qkv_and_gates(x), reference(layer, x)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
