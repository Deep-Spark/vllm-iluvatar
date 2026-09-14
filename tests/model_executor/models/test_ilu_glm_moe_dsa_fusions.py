# SPDX-License-Identifier: Apache-2.0
"""Shared-expert SwiGLU quantization integration."""

from unittest.mock import Mock

import ixformer.inference.functions as ixf_ops
import pytest
import torch
import torch.nn.functional as F

from vllm_iluvatar.models.glm_moe_dsa import model as model_mod


def make_mlp(monkeypatch, *, enabled=True, static=False, symmetric=True, w8a8=True):
    scheme = model_mod.IluCompressedTensorsW8A8Int8.__new__(
        model_mod.IluCompressedTensorsW8A8Int8
    )
    scheme.is_static_input_scheme = static
    scheme.input_symmetric = symmetric

    class GateUp(torch.nn.Module):
        def forward(self, x):
            return x, None

    class Down(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scheme = scheme if w8a8 else None

        def forward(self, x):
            # Keep the real MLP's output visible to test the linear input
            # contract; the W8A8 scheme accepts this tuple without re-quantizing.
            return x, None

    monkeypatch.setattr(
        model_mod, "MergedColumnParallelLinear", lambda *a, **k: GateUp()
    )
    monkeypatch.setattr(model_mod, "RowParallelLinear", lambda *a, **k: Down())
    monkeypatch.setattr(model_mod, "SiluAndMul", lambda: torch.nn.Identity())
    return model_mod.IluGlmMoeDsaMLP(
        hidden_size=6144,
        intermediate_size=2048,
        hidden_act="silu",
        fuse_act_quant=enabled,
    )


@pytest.mark.parametrize(
    "enabled,static,symmetric,w8a8,fused",
    [
        (True, False, True, True, True),
        (False, False, True, True, False),
        (True, True, True, True, False),
        (True, False, False, True, False),
        (True, False, True, False, False),
    ],
)
def test_shared_quant_only_for_dynamic_symmetric_w8a8(
    monkeypatch, enabled, static, symmetric, w8a8, fused
):
    mlp = make_mlp(
        monkeypatch, enabled=enabled, static=static, symmetric=symmetric, w8a8=w8a8
    )
    x = torch.zeros(2, 4096, dtype=torch.bfloat16)
    q, scales = torch.zeros(2, 2048, dtype=torch.int8), torch.ones(2)
    quant = Mock(return_value=(q, scales))
    monkeypatch.setattr(ixf_ops, "silu_and_mul_quant", quant)
    result = mlp(x)
    if fused:
        quant.assert_called_once_with(x)
        assert result[0] is q and result[1] is scales
        assert result[2] == torch.bfloat16
    else:
        quant.assert_not_called()
        assert result is x


@pytest.mark.parametrize("tokens", [8, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_shared_quant_numerics_and_graph(monkeypatch, cuda_device, tokens, dtype):
    mlp = make_mlp(monkeypatch)
    x = torch.zeros(tokens, 4096, device=cuda_device, dtype=dtype)
    mlp(x)  # Warm up native dispatch before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, scales, out_dtype = mlp(x)
    assert out_dtype == dtype
    for seed in [31, 67]:
        gen = torch.Generator().manual_seed(seed)
        x_cpu = (torch.randn(tokens, 4096, generator=gen) * 3).to(dtype)
        x_cpu[0].zero_()
        x.copy_(x_cpu)
        graph.replay()
        gate, up = x_cpu.float().chunk(2, -1)
        product = F.silu(gate) * up
        amax = product.abs().amax(-1)
        inv = torch.where(amax > 0, 127 / amax, 0)
        expected = (product * inv[:, None]).round().clamp(-127, 127).to(torch.int8)
        assert (q.cpu().int() - expected.int()).abs().max() <= 1
        torch.testing.assert_close(scales.cpu(), amax / 127, rtol=2e-5, atol=1e-7)
        eager_q, eager_s, _ = mlp(x)
        assert torch.equal(eager_q, q)
        assert torch.equal(eager_s, scales)
