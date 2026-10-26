# SPDX-License-Identifier: Apache-2.0
"""Framework owns the FP8 producer/consumer and TP reduction contracts."""

import pytest
import torch
from conftest import requires_cuda

from vllm_iluvatar.model_executor.layers.activation import swigluoai_fp8_linear
from vllm_iluvatar.models.minimax_m3.mxfp8 import model as model_module


@requires_cuda
def test_composition_graph_replay_and_empty_input():
    x = torch.randn(8, 768, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, 384, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.full((128,), 0.02, device="cuda")
    assert swigluoai_fp8_linear(x[:0], w, scale).shape == (0, 128)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        swigluoai_fp8_linear(x, w, scale, use_gemv=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = swigluoai_fp8_linear(x, w, scale, use_gemv=True)
    for _ in range(3):
        x.normal_()
        graph.replay()
        expected = swigluoai_fp8_linear(x, w, scale, use_gemv=True)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "reduce,tp,core30,converted",
    [
        (True, 2, True, True),
        (False, 2, True, True),
        (True, 1, True, True),
        (True, 2, False, True),
        (True, 2, True, False),
    ],
)
def test_mlp_preserves_tp_reduction_and_fallback(
    monkeypatch, reduce, tp, core30, converted
):
    mlp = model_module.MiniMaxM3MLP.__new__(model_module.MiniMaxM3MLP)
    torch.nn.Module.__init__(mlp)
    mlp.gate_up_proj = lambda x: (x, None)
    mlp.act_fn = lambda x: x[..., :4]
    mlp._swiglu_parameters = dict(limit=7.0, alpha=1.702, beta=1.0)
    calls = []

    class Down:
        input_is_parallel = True
        bias = None
        reduce_results = reduce
        tp_size = tp
        weight = object()
        weight_scale = object()

        def __call__(self, x):
            calls.append("fallback")
            return x, None

    down = Down()
    if converted:
        down._ilu_mxfp8_can_use_gemv = True
    mlp.down_proj = down
    monkeypatch.setattr(model_module, "is_ivcore30", lambda: core30)

    def fused(x, *args, **kwargs):
        assert x.shape == (6, 8) and kwargs["use_gemv"]
        calls.append("fused")
        return x[:, :4].contiguous()

    monkeypatch.setattr(model_module, "swigluoai_fp8_linear", fused)

    def allreduce(x):
        calls.append("allreduce")
        return x * tp

    monkeypatch.setattr(model_module, "tensor_model_parallel_all_reduce", allreduce)
    x = torch.ones(2, 3, 8)
    result = mlp(x)
    assert result.shape == (2, 3, 4)
    if core30 and converted:
        expected = ["fused"] + (["allreduce"] if reduce and tp > 1 else [])
        assert calls == expected
        assert torch.all(result == (tp if reduce and tp > 1 else 1))
    else:
        assert calls == ["fallback"]
