# SPDX-License-Identifier: Apache-2.0
"""BI iHC dispatch, FP32 projection preservation, numerics and graph replay."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tests.conftest import requires_cuda
from vllm_iluvatar.ilu_util import arch
from vllm_iluvatar.models.hy_v4 import hc


class _Projection(nn.Linear):
    def __init__(self, input_size, output_size, params_dtype, bias, prefix):
        super().__init__(input_size, output_size, bias=bias, dtype=params_dtype)
        nn.init.normal_(self.weight, std=6e-3)

    def forward(self, x):
        return super().forward(x), None


def _layers(monkeypatch, hidden):
    monkeypatch.setattr(hc, "ReplicatedLinear", _Projection)
    config = SimpleNamespace(
        rms_norm_eps=1e-5,
        hc_mult=4,
        hidden_size=hidden,
    )
    return (
        hc.HYV4HCPreLayer(config, hidden).cuda(),
        hc.HYV4HCPostLayer(config).cuda(),
        hc.HYV4HCHeadLayer(config, hidden).cuda(),
    )


@pytest.mark.parametrize("cuda_arch", ["ivcore11", "ivcore30", None])
def test_arch_dispatch_and_disable(monkeypatch, cuda_arch):
    monkeypatch.setattr(arch, "cuda_arch", lambda: cuda_arch)
    monkeypatch.delenv("VLLM_ILU_HYV4_FUSE_IHC", raising=False)
    assert (hc._ihc_ops() is not None) == (cuda_arch is not None)
    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_IHC", "0")
    assert hc._ihc_ops() is None


def test_bi_projection_keeps_original_fp32_storage(monkeypatch):
    monkeypatch.setattr(arch, "is_ivcore11", lambda: True)
    weight = torch.randn(8, 16, dtype=torch.float32)
    assert hc._native_projection(nn.Module(), weight) is weight


@pytest.mark.parametrize(
    "hc_mult,hidden_dim",
    [(2, 4096), (4, 2048)],
)
def test_non_native_model_config_warns_once(monkeypatch, hc_mult, hidden_dim):
    warning_once = Mock()
    monkeypatch.setattr(hc.logger, "warning_once", warning_once)
    residual = torch.empty(1, hc_mult, hidden_dim, dtype=torch.bfloat16)

    assert not hc._native_shape(
        residual, hc_mult=hc_mult, hidden_dim=hidden_dim
    )
    warning_once.assert_called_once()
    assert "model config" in warning_once.call_args.args[0]


@requires_cuda
@pytest.mark.parametrize("hidden", [4096, 6144])
@pytest.mark.parametrize("tokens", [1, 4, 64, 65, 256])
@torch.inference_mode()
def test_bi_ihc_matches_eager(monkeypatch, hidden, tokens):
    if not arch.is_ivcore11():
        pytest.skip("BI dispatch boundary test")
    torch.manual_seed(17)
    pre, post, head = _layers(monkeypatch, hidden)
    # Exercise nontrivial gates and FP32 weights, not just initialized biases.
    pre.hc_scale.copy_(torch.tensor([0.25, 0.5], device="cuda"))
    head.hc_head_scale.fill_(0.25)
    residual = torch.randn(tokens, 4, hidden, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")

    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_IHC", "0")
    y_ref, gates_ref = pre(residual)
    post_ref = post(x, residual, gates_ref)
    head_ref = head(residual)
    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_IHC", "1")
    assert hc._native_projection_shape(
        residual, hc_mult=pre.hc_mult, hidden_dim=pre.hidden_dim
    ) == (tokens <= 64)
    y, gates = pre(residual)
    torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(gates, gates_ref, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(post(x, residual, gates_ref), post_ref,
                               atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(head(residual), head_ref, atol=2e-2, rtol=1e-2)
    # The large-batch projection must retain the exact eager path.
    if tokens > 64:
        assert torch.equal(y, y_ref)
        assert torch.equal(gates, gates_ref)
        assert torch.equal(head(residual), head_ref)


@requires_cuda
@torch.inference_mode()
def test_bi_ihc_graph_replay_and_unsupported_inputs(monkeypatch):
    if not arch.is_ivcore11():
        pytest.skip("BI graph test")
    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_IHC", "1")
    pre, post, head = _layers(monkeypatch, 6144)
    residual = torch.randn(1, 4, 6144, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(1, 6144, dtype=torch.bfloat16, device="cuda")

    def run():
        y, gates = pre(residual)
        return y, head(post(x, residual, gates))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    residual.normal_()
    x.normal_()
    graph.replay()
    for got, expected in zip(actual, run()):
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
    native_shape = {
        "hc_mult": pre.hc_mult,
        "hidden_dim": pre.hidden_dim,
    }
    assert not hc._native_projection_shape(residual.float(), **native_shape)
    assert not hc._native_projection_shape(residual[:, :, ::2], **native_shape)
    assert not hc._native_projection_shape(
        residual[:, :, :1024].contiguous(), **native_shape
    )
    assert not hc._native_projection_shape(residual[:0], **native_shape)
