# SPDX-License-Identifier: Apache-2.0
"""Preserve mHC mixing while reusing the FFN RMSNorm INT8 output."""

import pytest
import torch
from conftest import requires_corex30

from vllm_iluvatar.models.deepseek_v41.model import IluDeepseekV41DecoderLayer


@requires_corex30
@pytest.mark.parametrize("rows", [1, 80])
def test_mhc_norm_quant_and_graph(rows):
    torch.manual_seed(43)
    decoder = IluDeepseekV41DecoderLayer.__new__(IluDeepseekV41DecoderLayer)
    torch.nn.Module.__init__(decoder)
    decoder.rms_norm_eps = decoder.hc_eps = 1e-6
    decoder.hc_mult, decoder.hc_sinkhorn_iters, decoder.hc_post_alpha = 4, 20, 2.0
    decoder._hc_fn_io_cache = {}
    norm = torch.nn.Module()
    norm.weight = torch.nn.Parameter(
        torch.ones(5120, device="cuda", dtype=torch.bfloat16), requires_grad=False
    )
    fn = torch.randn(24, 20480, device="cuda") * 0.001
    scale = torch.ones(3, device="cuda")
    base = torch.zeros(24, device="cuda")
    x = torch.randn(rows, 4, 5120, device="cuda", dtype=torch.bfloat16)
    pre = torch.full((rows, 4), 0.25, device="cuda")

    def call(quant):
        sqrsum = x.float().square().sum((1, 2))
        return decoder._hc_pre_mixes(
            x, "ffn", fn, scale, base, pre, sqrsum, norm, want_int8=quant
        )

    plain, quantized = call(False), call(True)
    for index in (0, 1, 3):
        torch.testing.assert_close(plain[index], quantized[index], atol=0, rtol=0)
    normed = quantized[2].float()
    error = (normed - plain[2].float()).norm() / plain[2].float().norm()
    assert error < 0.005
    restored = quantized[4].float() * quantized[5].reshape(-1, 1)
    assert (restored - normed).norm() / normed.norm() < 0.02
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call(True)
    x.neg_()
    graph.replay()
    expected = call(True)
    for actual, reference in zip(captured, expected):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize(
    "converted,override", [(True, None), (False, None), (True, "deepep_low_latency")]
)
def test_moe_selects_runner_from_quant_method(converted, override, monkeypatch):
    from types import SimpleNamespace

    from vllm_iluvatar.model_executor.layers.quantization.backends.w4a8.nn_moe import (
        IluW4A8NNMoEMethod,
    )
    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (
        IluCompressedTensorsW8A8Int8,
    )
    from vllm_iluvatar.models.deepseek_v41 import model as mod

    quant = object.__new__(IluW4A8NNMoEMethod) if converted else object()
    runner = torch.nn.Module()
    runner._quant_method = quant
    runner.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(all2all_backend="allgather_reducescatter")
    )
    shared = torch.nn.Module()
    scheme = object.__new__(IluCompressedTensorsW8A8Int8) if converted else None
    shared.gate_up_proj = SimpleNamespace(scheme=scheme)
    shared.down_proj = SimpleNamespace(scheme=scheme)
    enabled = []
    shared.act_fn = SimpleNamespace(enable_fuse_quant=lambda: enabled.append(True))
    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        mod, "GateLinear", lambda **kwargs: torch.nn.Linear(256, 8, bias=False)
    )
    monkeypatch.setattr(mod, "FusedMoEFactory", lambda **kwargs: runner)
    monkeypatch.setattr(mod, "IluDeepseekV4Fp8MLP", lambda **kwargs: shared)
    monkeypatch.setattr(mod.IluDeepseekV41MoE, "init_ep", lambda self: None)
    monkeypatch.setattr(mod, "_DECODE_EP_BACKEND", override)
    monkeypatch.setattr(mod.envs, "VLLM_ILU_FUSE_NORM_QUANT", True)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=256,
                n_routed_experts=8,
                n_shared_experts=1,
                moe_intermediate_size=256,
                hidden_act="silu",
                swiglu_limit=7.0,
                num_experts_per_tok=2,
                norm_topk_prob=True,
            )
        ),
        quant_config=None,
        parallel_config=SimpleNamespace(
            enable_eplb=False, eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
    )
    moe = mod.IluDeepseekV41MoE(config)
    assert moe._use_moe_runner == (converted and override is None)
    assert moe.reuse_norm_quant == (converted and override is None)
    assert enabled == ([True] if converted else [])
