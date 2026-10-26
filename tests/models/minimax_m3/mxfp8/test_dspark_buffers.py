# SPDX-License-Identifier: Apache-2.0
"""Context layout, native cache consumption and graph buffer lifetime regressions."""

from types import SimpleNamespace

import ixformer.inference.functions as ixf
import pytest
import torch
import torch.nn.functional as F
from conftest import requires_cuda

from vllm_iluvatar.model_executor.layers.layernorm import (
    batched_gemma_rms_norm as normalize_context_k,
)
from vllm_iluvatar.models.minimax_m3.dspark import IluMiniMaxDSparkModel
from vllm_iluvatar.v1.worker.gpu.spec_decode.dspark.buffers import IluTargetDraftBuffer


@pytest.mark.parametrize("tokens,heads", [(0, 1), (1, 1), (8, 1), (17, 2), (65, 4)])
@requires_cuda
@torch.inference_mode()
def test_strided_context_norm_matches_independent_math_and_native(tokens, heads):
    # Production D=128; TP4/TP2/TP1 have 1/2/4 local KV heads.
    torch.manual_seed(4301)
    packed = torch.randn(5, tokens, 2, heads, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    if tokens:
        packed[:, 0, 0] = 0
    k = packed[:, :, 0]
    before = packed.clone()
    got = normalize_context_k(k, w, 1e-6)
    assert got.is_contiguous() and torch.equal(packed, before)
    x = k.cpu().double()
    ref = (
        x
        * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        * (1 + w.cpu().double()[:, None, None])
    ).to(torch.bfloat16)
    # One BF16 rounding unit, independently checked per output element.
    torch.testing.assert_close(got.cpu(), ref, rtol=0.008, atol=1e-5)
    if tokens:
        old = torch.stack([ixf.gemma_rms_norm(k[i], w[i], eps=1e-6) for i in range(5)])
        torch.testing.assert_close(got, old, rtol=0.008, atol=1e-5)
        assert torch.count_nonzero(got[:, 0]) == 0
        # Different layer weights must be distinguishable.
        if tokens > 1:
            wrong = normalize_context_k(k, w.flip(0), 1e-6)
            assert not torch.allclose(got, wrong, rtol=0.008, atol=1e-5)


@pytest.mark.parametrize("tokens", [1, 8, 17])
@requires_cuda
@torch.inference_mode()
def test_kv_projection_direct_layout_and_strided_cache_writer(tokens):
    torch.manual_seed(4302)
    model = IluMiniMaxDSparkModel.__new__(IluMiniMaxDSparkModel)
    torch.nn.Module.__init__(model)
    model.hidden_norm = torch.nn.Identity()
    # Production TP4 draft: 6144 input, one K/V head of 128 per layer.
    weights, scales = [], []
    for layer in range(5):
        w = torch.randn(256, 6144, device="cuda", dtype=torch.bfloat16) * 0.01
        q, s = ixf.fp8_per_token_dynamic_scaled(w, backend="ixformer_native")
        weights.append(q)
        scales.append(s)
    model._context_kv_weights, model._context_kv_scales = weights, scales
    x = torch.randn(tokens, 6144, device="cuda", dtype=torch.bfloat16)
    qx, sx = ixf.fp8_per_token_dynamic_scaled(x, backend="ixformer_native")
    ref = []
    for w, s in zip(weights, scales):
        y = torch.empty(tokens, 256, device="cuda", dtype=torch.bfloat16)
        op = ixf.fp8_per_channel_gemv if tokens <= 16 else ixf.fp8_per_channel_mm
        op(qx, w, sx, s, y, backend="ixmm")
        ref.append(y.view(tokens, 2, 1, 128))
    expected = torch.stack(ref, 1).permute(2, 1, 0, 3, 4).contiguous()
    k, v = model._project_context_kv(x, tokens, 5, 1, 128)
    assert torch.equal(k, expected[0]) and torch.equal(v, expected[1])
    assert k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
    assert v.stride(1) == 256
    # Native FP8 cache writer must honor V's larger token stride and negative slots.
    import vllm_iluvatar._C  # noqa: F401

    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    slots[0] = -1
    cache = torch.zeros(2, 1, 128, 256, device="cuda", dtype=torch.float8_e4m3fn)
    gold = cache.clone()
    scale = torch.ones(1, device="cuda")
    for layer in range(5):
        torch.ops._C.reshape_and_cache_fp8(
            k[layer],
            v[layer],
            cache[..., :128],
            cache[..., 128:],
            slots,
            "fp8",
            scale,
            scale,
        )
        torch.ops._C.reshape_and_cache_fp8(
            expected[0, layer],
            expected[1, layer],
            gold[..., :128],
            gold[..., 128:],
            slots,
            "fp8",
            scale,
            scale,
        )
        assert torch.equal(cache.view(torch.int8), gold.view(torch.int8))
        assert torch.count_nonzero(cache[:, :, 0].float()) == 0


@requires_cuda
@torch.inference_mode()
def test_context_norm_graph_updates_values_without_repacking():
    torch.manual_seed(4303)
    packed = torch.randn(5, 8, 2, 1, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    normalize_context_k(packed[:, :, 0], weights, 1e-6)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = normalize_context_k(packed[:, :, 0], weights, 1e-6)
    ptr = got.data_ptr()
    for live in (8, 3, 0, 8):
        packed.normal_()
        packed[:, live:] = 0
        graph.replay()
        ref = normalize_context_k(packed[:, :, 0], weights, 1e-6)
        assert got.data_ptr() == ptr
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("width,out_width", [(128, 256), (6144, 6144)])
@requires_cuda
@torch.inference_mode()
def test_target_draft_bridge_eager_and_two_graph_buckets(width, out_width):
    torch.manual_seed(4304)
    output = torch.empty(8, out_width, device="cuda", dtype=torch.bfloat16)
    fc = SimpleNamespace(
        input_size=5 * width,
        weight=torch.randn(out_width, 5 * width, device="cuda", dtype=torch.bfloat16)
        * 0.001,
    )
    bridge = IluTargetDraftBuffer(fc, output, 5)
    states = [
        torch.randn(8, width, device="cuda", dtype=torch.bfloat16) for _ in range(5)
    ]
    graphs = {}
    for count in (2, 8):
        bridge([x[:count] for x in states])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bridge([x[:count] for x in states])
        graphs[count] = graph
    # A separately captured draft consumer must retain the same input address.
    consumer = torch.cuda.CUDAGraph()
    with torch.cuda.graph(consumer):
        consumed = output * 2
    for count in (8, 2, 8, 2):
        for x in states:
            x.normal_()
        output.fill_(11)
        graphs[count].replay()
        consumer.replay()
        gold = F.linear(torch.cat([x[:count] for x in states], -1), fc.weight)
        torch.testing.assert_close(output[:count], gold, rtol=0, atol=0)
        torch.testing.assert_close(consumed[:count], gold * 2, rtol=0, atol=0)
        assert torch.all(output[count:] == 11)
    with pytest.raises(ValueError, match="layer count"):
        bridge(states[:-1])
