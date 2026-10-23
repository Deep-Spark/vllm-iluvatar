# SPDX-License-Identifier: Apache-2.0
"""HY4 prefill/decode share the sink-capable ixformer SparseFlash facade."""

import ixformer.inference.functions as ops
import pytest
import torch

from tests.conftest import requires_cuda
from vllm_iluvatar.models.hy_v4.flashmla_sparse import HYV4FlashMLASparseImpl

SCALE = 256**-0.5


def _impl(sink):
    impl = HYV4FlashMLASparseImpl.__new__(HYV4FlashMLASparseImpl)
    impl.attn_sink = sink
    impl.softmax_scale = SCALE
    return impl


@pytest.mark.parametrize("tokens", [1, 4, 64, 256])
@pytest.mark.parametrize("has_sink", [False, True])
@pytest.mark.parametrize("has_lengths", [False, True])
def test_prefill_and_decode_forward_sparse_flash_contract(
    monkeypatch, tokens, has_sink, has_lengths
):
    q = torch.zeros(tokens, 16, 576)
    kv = torch.zeros(32, 64, 576)
    idx = torch.ones(tokens, 128, dtype=torch.int32)
    idx[:, 17:] = -1
    lengths = torch.full((tokens,), 17, dtype=torch.int32) if has_lengths else None
    sink = torch.zeros(16) if has_sink else None
    expected_output = torch.zeros(tokens, 16, 512)
    calls = []

    def sparse_flash(query, cache, indices, scale, **kwargs):
        calls.append(kwargs)
        assert query is q
        assert cache.shape == (2048, 1, 576)
        assert cache.data_ptr() == kv.data_ptr()
        assert indices.shape == (tokens, 1, 128)
        assert indices.data_ptr() == idx.data_ptr()
        assert scale == SCALE
        assert kwargs["v_head_dim"] == 512
        assert kwargs["backend"] == "ixinfer"
        assert kwargs["attn_sink"] is sink
        if has_lengths:
            assert kwargs["topk_length"] is lengths
        else:
            torch.testing.assert_close(
                kwargs["topk_length"], torch.full((tokens,), 17, dtype=torch.int32)
            )
        return expected_output

    monkeypatch.setattr(ops, "sparse_flash_attn", sparse_flash)
    actual = _impl(sink)._bf16_flash_mla_kernel(q, kv, idx, lengths)
    assert actual is expected_output
    assert len(calls) == 1


def test_empty_queries_do_not_launch_sparse_flash(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("an empty query must not launch SparseFlash")

    monkeypatch.setattr(ops, "sparse_flash_attn", unexpected)
    q = torch.empty(0, 16, 576, dtype=torch.bfloat16)
    result = _impl(None)._bf16_flash_mla_kernel(
        q, torch.empty(1, 576), torch.empty(0, 128, dtype=torch.int32)
    )
    assert result.shape == (0, 16, 512)
    assert result.dtype == q.dtype


@requires_cuda
@pytest.mark.parametrize("tokens,heads", [(1, 8), (4, 16), (64, 16), (256, 16)])
@pytest.mark.parametrize("sink_kind", ["none", "finite", "infinities"])
@torch.inference_mode()
def test_sparse_flash_backend_matches_fp32(tokens, heads, sink_kind):
    torch.manual_seed(67)
    q = torch.randn(tokens, heads, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(513, 1, 576, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(-4, 518, (tokens, 1, 128), device="cuda", dtype=torch.int32)
    lengths = torch.full((tokens,), 128, device="cuda", dtype=torch.int32)
    if tokens > 1:
        lengths[:3] = torch.tensor([0, 17, 64], device="cuda", dtype=torch.int32)
    sink = None if sink_kind == "none" else torch.randn(heads, device="cuda")
    if sink_kind == "infinities":
        sink[0] = float("inf")
        sink[1] = -float("inf")
    expected = ops.dsa_flash_mla_sparse_fwd_diff_kv(
        q, kv, idx, SCALE, topk_length=lengths, attn_sink=sink, backend="torch"
    )
    actual = _impl(sink)._bf16_flash_mla_kernel(q, kv, idx, lengths)
    assert torch.isfinite(actual).all()
    # BF16 TCU operator regression; this tolerance is not model-quality acceptance.
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.01)
    if tokens > 1:
        assert torch.count_nonzero(actual[0]) == 0
    if sink_kind == "infinities":
        assert torch.count_nonzero(actual[:, 0]) == 0


@requires_cuda
@torch.inference_mode()
def test_sparse_flash_graph_replays_updated_kv_sink_and_lengths():
    q = torch.zeros(4, 16, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.full((513, 1, 576), 2.0, device="cuda", dtype=torch.bfloat16)
    kv[0] = 1000
    idx = torch.ones(4, 1, 128, device="cuda", dtype=torch.int32)
    idx[2, 0, 3] = -1
    idx[2, 0, 7] = 999
    lengths = torch.tensor([0, 17, 64, 128], device="cuda", dtype=torch.int32)
    sink = torch.zeros(16, device="cuda")
    sink[0], sink[1] = float("inf"), -float("inf")
    impl = _impl(sink)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        impl._bf16_flash_mla_kernel(q, kv, idx, lengths)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = impl._bf16_flash_mla_kernel(q, kv, idx, lengths)
    kv[1:] = -3
    sink[2:] = 1.5
    lengths.copy_(torch.tensor([0, 3, 11, 96], device="cuda", dtype=torch.int32))
    graph.replay()
    expected = ops.dsa_flash_mla_sparse_fwd_diff_kv(
        q, kv, idx, SCALE, topk_length=lengths, attn_sink=sink, backend="torch"
    )
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0)
    assert torch.count_nonzero(actual[0]) == 0
    assert torch.count_nonzero(actual[:, 0]) == 0
