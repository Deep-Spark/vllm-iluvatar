"""Real device pointer arithmetic past 4 GiB (requires about 4.3 GiB free)."""

import pytest
import torch

from vllm_iluvatar.ilu_util.arch import is_ivcore30


@pytest.mark.skipif(not is_ivcore30(), reason="only supported on ivcore30")
def test_sparse_fp8_cache_past_4gib_matches_small_cache():
    from ixformer.inference.functions import deepseekv41 as formal_ops
    from ixformer.inference.functions.deepseek import fp8_cache as shared_ops

    # Page's byte offset exceeds uint32, not merely the token position.
    page = 2**32 // (64 * 584) + 1
    required = (page + 1) * 64 * 584
    free, _ = torch.cuda.mem_get_info()
    if free < required + 512 * 2**20:
        pytest.skip("large physical-cache test needs 4.5 GiB free")
    big = torch.empty(page + 1, 64, 584, device="cuda", dtype=torch.uint8)
    small = torch.zeros(2, 64, 584, device="cuda", dtype=torch.uint8)
    big[0].zero_()
    big[page].zero_()
    kv = torch.randn(512, device="cuda") * 0.1
    fp8, scales = shared_ops._fp8_quantize_nope(kv[:448])
    for cache, p in [(big, page), (small, 1)]:
        shared_ops._pack_fp8_token_to_cache(
            cache, p, 0, 64, fp8, kv[448:].bfloat16(), scales
        )
    q, qs = formal_ops.dsv41_quant_mla_q_packed_fp8(
        torch.randn(1, 32, 512, device="cuda", dtype=torch.bfloat16)
    )
    idx = torch.full((1, 256), -1, device="cuda", dtype=torch.int32)
    lens = torch.ones(1, device="cuda", dtype=torch.int32)
    idx[0, 0] = 64
    expected = formal_ops.dsv41_sparse_mla_packed_fp8(q, qs, small, idx, 0.04, lens)
    idx[0, 0] = page * 64
    actual = formal_ops.dsv41_sparse_mla_packed_fp8(q, qs, big, idx, 0.04, lens)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = formal_ops.dsv41_sparse_mla_packed_fp8(q, qs, big, idx, 0.04, lens)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, expected, rtol=0, atol=0)
