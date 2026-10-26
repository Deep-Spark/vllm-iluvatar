"""Actual SWA varlen backend reads valid pages beyond a real 4 GiB backing."""

import math

import pytest
import torch
from conftest import requires_corex30

pytestmark = requires_corex30


@pytest.mark.ixinfer
@pytest.mark.parametrize("heads,groups,queries", [(64, 4, 1), (64, 4, 4), (32, 2, 4)])
def test_swa_varlen_large_packed_backing_and_graph(heads, groups, queries):
    from ixformer.contrib.vllm_flash_attn import flash_attn_varlen_func

    torch.manual_seed(191)
    page_bytes = groups * 16 * 384 * 2
    high = math.ceil(4 * 1024**3 / page_bytes) + 2
    backing = torch.empty(
        (high + 1, groups, 16, 384), device="cuda", dtype=torch.bfloat16
    )
    kc, vc = backing[..., :192], backing[..., 192:]
    assert backing.untyped_storage().nbytes() > 4 * 1024**3
    assert high * kc.stride(0) * kc.element_size() > 4 * 1024**3
    for page, offset in [(0, 0.25), (high, 1.25)]:
        kc[page].copy_(torch.randn_like(kc[page]) / 8)
        vc[page].copy_(torch.randn_like(vc[page]) / 16 + offset)
    q = torch.randn(queries, heads, 192, device="cuda", dtype=torch.bfloat16) / 8
    table = torch.tensor([[0, high]], device="cuda", dtype=torch.int32)
    cuq = torch.tensor([0, queries], device="cuda", dtype=torch.int32)
    cuk = torch.tensor([0, 32], device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)

    def run():
        flash_attn_varlen_func(
            q=q,
            k=kc,
            v=vc,
            out=out,
            cu_seqlens_q=cuq,
            cu_seqlens_k=cuk,
            max_seqlen_q=queries,
            max_seqlen_k=32,
            softmax_scale=192**-0.5,
            causal=True,
            window_size=(511, 0),
            block_table=table,
            backend="ixinfer",
        )
        return out

    def reference(pages):
        k = torch.cat(
            [kc[p].transpose(0, 1).cpu().double() for p in pages], 0
        ).repeat_interleave(heads // groups, 1)
        v = torch.cat(
            [vc[p].transpose(0, 1).cpu().double() for p in pages], 0
        ).repeat_interleave(heads // groups, 1)
        scores = torch.einsum("thd,shd->hts", q.cpu().double(), k) * 192**-0.5
        mask = torch.arange(32)[None, :] <= torch.arange(32 - queries, 32)[:, None]
        scores.masked_fill_(~mask[None], float("-inf"))
        return torch.einsum("hts,shd->thd", scores.softmax(-1), v).to(torch.bfloat16)

    for _ in range(3):
        run()
    torch.testing.assert_close(out.cpu(), reference([0, high]), atol=0.02, rtol=0.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    table.copy_(torch.tensor([[high, 0]], device="cuda", dtype=torch.int32))
    q.add_(0.01)
    graph.replay()
    torch.testing.assert_close(out.cpu(), reference([high, 0]), atol=0.02, rtol=0.02)
