# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage of valid-only GLM5Next indexer logits."""

import pytest
import torch

from vllm_iluvatar.models.glm5next import sparse_indexer as indexer


@pytest.mark.parametrize(
    "rows,width,kv_len",
    [(16, 768, 640), (16, 262144, 28160), (528, 8192, 1024)],
)
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_valid_logits_topk_replays_with_dirty_exterior(
    monkeypatch, rows, width, kv_len, compact
):
    torch.manual_seed(43)
    device = "cuda"
    dtype = torch.bfloat16
    page = 128
    pages = (kv_len + page - 1) // page
    q = torch.randn(rows, 32, 128, device=device, dtype=dtype) * 0.1
    kv = torch.randn(2 * pages, page, 128, device=device, dtype=dtype) * 0.1
    weights = torch.randn(rows, 32, device=device, dtype=dtype).abs()
    table = torch.arange(2 * pages, device=device, dtype=torch.int32).reshape(2, pages)
    real = rows - 1
    split = real // 2
    cu_q = torch.tensor([0, split, real], device=device, dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_len, 2 * kv_len], device=device, dtype=torch.int32)
    ks = torch.cat((torch.zeros(split), torch.full((rows - split,), kv_len))).to(
        device=device, dtype=torch.int32
    )
    valid = torch.empty(rows, device=device, dtype=torch.int32)
    ke = torch.empty_like(valid)
    positions = torch.empty(rows, device=device, dtype=torch.int64)
    dirty = torch.empty(rows, width, device=device, dtype=dtype)
    clean = torch.empty_like(dirty)
    columns = torch.arange(width, device=device)[None, :]
    select_k = 512
    out_width = select_k if compact else select_k * 4 + 3
    actual = torch.empty(rows, out_width, device=device, dtype=torch.int32)
    expected = torch.empty_like(actual)
    original = indexer.ixfops.dsa_indexer_mqa_logits_with_kvcache

    def supplied_logits(*args, **kwargs):
        # Exercise the plugin's flag and its real producer/consumer calls.
        assert kwargs["mask_invalid"] is False
        return original(*args, **kwargs, logits=dirty)

    monkeypatch.setattr(
        indexer.ixfops, "dsa_indexer_mqa_logits_with_kvcache", supplied_logits
    )

    def run_actual():
        logits = indexer._kpool_mqa_logits(
            q,
            weights,
            kv,
            table,
            cu_q,
            cu_kv,
            ks,
            ke,
            valid,
            max_q_len=max(split, real - split),
            max_kv_len=kv_len,
            max_context_len=width,
            indexer_type=indexer.IndexerType.BF16,
        )
        indexer._select_pools_and_expand(
            logits, ks, ke, positions, 2048, 4, actual, compact
        )

    def run_reference():
        original(
            q,
            cu_q,
            cu_kv,
            kv,
            table,
            weights,
            logits=clean,
            max_q_len=max(split, real - split),
            max_kv_len=kv_len,
            max_context_len=width,
            valid_lens=valid,
            backend="ixinfer",
            mask_invalid=True,
        )
        indexer._select_pools_and_expand(
            clean, ks, ke, positions, 2048, 4, expected, compact
        )

    boundaries = [0, 1, 3, 127, 128, 129, 511, 512, 513, kv_len]
    lengths = (boundaries * ((rows + 9) // 10))[:rows]

    def update_lengths(values):
        values[-1] = 0  # graph padding row is not mapped by cu_q
        valid.copy_(torch.tensor(values, device=device, dtype=torch.int32))
        torch.add(ks, valid, out=ke)
        positions.copy_(valid * 4 + 2)

    update_lengths(lengths.copy())
    run_actual()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_actual()
    cases = (
        (lengths.copy(), float("nan")),
        (list(reversed(lengths)), float("inf")),
        (lengths.copy(), 10000.0),
    )
    for values, poison in cases:
        update_lengths(values)
        dirty.fill_(poison)
        run_reference()
        graph.replay()
        live = columns < valid[:, None]
        torch.testing.assert_close(dirty[live], clean[live], rtol=0, atol=0)
        # The unchanged deterministic selection and expansion must ignore exterior data.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        pools = actual if compact else actual[:, :2048:4] // 4
        assert ((pools < valid[:, None]) | (pools == -1)).all()
        assert (pools[valid == 0] == -1).all()
