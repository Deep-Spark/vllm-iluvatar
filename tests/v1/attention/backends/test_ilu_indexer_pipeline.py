# SPDX-License-Identifier: Apache-2.0
"""Actual metadata builder, native logits producer and TopK integration."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_iluvatar.model_executor.layers import sparse_attn_indexer as sparse
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV32IndexerMetadataBuilder,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

RATIO, PAGE, HEADS, TOPK = 2, 64, 16, 8

# Query / sequence lengths per case, and which requests are still prefilling.
CASES = {
    "subchunk": ([513], [1025], [True]),
    "prefill": ([9, 7], [74, 130], [True, True]),
    "mixed": ([4, 1, 9], [130, 73, 151], [False, False, True]),
    "padded": ([4, 1, 0], [130, 73, 0], [False] * 3),
    "ragged": ([4, 1], [130, 73], [False, False]),
}


def _build_case(case, monkeypatch, device):
    """The builder's real output for one batch shape, plus what it was built from.

    Everything here is shared by the BF16 and INT8 records: they differ only in
    what a key row looks like and which scoring kernel reads it.
    """
    queries, lengths, prefilling = CASES[case]
    total = sum(queries)
    padded = total + 3 if case == "padded" else total
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=2048),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024, max_num_seqs=4),
        speculative_config=SimpleNamespace(num_speculative_tokens=4),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    spec = MLAAttentionSpec(
        block_size=PAGE * RATIO,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=RATIO,
    )
    builder = IluDeepseekV32IndexerMetadataBuilder(spec, ["indexer"], config, device)
    # Use the real splitting rule; a 1 MiB budget makes M=513,N=512 split.
    monkeypatch.setattr(sparse.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 1)
    qsl = [0]
    positions_cpu = []
    requests = []
    for req, (count, length) in enumerate(zip(queries, lengths)):
        qsl.append(qsl[-1] + count)
        positions_cpu.extend(range(length - count, length))
        requests.extend([req] * count)
    qsl_cpu = torch.tensor(qsl, dtype=torch.int32)
    # Match the runner's persistent table width, including unused tail pages.
    pages_per_req = (config.model_config.max_model_len + PAGE * RATIO - 1) // (
        PAGE * RATIO
    )
    bt = (
        torch.arange(
            1, 1 + len(queries) * pages_per_req, dtype=torch.int32, device=device
        )
        .reshape(len(queries), pages_per_req)
        .flip(1)
        .contiguous()
    )
    slots = [
        int(bt[req, pos // (PAGE * RATIO)].item()) * PAGE * RATIO + pos % (PAGE * RATIO)
        for req, pos in zip(requests, positions_cpu)
    ]
    slots += [-1] * (padded - total)
    common = CommonAttentionMetadata(
        query_start_loc=qsl_cpu.to(device),
        query_start_loc_cpu=qsl_cpu,
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
        num_reqs=len(queries),
        num_actual_tokens=padded,
        max_query_len=max(queries),
        max_seq_len=max(lengths),
        block_table_tensor=bt,
        slot_mapping=torch.tensor(slots, device=device),
        seq_lens_cpu_upper_bound=torch.tensor(lengths, dtype=torch.int32),
        is_prefilling=torch.tensor(prefilling, dtype=torch.bool),
    )
    metadata = builder.build(0, common)
    assert metadata.num_states == PAGE
    # The INT8 scoring kernel masks from ``valid_lens`` where the span kernel
    # reads ``[ks, ke)``, so the two have to agree whichever way the builder
    # produced them -- including on the CUDA-graph padding rows, which the
    # decode path zeroes in a persistent buffer rather than computing.
    for part in (metadata.prefill, metadata.decode):
        if part is None:
            continue
        for md in getattr(part, "chunks", [part]):
            torch.testing.assert_close(
                md.valid_lens, md.cu_seqlen_ke - md.cu_seqlen_ks, rtol=0, atol=0
            )
    monkeypatch.setattr(
        sparse,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"indexer": metadata}),
    )
    return SimpleNamespace(
        metadata=metadata,
        block_table=bt,
        lengths=lengths,
        queries=queries,
        qsl=qsl,
        positions_cpu=positions_cpu,
        total=total,
        padded=padded,
        num_pages=1 + bt.numel(),
    )


def _check_selection(output, positions_cpu, total):
    """Every row's top-k, against the exact scores the caches were built for.

    Key ``k`` scores ``(k + 1) * HEADS``, so the ranking is the key order and
    the cutoff is exact. Ties at the cutoff may go either way, hence the
    two-sided check rather than an equality.
    """
    for row, pos in enumerate(positions_cpu):
        visible = (pos + 1) // RATIO
        indices = output[row].cpu().tolist()
        selected = [key for key in indices if key >= 0]
        assert len(selected) == min(TOPK, visible)
        assert len(set(selected)) == len(selected)
        assert all(0 <= key < visible for key in selected)
        scores = (torch.arange(1, visible + 1).to(torch.bfloat16) * HEADS).float()
        cutoff = float(scores.topk(min(TOPK, visible)).values[-1])
        assert all(float(scores[key]) >= cutoff for key in selected)
        assert set(torch.where(scores > cutoff)[0].tolist()) <= set(selected)
    assert torch.all(output[total:] == -1)


@pytest.mark.parametrize("case", ["prefill", "subchunk", "ragged", "mixed", "padded"])
@torch.inference_mode()
def test_real_builder_logits_topk(case, monkeypatch):
    device = torch.device("cuda:0")
    setup = _build_case(case, monkeypatch, device)
    bt, metadata = setup.block_table, setup.metadata
    if case == "subchunk":
        assert len(metadata.prefill.chunks) == 2
        assert [c.max_q_len for c in metadata.prefill.chunks] == [512, 1]
    if case in ("ragged", "mixed", "padded"):
        assert metadata.decode.requires_padding
        assert metadata.decode.cu_seqlens_q.cpu().tolist() == setup.qsl[:3] + (
            [5] if case == "padded" else []
        )

    cache = torch.full(
        (setup.num_pages, PAGE, 128),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    for req, length in enumerate(setup.lengths):
        for key in range(length // RATIO):
            physical = int(bt[req, key // PAGE].item())
            cache[physical, key % PAGE].zero_()
            cache[physical, key % PAGE, 0] = key + 1

    recorded = []
    native = sparse.ixfops.dsa_indexer_mqa_logits_with_blocks

    def capture(*args, **kwargs):
        output = native(*args, **kwargs)
        recorded.append((output, kwargs["cu_seqlen_ke"] - kwargs["cu_seqlen_ks"]))
        return output

    monkeypatch.setattr(sparse.ixfops, "dsa_indexer_mqa_logits_with_blocks", capture)
    q = torch.ones(setup.padded, HEADS, 128, dtype=torch.bfloat16, device=device)
    weights = torch.ones(setup.padded, HEADS, dtype=q.dtype, device=device)
    output = torch.full((setup.padded, TOPK), -99, dtype=torch.int32, device=device)
    result = sparse.ilu_sparse_attn_indexer(
        q,
        "indexer",
        cache,
        q,
        None,
        weights,
        TOPK,
        128,
        2048,
        sum(setup.lengths),
        output,
        skip_k_cache_insert=True,
    )
    assert result is output
    # Integer/power-of-two scores have an exact BF16 reference. No NaN from
    # poisoned invalid cache/tail columns may enter any visible logits row.
    for logits, spans in recorded:
        for row, length in enumerate(spans.cpu().tolist()):
            reference = torch.arange(1, length + 1).to(torch.bfloat16) * HEADS
            torch.testing.assert_close(
                logits[row, :length].cpu(), reference, rtol=0, atol=0
            )
    _check_selection(output, setup.positions_cpu, setup.total)


# ``key + 1`` has to fit in an int8 byte, which rules out the 512-key subchunk
# case. "prefill" and "mixed" between them cover both branches of the dispatch.
@pytest.mark.parametrize("case", ["prefill", "mixed"])
@torch.inference_mode()
def test_int8_record_logits_topk(case, monkeypatch):
    """The INT8 record end to end, minus the index-key store.

    ``VLLM_ILU_DS4_INDEXER_KV_DTYPE=int8`` is what routes scoring to the
    ixinfer kernel and makes the builder hand it ``valid_lens``; an INT8 query
    is what the op dispatches on. The key pages are written here because the
    store kernel has no INT8 record yet.
    """
    device = torch.device("cuda:0")
    monkeypatch.setenv("VLLM_ILU_DS4_INDEXER_KV_DTYPE", "int8")
    setup = _build_case(case, monkeypatch, device)
    bt = setup.block_table

    # 128 INT8 value bytes per row then one FP32 scale per row, segregated
    # inside the page -- all rows' values, then all rows' scales.
    row_bytes, scale_base = 128 + 4, PAGE * 128
    cache = torch.empty(
        (setup.num_pages, PAGE, row_bytes), dtype=torch.uint8, device=device
    )
    flat = cache.reshape(setup.num_pages, -1)
    # Poison every row first, so that rows left untouched below -- unused
    # pages and the tail of a request's last page -- score far above every
    # real key. A wrong page or row offset then shows up as a mismatch
    # instead of reading plausible zeros. Reads past ``valid_lens`` are not
    # what this catches: top-k rescans ``[ks, ke)`` and never sees them.
    flat[:, :scale_base] = 127
    flat[:, scale_base:] = (
        torch.tensor([1000.0], dtype=torch.float32)
        .view(torch.uint8)
        .to(device)
        .repeat(PAGE)
    )
    values = torch.zeros(128, dtype=torch.int8)
    unit_scale = torch.tensor([1.0], dtype=torch.float32).view(torch.uint8).to(device)
    for req, length in enumerate(setup.lengths):
        for key in range(length // RATIO):
            physical = int(bt[req, key // PAGE].item())
            row = key % PAGE
            values.zero_()
            values[0] = key + 1
            flat[physical, row * 128 : (row + 1) * 128] = values.view(torch.uint8).to(
                device
            )
            flat[physical, scale_base + row * 4 : scale_base + (row + 1) * 4] = (
                unit_scale
            )

    recorded = []
    native = sparse.ixfops.dsa_indexer_mqa_logits

    def capture(*args, **kwargs):
        output = native(*args, **kwargs)
        recorded.append((output, kwargs["valid_lens"]))
        return output

    monkeypatch.setattr(sparse.ixfops, "dsa_indexer_mqa_logits", capture)
    # A query of all ones against a row whose only non-zero byte is ``key + 1``
    # contracts to ``key + 1``, and the per-head weights sum that HEADS times.
    q = torch.ones(setup.padded, HEADS, 128, dtype=torch.int8, device=device)
    weights = torch.ones(setup.padded, HEADS, dtype=torch.float32, device=device)
    output = torch.full((setup.padded, TOPK), -99, dtype=torch.int32, device=device)
    result = sparse.ilu_sparse_attn_indexer(
        q,
        "indexer",
        cache,
        q,
        None,
        weights,
        TOPK,
        128,
        2048,
        sum(setup.lengths),
        output,
        skip_k_cache_insert=True,
    )
    assert result is output
    assert recorded, "the INT8 query did not reach dsa_indexer_mqa_logits"
    for logits, spans in recorded:
        for row, length in enumerate(spans.cpu().tolist()):
            reference = torch.arange(1, length + 1).to(torch.bfloat16) * HEADS
            torch.testing.assert_close(
                logits[row, :length].cpu().float(),
                reference.float(),
                rtol=0,
                atol=0,
            )
    _check_selection(output, setup.positions_cpu, setup.total)
