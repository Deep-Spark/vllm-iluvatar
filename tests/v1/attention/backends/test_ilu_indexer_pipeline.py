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


@pytest.mark.parametrize("case", ["prefill", "subchunk", "ragged", "mixed", "padded"])
@torch.inference_mode()
def test_real_builder_logits_topk(case, monkeypatch):
    device = torch.device("cuda:0")
    ratio, page, heads, topk = 2, 64, 16, 8
    if case == "subchunk":
        queries, lengths, prefilling = [513], [1025], [True]
    elif case == "prefill":
        queries, lengths, prefilling = [9, 7], [74, 130], [True, True]
    elif case == "mixed":
        queries, lengths, prefilling = [4, 1, 9], [130, 73, 151], [False, False, True]
    elif case == "padded":
        queries, lengths, prefilling = [4, 1, 0], [130, 73, 0], [False] * 3
    else:
        queries, lengths, prefilling = [4, 1], [130, 73], [False, False]
    total = sum(queries)
    padded = total + 3 if case == "padded" else total
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=2048),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024, max_num_seqs=4),
        speculative_config=SimpleNamespace(num_speculative_tokens=4),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    spec = MLAAttentionSpec(
        block_size=page * ratio,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=ratio,
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
    pages_per_req = (config.model_config.max_model_len + page * ratio - 1) // (
        page * ratio
    )
    bt = (
        torch.arange(
            1, 1 + len(queries) * pages_per_req, dtype=torch.int32, device=device
        )
        .reshape(len(queries), pages_per_req)
        .flip(1)
        .contiguous()
    )
    cache = torch.full(
        (1 + bt.numel(), page, 128), float("nan"), dtype=torch.bfloat16, device=device
    )
    for req, length in enumerate(lengths):
        for key in range(length // ratio):
            physical = int(bt[req, key // page].item())
            cache[physical, key % page].zero_()
            cache[physical, key % page, 0] = key + 1
    slots = [
        int(bt[req, pos // (page * ratio)].item()) * page * ratio + pos % (page * ratio)
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
    assert metadata.num_states == page
    if case == "subchunk":
        assert len(metadata.prefill.chunks) == 2
        assert [c.max_q_len for c in metadata.prefill.chunks] == [512, 1]
    if case in ("ragged", "mixed", "padded"):
        assert metadata.decode.requires_padding
        assert metadata.decode.cu_seqlens_q.cpu().tolist() == qsl[:3] + (
            [5] if case == "padded" else []
        )
    monkeypatch.setattr(
        sparse,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"indexer": metadata}),
    )
    recorded = []
    native = sparse.ixfops.dsa_indexer_mqa_logits_with_blocks

    def capture(*args, **kwargs):
        output = native(*args, **kwargs)
        recorded.append((output, kwargs["cu_seqlen_ke"] - kwargs["cu_seqlen_ks"]))
        return output

    monkeypatch.setattr(sparse.ixfops, "dsa_indexer_mqa_logits_with_blocks", capture)
    q = torch.ones(padded, heads, 128, dtype=torch.bfloat16, device=device)
    weights = torch.ones(padded, heads, dtype=q.dtype, device=device)
    output = torch.full((padded, topk), -99, dtype=torch.int32, device=device)
    result = sparse.ilu_sparse_attn_indexer(
        q,
        "indexer",
        cache,
        q,
        None,
        weights,
        topk,
        128,
        2048,
        sum(lengths),
        output,
        skip_k_cache_insert=True,
    )
    assert result is output
    # Integer/power-of-two scores have an exact BF16 reference. No NaN from
    # poisoned invalid cache/tail columns may enter any visible logits row.
    for logits, spans in recorded:
        for row, length in enumerate(spans.cpu().tolist()):
            reference = torch.arange(1, length + 1).to(torch.bfloat16) * heads
            torch.testing.assert_close(
                logits[row, :length].cpu(), reference, rtol=0, atol=0
            )
    for row, pos in enumerate(positions_cpu):
        visible = (pos + 1) // ratio
        indices = output[row].cpu().tolist()
        selected = [key for key in indices if key >= 0]
        assert len(selected) == min(topk, visible)
        assert len(set(selected)) == len(selected)
        assert all(0 <= key < visible for key in selected)
        scores = (torch.arange(1, visible + 1).to(torch.bfloat16) * heads).float()
        cutoff = float(scores.topk(min(topk, visible)).values[-1])
        assert all(float(scores[key]) >= cutoff for key in selected)
        assert set(torch.where(scores > cutoff)[0].tolist()) <= set(selected)
    assert torch.all(output[total:] == -1)
