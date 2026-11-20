# SPDX-License-Identifier: Apache-2.0
"""Storage-page addressing across index-cache writes, reads and graph replay."""

from types import SimpleNamespace as NS

import pytest
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_iluvatar.v1.attention.backends.mla import ilu_indexer
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV32IndexerMetadataBuilder,
    make_storage_block_size_build,
)

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
MANAGER_TOKENS = 2304
STORAGE_TOKENS = 256
POOL = 4


def make_builder(
    kernel_tokens=64,
    *,
    device="cuda:0",
    spec_tokens=STORAGE_TOKENS,
    storage_override=STORAGE_TOKENS,
    speculative_tokens=5,
):
    cfg = NS(
        model_config=NS(max_model_len=4608),
        scheduler_config=NS(max_num_seqs=4, max_num_batched_tokens=64),
        speculative_config=NS(num_speculative_tokens=speculative_tokens),
        parallel_config=NS(
            prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
    )
    spec = MLAAttentionSpec(
        block_size=spec_tokens,
        storage_block_size=storage_override,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=POOL,
    )
    width = 2 * MANAGER_TOKENS // kernel_tokens
    builder = IluDeepseekV32IndexerMetadataBuilder(
        spec,
        ["indexer.k_cache"],
        cfg,
        torch.device(device),
        block_table_width=width,
    )
    builder.set_kernel_block_size(kernel_tokens)
    return builder


def make_common(builder, lengths, previous, manager_ids=None):
    b = len(lengths)
    device = builder.device
    ids = manager_ids or [[1, 4], [2, 5], [0, 0]][:b]
    manager = torch.tensor(ids, dtype=torch.int32, device=device)
    factor = MANAGER_TOKENS // builder.kernel_block_size
    table = (
        manager[:, :, None] * factor
        + torch.arange(factor, device=device, dtype=torch.int32)
    ).reshape(b, -1)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32)
    seq = torch.tensor(previous, dtype=torch.int32) + torch.tensor(
        lengths, dtype=torch.int32
    )
    positions = torch.cat(
        [
            torch.arange(p, p + n, device=device, dtype=torch.int64)
            for p, n in zip(previous, lengths)
        ]
    )
    request = torch.repeat_interleave(
        torch.arange(b, device=device), torch.tensor(lengths, device=device)
    )
    physical = (
        manager[request, positions // MANAGER_TOKENS].long() * MANAGER_TOKENS
        + positions % MANAGER_TOKENS
    )
    common = CommonAttentionMetadata(
        query_start_loc=cu.to(device),
        query_start_loc_cpu=cu,
        seq_lens=seq.to(device),
        seq_lens_cpu_upper_bound=seq,
        num_reqs=b,
        num_actual_tokens=int(cu[-1]),
        max_query_len=max(lengths),
        max_seq_len=int(seq.max()),
        block_table_tensor=table,
        slot_mapping=physical,
    )
    common._token_to_req_indices_cache = request.to(torch.int32)
    expected_table = (
        manager[:, :, None] * (MANAGER_TOKENS // STORAGE_TOKENS)
        + torch.arange(
            MANAGER_TOKENS // STORAGE_TOKENS, device=device, dtype=torch.int32
        )
    ).reshape(b, -1)
    expected_slots = torch.where((positions + 1) % POOL == 0, physical // POOL, -1)
    return common, expected_table, expected_slots


@requires_gpu
@pytest.mark.parametrize("kernel_tokens", [64, 256])
@pytest.mark.parametrize(
    "lengths,previous",
    [([8, 12], [0, 256]), ([6, 2], [258, 510]), ([6, 8], [258, 256])],
)
def test_storage_page_table_matches_pool_writer(kernel_tokens, lengths, previous):
    builder = make_builder(kernel_tokens)
    common, expected_table, expected_slots = make_common(builder, lengths, previous)
    original = common.block_table_tensor.clone()
    metadata = builder.build(0, common)
    torch.testing.assert_close(metadata.slot_mapping, expected_slots, rtol=0, atol=0)
    torch.testing.assert_close(metadata.block_table, expected_table, rtol=0, atol=0)
    torch.testing.assert_close(common.block_table_tensor, original, rtol=0, atol=0)
    if metadata.prefill is not None:
        for chunk in metadata.prefill.chunks:
            first = metadata.num_decodes
            torch.testing.assert_close(
                chunk.block_table, expected_table[first:], rtol=0, atol=0
            )
    if metadata.decode is not None:
        torch.testing.assert_close(
            metadata.decode.block_table,
            expected_table[: metadata.num_decodes],
            rtol=0,
            atol=0,
        )
    assert metadata.block_table.is_contiguous()


@requires_gpu
def test_compressed_cache_producer_and_score_read_same_pages():
    import ixformer.inference.functions as ops
    from ixformer.inference.functions.models.glm5next.kpool.float.glm5next_kpool import (
        kpool_compress_insert,
    )

    device = "cuda"

    torch.manual_seed(41)
    builder = make_builder()
    common, expected_table, _ = make_common(builder, [12, 12], [0, 256])
    metadata = builder.build(0, common)
    chunk = metadata.prefill.chunks[0]
    kv = torch.zeros(256, 64, 132, device=device, dtype=torch.uint8)
    k = torch.randn(24, 128, device=device, dtype=torch.bfloat16)
    gate = torch.randn_like(k)
    ape = torch.randn(4, 128, device=device)
    kpool_compress_insert(k, gate, ape, kv, metadata.slot_mapping, 4, 128, False, True)
    q = torch.randint(-127, 128, (24, 32, 128), dtype=torch.int8, device=device)
    weights = torch.ones(24, 32, device=device)
    arguments = dict(
        max_q_len=chunk.max_q_len,
        max_kv_len=chunk.max_kv_len,
        max_context_len=chunk.max_context_len,
        valid_lens=chunk.valid_lens,
        mask_invalid=True,
        backend="ixinfer",
    )
    actual = ops.dsa_indexer_mqa_logits_with_kvcache(
        q,
        chunk.cu_seqlens_q,
        chunk.cu_seq_lens,
        kv,
        chunk.block_table,
        weights,
        **arguments,
    )
    expected = ops.dsa_indexer_mqa_logits_with_kvcache(
        q,
        chunk.cu_seqlens_q,
        chunk.cu_seq_lens,
        kv,
        expected_table,
        weights,
        **arguments,
    )
    assert (expected[torch.isfinite(expected)] > 0).any()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_gpu
def test_storage_page_table_address_survives_graph_and_request_reorder():
    builder = make_builder()
    common, expected, _ = make_common(builder, [6, 6, 0], [58, 58, 0])
    meta = builder.build(0, common)
    table = meta.decode.block_table
    pointer = table.data_ptr()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        result = table.clone()
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph, stream=stream):
        result.copy_(table)
    for ids in [[[2, 5], [1, 4], [0, 0]], [[4, 1], [5, 2], [0, 0]]]:
        update, expected, _ = make_common(builder, [6, 6, 0], [58, 58, 0], ids)
        meta = builder.build(0, update)
        assert meta.decode.block_table.data_ptr() == pointer
        graph.replay()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    smaller, expected, _ = make_common(builder, [6], [58], [[3, 1]])
    meta = builder.build(0, smaller)
    assert meta.decode.block_table.data_ptr() == pointer
    torch.testing.assert_close(meta.decode.block_table, expected, rtol=0, atol=0)


def _cpu_compressed_slots(
    num_tokens,
    slot_mapping,
    query_start_loc,
    seq_lens,
    block_table,
    block_size,
    compress_ratio,
    out=None,
):
    assert out is not None
    out.fill_(-1)
    for row in range(seq_lens.numel()):
        begin, end = map(int, query_start_loc[row : row + 2])
        start_pos = int(seq_lens[row]) - (end - begin)
        for token in range(begin, end):
            pos = start_pos + token - begin
            if (pos + 1) % compress_ratio == 0 and slot_mapping[token] >= 0:
                state = pos // compress_ratio
                out[token] = (
                    block_table[row, state // block_size] * block_size
                    + state % block_size
                )
    return out[:num_tokens]


@pytest.fixture
def cpu_metadata(monkeypatch):
    monkeypatch.setattr(ilu_indexer, "num_compute_units", lambda index: 16)
    monkeypatch.setattr(
        ilu_indexer, "get_compressed_slot_mapping", _cpu_compressed_slots
    )
    monkeypatch.setattr(ilu_indexer, "has_deep_gemm", lambda: False)
    # These tests exercise page metadata only; pinned transfers are GPU tests.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)


@pytest.mark.parametrize(
    "lengths,previous",
    [([8, 12], [0, 256]), ([1, 1, 0], [259, 511, 0]), ([1, 8], [259, 256])],
)
@pytest.mark.parametrize("kernel_tokens", [64, 256])
@pytest.mark.parametrize("wrapped", [False, True])
def test_cpu_storage_writer_and_all_readers_agree(
    cpu_metadata, kernel_tokens, lengths, previous, wrapped
):
    builder = make_builder(kernel_tokens, device="cpu", speculative_tokens=0)
    common, expected_table, expected_slots = make_common(builder, lengths, previous)
    original = common.block_table_tensor.clone()
    build = builder.build
    if wrapped:
        build = make_storage_block_size_build(
            IluDeepseekV32IndexerMetadataBuilder.build
        ).__get__(builder)
    meta = build(0, common)
    torch.testing.assert_close(meta.slot_mapping, expected_slots, rtol=0, atol=0)
    torch.testing.assert_close(meta.block_table, expected_table, rtol=0, atol=0)
    torch.testing.assert_close(common.block_table_tensor, original, rtol=0, atol=0)
    if meta.decode is not None:
        torch.testing.assert_close(
            meta.decode.block_table, expected_table[: meta.num_decodes], rtol=0, atol=0
        )
        assert meta.decode.block_table.data_ptr() == meta.block_table.data_ptr()
    if meta.prefill is not None:
        for chunk in meta.prefill.chunks:
            torch.testing.assert_close(
                chunk.block_table, expected_table[meta.num_decodes :], rtol=0, atol=0
            )


def test_cpu_storage_buffer_is_reserved_and_stable(cpu_metadata):
    builder = make_builder(device="cpu", speculative_tokens=0)
    pointer = builder._storage_block_table_buffer.data_ptr()
    for lengths, previous, ids in [
        ([8, 12], [0, 256], [[1, 4], [2, 5]]),
        ([1, 1, 0], [259, 511, 0], [[2, 5], [1, 4], [0, 0]]),
        ([1], [259], [[3, 1]]),
    ]:
        common, expected, _ = make_common(builder, lengths, previous, ids)
        meta = builder.build(0, common)
        assert meta.block_table.data_ptr() == pointer
        torch.testing.assert_close(meta.block_table, expected, rtol=0, atol=0)
        if meta.decode is not None:
            assert meta.decode.block_table.data_ptr() == pointer


@pytest.mark.parametrize(
    "spec_tokens,kernel_tokens,expected_factor",
    [(1024, 64, 4), (256, 64, 1), (256, 256, 1)],
)
def test_cpu_storage_reservation_uses_wrapper_page_units(
    cpu_metadata, spec_tokens, kernel_tokens, expected_factor
):
    builder = make_builder(
        kernel_tokens,
        device="cpu",
        spec_tokens=spec_tokens,
        storage_override=None,
        speculative_tokens=0,
    )
    original_spec = builder.kv_cache_spec
    assert original_spec.num_states == spec_tokens // POOL
    if expected_factor == 1:
        assert builder._storage_block_table_buffer is None
    else:
        assert builder._storage_block_table_buffer.shape[1] == (
            builder._kernel_block_table_width // expected_factor
        )
    common, _, _ = make_common(builder, [1], [259])
    build = make_storage_block_size_build(
        IluDeepseekV32IndexerMetadataBuilder.build
    ).__get__(builder)
    meta = build(0, common)
    expected = common.block_table_tensor[:, ::expected_factor] // expected_factor
    torch.testing.assert_close(meta.block_table, expected, rtol=0, atol=0)
    assert builder.kv_cache_spec is original_spec
