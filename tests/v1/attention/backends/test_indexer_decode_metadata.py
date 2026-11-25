# SPDX-License-Identifier: Apache-2.0
"""Integer bounds and replay validation for fused indexer metadata.

TRITON_INTERPRET=1 runs the numerical cases on CPU without a GPU.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV32IndexerMetadataBuilder,
)
from vllm_iluvatar.v1.attention.backends.mla.indexer_decode_metadata import (
    build_indexer_decode_metadata,
)

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"
pytestmark = pytest.mark.skipif(
    not INTERPRET and not torch.cuda.is_available(),
    reason="requires GPU or interpreter",
)


def reference(q, seq, ratio, padded, flat):
    counts = q[1:] - q[:-1]
    compressed = seq // ratio
    prefix = torch.cat((torch.zeros(1, dtype=torch.int32), compressed.cumsum(0).int()))
    ks = torch.zeros(padded, dtype=torch.int32)
    ke = torch.zeros_like(ks)
    for b, count in enumerate(counts.tolist()):
        for j in range(count):
            row = int(q[b] - q[0]) + j
            ks[row] = prefix[b]
            ke[row] = prefix[b] + (int(seq[b]) - count + j + 1) // ratio
    return dict(
        cu_seqlens_q=q - q[0],
        cu_seqlens_kv=prefix,
        per_req_decode_lens=counts,
        compressed_seq_lens=compressed,
        decode_lens=torch.ones(padded, dtype=torch.int32) if flat else counts,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
        valid_lens=ke - ks,
    )


def inputs(counts, padded, origin=0):
    counts = torch.tensor(counts, dtype=torch.int32)
    q = torch.cat((torch.zeros(1, dtype=torch.int32), counts.cumsum(0).int())) + origin
    seq = 112637 + torch.arange(len(counts), dtype=torch.int32) * 17 + counts
    if len(seq):
        seq[0] = counts[0] + 3
    assert int(counts.sum()) <= padded
    return q, seq


def request_ids(q, padded):
    ids = torch.full((padded,), -1, dtype=torch.int32)
    for b in range(len(q) - 1):
        ids[int(q[b] - q[0]) : int(q[b + 1] - q[0])] = b
    return ids.to(DEVICE)


def assert_outputs(actual, expected):
    for name in expected:
        torch.testing.assert_close(
            actual[name].cpu(),
            expected[name],
            rtol=0,
            atol=0,
            msg=lambda msg, name=name: name + ": " + msg,
        )


@pytest.mark.parametrize(
    "counts,padded,origin",
    [
        ([4, 2, 0], 8, 0),
        ([1, 0, 1, 0], 8, 7),
        ([6] * 16, 96, 0),
        ([0] * 16, 96, 0),
        ([i % 4 for i in range(129)], 512, 0),
        ([200, 200, 200, 0], 640, 11),
        ([], 8, 0),
        ([], 0, 0),
    ],
)
@pytest.mark.parametrize(
    "ratio,flat,with_ids",
    [(1, False, False), (2, True, True), (4, True, False), (4, False, True)],
)
def test_decode_bounds(counts, padded, origin, ratio, flat, with_ids):
    q, seq = inputs(counts, padded, origin)
    expected = reference(q, seq, ratio, padded, flat)
    actual = {
        name: torch.full_like(value, -99, device=DEVICE)
        for name, value in expected.items()
    }
    build_indexer_decode_metadata(
        q.to(DEVICE),
        seq.to(DEVICE),
        compress_ratio=ratio,
        flatten_decode_lens=flat,
        token_to_req_indices=request_ids(q, padded) if with_ids else None,
        **actual,
    )
    assert_outputs(actual, expected)


@pytest.mark.skipif(INTERPRET, reason="CUDA graph requires compiled GPU kernels")
@pytest.mark.parametrize("with_ids", [False, True])
def test_graph_updates_lengths_and_padding(with_ids):
    padded = 96
    q, seq = inputs([6] * 16, padded)
    gpu_q, gpu_seq = q.to(DEVICE), seq.to(DEVICE)
    expected = reference(q, seq, 4, padded, True)
    actual = {
        name: torch.empty_like(value, device=DEVICE) for name, value in expected.items()
    }
    ids = request_ids(q, padded) if with_ids else None

    def run():
        build_indexer_decode_metadata(
            gpu_q,
            gpu_seq,
            compress_ratio=4,
            flatten_decode_lens=True,
            token_to_req_indices=ids,
            **actual,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for counts in ([1, 0, 3, 2] * 4, [0] * 16, [6] * 16):
        q, seq = inputs(counts, padded)
        gpu_q.copy_(q)
        gpu_seq.copy_(seq)
        if with_ids:
            ids.copy_(request_ids(q, padded))
        for value in actual.values():
            value.fill_(-99)
        graph.replay()
        assert_outputs(actual, reference(q, seq, 4, padded, True))


@pytest.mark.parametrize(
    "counts,padded", [([6, 6, 6], 18), ([3, 0, 1], 8), ([1, 1, 0], 8)]
)
def test_builder_preserves_request_and_legacy_lengths(counts, padded):
    q, seq = inputs(counts, padded)
    builder = object.__new__(IluDeepseekV32IndexerMetadataBuilder)
    builder.num_speculative_tokens = 5
    builder.use_flattening = True
    builder.compress_ratio = 4
    builder.max_decode_context_len = 262144
    b = len(counts)
    flat = max(counts) > 1
    expected = reference(q, seq, 4, padded, flat)
    fields = {
        "cu_seqlens_q": "cu_seqlens_q_buffer",
        "cu_seqlens_kv": "cu_seqlens_kv_buffer",
        "per_req_decode_lens": "per_req_decode_lens_buffer",
        "compressed_seq_lens": "decode_compressed_seq_lens_buffer",
        "decode_lens": "decode_lens_buffer",
        "cu_seqlen_ks": "cu_seqlen_ks_buffer",
        "cu_seqlen_ke": "cu_seqlen_ke_buffer",
        "valid_lens": "valid_lens_buffer",
    }
    for name, field in fields.items():
        setattr(builder, field, torch.empty_like(expected[name], device=DEVICE))
    common = SimpleNamespace(
        query_start_loc=q.to(DEVICE),
        query_start_loc_cpu=q,
        seq_lens=seq.to(DEVICE),
        _token_to_req_indices_cache=None,
        block_table_tensor=torch.zeros(b, 2, device=DEVICE, dtype=torch.int32),
    )
    meta = builder._build_fused_decode_metadata(common, b, padded)
    actual = {
        name: getattr(meta, "seq_lens" if name == "compressed_seq_lens" else name)
        for name in expected
    }
    assert_outputs(actual, expected)
    assert meta.max_q_len == max(counts)
    assert meta.write_max_decode_len == max(counts)
    assert meta.decode_is_uniform == (min(counts) == max(counts))
    nonzero = [n for n in counts if n]
    assert meta.requires_padding == (len(set(nonzero)) > 1)


@pytest.mark.skipif(INTERPRET, reason="paged logits requires GPU kernels")
def test_metadata_graph_drives_valid_only_logits_and_topk():
    import ixformer.inference.functions as ops

    torch.manual_seed(29)
    padded, width = 16, 4096
    q_offsets = torch.tensor([0, 6, 9, 9], dtype=torch.int32)
    seq = torch.tensor([2307, 4097, 0], dtype=torch.int32)
    gpu_q, gpu_seq = q_offsets.to(DEVICE), seq.to(DEVICE)
    expected = reference(q_offsets, seq, 4, padded, True)
    metadata = {k: torch.empty_like(v, device=DEVICE) for k, v in expected.items()}
    q = torch.randn(padded, 32, 128, device=DEVICE, dtype=torch.bfloat16) * 0.1
    kv = torch.randn(27, 128, 128, device=DEVICE, dtype=torch.bfloat16) * 0.1
    weights = torch.rand(padded, 32, device=DEVICE, dtype=torch.bfloat16)
    table = torch.arange(27, device=DEVICE, dtype=torch.int32).reshape(3, 9)
    dirty = torch.empty(padded, width, device=DEVICE, dtype=torch.bfloat16)
    clean = torch.empty_like(dirty)
    actual_ids = torch.empty(padded, 512, device=DEVICE, dtype=torch.int32)
    expected_ids = torch.empty_like(actual_ids)

    def select(md, logits, indices, mask):
        ops.dsa_indexer_mqa_logits_with_kvcache(
            q,
            md["cu_seqlens_q"],
            md["cu_seqlens_kv"],
            kv,
            table,
            weights,
            logits=logits,
            max_q_len=6,
            max_kv_len=1024,
            max_context_len=width,
            valid_lens=md["valid_lens"],
            mask_invalid=mask,
        )
        ops.dsa_update_topk_indices(
            logits, md["cu_seqlen_ks"], md["cu_seqlen_ke"], 512, indices
        )

    def run():
        build_indexer_decode_metadata(
            gpu_q, gpu_seq, compress_ratio=4, flatten_decode_lens=True, **metadata
        )
        select(metadata, dirty, actual_ids, False)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for lengths in ([2307, 4097, 0], [7, 13, 0], [2307, 4097, 0]):
        seq = torch.tensor(lengths, dtype=torch.int32)
        gpu_seq.copy_(seq)
        expected = reference(q_offsets, seq, 4, padded, True)
        reference_gpu = {k: v.to(DEVICE) for k, v in expected.items()}
        dirty.fill_(float("nan"))
        graph.replay()
        select(reference_gpu, clean, expected_ids, True)
        assert_outputs(metadata, expected)
        torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)


def _mla_mapping_builder(capacity):
    from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
        IluFlashMLASparseMetadataBuilder,
    )

    builder = object.__new__(IluFlashMLASparseMetadataBuilder)
    builder.req_id_per_token_buffer = torch.full(
        (capacity,), -99, dtype=torch.int32, device=DEVICE
    )
    return builder


def _mla_mapping_metadata(query_start_loc_cpu, query_start_loc, padded):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    num_reqs = query_start_loc.shape[0] - 1
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        num_reqs=num_reqs,
        num_actual_tokens=padded,
        max_query_len=padded,
        max_seq_len=padded,
        block_table_tensor=torch.zeros((num_reqs, 1), dtype=torch.int32, device=DEVICE),
        slot_mapping=torch.zeros(padded, dtype=torch.int64, device=DEVICE),
    )


@pytest.mark.parametrize(
    "cpu_starts,device_starts,padded,expected",
    [
        ([0, 1, 3, 4, 4], [0, 2, 2, 3, 4], 8, [0, 0, 2, 3, 0, 0, 0, 0]),
        ([0, 6, 12, 12], [0, 6, 12, 12], 18, [0] * 6 + [1] * 6 + [0] * 6),
        ([0, 0, 0], [0, 0, 0], 4, [0] * 4),
        ([0], [0], 0, []),
    ],
)
def test_mla_request_mapping_uses_device_boundaries(
    cpu_starts, device_starts, padded, expected
):
    """Device boundaries can differ from the CPU shape used for dispatch."""
    builder = _mla_mapping_builder(max(padded, 1))
    common = _mla_mapping_metadata(
        torch.tensor(cpu_starts, dtype=torch.int32),
        torch.tensor(device_starts, dtype=torch.int32, device=DEVICE),
        padded,
    )
    result = builder._build_req_id_per_token(common)
    assert result.shape == (padded,)
    if padded:
        assert result.data_ptr() == builder.req_id_per_token_buffer.data_ptr()
    torch.testing.assert_close(
        result.cpu(), torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0
    )


def test_mla_request_mapping_keeps_capture_buffer_with_shared_cache():
    builder = _mla_mapping_builder(8)
    query = torch.tensor([0, 2, 3, 3], dtype=torch.int32)
    common = _mla_mapping_metadata(query, query.to(DEVICE), 4)
    cached = torch.tensor([0, 0, 1, 0], dtype=torch.int32, device=DEVICE)
    common._token_to_req_indices_cache = cached
    result = builder._build_req_id_per_token(common)
    assert result.data_ptr() == builder.req_id_per_token_buffer.data_ptr()
    assert result.data_ptr() != cached.data_ptr()
    cached.fill_(7)
    torch.testing.assert_close(
        result.cpu(), torch.tensor([0, 0, 1, 0], dtype=torch.int32), rtol=0, atol=0
    )


@pytest.mark.skipif(INTERPRET, reason="CUDA graph requires compiled GPU kernels")
def test_mla_request_mapping_queued_mtp_graph_batches():
    """Queue four draft updates per proposal across 4 -> 3 padded requests."""
    batches = [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 3],
        [0, 1, 2, 2, 2],
        [0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4],
    ]
    cpu_queries = [torch.tensor(q, dtype=torch.int32) for q in batches]
    device_queries = [q.to(DEVICE) for q in cpu_queries]
    query = device_queries[0].clone()
    builder = _mla_mapping_builder(4)
    common = _mla_mapping_metadata(cpu_queries[0], query, 4)
    mapping = builder._build_req_id_per_token(common)
    output = torch.empty_like(mapping)
    output.copy_(mapping)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output.copy_(mapping)

    snapshots = []
    expected = []
    for cpu_query, device_query in zip(cpu_queries, device_queries):
        query.copy_(device_query)
        common.query_start_loc_cpu = cpu_query
        for _ in range(4):
            common._token_to_req_indices_cache = None
            current = builder._build_req_id_per_token(common)
            assert current.data_ptr() == mapping.data_ptr()
            graph.replay()
            snapshots.append(output.clone())
            counts = (cpu_query[1:] - cpu_query[:-1]).tolist()
            ids = [req for req, count in enumerate(counts) for _ in range(count)]
            expected.append(ids + [0] * (4 - len(ids)))
    # No synchronization between proposal builds or graph replays above.
    actual = torch.stack(snapshots).cpu()
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0
    )
