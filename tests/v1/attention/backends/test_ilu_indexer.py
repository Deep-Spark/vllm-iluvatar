# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar DeepSeek V3.2 sparse-indexer metadata helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    get_max_decode_context_len,
    get_max_prefill_buffer_size,
    kv_spans_from_batches,
    kv_spans_from_batches_device,
    make_storage_block_size_build,
    split_indexer_prefill_chunks,
)


def _spans_device(start_seq_loc, seq_len_per_batch, compress_ratio):
    """Call the device variant with the arange the builder caches for it."""
    num_tokens = int(start_seq_loc[-1].item())
    return kv_spans_from_batches_device(
        cu_seqlens_q=start_seq_loc.int(),
        seq_len_per_batch=seq_len_per_batch.int(),
        token_arange_p1=torch.arange(1, num_tokens + 9, dtype=torch.int32),
        num_tokens=num_tokens,
        compress_ratio=compress_ratio,
    )


def _slice_tuple(chunk: tuple[slice, slice]) -> tuple[tuple[int, int], tuple[int, int]]:
    req_slice, query_slice = chunk
    return (
        (req_slice.start, req_slice.stop),
        (query_slice.start, query_slice.stop),
    )


def test_split_indexer_prefill_chunks_respects_workspace_and_logits_budget():
    chunks = split_indexer_prefill_chunks(
        seq_lens_cpu=torch.tensor([5, 7, 11]),
        query_lens_cpu=torch.tensor([2, 3, 4]),
        workspace_size=12,
        max_logits_bytes=12 * 4,
        request_offset=10,
    )

    assert [_slice_tuple(chunk) for chunk in chunks] == [
        ((10, 11), (0, 2)),
        ((11, 12), (0, 1)),
        ((11, 12), (1, 2)),
        ((11, 12), (2, 3)),
        ((12, 13), (0, 1)),
        ((12, 13), (1, 2)),
        ((12, 13), (2, 3)),
        ((12, 13), (3, 4)),
    ]


def test_kv_spans_from_batches_token_domain():
    starts, ends = kv_spans_from_batches(
        start_seq_loc=torch.tensor([0, 2, 5]),
        seq_len_per_batch=torch.tensor([5, 9]),
        device=torch.device("cpu"),
        compress_ratio=1,
    )

    assert starts.tolist() == [0, 0, 5, 5, 5]
    assert ends.tolist() == [4, 5, 12, 13, 14]
    assert starts.dtype == torch.int32
    assert ends.dtype == torch.int32


def test_kv_spans_from_batches_compressed_domain():
    starts, ends = kv_spans_from_batches(
        start_seq_loc=torch.tensor([0, 2, 5]),
        seq_len_per_batch=torch.tensor([8, 12]),
        device=torch.device("cpu"),
        compress_ratio=4,
    )

    assert starts.tolist() == [0, 0, 2, 2, 2]
    assert ends.tolist() == [1, 2, 4, 4, 5]


def test_kv_spans_from_batches_handles_empty_selection():
    starts, ends = kv_spans_from_batches(
        start_seq_loc=torch.tensor([0, 0, 0]),
        seq_len_per_batch=torch.tensor([5, 9]),
        device=torch.device("cpu"),
        compress_ratio=1,
    )

    assert starts.tolist() == []
    assert ends.tolist() == []
    assert starts.dtype == torch.int32
    assert ends.dtype == torch.int32


def test_kv_spans_device_matches_host_variant():
    """The decode path builds ks/ke on device to dodge a per-draft-step D2H;
    it must stay bit-identical to the host builder the prefill path still uses.
    """
    cases = [
        # token domain, ragged selection
        (torch.tensor([0, 2, 5]), torch.tensor([5, 9]), 1),
        # compressed domain (V4)
        (torch.tensor([0, 2, 5]), torch.tensor([8, 12]), 4),
        # uniform MTP decode: next_n=4 tokens per request
        (torch.tensor([0, 4, 8, 12]), torch.tensor([131, 4096, 77]), 4),
        # CUDA-graph padding: trailing requests carry query_len == 0
        (torch.tensor([0, 4, 8, 8, 8]), torch.tensor([64, 4096, 0, 0]), 4),
        # plain single-token decode
        (torch.tensor([0, 1, 2, 3]), torch.tensor([17, 33, 65]), 1),
        # a request in the *middle* contributes no token: the duplicate entry
        # it leaves in cu_seqlens_q must not capture any of its neighbours.
        (torch.tensor([0, 2, 2, 5, 6]), torch.tensor([100, 55, 4096, 7]), 1),
        (torch.tensor([0, 2, 2, 5, 6]), torch.tensor([100, 55, 4096, 7]), 4),
    ]
    for start_seq_loc, seq_lens, ratio in cases:
        host_s, host_e = kv_spans_from_batches(
            start_seq_loc=start_seq_loc,
            seq_len_per_batch=seq_lens,
            device=torch.device("cpu"),
            compress_ratio=ratio,
        )
        dev_s, dev_e = _spans_device(start_seq_loc, seq_lens, ratio)

        assert dev_s.tolist() == host_s.tolist(), (start_seq_loc, seq_lens, ratio)
        assert dev_e.tolist() == host_e.tolist(), (start_seq_loc, seq_lens, ratio)
        assert dev_s.dtype == torch.int32
        assert dev_e.dtype == torch.int32


def test_kv_spans_device_handles_empty_selection():
    starts, ends = _spans_device(
        torch.tensor([0, 0, 0]), torch.tensor([5, 9]), compress_ratio=1
    )

    assert starts.tolist() == []
    assert ends.tolist() == []
    assert starts.dtype == torch.int32
    assert ends.dtype == torch.int32


def test_kv_spans_device_needs_no_cpu_copy_of_seq_lens():
    """Regression: the builder must not touch ``seq_lens`` on the host.

    ``.cpu()`` / ``.item()`` on the device seq_lens is what cost a blocking
    16-byte D2H per draft step, so fail loudly if the implementation reaches
    for it again.
    """

    class NoHostAccess(torch.Tensor):
        @staticmethod
        def __new__(cls, data):
            return torch.Tensor._make_subclass(cls, data)

        def cpu(self, *args, **kwargs):  # noqa: D102
            raise AssertionError("seq_lens must not be copied to host")

        def tolist(self):  # noqa: D102
            raise AssertionError("seq_lens must not be copied to host")

    seq_lens = NoHostAccess(torch.tensor([64, 4096, 0, 0], dtype=torch.int32))
    starts, ends = kv_spans_from_batches_device(
        cu_seqlens_q=torch.tensor([0, 4, 8, 8, 8], dtype=torch.int32),
        seq_len_per_batch=seq_lens,
        token_arange_p1=torch.arange(1, 33, dtype=torch.int32),
        num_tokens=8,
        compress_ratio=4,
    )

    assert starts.numel() == 8
    assert ends.numel() == 8


def test_get_max_prefill_buffer_size_scales_with_model_len():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=163840),
    )

    assert get_max_prefill_buffer_size(vllm_config) == 163840 * 40


def test_get_max_decode_context_len_uses_cache_domain_upper_bound():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=4097),
    )

    assert get_max_decode_context_len(vllm_config, compress_ratio=1) == 4097
    assert get_max_decode_context_len(vllm_config, compress_ratio=4) == 1025


def test_storage_block_size_wrapper_is_scoped_to_compressed_build():
    seen = []

    def original(self, common_prefix_len, common_attn_metadata, fast_build=False):
        seen.append(self.kv_cache_spec.block_size)
        return common_prefix_len, common_attn_metadata, fast_build

    wrapped = make_storage_block_size_build(original)
    builder = SimpleNamespace(
        compress_ratio=4,
        kv_cache_spec=SimpleNamespace(block_size=256, storage_block_size=64),
    )

    assert wrapped(builder, 3, "meta", True) == (3, "meta", True)
    assert seen == [64]
    assert builder.kv_cache_spec.block_size == 256

    builder.compress_ratio = 1
    wrapped(builder, 0, None)
    assert seen == [64, 256]


def test_pcp_dual_chunk_causal_spans_and_query_subchunks():
    """Runner represents each PCP segment as a request ending at its own position."""
    from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
        IluDeepseekV32IndexerMetadataBuilder,
    )

    builder = object.__new__(IluDeepseekV32IndexerMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.compress_ratio = 1
    builder.max_prefill_buffer_size = 100
    # One request with cached prefix 8, local segments [8,9] and [14,15].
    starts = torch.tensor([0, 2, 4], dtype=torch.int32)
    ends = torch.tensor([10, 16], dtype=torch.int32)
    blocks = torch.tensor([[3], [3]], dtype=torch.int32)
    chunk = builder.build_one_prefill_chunk(
        slice(0, 2), slice(0, 4), starts, ends, blocks
    )
    assert chunk.valid_lens.tolist() == [9, 10, 15, 16]
    assert chunk.cu_seqlens_q.tolist() == [0, 2, 4]
    sub = builder.build_one_prefill_chunk(
        slice(1, 2), slice(0, 1), starts, ends, blocks
    )
    assert sub.valid_lens.tolist() == [15]
    assert (sub.token_start, sub.token_end) == (2, 3)


@pytest.mark.parametrize("seq_lens", [[0], [0, 0], [7, 0]])
def test_split_prefill_chunks_no_local_queries(seq_lens):
    assert (
        split_indexer_prefill_chunks(
            torch.tensor(seq_lens),
            torch.zeros(len(seq_lens), dtype=torch.int32),
            workspace_size=32,
            max_logits_bytes=64,
        )
        == []
    )
