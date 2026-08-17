# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar DeepSeek V3.2 sparse-indexer metadata helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV32IndexerBackend,
    get_max_decode_context_len,
    get_max_prefill_buffer_size,
    kv_spans_from_batches,
    split_indexer_prefill_chunks,
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


def test_deepseek_v32_indexer_backend_cache_layout():
    assert IluDeepseekV32IndexerBackend.get_name() == "DEEPSEEK_V32_INDEXER"
    assert IluDeepseekV32IndexerBackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=64,
        num_kv_heads=1,
        head_size=576,
    ) == (3, 64, 576)
    assert IluDeepseekV32IndexerBackend.get_kv_cache_stride_order() == (0, 1, 2)
    assert IluDeepseekV32IndexerBackend.get_kv_cache_stride_order(
        include_num_layers_dimension=True,
    ) == (0, 1, 2, 3)


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
