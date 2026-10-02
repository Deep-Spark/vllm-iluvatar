# SPDX-License-Identifier: Apache-2.0
"""IluDeepseekSparseSWAMetadataBuilder.build() branch coverage (mocked ixformer)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import CommonAttentionMetadata

from vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa import (
    IluDeepseekSparseSWAMetadataBuilder,
    build_dspark_noncausal_swa_indices,
)


def _make_cad(
    *,
    query_lens: list[int],
    seq_lens: list[int],
    causal: bool,
    block_size: int = 4,
    is_prefilling: list[bool] | None = None,
) -> CommonAttentionMetadata:
    num_reqs = len(query_lens)
    num_tokens = sum(query_lens)
    qsl = [0]
    for q in query_lens:
        qsl.append(qsl[-1] + q)
    query_start_loc = torch.tensor(qsl, dtype=torch.int32)
    max_blocks = max((s + block_size - 1) // block_size for s in seq_lens)
    block_table = torch.stack(
        [torch.arange(max_blocks, dtype=torch.int32) + r * 100 for r in range(num_reqs)]
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    cad = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=causal,
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        is_prefilling=(
            None
            if is_prefilling is None
            else torch.tensor(is_prefilling, dtype=torch.bool)
        ),
    )
    # Avoid pinned-memory CPU→device copy in token_to_req_indices().
    cad._token_to_req_indices_cache = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32),
        torch.tensor(query_lens, dtype=torch.int32),
    )
    return cad


def _make_builder(
    *,
    is_dspark: bool,
    window_size: int = 8,
    num_spec: int = 5,
    decode_threshold: int | None = None,
    block_size: int = 4,
    max_tokens: int = 64,
) -> IluDeepseekSparseSWAMetadataBuilder:
    builder = object.__new__(IluDeepseekSparseSWAMetadataBuilder)
    builder.is_dspark = is_dspark
    builder.window_size = window_size
    builder.num_speculative_tokens = num_spec
    builder.noncausal_index_width = (
        cdiv(window_size + num_spec, 128) * 128 if is_dspark else 0
    )
    builder.block_size = block_size
    builder.decode_threshold = (
        decode_threshold
        if decode_threshold is not None
        else (1 + 2 * num_spec if is_dspark else 1)
    )
    builder.device = torch.device("cpu")
    builder.vllm_config = SimpleNamespace(
        speculative_config=(
            SimpleNamespace(use_dspark=lambda: is_dspark, method="dspark")
            if is_dspark
            else SimpleNamespace(use_dspark=lambda: False, method="mtp")
        )
    )
    builder.max_model_len = 1024
    builder.max_num_batched_tokens = max_tokens
    builder._max_tokens = max_tokens
    builder.token_to_req_indices = torch.zeros(max_tokens, dtype=torch.int32)
    builder.decode_swa_indices = torch.zeros(
        max_tokens, 1, window_size, dtype=torch.int32
    )
    builder.decode_swa_lens = torch.zeros(max_tokens, dtype=torch.int32)
    builder.prefill_swa_indices = torch.zeros(
        max_tokens, 1, window_size, dtype=torch.int32
    )
    builder.prefill_swa_lens = torch.zeros(max_tokens, dtype=torch.int32)
    builder.is_valid_token = torch.zeros(max_tokens, dtype=torch.bool)
    builder.no_replay_start = torch.zeros(max_tokens, dtype=torch.int32)
    builder._padded_replay_start = torch.zeros_like(builder.no_replay_start)
    builder.decode_swa_indices_noncausal = (
        torch.empty(
            max_tokens, 1, builder.noncausal_index_width, dtype=torch.int32
        )
        if is_dspark
        else None
    )
    builder.max_image_tokens = 0
    builder._build_deepseek_v4_metadata = MagicMock(return_value={})
    builder.build_tile_scheduler = MagicMock(
        return_value={
            "swaonly": None,
            "c4a": None,
            "c128a": None,
            "c1a": None,
            "c2a": None,
        }
    )
    builder._populate_prefill_opt_fields = MagicMock()
    return builder


def test_replay_start_is_zero_padded_to_cudagraph_request_count():
    builder = _make_builder(is_dspark=True)
    replay_start = torch.tensor([16, 0, 0], dtype=torch.int32)
    builder._padded_replay_start[3] = 123

    padded = builder._pad_replay_start(replay_start, num_reqs=4)

    assert padded.tolist() == [16, 0, 0, 0]
    assert padded.data_ptr() == builder._padded_replay_start.data_ptr()
    assert builder._pad_replay_start(replay_start, num_reqs=3).data_ptr() == (
        replay_start.data_ptr()
    )


def test_causal_decode_calls_plugin_and_sets_metadata_causal_true():
    builder = _make_builder(is_dspark=True, num_spec=5)
    cad = _make_cad(query_lens=[1], seq_lens=[12], causal=True, is_prefilling=[False])

    def _fake_plugin(*args, **kwargs):
        n = kwargs["num_tokens"]
        args[0][:n].fill_(42)
        args[1][:n] = 7

    with (
        patch(
            "torch.ops._C.v41_swa_indices_and_lens",
            side_effect=_fake_plugin,
            create=True,
        ) as mock_op,
        patch(
            "vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa.envs."
            "VLLM_ILU_DS4_PREFILL_AS_DECODE",
            False,
        ),
        patch(
            "vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa.envs."
            "VLLM_ILU_DS4_PREFILL_OPT",
            False,
        ),
    ):
        md = builder.build(0, cad)

    mock_op.assert_called_once()
    assert md.causal is True
    assert md.num_decode_tokens == 1
    assert md.decode_swa_lens.tolist() == [7]
    assert int(md.decode_swa_indices[0, 0, 0]) == 42


def test_noncausal_dspark_uses_python_indices_not_plugin():
    window_size, num_spec, block_size = 8, 5, 4
    builder = _make_builder(
        is_dspark=True,
        window_size=window_size,
        num_spec=num_spec,
        block_size=block_size,
    )
    ctx = 10
    cad = _make_cad(
        query_lens=[num_spec],
        seq_lens=[ctx + num_spec],
        causal=False,
        block_size=block_size,
        is_prefilling=[True],
    )

    with (
        patch(
            "torch.ops._C.v41_swa_indices_and_lens",
            create=True,
        ) as mock_op,
        patch(
            "vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa.envs."
            "VLLM_ILU_DS4_PREFILL_AS_DECODE",
            False,
        ),
        patch(
            "vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa.envs."
            "VLLM_ILU_DS4_PREFILL_OPT",
            False,
        ),
    ):
        md = builder.build(0, cad)

    mock_op.assert_not_called()
    assert md.causal is False
    assert md.num_decode_tokens == num_spec

    expected_indices, expected_lens = build_dspark_noncausal_swa_indices(
        seq_lens=cad.seq_lens,
        query_start_loc=cad.query_start_loc,
        block_table=cad.block_table_tensor,
        is_valid_token=cad.slot_mapping >= 0,
        block_size=block_size,
        window_size=window_size,
        index_width=builder.noncausal_index_width,
        num_decode_tokens=num_spec,
        device=torch.device("cpu"),
    )
    assert torch.equal(md.decode_swa_indices, expected_indices)
    assert torch.equal(md.decode_swa_lens, expected_lens)
    # All queries in the block share the same non-causal key set.
    first = md.decode_swa_indices[0, 0, : md.decode_swa_lens[0]].tolist()
    for q in range(1, num_spec):
        assert md.decode_swa_indices[q, 0, : md.decode_swa_lens[q]].tolist() == first


def test_noncausal_without_dspark_raises():
    builder = _make_builder(is_dspark=False, decode_threshold=8)
    cad = _make_cad(query_lens=[5], seq_lens=[15], causal=False)

    with (
        patch(
            "vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa.envs."
            "VLLM_ILU_DS4_PREFILL_AS_DECODE",
            False,
        ),
        pytest.raises(AssertionError, match="only supported for the DSpark"),
    ):
        builder.build(0, cad)
