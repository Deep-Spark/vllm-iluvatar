# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the Iluvatar DeepSeek-V4 DSpark non-causal SWA path.

These tests run on CPU because the non-causal branch uses a pure-PyTorch
metadata helper. They verify the private W4A8 builder and independent non-causal metadata
helper. The frozen baseline raw-FP8 builder must continue rejecting DSpark;
this test does not introduce raw-FP8 support into either route.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

from vllm_iluvatar.backends.ixaite.attention.sparse_swa import (
    IluDeepseekSparseSWAMetadataBuilder,
)
from vllm_iluvatar.backends.ixaite.attention.sparse_swa import (
    build_dspark_noncausal_swa_indices as _w4a8_compute_dspark,
)
from vllm_iluvatar.models.deepseek_v4.sparse_swa import (
    IluDeepseekV4Fp8SWAMetadataBuilder,
)

WINDOW_SIZE = 64
BLOCK_SIZE = 16
HEAD_SIZE = 512
NUM_SPEC_TOKENS = 5


def _make_spec_config(num_speculative_tokens: int, parallel_drafting: bool):
    return SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        parallel_drafting=parallel_drafting,
        use_dspark=lambda: True,
    )


def _make_vllm_config(
    *,
    num_speculative_tokens: int = 0,
    parallel_drafting: bool = False,
    compress_ratios: list[int] | None = None,
    max_num_batched_tokens: int = 128,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                sliding_window=WINDOW_SIZE,
                compress_ratios=compress_ratios or [1, 4, 128],
            ),
            max_model_len=16384,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=64,
        ),
        speculative_config=(
            _make_spec_config(num_speculative_tokens, parallel_drafting)
            if num_speculative_tokens > 0
            else None
        ),
    )


def _make_kv_cache_spec():
    return SlidingWindowMLASpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=HEAD_SIZE,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=WINDOW_SIZE,
    )


def _make_common_attn_metadata(
    seq_lens: list[int],
    query_lens: list[int],
    block_table: torch.Tensor,
    causal: bool,
    device: torch.device,
) -> Any:
    num_reqs = len(seq_lens)
    num_tokens = sum(query_lens)
    query_start_loc = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device=device
    ).cumsum(0, dtype=torch.int32)
    query_start_loc_cpu = query_start_loc.cpu()
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device=device)

    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    offset = 0
    for r, ql in enumerate(query_lens):
        token_to_req[offset : offset + ql] = r
        offset += ql

    return SimpleNamespace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_t,
        seq_lens_cpu_upper_bound=seq_lens_t.cpu(),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=causal,
        token_to_req_indices=lambda buffer: token_to_req,
    )


def _reference_noncausal_indices_lens(
    seq_lens: list[int],
    query_lens: list[int],
    block_table: torch.Tensor,
    window_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """CPU reference for the DSpark non-causal window.

    Returns ``(indices [num_tokens, width], lens [num_tokens], width)``.
    """
    num_reqs = len(seq_lens)
    num_tokens = sum(query_lens)
    max_query_len = max(query_lens) if query_lens else 0
    width = cdiv(window_size + max_query_len, 128) * 128

    indices = torch.full((num_tokens, width), -1, dtype=torch.int32, device="cpu")
    lens = torch.zeros(num_tokens, dtype=torch.int32, device="cpu")
    bt_cpu = block_table.detach().to("cpu")

    t = 0
    for r in range(num_reqs):
        seq_len = seq_lens[r]
        query_len = query_lens[r]
        prefix_len = seq_len - query_len
        start_pos = max(prefix_len - window_size, 0)
        end_pos = seq_len
        swa_len = end_pos - start_pos
        for _ in range(query_len):
            lens[t] = swa_len
            for off in range(swa_len):
                p = start_pos + off
                blk = int(bt_cpu[r, p // block_size].item())
                indices[t, off] = blk * block_size + (p % block_size)
            t += 1

    return indices, lens, width


def _make_block_table(
    num_reqs: int, seq_lens: list[int], device: torch.device
) -> torch.Tensor:
    max_blocks = (max(seq_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32, device=device)
    next_block = 0
    for r, sl in enumerate(seq_lens):
        blocks = (sl + BLOCK_SIZE - 1) // BLOCK_SIZE
        table[r, :blocks] = torch.arange(
            next_block, next_block + blocks, dtype=torch.int32, device=device
        )
        next_block += blocks
    return table


def _test_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _assert_int_tensors_equal(got: torch.Tensor, ref: torch.Tensor) -> None:
    # Avoid torch.equal under a leftover torch.device("cuda") context:
    # .cpu() can stay on CUDA and equal then mixes devices.
    assert got.detach().tolist() == ref.detach().tolist()


def test_w4a8_builder_dspark_noncausal_width():
    device = torch.device("cpu")
    vllm_config = _make_vllm_config(
        num_speculative_tokens=NUM_SPEC_TOKENS,
        parallel_drafting=False,
    )
    builder = IluDeepseekSparseSWAMetadataBuilder(
        kv_cache_spec=_make_kv_cache_spec(),
        layer_names=["layer_0.swa_cache"],
        vllm_config=vllm_config,
        device=device,
    )

    seq_lens = [80, 50]
    query_lens = [1 + NUM_SPEC_TOKENS, 1 + NUM_SPEC_TOKENS]
    block_table = _make_block_table(len(seq_lens), seq_lens, device)
    metadata = _make_common_attn_metadata(
        seq_lens, query_lens, block_table, causal=False, device=device
    )

    swa_metadata = builder.build(0, metadata)

    expected_width = cdiv(WINDOW_SIZE + max(query_lens), 128) * 128
    assert swa_metadata.decode_swa_indices.shape == (
        sum(query_lens),
        1,
        expected_width,
    )
    assert swa_metadata.decode_swa_lens.shape == (sum(query_lens),)

    ref_indices, ref_lens, ref_width = _reference_noncausal_indices_lens(
        seq_lens, query_lens, block_table, WINDOW_SIZE, BLOCK_SIZE
    )
    assert expected_width == ref_width
    _assert_int_tensors_equal(
        swa_metadata.decode_swa_indices.view(sum(query_lens), -1),
        ref_indices,
    )
    _assert_int_tensors_equal(swa_metadata.decode_swa_lens, ref_lens)


# The causal branch requires the selected ixaite metadata kernel.
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ixaite causal metadata requires GPU"
)
def test_w4a8_builder_causal_unchanged():
    device = _test_device()
    vllm_config = _make_vllm_config(num_speculative_tokens=NUM_SPEC_TOKENS)
    builder = IluDeepseekSparseSWAMetadataBuilder(
        kv_cache_spec=_make_kv_cache_spec(),
        layer_names=["layer_0.swa_cache"],
        vllm_config=vllm_config,
        device=device,
    )

    seq_lens = [80, 50]
    query_lens = [1, 1]
    block_table = _make_block_table(len(seq_lens), seq_lens, device)
    metadata = _make_common_attn_metadata(
        seq_lens, query_lens, block_table, causal=True, device=device
    )

    swa_metadata = builder.build(0, metadata)
    assert swa_metadata.decode_swa_indices.shape == (sum(query_lens), 1, WINDOW_SIZE)


def test_baseline_fp8_builder_rejects_dspark_noncausal():
    device = torch.device("cpu")
    vllm_config = _make_vllm_config(
        num_speculative_tokens=NUM_SPEC_TOKENS,
        parallel_drafting=False,
    )
    builder = IluDeepseekV4Fp8SWAMetadataBuilder(
        kv_cache_spec=_make_kv_cache_spec(),
        layer_names=["layer_0.swa_cache"],
        vllm_config=vllm_config,
        device=device,
    )

    seq_lens = [80, 50]
    query_lens = [1 + NUM_SPEC_TOKENS, 1 + NUM_SPEC_TOKENS]
    block_table = _make_block_table(len(seq_lens), seq_lens, device)
    metadata = _make_common_attn_metadata(
        seq_lens, query_lens, block_table, causal=False, device=device
    )

    # Baseline never implemented this non-causal FP8 route; retain that boundary.
    with pytest.raises(NotImplementedError, match="only supports causal"):
        builder.build(0, metadata)


def test_baseline_fp8_builder_rejects_dspark_without_spec_config():
    device = _test_device()
    vllm_config = _make_vllm_config()
    builder = IluDeepseekV4Fp8SWAMetadataBuilder(
        kv_cache_spec=_make_kv_cache_spec(),
        layer_names=["layer_0.swa_cache"],
        vllm_config=vllm_config,
        device=device,
    )

    seq_lens = [80]
    query_lens = [1]
    block_table = _make_block_table(len(seq_lens), seq_lens, device)
    metadata = _make_common_attn_metadata(
        seq_lens, query_lens, block_table, causal=False, device=device
    )

    with pytest.raises(NotImplementedError, match="only supports causal"):
        builder.build(0, metadata)


def test_noncausal_helper_invalid_token_zeros_len():
    device = torch.device("cpu")
    width = 128
    swa_indices = torch.full((4, 1, width), -1, dtype=torch.int32)
    swa_lens = torch.ones(4, dtype=torch.int32)
    is_valid = torch.tensor([True, False, True, True], dtype=torch.bool)

    seq_lens = torch.tensor([10, 10, 10, 10], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    block_table = _make_block_table(4, [10, 10, 10, 10], device)

    _w4a8_compute_dspark(
        out_indices=swa_indices,
        out_lens=swa_lens,
        window_size=WINDOW_SIZE,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        is_valid_token=is_valid,
        block_table=block_table,
        block_size=BLOCK_SIZE,
        index_width=width,
        num_decode_tokens=4,
        device=device,
    )

    assert swa_lens[1].item() == 0
    assert swa_lens[0].item() > 0
    assert swa_lens[2].item() > 0
    assert swa_lens[3].item() > 0
