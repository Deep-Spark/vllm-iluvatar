# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""CPU checks for the Iluvatar slot-mapping fallback."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.block_table import SlotMappingMode

from vllm_iluvatar.backends.ixaite.patches.block_table_slot_mapping import (
    _compute_slot_mapping_torch,
)


def _table(rows: list[list[int]], *, width: int) -> torch.Tensor:
    table = torch.zeros((len(rows), width), dtype=torch.int32)
    for row, ids in enumerate(rows):
        table[row, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
    return table


def _run(
    *,
    positions: list[int],
    query_start_loc: list[int],
    block_table: list[list[int]],
    block_size: int,
    blocks_per_kv_block: int = 1,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    interleave: int = 1,
    max_num_batched_tokens: int | None = None,
) -> torch.Tensor:
    num_tokens = len(positions)
    width = max(len(row) for row in block_table)
    slot_mapping = torch.zeros(max_num_batched_tokens or num_tokens, dtype=torch.int64)
    table = SimpleNamespace(
        slot_mapping_mode=SlotMappingMode.TOKEN_TO_KV_SLOT,
        max_num_batched_tokens=slot_mapping.numel(),
        slot_mapping=SimpleNamespace(gpu=slot_mapping),
        block_table=SimpleNamespace(gpu=_table(block_table, width=width)),
        block_size=block_size,
        kv_cache_block_size=block_size * blocks_per_kv_block,
        blocks_per_kv_block=blocks_per_kv_block,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=interleave,
    )
    _compute_slot_mapping_torch(
        table,
        len(query_start_loc) - 1,
        torch.tensor(query_start_loc, dtype=torch.int32),
        torch.tensor(positions, dtype=torch.int64),
    )
    return slot_mapping


def test_standard_blocks_match_position_within_kernel_block():
    slots = _run(
        positions=[0, 1, 16, 17],
        query_start_loc=[0, 4],
        block_table=[[4, 9]],
        block_size=16,
    )
    assert slots.tolist() == [4 * 16 + 0, 4 * 16 + 1, 9 * 16 + 0, 9 * 16 + 1]


def test_hybrid_blocks_index_the_expanded_kernel_table():
    # Allocator block is 33 kernel blocks, as on Qwen3.5 hybrid attention.
    # Position 528 is the first token of kernel block 33.
    slots = _run(
        positions=[0, 527, 528],
        query_start_loc=[0, 3],
        block_table=[list(range(40))],
        block_size=16,
        blocks_per_kv_block=33,
    )
    assert slots.tolist() == [0, 32 * 16 + 15, 33 * 16 + 0]


def test_context_parallel_rank_pads_non_local_tokens():
    rank0 = _run(
        positions=[0, 1, 2],
        query_start_loc=[0, 3],
        block_table=[[4, 5]],
        block_size=16,
        dcp_world_size=2,
        dcp_rank=0,
    )
    rank1 = _run(
        positions=[0, 1, 2],
        query_start_loc=[0, 3],
        block_table=[[4, 5]],
        block_size=16,
        dcp_world_size=2,
        dcp_rank=1,
    )
    assert rank0.tolist() == [4 * 16 + 0, PAD_SLOT_ID, 4 * 16 + 1]
    assert rank1.tolist() == [PAD_SLOT_ID, 4 * 16 + 0, PAD_SLOT_ID]


def test_two_requests_and_graph_padding():
    slots = _run(
        positions=[0, 15, 0],
        query_start_loc=[0, 2, 3],
        block_table=[[3], [6]],
        block_size=16,
        max_num_batched_tokens=5,
    )
    assert slots.tolist() == [3 * 16, 3 * 16 + 15, 6 * 16, PAD_SLOT_ID, PAD_SLOT_ID]


def _reference(
    *,
    positions: list[int],
    query_start_loc: list[int],
    block_table: list[list[int]],
    block_size: int,
    blocks_per_kv_block: int = 1,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    interleave: int = 1,
    max_num_batched_tokens: int | None = None,
) -> list[int]:
    num_tokens = len(positions)
    out = [PAD_SLOT_ID] * (max_num_batched_tokens or num_tokens)
    if num_tokens == 0:
        return out
    virtual_block_size = block_size * blocks_per_kv_block * dcp_world_size
    num_reqs = len(query_start_loc) - 1
    for token, pos in enumerate(positions):
        req = next(
            r
            for r in range(num_reqs)
            if query_start_loc[r] <= token < query_start_loc[r + 1]
        )
        virtual_block_index = pos // virtual_block_size
        virtual_block_offset = pos - virtual_block_index * virtual_block_size
        is_local = (virtual_block_offset // interleave) % dcp_world_size == dcp_rank
        local_offset = (
            virtual_block_offset // (dcp_world_size * interleave)
        ) * interleave + (virtual_block_offset % interleave)
        block_index = (
            virtual_block_index * blocks_per_kv_block + local_offset // block_size
        )
        if not is_local:
            out[token] = PAD_SLOT_ID
            continue
        block_number = block_table[req][block_index]
        out[token] = block_number * block_size + local_offset % block_size
    return out


def test_hybrid_context_parallel_matches_kernel_address():
    kwargs = dict(
        positions=[0, 1, 527, 528, 1056, 0, 16],
        query_start_loc=[0, 5, 7],
        block_table=[list(range(80)), [9, 10, 11, 12]],
        block_size=16,
        blocks_per_kv_block=33,
        dcp_world_size=2,
        interleave=1,
        max_num_batched_tokens=9,
    )
    for rank in (0, 1):
        got = _run(dcp_rank=rank, **kwargs).tolist()
        expect = _reference(dcp_rank=rank, **kwargs)
        assert got == expect
