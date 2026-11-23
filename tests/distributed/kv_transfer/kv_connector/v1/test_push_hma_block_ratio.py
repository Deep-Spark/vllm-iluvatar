# SPDX-License-Identifier: Apache-2.0
"""HMA handshake and page pairing when P and D scheduler blocks differ."""

from __future__ import annotations

import numpy as np
import pytest

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_common.semantic_pairing import (
    validate_pp_hma_region_lengths,
    validate_region_geometry_pair,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_common.types import (
    HmaTransferRoute,
    SemanticTransferRegion,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_push.xfer_plan import (
    nixl_block_size_ratio,
    pair_kernel_pages,
)


def _region(region_id: int, tokens: int, kernels: int, block_len: int) -> SemanticTransferRegion:
    return SemanticTransferRegion(
        region_id=region_id,
        block_len=block_len,
        shared_by=["layer"],
        layer_indices=[0],
        group_ids=[0],
        cache_kinds=["kv"],
        identities=["attn"],
        kernel_blocks=kernels,
        block_tokens=tokens,
    )


def _route(local_g: int, remote_g: int, is_ssm: bool, local_r: int, remote_r: int):
    return HmaTransferRoute(
        local_group_idx=local_g,
        remote_group_idx=remote_g,
        is_ssm=is_ssm,
        local_desc_region_ids=np.asarray([local_r], dtype=np.int64),
        remote_desc_region_ids=np.asarray([remote_r], dtype=np.int64),
        raw_local_region_ids=(local_r,),
        raw_remote_region_ids=(remote_r,),
    )


def test_geometry_accepts_4480_over_640_attention_and_mamba_state():
    attn_p = _region(0, 4480, 70, 4096)
    attn_d = _region(0, 640, 10, 4096)
    # Same state bytes: mamba block_tokens differ, kernel page count stays 1.
    ssm_p = _region(1, 4480, 1, 16384)
    ssm_d = _region(1, 640, 1, 16384)
    assert (
        validate_region_geometry_pair(
            attn_p,
            attn_d,
            identity="attn",
            local_region_id=0,
            remote_region_id=0,
        )
        == 7
    )
    validate_pp_hma_region_lengths(
        local_block_lens=[4096, 16384],
        remote_block_lens=[4096, 16384],
        local_region_ids_by_group=[],
        remote_region_ids_by_group=[],
        selected_remote_region_ids=[0, 1],
        block_size_ratio=1,
        hybrid_mla_mamba=True,
        local_transfer_regions=[attn_p, ssm_p],
        remote_transfer_regions=[attn_d, ssm_d],
        routes=[
            _route(0, 0, False, 0, 0),
            _route(2, 2, True, 1, 1),
        ],
    )


def test_geometry_accepts_padded_mamba_page_and_the_reverse_direction():
    # D validating P: local scheduler block is the smaller one.
    attn_d = _region(0, 640, 10, 4096)
    attn_p = _region(0, 4480, 70, 28672)
    ssm_d = _region(1, 640, 1, 8192)
    ssm_p = _region(1, 4480, 1, 8192 * 7)
    validate_pp_hma_region_lengths(
        local_block_lens=[4096, 8192],
        remote_block_lens=[28672, 8192 * 7],
        local_region_ids_by_group=[],
        remote_region_ids_by_group=[],
        selected_remote_region_ids=[0, 1],
        block_size_ratio=1,
        hybrid_mla_mamba=True,
        local_transfer_regions=[attn_d, ssm_d],
        remote_transfer_regions=[attn_p, ssm_p],
        routes=[
            _route(0, 0, False, 0, 0),
            _route(2, 2, True, 1, 1),
        ],
    )


def test_geometry_accepts_kernel_pages_against_one_scheduler_slot():
    # Live handshake: P kernel_blocks=70, D kernel_blocks=1, tokens 4480 vs 640.
    attn_p = _region(0, 4480, 70, 4096)
    attn_d = _region(0, 640, 1, 4096)
    ssm_p = _region(1, 4480, 1, 16384)
    ssm_d = _region(1, 640, 1, 16384)
    validate_pp_hma_region_lengths(
        local_block_lens=[4096, 16384],
        remote_block_lens=[4096, 16384],
        local_region_ids_by_group=[],
        remote_region_ids_by_group=[],
        selected_remote_region_ids=[0, 1],
        block_size_ratio=1,
        hybrid_mla_mamba=True,
        local_transfer_regions=[attn_p, ssm_p],
        remote_transfer_regions=[attn_d, ssm_d],
        routes=[
            _route(0, 0, False, 0, 0),
            _route(5, 2, True, 0, 0),
        ],
    )


def test_geometry_rejects_max_model_len_mamba_span():
    attn_p = _region(0, 4480, 70, 4096)
    mamba_d = _region(0, 32768, 1, 4096)
    with pytest.raises(ValueError, match="not divisible"):
        validate_region_geometry_pair(
            attn_p,
            mamba_d,
            identity="group0",
            local_region_id=0,
            remote_region_id=0,
        )


def test_pair_kernel_pages_covers_the_shorter_side():
    # 1000 tokens: one 4480-block on P (70 pages), two 640-blocks on D (20 pages).
    local, remote = pair_kernel_pages(
        [5],
        [9, 10],
        local_ratio=70,
        remote_ratio=10,
    )
    assert local == [5 * 70 + i for i in range(20)]
    assert remote == [9 * 10 + i for i in range(10)] + [10 * 10 + i for i in range(10)]

    # 300 tokens fits in one D block: 10 kernel pages.
    local, remote = pair_kernel_pages([5], [9], local_ratio=70, remote_ratio=10)
    assert len(local) == len(remote) == 10
    assert local[0] == 5 * 70
    assert remote[-1] == 9 * 10 + 9


def test_nixl_block_size_ratio_keeps_kernel_page_facing_larger_remote_block():
    assert nixl_block_size_ratio(64, 64) == 1
    assert nixl_block_size_ratio(4480, 640) == 7
    assert nixl_block_size_ratio(64, 640) == 1
    with pytest.raises(AssertionError):
        nixl_block_size_ratio(100, 64)


def test_geometry_accepts_non_divisible_scheduler_blocks_sharing_kernel_page():
    # P PP8 (4480 = 70 * 64) vs D TP2 (2304 = 36 * 64)
    attn_p = _region(0, 4480, 70, 4096)
    attn_d = _region(0, 2304, 36, 4096)
    ssm_p = _region(1, 4480, 1, 16384)
    ssm_d = _region(1, 2304, 1, 16384)
    assert (
        validate_region_geometry_pair(
            attn_p,
            attn_d,
            identity="attn",
            local_region_id=0,
            remote_region_id=0,
        )
        == 1
    )
    validate_pp_hma_region_lengths(
        local_block_lens=[4096, 16384],
        remote_block_lens=[4096, 16384],
        local_region_ids_by_group=[],
        remote_region_ids_by_group=[],
        selected_remote_region_ids=[0, 1],
        block_size_ratio=1,
        hybrid_mla_mamba=True,
        local_transfer_regions=[attn_p, ssm_p],
        remote_transfer_regions=[attn_d, ssm_d],
        routes=[
            _route(0, 0, False, 0, 0),
            _route(2, 2, True, 1, 1),
        ],
    )
    local, remote = pair_kernel_pages(
        [12],
        [24, 25],
        local_ratio=70,
        remote_ratio=36,
        local_page_offset=12 * 70,
        remote_page_offset=24 * 36,
    )
    assert local == [12 * 70 + 24 + i for i in range(46)]
    assert remote == [24 * 36 + i for i in range(36)] + [25 * 36 + i for i in range(10)]

