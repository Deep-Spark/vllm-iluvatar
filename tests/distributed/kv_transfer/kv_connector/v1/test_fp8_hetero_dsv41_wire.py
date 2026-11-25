# SPDX-License-Identifier: Apache-2.0
"""DSv4.1 host-wire geometry for hetero FP8 push."""

from __future__ import annotations

import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_common.dsv41 import (
    Dsv41Layout,
    wire_dsv41_layout,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_common.fp8_hetero.wire_layout import (
    QUANT_INDEXER,
    QUANT_PACK,
    QUANT_PACK_COMPRESSED,
    _to_wire_spec,
    quant_kind_for,
    subpage_ratio_for,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_common.semantic_pairing import (
    SemanticTransferRegion,
    select_pp_regions_and_pairings,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_push.xfer_plan import (
    compile_xfer_plan,
    pair_kernel_pages,
)


def test_dsv41_compressed_main_maps_to_fp8_wire_alignment_64():
    compute = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=512,
        tokens_per_state=2,
        cache_dtype_str="i8g64",
        storage_block_size=256,
    )
    wire = _to_wire_spec(compute, "layers.0.main")
    assert wire.cache_dtype_str == "fp8_ds_mla"
    assert wire.model_version == "deepseek_v4"
    assert wire.state_content_bytes == 584
    assert wire.alignment == 64
    assert quant_kind_for("layers.0.main", wire, compute) == QUANT_PACK_COMPRESSED


def test_dsv41_bf16_main_maps_to_fp8_wire_alignment_64():
    compute = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        tokens_per_state=1,
        cache_dtype_str="bfloat16",
        model_version="deepseek_v4",
        state_content_bytes=584,
        alignment=64,
    )
    wire = _to_wire_spec(compute, "layers.1.main", prefer_v41=True)
    assert wire.cache_dtype_str == "fp8_ds_mla"
    assert wire.model_version == "deepseek_v4"
    assert wire.alignment == 64
    assert quant_kind_for("layers.1.main", wire, compute) == QUANT_PACK


def test_dsv41_split_indexer_maps_to_fp8_wire_record():
    compute = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        state_content_bytes=128,
        tokens_per_state=1,
        cache_dtype_str="int8_indexer_split",
        alignment=64,
    )
    wire = _to_wire_spec(compute, "layers.2.indexer.k_cache")
    assert wire.dtype == torch.uint8
    assert wire.head_size == 132
    assert wire.cache_dtype_str is None
    assert quant_kind_for("layers.2.indexer.k_cache", wire, compute) == QUANT_INDEXER


def test_dsv4_main_keeps_alignment_576():
    compute = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        tokens_per_state=1,
        cache_dtype_str="bfloat16",
        model_version="deepseek_v4",
    )
    wire = _to_wire_spec(compute, "layers.0.main")
    assert wire.model_version == "deepseek_v4"
    assert wire.alignment == 576
    assert quant_kind_for("layers.0.main", wire, compute) == QUANT_PACK


def test_dsv41_split_indexer_576_compute_align_drops_wire_padding():
    compute = MLAAttentionSpec(
        block_size=1024,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        state_content_bytes=128,
        tokens_per_state=2,
        cache_dtype_str="int8_indexer_split",
        alignment=576,
    )
    wire = _to_wire_spec(compute, "layers.2.attn.indexer.k_cache")
    assert wire.alignment == 64
    assert wire.page_size_bytes == 512 * 132
    assert subpage_ratio_for(wire, 64) == 16
    assert wire.page_size_bytes // subpage_ratio_for(wire, 64) == 32 * 132


def test_wire_dsv41_layout_maps_hetero_staging_to_fp8_legacy():
    layout = Dsv41Layout(
        num_hidden_layers=40,
        kv_source_layers=(2, 8, 14, 20),
        index_source_layers=(2, 8, 14, 20),
        compress_ratios=(2,) * 20 + (1,) * 20,
        swa_bounded_replay=True,
        producer_layer_end=20,
        main_kv_format="i4g32",
        indexer_kv_layout="split",
    )
    wire = wire_dsv41_layout(layout, host_staging_producer=True, hetero_enabled=True)
    assert wire is not None
    assert wire.main_kv_format == "none"
    assert wire.indexer_kv_layout == "legacy"
    assert wire.kv_source_layers == (2, 8, 14, 20)
    assert (
        wire_dsv41_layout(layout, host_staging_producer=False, hetero_enabled=False)
        == layout
    )


def test_compile_xfer_plan_carries_per_bucket_subpage_ratios():
    import numpy as np

    plan = compile_xfer_plan(
        local_num_blocks=10,
        remote_num_blocks=100,
        local_physical_per_logical=1,
        remote_physical_per_logical=1,
        block_size_ratio=1,
        num_local_fa_regions=2,
        num_remote_fa_regions=2,
        group_spec_types=(),
        conv_decomp=None,
        wire_desc_offsets=(0, 80),
        local_region_page_ratios=(8, 16),
        local_region_ids_by_group=[
            np.asarray([], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
        ],
        remote_region_ids_by_group=[
            np.asarray([], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
        ],
    )
    assert [(b.local_group_idx, b.local_page_ratio) for b in plan.buckets] == [
        (0, 1),
        (1, 8),
        (2, 16),
    ]
    # 1500 tokens: P main=3 blocks (8x -> 24 subpages), D=24 blocks (1x);
    # P indexer=2 blocks (16x -> 32 subpages), D=24 blocks -> truncated to 24.
    l_main, r_main = pair_kernel_pages(
        [10, 11, 12], list(range(24)), local_ratio=8, remote_ratio=1
    )
    assert len(l_main) == 24 and l_main[0] == 80 and l_main[-1] == 103
    l_idx, r_idx = pair_kernel_pages(
        [5, 6], list(range(24)), local_ratio=16, remote_ratio=1
    )
    assert len(l_idx) == 24 and l_idx[0] == 80 and l_idx[-1] == 103
    assert r_idx == list(range(24))


def test_select_pp_regions_empty_remote_producer_stage_is_inactive():
    import numpy as np

    d_reg = SemanticTransferRegion(
        region_id=0,
        block_len=18688,
        shared_by=["layers.2.attn"],
        layer_indices=[2],
        group_ids=[1],
        cache_kinds=["main_mla"],
        identities=["2:main_mla"],
        kernel_blocks=1,
        block_tokens=64,
    )
    _, _, sel_l, sel_r, pairings, inactive = select_pp_regions_and_pairings(
        local_pp_split="40",
        local_pp_rank=0,
        local_layer_region_ids={"layers.2.attn": [0]},
        local_region_ids_by_group=[np.asarray([0], dtype=np.int64)],
        remote_pp_split="3,3,34",
        remote_pp_rank=1,
        remote_layer_region_ids={},
        remote_region_ids_by_group=[np.asarray([], dtype=np.int64)],
        is_hma_required=True,
        local_transfer_regions=[d_reg],
        remote_transfer_regions=[],
        allow_empty_skipped_overlap=True,
    )
    assert inactive is True
    assert sel_l == [] and sel_r == [] and pairings == []
