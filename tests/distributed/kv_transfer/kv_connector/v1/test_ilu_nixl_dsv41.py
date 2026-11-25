# SPDX-License-Identifier: Apache-2.0
"""Unit tests for IluNixl Pull Connector DeepSeek-V4.1 support."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1 import skip_layers
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.base_scheduler import (
    IluNixlBaseConnectorScheduler,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.base_worker import (
    IluNixlWorkerMixin,
    RegionPairing,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.metadata import (
    IluNixlAgentMetadata,
    IluNixlConnectorMetadata,
    IluNixlTransferRegion,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_common.dsv41 import (
    Dsv41Layout,
    compute_prefix_replay_alignment,
    legal_decode_cut_layers,
    resolve_dsv41_layout,
    validate_decode_pp_split,
    validate_dsv41_peer_handshake,
)


def _v41_hf_config():
    # 40-layer V4.1 layout: layers 0,1 pure SWA (cr=0); 2..19 cr=2; 20..39 cr=1.
    return SimpleNamespace(
        model_type="deepseek_v41",
        num_hidden_layers=40,
        kv_source_layer_ids=(2, 8, 14, 20),
        index_source_layer_ids=(2, 8, 14, 20, 24, 28, 32, 36),
        compress_ratios=tuple([0, 0] + [2] * 18 + [1] * 20),
    )


def _v41_vllm_config(
    *,
    role: str = "kv_consumer",
    pp_size: int = 1,
    tp_size: int = 2,
    swa_bounded_replay: bool = True,
    use_v2_model_runner: bool = True,
):
    hf = _v41_hf_config()
    return SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        model_config=SimpleNamespace(hf_text_config=hf, hf_config=hf),
        cache_config=SimpleNamespace(
            block_size=128,
            cache_dtype="fp8_ds_mla",
            swa_bounded_replay=swa_bounded_replay,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size,
            tensor_parallel_size=tp_size,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_index=0,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        kv_transfer_config=SimpleNamespace(
            kv_role=role,
            kv_connector="IluNixlPullConnector",
            engine_id=f"eng-{role}",
            kv_buffer_device="cuda",
            get_from_extra_config=lambda _k, default=None: default,
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def test_resolve_dsv41_layout_truncates_producer_and_keeps_consumer_full(monkeypatch):
    monkeypatch.delenv("VLLM_ILU_PD_PREFILL_FULL_DEPTH", raising=False)
    monkeypatch.delenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", raising=False)
    monkeypatch.delenv("VLLM_ILU_DS41_INDEXER_KV_LAYOUT", raising=False)

    p_cfg = _v41_vllm_config(role="kv_producer", swa_bounded_replay=True)
    p_layout = resolve_dsv41_layout(p_cfg)
    assert p_layout is not None
    assert p_layout.swa_bounded_replay is True
    assert p_layout.producer_layer_end == 20
    assert p_layout.kv_layer_end == 21

    d_cfg = _v41_vllm_config(role="kv_consumer", swa_bounded_replay=True)
    d_layout = resolve_dsv41_layout(d_cfg)
    assert d_layout is not None
    assert d_layout.producer_layer_end is None
    assert d_layout.kv_layer_end is None

    # Non-V4.1 returns None.
    other = _v41_vllm_config()
    other.model_config.hf_text_config.model_type = "qwen3"
    assert resolve_dsv41_layout(other) is None


def test_decode_pp_split_validation_rejects_cut_inside_kv_group():
    d_cfg = _v41_vllm_config(role="kv_consumer", pp_size=8)
    layout = resolve_dsv41_layout(d_cfg)
    assert layout is not None
    assert legal_decode_cut_layers(layout) == (1, 2, 8, 14, 20)

    # Default even PP=8 cut [0,5,10,15,20,25,30,35] breaks kv groups on D.
    with pytest.raises(ValueError, match="cannot cut inside a KV-sharing group"):
        validate_decode_pp_split(d_cfg, layout, pp_split="5,5,5,5,5,5,5,5")

    # Legal PP=4 cut landing on [0, 2, 8, 20].
    d_cfg_pp4 = _v41_vllm_config(role="kv_consumer", pp_size=4)
    validate_decode_pp_split(d_cfg_pp4, layout, pp_split="2,6,12,20")


def test_handshake_consistency_hard_fails_on_v41_mismatches():
    local = resolve_dsv41_layout(_v41_vllm_config(role="kv_consumer"))
    assert local is not None

    with pytest.raises(RuntimeError, match="swa_bounded_replay mismatch"):
        validate_dsv41_peer_handshake(
            local,
            remote_engine_id="prefill-1",
            remote_swa_bounded_replay=False,
            remote_kv_source_layers=local.kv_source_layers,
            remote_index_source_layers=local.index_source_layers,
            remote_compress_ratios=local.compress_ratios,
        )

    with pytest.raises(RuntimeError, match="kv_source_layer_ids mismatch"):
        validate_dsv41_peer_handshake(
            local,
            remote_engine_id="prefill-1",
            remote_swa_bounded_replay=True,
            remote_kv_source_layers=(2, 8, 14, 22),
            remote_index_source_layers=local.index_source_layers,
            remote_compress_ratios=local.compress_ratios,
        )

    with pytest.raises(RuntimeError, match="index_source_layer_ids mismatch"):
        validate_dsv41_peer_handshake(
            local,
            remote_engine_id="prefill-1",
            remote_swa_bounded_replay=True,
            remote_kv_source_layers=local.kv_source_layers,
            remote_index_source_layers=(2, 8, 14, 20),
            remote_compress_ratios=local.compress_ratios,
        )


def test_scheduler_clips_bounded_replay_and_ring_preserving_group_seats():
    main_spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
    )
    swa_replay = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        bounded_replay=True,
    )
    ring_spec = CircularBufferSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1024,
        head_size_v=0,
        dtype=torch.float32,
    )
    groups = [
        KVCacheGroupSpec(["layers.2.attn"], main_spec),
        KVCacheGroupSpec(["layers.2.attn.swa_cache"], swa_replay),
        KVCacheGroupSpec(["layers.2.attn.compressor.state_cache"], ring_spec),
    ]
    kv_cfg = SimpleNamespace(
        kv_cache_groups=groups,
        transfer_groups=tuple(groups),
        transfer_group_ids=(0, 1, 2),
        select_transfer_block_ids=lambda ids: tuple(ids),
    )

    # Bounded replay ON -> SWA and ring both clipped to [] while seats stay len=3.
    sched_on = object.__new__(IluNixlBaseConnectorScheduler)
    sched_on.kv_cache_config = kv_cfg
    sched_on._is_hma_required = True
    sched_on.blocks_per_sw = [0, 3, 0]
    sched_on._ssm_spec_blocks = [None, None, None]
    sched_on.dsv41_layout = Dsv41Layout(
        num_hidden_layers=40,
        kv_source_layers=(2,),
        index_source_layers=(2,),
        compress_ratios=(0, 0, 2),
        swa_bounded_replay=True,
        producer_layer_end=None,
    )
    sched_on._init_ilu_transfer_clip_rules()
    clipped_on = sched_on.get_exchange_clipped_blocks(
        ([10, 11, 12], [20, 21, 22, 23], [30, 31])
    )
    assert clipped_on == ([10, 11, 12], [], [])

    # Bounded replay OFF -> SWA keeps tail window (3 blocks), ring keeps first block.
    swa_plain = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        bounded_replay=False,
    )
    groups_off = [
        KVCacheGroupSpec(["layers.2.attn"], main_spec),
        KVCacheGroupSpec(["layers.2.attn.swa_cache"], swa_plain),
        KVCacheGroupSpec(["layers.2.attn.compressor.state_cache"], ring_spec),
    ]
    kv_cfg_off = SimpleNamespace(
        kv_cache_groups=groups_off,
        transfer_groups=tuple(groups_off),
        transfer_group_ids=(0, 1, 2),
        select_transfer_block_ids=lambda ids: tuple(ids),
    )
    sched_off = object.__new__(IluNixlBaseConnectorScheduler)
    sched_off.kv_cache_config = kv_cfg_off
    sched_off._is_hma_required = True
    sched_off.blocks_per_sw = [0, 3, 0]
    sched_off._ssm_spec_blocks = [None, None, None]
    sched_off.dsv41_layout = Dsv41Layout(
        num_hidden_layers=40,
        kv_source_layers=(2,),
        index_source_layers=(2,),
        compress_ratios=(0, 0, 2),
        swa_bounded_replay=False,
        producer_layer_end=None,
    )
    sched_off._init_ilu_transfer_clip_rules()
    clipped_off = sched_off.get_exchange_clipped_blocks(
        ([10, 11, 12], [20, 21, 22, 23], [30, 31])
    )
    assert clipped_off == ([10, 11, 12], [21, 22, 23], [30])

    # remote_block_starts: group 0 has 2 cached blocks + 2 unhashed; groups 1/2 clipped -> -1.
    def _blk(hashed: bool, null: bool = False):
        return SimpleNamespace(block_hash="h" if hashed else None, is_null=null)

    blocks_obj = SimpleNamespace(
        blocks=[
            [_blk(True), _blk(True), _blk(False), _blk(False)],
            [_blk(False, null=True), _blk(False)],
            [_blk(False)],
        ]
    )
    assert sched_on.compute_remote_block_starts(blocks_obj) == [2, -1, -1]


def test_prefix_replay_alignment_returns_2_for_v41_consumer(monkeypatch):
    monkeypatch.delenv("VLLM_ILU_V41_REMOTE_REPLAY_ALIGN2", raising=False)
    d_cfg = _v41_vllm_config(role="kv_consumer")
    main_cr2 = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        tokens_per_state=2,
        state_content_bytes=584,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
    )
    swa = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=128,
        bounded_replay=True,
        state_content_bytes=584,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
    )
    kv_cfg = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(["layers.2.attn"], main_cr2),
            KVCacheGroupSpec(["layers.2.attn.swa_cache"], swa),
        ]
    )
    assert (
        compute_prefix_replay_alignment(
            d_cfg, kv_cfg, is_consumer=True, has_mamba=False
        )
        == 2
    )
    assert (
        compute_prefix_replay_alignment(
            d_cfg, kv_cfg, is_consumer=False, has_mamba=False
        )
        is None
    )


def test_packed_registration_supports_multi_ratio_kernel_blocks_and_skip_layers(
    monkeypatch,
):
    num_blocks = 4
    # Group 0 (r=1): [4, 1, 64, 16] uint8 -> 1024 B/scheduler block
    # Group 1 (r=2): [8, 1, 32, 16] uint8 -> 512 B/kernel block * 2 = 1024 B/scheduler block
    # Group 2 (r=4): [16, 1, 16, 16] uint8 -> 256 B/kernel block * 4 = 1024 B/scheduler block
    # Skipped SWA layer: also in kv_caches, registered in skip_layers.
    page = 1024
    block_stride = page * 4
    backing = torch.zeros(num_blocks * block_stride, dtype=torch.uint8)

    l_r1 = "model.layers.2.attn"
    l_r2 = "model.layers.8.attn"
    l_r4 = "model.layers.20.attn.indexer.k_cache"
    l_skip = "model.layers.2.attn.swa_cache"

    spec_r1 = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.uint8,
        tokens_per_state=2,
    )
    spec_r2 = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.uint8,
        tokens_per_state=1,
    )
    spec_r4 = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.uint8,
        tokens_per_state=1,
    )
    spec_skip = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.uint8,
        sliding_window=128,
        bounded_replay=True,
    )

    kv_caches = {
        l_r1: torch.as_strided(
            backing,
            size=(num_blocks * 1, 1, 64, 16),
            stride=(block_stride, 1024, 16, 1),
            storage_offset=0,
        ),
        l_r2: torch.as_strided(
            backing,
            size=(num_blocks * 2, 1, 32, 16),
            stride=(block_stride // 2, 512, 16, 1),
            storage_offset=0,
        ),
        l_r4: torch.as_strided(
            backing,
            size=(num_blocks * 4, 1, 16, 16),
            stride=(block_stride // 4, 256, 16, 1),
            storage_offset=0,
        ),
        l_skip: torch.as_strided(
            backing,
            size=(num_blocks, 1, 64, 16),
            stride=(block_stride, 1024, 16, 1),
            storage_offset=page * 3,
        ),
    }
    # For dense multi-kernel splitting in LBNHC, each layer uses dense stride = page (1024):
    # rebuild views with dense LBNHC strides matching upstream create_kv_cache_views.
    lbnhc_backing = torch.zeros(num_blocks * page * 4, dtype=torch.uint8)
    kv_caches = {
        l_r1: torch.as_strided(
            lbnhc_backing,
            size=(num_blocks * 1, 1, 64, 16),
            stride=(page, page, 16, 1),
            storage_offset=0,
        ),
        l_r2: torch.as_strided(
            lbnhc_backing,
            size=(num_blocks * 2, 1, 32, 16),
            stride=(page // 2, page // 2, 16, 1),
            storage_offset=num_blocks * page,
        ),
        l_r4: torch.as_strided(
            lbnhc_backing,
            size=(num_blocks * 4, 1, 16, 16),
            stride=(page // 4, page // 4, 16, 1),
            storage_offset=2 * num_blocks * page,
        ),
        l_skip: torch.as_strided(
            lbnhc_backing,
            size=(num_blocks, 1, 64, 16),
            stride=(page, page, 16, 1),
            storage_offset=3 * num_blocks * page,
        ),
    }
    groups = [
        SimpleNamespace(layer_names=[l_r1], kv_cache_spec=spec_r1),
        SimpleNamespace(layer_names=[l_r2], kv_cache_spec=spec_r2),
        SimpleNamespace(layer_names=[l_r4], kv_cache_spec=spec_r4),
        SimpleNamespace(layer_names=[l_skip], kv_cache_spec=spec_skip),
    ]
    tensors = [
        SimpleNamespace(
            size=lbnhc_backing.numel(),
            layers=[l_r1, l_r2, l_r4, l_skip],
            offset=0,
            layer_stride=num_blocks * page,
            block_stride=page,
        )
    ]
    kv_cfg = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=groups,
        transfer_groups=tuple(groups),
        kv_cache_tensors=tensors,
    )

    monkeypatch.setattr(skip_layers, "_SKIP_LAYERS", {l_skip})
    monkeypatch.delenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", raising=False)
    monkeypatch.delenv("VLLM_ILU_DS41_INDEXER_KV_LAYOUT", raising=False)

    worker = object.__new__(IluNixlWorkerMixin)
    worker.vllm_config = _v41_vllm_config(role="kv_producer")
    worker.dsv41_layout = resolve_dsv41_layout(worker.vllm_config)
    worker.fp8_wire_staging_enabled = False
    worker.use_host_buffer = False
    worker._logical_num_blocks = num_blocks
    worker._scheduler_block_size = 128
    worker._physical_blocks_per_logical_kv_block = 4  # simulate upstream scalar
    worker.num_blocks = num_blocks * 4
    worker.block_size = 32
    worker.kv_cache_config = kv_cfg
    worker.engine_id = "producer"
    worker.tp_rank = 0
    worker.world_size = 1
    worker.pp_rank = 0
    worker.pp_world_size = 1
    worker.pp_split = "40"
    worker.use_mla = True
    worker._has_mamba = False
    worker.model_config = SimpleNamespace(get_total_num_kv_heads=lambda: 1)
    worker.attn_backends = []
    worker.backend_name = "FLASHMLA_SPARSE"
    worker.kv_cache_layout = "LBNHC"
    worker._mamba_ssm_size = (0, 0)
    worker.nixl_memory_type = "DRAM"
    worker.nixl_backends = ["UCX"]
    worker._registered_descs = []
    worker.region_mem_types = []
    worker.dst_region_mem_types = {}
    worker.kv_caches_base_addr = {"producer": {}}
    worker.dst_num_blocks = {}
    worker.src_xfer_handles_by_block_size = {}
    worker.register_local_xfer_handler = lambda _bs: (
        99,
        np.zeros((num_blocks * 3, 3), dtype=np.uint64),
    )
    worker._compute_compatibility_hash = lambda _=None: "compat-v41"
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_agent_metadata.return_value = b"agent"
    worker.nixl_wrapper.prep_xfer_dlist.return_value = 99

    worker.register_kv_caches(kv_caches)

    assert worker.num_blocks == num_blocks
    assert worker._physical_blocks_per_logical_kv_block == 1
    assert l_skip not in worker.layer_region_ids
    assert worker._skipped_layers == [l_skip]
    by_layer = {r.shared_by[0]: r for r in worker.transfer_regions}
    assert by_layer[l_r1].kernel_blocks == 1
    assert by_layer[l_r2].kernel_blocks == 2
    assert by_layer[l_r4].kernel_blocks == 4
    assert by_layer[l_r1].block_tokens == 128
    assert by_layer[l_r2].block_tokens == 64
    assert by_layer[l_r4].block_tokens == 64
    assert [r.num_blocks for r in worker.transfer_regions] == [4, 4, 4]
    assert worker._local_desc_cumsum.tolist() == [0, 4, 8]


def test_same_layer_across_two_groups_pairs_and_routes_correctly():
    # Regression test for Stage 2 risk: one layer's same kind appearing in two
    # groups on both P (with extra forwarder group shifting group IDs) and D.
    d_regions = [
        IluNixlTransferRegion(
            region_id=0,
            block_len=1024,
            shared_by=["layers.2.attn"],
            layer_indices=[2],
            group_ids=[0],
            cache_kinds=["main_mla"],
            identities=["2:main_mla"],
            kernel_blocks=1,
            block_tokens=128,
            num_blocks=8,
            identity_groups={"2:main_mla": [0]},
        ),
        IluNixlTransferRegion(
            region_id=1,
            block_len=1024,
            shared_by=["layers.2.attn"],
            layer_indices=[2],
            group_ids=[2],
            cache_kinds=["main_mla"],
            identities=["2:main_mla"],
            kernel_blocks=1,
            block_tokens=128,
            num_blocks=8,
            identity_groups={"2:main_mla": [2]},
        ),
    ]
    # Producer has an extra forwarder group at group 1, so D's group 2 maps to P's group 3.
    p_regions = [
        IluNixlTransferRegion(
            region_id=10,
            block_len=1024,
            shared_by=["layers.2.attn"],
            layer_indices=[2],
            group_ids=[0],
            cache_kinds=["main_mla"],
            identities=["2:main_mla"],
            kernel_blocks=1,
            block_tokens=128,
            num_blocks=16,
            identity_groups={"2:main_mla": [0]},
        ),
        IluNixlTransferRegion(
            region_id=11,
            block_len=1024,
            shared_by=["layers.2.attn"],
            layer_indices=[2],
            group_ids=[3],
            cache_kinds=["main_mla"],
            identities=["2:main_mla"],
            kernel_blocks=1,
            block_tokens=128,
            num_blocks=16,
            identity_groups={"2:main_mla": [3]},
        ),
    ]

    pairings, sel_local, sel_remote = IluNixlWorkerMixin._pair_hma_transfer_regions(
        [0, 1], [10, 11], d_regions, p_regions, num_groups=3
    )
    assert sel_local == [0, 1]
    assert sel_remote == [10, 11]
    assert pairings == [
        RegionPairing(0, 10, 0, 0, "2:main_mla"),
        RegionPairing(1, 11, 2, 3, "2:main_mla"),
    ]


def test_kernel_blocks_mismatch_raises_with_region_and_identity():
    worker = object.__new__(IluNixlWorkerMixin)
    worker._has_mamba = False
    worker.use_mla = True
    worker._group_spec_types = (MLAAttentionSpec,)
    worker.block_len_per_layer = [65536]
    worker._is_region_replicated = lambda _: True
    worker.dst_num_blocks = {"prefill": 10}
    worker.transfer_topo = SimpleNamespace(
        block_size_ratio=lambda _: 1, tp_ratio=lambda _: 1
    )
    worker.transfer_regions = [
        IluNixlTransferRegion(
            region_id=0,
            block_len=65536,
            shared_by=["layers.8.attn"],
            layer_indices=[8],
            group_ids=[0],
            cache_kinds=["main_mla"],
            identities=["8:main_mla"],
            kernel_blocks=2,
            block_tokens=128,
            num_blocks=10,
        )
    ]
    remote_meta = SimpleNamespace(
        engine_id="prefill",
        block_size=128,
        block_lens=[65536],
        num_blocks=10,
        transfer_regions=[
            IluNixlTransferRegion(
                region_id=3,
                block_len=65536,
                shared_by=["layers.8.attn"],
                layer_indices=[8],
                group_ids=[1],
                cache_kinds=["main_mla"],
                identities=["8:main_mla"],
                kernel_blocks=4,
                block_tokens=128,
                num_blocks=10,
            )
        ],
    )
    pairings = [RegionPairing(0, 3, 0, 1, "8:main_mla")]

    with pytest.raises(
        ValueError,
        match=r"8:main_mla \(local r0, remote r3\): kernel blocks per scheduler block differ \(local=2, remote=4\)",
    ):
        worker._validate_pp_hma_region_lengths(
            remote_meta,
            local_region_ids_by_group=[],
            remote_region_ids_by_group=[],
            selected_remote_region_ids=[3],
            remote_tp_size=1,
            pairings=pairings,
        )


def test_truncated_producer_overlap_skips_inactive_ranks_and_adjusts_notif_count():
    # P: PP=1 (40 layers, truncated at kv_layer_end=21), TP=2
    # D: PP=4 ("2,6,12,20" -> stages [0,2), [2,8), [8,20), [20,40)), TP=1
    # All 3 stages [2,8), [8,20), [20,21) overlap P's [0,21); stage 0 [0,2) is pure SWA.
    # For D PP=2 ("20,20"): both D stages [0,20) and [20,40) overlap [0,21) -> pp_consumers=2.
    # For D PP=4 ("10,10,10,10") vs P PP=1 kv_layer_end=20: only D stages 0,1 overlap [0,20) -> pp_consumers=2, D stage 2 [20,30) is inactive!
    assert (
        IluNixlWorkerMixin()._effective_overlap_pp_count
        if False
        else IluNixlWorkerMixin._overlap_pp_count(0, "40", "10,10,10,10")
    ) == 4
    worker = object.__new__(IluNixlWorkerMixin)
    assert (
        worker._effective_overlap_pp_count(0, "40", "10,10,10,10", kv_layer_end=20) == 2
    )

    # D stage 2 owns [20, 30), while P has kv_layer_end=20 -> inactive overlap!
    worker.pp_split = "10,10,10,10"
    worker.pp_rank = 2
    worker.pp_world_size = 4
    worker.tp_rank = 0
    worker.world_size = 1
    worker.use_mla = True
    worker._has_mamba = False
    worker._is_hma_required = True
    worker._virtual_packed_regions = True
    worker.dsv41_layout = _v41_vllm_config(role="kv_consumer")
    worker.dsv41_layout = resolve_dsv41_layout(worker.dsv41_layout)
    worker.layer_region_ids = {"model.layers.20.attn": [0]}
    worker.region_ids_by_group = [np.asarray([0], dtype=np.int64)]
    worker.transfer_regions = [
        IluNixlTransferRegion(
            region_id=0,
            block_len=1024,
            shared_by=["model.layers.20.attn"],
            layer_indices=[20],
            group_ids=[0],
            cache_kinds=["main_mla"],
            identities=["20:main_mla"],
            num_blocks=8,
        )
    ]
    worker.block_len_per_layer = [1024]
    worker._group_spec_types = (MLAAttentionSpec,)
    worker.dst_num_blocks = {}
    worker.local_region_ids_by_group = {}
    worker.remote_region_ids_by_group = {}
    worker.region_pairings = {}
    worker._inactive_remote_keys = {}
    worker._remote_pp_size_by_engine = {}
    worker._remote_pp_split_by_engine = {}
    worker._remote_kv_layer_end_by_engine = {}
    worker._remote_supports_block_starts = {}
    worker._semantic_region_engines = set()
    worker._remote_agents = defaultdict(dict)
    worker.tp_mappings = {}
    worker.dst_xfer_side_handles = defaultdict(dict)
    worker.transfer_topo = SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        total_num_kv_heads=1,
        is_mla=True,
        dcp_size=1,
        register_remote_engine=lambda *_: None,
        dcp_consumer_count=lambda *_: 1,
        tp_ratio=lambda _: -2,
        get_engine_info=lambda _: SimpleNamespace(
            remote_tp_size=2,
            remote_physical_blocks_per_logical=1,
            remote_block_size=128,
        ),
    )
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.add_remote_agent.return_value = "p_agent_0"

    p_meta = IluNixlAgentMetadata(
        engine_id="prefill",
        agent_metadata=b"p",
        kv_caches_base_addr=[1000],
        device_id=0,
        num_blocks=8,
        block_lens=[1024],
        block_strides=[1024],
        kv_cache_layout="LBNHC",
        block_size=128,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASHMLA_SPARSE",
        physical_blocks_per_logical_kv_block=1,
        pp_split="40",
        pp_rank=0,
        tp_rank=0,
        layer_region_ids={"model.layers.2.attn": [0]},
        region_ids_by_group=[[0]],
        transfer_regions=[
            IluNixlTransferRegion(
                region_id=0,
                block_len=1024,
                shared_by=["model.layers.2.attn"],
                layer_indices=[2],
                group_ids=[0],
                cache_kinds=["main_mla"],
                identities=["2:main_mla"],
                num_blocks=8,
            )
        ],
        virtual_packed=True,
        kv_layer_end=20,
        swa_bounded_replay=True,
        kv_source_layers=list(worker.dsv41_layout.kv_source_layers),
        index_source_layers=list(worker.dsv41_layout.index_source_layers),
        compress_ratios=list(worker.dsv41_layout.compress_ratios),
        supports_remote_block_starts=True,
    )

    agent = worker.add_remote_agent(
        p_meta, remote_tp_rank=0, remote_tp_size=2, remote_pp_rank=0, remote_pp_size=1
    )
    assert agent == "p_agent_0"
    assert 0 in worker._inactive_remote_keys["prefill"]
    assert "prefill" not in worker.dst_xfer_side_handles

    # Reading a request on this inactive D stage sends no READ and no spurious
    # notification, while still marking awaiting_kvs request as complete.
    worker._remote_agents["prefill"][0] = "p_agent_0"
    worker._engine_last_active = {}
    worker._recving_transfers = {}
    req_meta = SimpleNamespace(
        remote=SimpleNamespace(
            engine_id="prefill",
            request_id="req-1",
            block_ids=([1, 2, 3],),
        ),
        local_physical_block_ids=([4, 5],),
        awaiting_kvs=True,
        remote_block_starts=[0],
    )
    worker._read_blocks_for_req("req-1", req_meta)
    worker.nixl_wrapper.send_notif.assert_not_called()
    worker.nixl_wrapper.make_prepped_xfer.assert_not_called()
    assert worker._recving_transfers["req-1"] == []


def test_read_blocks_uses_cumsum_offsets_and_middle_remote_block_starts():
    # D has 2 local groups (main, swa-clipped) and 2 regions with num_blocks [6, 10] -> cumsum [0, 6].
    # P has 3 remote groups (extra forwarder group at 0; main at 1; swa at 2) and
    # selected remote regions [5, 8] with num_blocks [12, 20] -> cumsum [0, 12].
    # Bounded replay: P prompt blocks for main group = [10, 11, 12, 13, 14],
    # D prefix hit = 1 block, replay tail = 1 block -> D asks for 3 blocks starting at index 1 ([11, 12, 13]).
    worker = object.__new__(IluNixlWorkerMixin)
    worker.engine_id = "decode"
    worker.world_size = 2
    worker.pp_world_size = 1
    worker.pp_split = "40"
    worker.use_mla = True
    worker._has_mamba = False
    worker._physical_blocks_per_logical_kv_block = 1
    worker._group_spec_types = (MLAAttentionSpec, SlidingWindowMLASpec)
    worker.dst_num_blocks = {"decode": 6, "prefill": 12}
    worker._local_desc_cumsum = np.asarray([0, 6], dtype=np.int64)
    worker._remote_desc_cumsum = {"prefill": {0: np.asarray([0, 12], dtype=np.int64)}}
    worker._remote_pp_size_by_engine = {"prefill": 2}
    worker._remote_pp_split_by_engine = {"prefill": "20,20"}
    worker._remote_kv_layer_end_by_engine = {"prefill": 21}
    worker._remote_supports_block_starts = {"prefill": True}
    worker._semantic_region_engines = {"prefill"}
    worker.dsv41_layout = resolve_dsv41_layout(_v41_vllm_config(role="kv_consumer"))
    worker.region_pairings = {
        "prefill": {
            0: [
                RegionPairing(0, 5, 0, 1, "2:main_mla"),
                RegionPairing(1, 8, 0, 1, "2:main_mla_scales"),
            ]
        }
    }
    worker.transfer_topo = SimpleNamespace(
        get_engine_info=lambda _: SimpleNamespace(
            remote_tp_size=2,
            remote_physical_blocks_per_logical=1,
            remote_block_size=128,
        ),
        tp_ratio=lambda _: 1,
        block_size_ratio=lambda _: 1,
    )
    worker.tp_mappings = {
        "prefill": SimpleNamespace(
            source_ranks_per_group=((0,), (0,)),
            all_source_ranks=(0,),
        )
    }
    worker._engine_last_active = {}
    worker._remote_agents = {"prefill": {0: "p_agent_0"}}
    worker.src_xfer_handles_by_block_size = {128: 11}
    worker.dst_xfer_side_handles = {"prefill": {0: 22}}
    worker._recving_transfers = defaultdict(list)

    xfer_calls = []
    worker.nixl_wrapper = SimpleNamespace(
        make_prepped_xfer=lambda op, lh, lids, rh, rids, notif_msg=None: (
            xfer_calls.append((op, lh, lids.tolist(), rh, rids.tolist(), notif_msg))
            or 33
        ),
        transfer=lambda _h: None,
        send_notif=lambda *_a, **_kw: None,
    )

    meta = IluNixlConnectorMetadata()
    meta.add_new_req_to_recv(
        request_id="req-mid",
        local_block_ids=([2, 4, 5], []),
        kv_transfer_params={
            "remote_block_ids": ([99], [10, 11, 12, 13, 14], []),
            "remote_engine_id": "prefill",
            "remote_request_id": "p-req-mid",
            "remote_host": "127.0.0.1",
            "remote_port": 5555,
            "tp_size": 2,
            "pp_size": 2,
            "pp_split": "20,20",
        },
        awaiting_kvs=True,
        remote_block_starts=[1, -1],
    )
    req_meta = meta.reqs_to_recv["req-mid"]
    worker._read_blocks_for_req("req-mid", req_meta)

    assert len(xfer_calls) == 1
    op, lh, lids, rh, rids, notif = xfer_calls[0]
    assert op == "READ"
    assert lh == 11 and rh == 22
    # Local desc IDs: region 0 (base 0) + [2,4,5] -> [2,4,5]; region 1 (base 6) + [2,4,5] -> [8,10,11]
    assert lids == [2, 4, 5, 8, 10, 11]
    # Remote desc IDs: compact 0 (base 0) + [11,12,13] -> [11,12,13]; compact 1 (base 12) + [11,12,13] -> [23,24,25]
    assert rids == [11, 12, 13, 23, 24, 25]
    assert notif == b"p-req-mid:2:1"


def test_paired_quantized_main_and_split_indexer_scale_planes_register_and_pair(
    monkeypatch,
):
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_common.dsv41 import (
        resolve_quantized_kv_format,
    )
    from vllm_iluvatar.models.deepseek_v41 import compressed_kv
    from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
        PAYLOAD_PAGE_BYTES,
        plan_scale_layout,
    )
    from vllm_iluvatar.models.deepseek_v41.indexer_kv import (
        SPLIT_INDEXER_FORMAT,
        plan_indexer_scales,
    )

    I8G64 = resolve_quantized_kv_format("i8g64")
    monkeypatch.setattr(compressed_kv, "main_kv_format", lambda: I8G64)

    monkeypatch.setattr(skip_layers, "_SKIP_LAYERS", set())
    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "i8g64")
    monkeypatch.setenv("VLLM_ILU_DS41_INDEXER_KV_LAYOUT", "split")
    monkeypatch.setenv("VLLM_ILU_V41_INDEXER_KV_DTYPE", "int8")

    num_blocks = 3
    main_layer = "model.layers.2.attn"
    idx_layer = "model.layers.2.attn.indexer.k_cache"

    main_spec = MLAAttentionSpec(
        block_size=I8G64.rows_per_page(512) * 2,
        tokens_per_state=2,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=I8G64.payload_bytes(512),
        cache_dtype_str=I8G64.name,
    )
    # Split indexer page: 512 states * 128 B = 65536 B (matches 64 KiB page pattern).
    idx_spec = MLAAttentionSpec(
        block_size=1024,
        tokens_per_state=2,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        state_content_bytes=128,
        cache_dtype_str=SPLIT_INDEXER_FORMAT,
    )

    page = PAYLOAD_PAGE_BYTES
    total_size = num_blocks * 2 * page
    tensors = [
        SimpleNamespace(
            size=total_size,
            layers=[main_layer],
            offset=0,
            layer_stride=page,
            block_stride=2 * page,
        ),
        SimpleNamespace(
            size=total_size,
            layers=[idx_layer],
            offset=page,
            layer_stride=page,
            block_stride=2 * page,
        ),
    ]
    groups = [
        SimpleNamespace(layer_names=[main_layer], kv_cache_spec=main_spec),
        SimpleNamespace(layer_names=[idx_layer], kv_cache_spec=idx_spec),
    ]
    kv_cfg = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=groups,
        transfer_groups=tuple(groups),
        kv_cache_tensors=tensors,
    )

    payload_buf = torch.zeros(total_size, dtype=torch.uint8)
    kv_caches = {
        main_layer: torch.as_strided(
            payload_buf,
            size=(num_blocks, I8G64.rows_per_page(512), I8G64.payload_bytes(512)),
            stride=(2 * page, I8G64.payload_bytes(512), 1),
            storage_offset=0,
        ),
        idx_layer: torch.as_strided(
            payload_buf,
            size=(num_blocks, 512, 128),
            stride=(2 * page, 128, 1),
            storage_offset=page,
        ),
    }

    main_scale_layout = plan_scale_layout(kv_cfg, I8G64)
    main_scale_buf = torch.zeros(
        main_scale_layout.scale_bytes // 2, dtype=torch.float16
    )
    mp = main_scale_layout.placements[main_layer]
    main_scale_view = torch.as_strided(
        main_scale_buf,
        size=(num_blocks, mp.rows, mp.groups),
        stride=(mp.block_stride_elements, mp.groups, 1),
        storage_offset=mp.offset_elements,
    )

    idx_scale_layout = plan_indexer_scales(kv_cfg)
    idx_scale_buf = torch.zeros(idx_scale_layout.nbytes // 4, dtype=torch.float32)
    ip = idx_scale_layout.placements[idx_layer]
    idx_scale_view = idx_scale_buf[
        ip.offset : ip.offset + num_blocks * ip.rows_per_manager_block
    ].view(num_blocks, ip.rows_per_manager_block)

    vllm_cfg = _v41_vllm_config(role="kv_producer")
    vllm_cfg.compilation_config.static_forward_context = {
        main_layer: SimpleNamespace(kv_cache_scales=main_scale_view),
        idx_layer: SimpleNamespace(kv_cache_scales=idx_scale_view),
    }

    worker = object.__new__(IluNixlWorkerMixin)
    worker.vllm_config = vllm_cfg
    worker.dsv41_layout = resolve_dsv41_layout(vllm_cfg)
    worker.fp8_wire_staging_enabled = False
    worker.use_host_buffer = False
    worker._logical_num_blocks = num_blocks
    worker._scheduler_block_size = 256
    worker._physical_blocks_per_logical_kv_block = 1
    worker.num_blocks = num_blocks
    worker.block_size = 256
    worker.kv_cache_config = kv_cfg
    worker.engine_id = "producer"
    worker.tp_rank = 0
    worker.world_size = 1
    worker.pp_rank = 0
    worker.pp_world_size = 1
    worker.pp_split = "40"
    worker.use_mla = True
    worker._has_mamba = False
    worker.model_config = SimpleNamespace(get_total_num_kv_heads=lambda: 1)
    worker.attn_backends = []
    worker.backend_name = "FLASHMLA_SPARSE"
    worker.kv_cache_layout = "BLHNC"
    worker._mamba_ssm_size = (0, 0)
    worker.nixl_memory_type = "DRAM"
    worker.nixl_backends = ["UCX"]
    worker._registered_descs = []
    worker.region_mem_types = []
    worker.dst_region_mem_types = {}
    worker.kv_caches_base_addr = {"producer": {}}
    worker.dst_num_blocks = {}
    worker.src_xfer_handles_by_block_size = {}
    worker.register_local_xfer_handler = lambda _bs: (
        99,
        np.zeros((num_blocks * 4, 3), dtype=np.uint64),
    )
    worker._compute_compatibility_hash = lambda _=None: "compat-paired"
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_agent_metadata.return_value = b"agent"

    worker.register_kv_caches(kv_caches)

    # 3 distinct physical backing buffers registered: payload + main_scales + idx_scales
    assert len(worker.kv_caches_base_addr["producer"][0]) == 3
    # 4 virtual transfer regions: main_mla, indexer_k_cache, main_mla_scales, indexer_k_cache_scales
    assert worker.num_regions == 4
    kinds = {r.cache_kinds[0] for r in worker.transfer_regions}
    assert kinds == {
        "main_mla",
        "indexer_k_cache",
        "main_mla_scales",
        "indexer_k_cache_scales",
    }
    assert len(worker.layer_region_ids[main_layer]) == 2
    assert len(worker.layer_region_ids[idx_layer]) == 2
    assert worker.xfer_handshake_metadata.main_kv_format == "i8g64"
    assert worker.xfer_handshake_metadata.indexer_kv_layout == "split"

    # Self-pairing verifies all 4 regions pair cleanly onto their owning groups (0 and 1).
    pairings, sel_l, sel_r = IluNixlWorkerMixin._pair_hma_transfer_regions(
        list(range(4)),
        list(range(4)),
        worker.transfer_regions,
        worker.transfer_regions,
        num_groups=2,
    )
    assert len(pairings) == 4
    assert {p.identity for p in pairings} == {
        "2:main_mla",
        "2:indexer_k_cache",
        "2:main_mla_scales",
        "2:indexer_k_cache_scales",
    }


def test_native_nixl_vram_v41_pull_read_end_to_end(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/Iluvatar GPU required for native NIXL VRAM smoke test")
    nixl_api = pytest.importorskip("nixl._api")
    import time
    import uuid

    import msgspec

    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.metadata import (
        IluNixlHandshakePayload,
    )
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.pull_worker import (
        IluNixlPullConnectorWorker,
    )
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_common.dsv41 import (
        resolve_quantized_kv_format,
    )
    from vllm_iluvatar.models.deepseek_v41 import compressed_kv
    from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
        PAYLOAD_PAGE_BYTES,
        plan_scale_layout,
    )
    from vllm_iluvatar.models.deepseek_v41.indexer_kv import (
        SPLIT_INDEXER_FORMAT,
        plan_indexer_scales,
    )

    I8G64 = resolve_quantized_kv_format("i8g64")
    monkeypatch.setattr(compressed_kv, "main_kv_format", lambda: I8G64)
    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "i8g64")
    monkeypatch.setenv("VLLM_ILU_DS41_INDEXER_KV_LAYOUT", "split")
    monkeypatch.setenv("VLLM_ILU_V41_INDEXER_KV_DTYPE", "int8")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    main_layer = "model.layers.2.attn"
    idx_layer = "model.layers.2.attn.indexer.k_cache"
    fwd_layer = "model.layers.10.attn"
    swa_layer = "model.layers.2.attn.swa_cache"
    monkeypatch.setattr(skip_layers, "_SKIP_LAYERS", {fwd_layer, swa_layer})

    main_spec = MLAAttentionSpec(
        block_size=I8G64.rows_per_page(512) * 2,
        tokens_per_state=2,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=I8G64.payload_bytes(512),
        cache_dtype_str=I8G64.name,
    )
    idx_spec = MLAAttentionSpec(
        block_size=I8G64.rows_per_page(512) * 2,
        tokens_per_state=2,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        state_content_bytes=128,
        cache_dtype_str=SPLIT_INDEXER_FORMAT,
    )
    swa_spec = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=128,
        bounded_replay=True,
        state_content_bytes=584,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
    )

    page = PAYLOAD_PAGE_BYTES

    # Build LBNHC layout so r=1 (main) and r=2 (indexer) coexist on one VRAM buffer!
    def _build_lbnhc_side(
        role: str, num_blocks: int, include_forwarder: bool, seed: int
    ):
        layer_list = [main_layer, idx_layer, swa_layer]
        if include_forwarder:
            layer_list.insert(0, fwd_layer)
        n_slots = len(layer_list)
        total_bytes = num_blocks * page * n_slots

        tensors = [
            SimpleNamespace(
                size=total_bytes,
                layers=[lname],
                offset=slot_idx * num_blocks * page,
                layer_stride=num_blocks * page,
                block_stride=page,
            )
            for slot_idx, lname in enumerate(layer_list)
        ]
        groups = []
        if include_forwarder:
            groups.append(
                SimpleNamespace(
                    layer_names=[fwd_layer],
                    kv_cache_spec=main_spec,
                    is_eagle_group=False,
                )
            )
        groups.extend(
            [
                SimpleNamespace(
                    layer_names=[main_layer],
                    kv_cache_spec=main_spec,
                    is_eagle_group=False,
                ),
                SimpleNamespace(
                    layer_names=[idx_layer],
                    kv_cache_spec=idx_spec,
                    is_eagle_group=False,
                ),
                SimpleNamespace(
                    layer_names=[swa_layer],
                    kv_cache_spec=swa_spec,
                    is_eagle_group=False,
                ),
            ]
        )
        kv_cfg = SimpleNamespace(
            num_blocks=num_blocks,
            kv_cache_groups=groups,
            transfer_groups=tuple(groups),
            kv_cache_tensors=tensors,
        )

        if role == "kv_producer":
            idx_t = torch.arange(total_bytes, dtype=torch.int64, device=device)
            payload_buf = ((idx_t + (idx_t // page) * 31 + seed) % 251).to(torch.uint8)
        else:
            payload_buf = torch.full(
                (total_bytes,), 0xEE, dtype=torch.uint8, device=device
            )

        slot_of = {lname: i for i, lname in enumerate(layer_list)}
        kv_caches = {}
        if include_forwarder:
            kv_caches[fwd_layer] = torch.as_strided(
                payload_buf,
                size=(num_blocks, I8G64.rows_per_page(512), I8G64.payload_bytes(512)),
                stride=(page, I8G64.payload_bytes(512), 1),
                storage_offset=slot_of[fwd_layer] * num_blocks * page,
            )
        kv_caches[main_layer] = torch.as_strided(
            payload_buf,
            size=(num_blocks, I8G64.rows_per_page(512), I8G64.payload_bytes(512)),
            stride=(page, I8G64.payload_bytes(512), 1),
            storage_offset=slot_of[main_layer] * num_blocks * page,
        )
        # r=2 kernel blocks per scheduler block: shape=(num_blocks*2, 256, 128), stride(0)=page//2=32768!
        kv_caches[idx_layer] = torch.as_strided(
            payload_buf,
            size=(num_blocks * 2, 256, 128),
            stride=(page // 2, 128, 1),
            storage_offset=slot_of[idx_layer] * num_blocks * page,
        )
        kv_caches[swa_layer] = torch.as_strided(
            payload_buf,
            size=(num_blocks, 64, 584),
            stride=(page, 584, 1),
            storage_offset=slot_of[swa_layer] * num_blocks * page,
        )

        main_scale_layout = plan_scale_layout(kv_cfg, I8G64)
        main_scale_elems = main_scale_layout.scale_bytes // 2
        if role == "kv_producer":
            s_idx = torch.arange(
                main_scale_layout.scale_bytes, dtype=torch.int64, device=device
            )
            main_scale_raw = ((s_idx + 53) % 251).to(torch.uint8)
            main_scale_buf = main_scale_raw.view(torch.float16)
        else:
            main_scale_buf = torch.empty(
                main_scale_elems, dtype=torch.float16, device=device
            )
            main_scale_buf.view(torch.uint8).fill_(0xEE)

        scale_views = {}
        for lname in [fwd_layer, main_layer] if include_forwarder else [main_layer]:
            mp = main_scale_layout.placements[lname]
            scale_views[lname] = torch.as_strided(
                main_scale_buf,
                size=(num_blocks, mp.rows, mp.groups),
                stride=(mp.block_stride_elements, mp.groups, 1),
                storage_offset=mp.offset_elements,
            )

        idx_scale_layout = plan_indexer_scales(kv_cfg)
        idx_scale_elems = idx_scale_layout.nbytes // 4
        if role == "kv_producer":
            is_idx = torch.arange(
                idx_scale_layout.nbytes, dtype=torch.int64, device=device
            )
            idx_scale_raw = ((is_idx + 97) % 251).to(torch.uint8)
            idx_scale_buf = idx_scale_raw.view(torch.float32)
        else:
            idx_scale_buf = torch.empty(
                idx_scale_elems, dtype=torch.float32, device=device
            )
            idx_scale_buf.view(torch.uint8).fill_(0xEE)
        ip = idx_scale_layout.placements[idx_layer]
        # r=2 kernel blocks on indexer scales too: [num_blocks * 2, 256]
        scale_views[idx_layer] = idx_scale_buf[
            ip.offset : ip.offset + num_blocks * ip.rows_per_manager_block
        ].view(num_blocks * 2, ip.rows_per_manager_block // 2)

        vllm_cfg = _v41_vllm_config(
            role=role, pp_size=2 if include_forwarder else 1, tp_size=1
        )
        vllm_cfg.compilation_config.static_forward_context = {
            k: SimpleNamespace(kv_cache_scales=v) for k, v in scale_views.items()
        }

        token = uuid.uuid4().hex[:8]
        engine_id = f"{role}-{token}"
        agent_cfg = nixl_api.nixl_agent_config(backends=["UCX"], num_threads=1)
        w = object.__new__(IluNixlPullConnectorWorker)
        w.vllm_config = vllm_cfg
        w.kv_transfer_config = vllm_cfg.kv_transfer_config
        w.dsv41_layout = resolve_dsv41_layout(vllm_cfg)
        w.fp8_wire_staging_enabled = False
        w.use_host_buffer = False
        w.enable_permute_local_kv = False
        w.enable_heterogeneous_attn_post_process = False
        w._mixed_mem_types = False
        w._logical_num_blocks = num_blocks
        w._scheduler_block_size = 256
        w._physical_blocks_per_logical_kv_block = 1
        w.num_blocks = num_blocks
        w.block_size = 256
        w.kv_cache_config = kv_cfg
        w.engine_id = engine_id
        w.tp_rank = 0
        w.world_size = 1
        w.transfer_tp_rank = 0
        w.transfer_tp_size = 1
        w.dcp_size = 1
        w.pcp_size = 1
        w.pcp_rank = 0
        w.pcp_dcp_sharded = False
        w.pp_rank = 0
        w.pp_world_size = 2 if include_forwarder else 1
        w.pp_split = "20,20" if include_forwarder else "40"
        w.use_mla = True
        w._has_mamba = False
        w._is_hma_required = True
        w._group_spec_types = tuple(type(g.kv_cache_spec) for g in groups)
        w.model_config = SimpleNamespace(
            get_total_num_kv_heads=lambda: 1,
            get_head_size=lambda: 512,
            get_total_num_hidden_layers=lambda: 40,
            model="dsv41-smoke",
            dtype=torch.bfloat16,
        )
        w.attn_backends = []
        w.backend_name = "FLASHMLA_SPARSE"
        w.kv_cache_layout = "LBNHC"
        w._mamba_ssm_size = (0, 0)
        w.nixl_memory_type = "VRAM"
        w.nixl_backends = ["UCX"]
        w._registered_descs = []
        w.region_mem_types = []
        w.dst_region_mem_types = {}
        w.kv_caches_base_addr = defaultdict(dict)
        w.dst_num_blocks = {}
        w.src_xfer_handles_by_block_size = {}
        w.src_xfer_handles_by_tp_ratio = {}
        w._dram_src_handles_by_block_size = {}
        w._dram_src_handles_by_tp_ratio = {}
        w.dst_xfer_side_handles = defaultdict(dict)
        w._remote_agents = defaultdict(dict)
        w.tp_mappings = {}
        w.region_pairings = {}
        w._remote_desc_cumsum = {}
        w._remote_kv_layer_end_by_engine = {}
        w._remote_supports_block_starts = {}
        w._inactive_remote_keys = {}
        w.local_region_ids_by_group = {}
        w.remote_region_ids_by_group = {}
        w._remote_pp_size_by_engine = {}
        w._remote_pp_split_by_engine = {}
        w._semantic_region_engines = set()
        w._engine_last_active = {}
        w._engine_by_address = {}
        w._handshake_lock = __import__("threading").RLock()
        w._handshake_futures = {}
        w._recving_metadata = {}
        w._recving_transfers = defaultdict(list)
        w._reqs_to_send = {}
        w._reqs_to_process = set()
        w._replicated_pcp_done_sending = set()
        w._failed_recv_reqs = __import__("queue").Queue()
        w._recv_failures = set()
        w._pending_recv_notifs = {}
        w.consumer_notification_counts_by_req = defaultdict(int)
        w.expected_consumer_notifications_by_req = {}
        w._is_region_replicated = lambda _: True
        w._compute_compatibility_hash = lambda _=None: "v41-e2e-hash"
        w.nixl_wrapper = nixl_api.nixl_agent(engine_id, agent_cfg)
        w.register_kv_caches(kv_caches, scale_views)
        return SimpleNamespace(
            worker=w,
            payload_buf=payload_buf,
            main_scale_buf=main_scale_buf,
            idx_scale_buf=idx_scale_buf,
            kv_caches=kv_caches,
            scale_views=scale_views,
        )

    p_side = d_side = None
    try:
        # Producer: 5 blocks, PP=2 ("20,20"), 4 groups (group 0=forwarder, 1=main, 2=idx, 3=swa)
        p_side = _build_lbnhc_side(
            "kv_producer", num_blocks=5, include_forwarder=True, seed=17
        )
        # Consumer: 4 blocks, PP=1 ("40"), 3 groups (group 0=main, 1=idx, 2=swa)
        d_side = _build_lbnhc_side(
            "kv_consumer", num_blocks=4, include_forwarder=False, seed=0
        )
        torch.cuda.synchronize()

        p_w, d_w = p_side.worker, d_side.worker

        # Decode P's wire handshake payload on D and run add_remote_agent.
        wire_bytes = msgspec.msgpack.encode(p_w.xfer_handshake_metadata)
        payload = msgspec.msgpack.Decoder(IluNixlHandshakePayload).decode(wire_bytes)
        p_meta = msgspec.msgpack.Decoder(IluNixlAgentMetadata).decode(
            payload.agent_metadata_bytes
        )
        remote_name = d_w.add_remote_agent(
            p_meta,
            remote_tp_rank=0,
            remote_tp_size=1,
            remote_pp_rank=0,
            remote_pp_size=2,
        )
        d_w._remote_agents[p_w.engine_id][0] = remote_name
        # Register D on P so UCX reverse notification delivery works.
        d_meta = msgspec.msgpack.Decoder(IluNixlAgentMetadata).decode(
            d_w.xfer_handshake_metadata.agent_metadata_bytes
        )
        p_w.nixl_wrapper.add_remote_agent(d_meta.agent_metadata)

        # Register request on P: waiting for consumer notification.
        req_id = "req-v41-vram"
        p_w._reqs_to_process.add(req_id)
        p_w._reqs_to_send[req_id] = time.perf_counter() + 60.0

        # P blocks: group 0 (fwd)=[], group 1 (main)=[0, 4, 2, 1], group 2 (idx)=[3, 1, 4, 0], group 3 (swa)=[]
        # D prefix hit=1 block, replay tail=1 block -> D pulls 2 middle blocks (starts=[1, 1, -1]):
        #   main: P[4, 2] -> D[2, 0]
        #   idx : P[1, 4] -> D[3, 1]  (note: idx has r=2 kernel blocks per scheduler block!)
        conn_meta = IluNixlConnectorMetadata()
        conn_meta.add_new_req_to_recv(
            request_id=req_id,
            local_block_ids=([2, 0], [3, 1], []),
            kv_transfer_params={
                "remote_block_ids": ([], [0, 4, 2, 1], [3, 1, 4, 0], []),
                "remote_engine_id": p_w.engine_id,
                "remote_request_id": req_id,
                "remote_host": "127.0.0.1",
                "remote_port": 5555,
                "tp_size": 1,
                "pp_size": 2,
                "pp_split": "20,20",
            },
            awaiting_kvs=True,
            remote_block_starts=[1, 1, -1],
        )
        req_meta = conn_meta.reqs_to_recv[req_id]
        d_w._recving_metadata[req_id] = req_meta
        d_w._read_blocks_for_req(req_id, req_meta)

        # Wait for D's NIXL READ handle(s) to finish and P's notification to arrive.
        handles = d_w._recving_transfers[req_id]
        assert len(handles) == 1
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            state = d_w.nixl_wrapper.check_xfer_state(handles[0])
            if state == "DONE":
                break
            assert state == "PROC", f"Unexpected NIXL xfer state: {state}"
            time.sleep(0.002)
        assert state == "DONE"
        d_w.nixl_wrapper.release_xfer_handle(handles[0])

        notified = set()
        while not notified and time.monotonic() < deadline:
            notified = p_w._get_new_notifs()
            if not notified:
                time.sleep(0.002)
        assert notified == {req_id}
        assert req_id not in p_w._reqs_to_send

        torch.cuda.synchronize()
        # Verify main_layer blocks: P[4]->D[2], P[2]->D[0]; D[1], D[3] untouched (0xEE)!
        for src_b, dst_b in ((4, 2), (2, 0)):
            assert torch.equal(
                d_side.kv_caches[main_layer][dst_b].cpu(),
                p_side.kv_caches[main_layer][src_b].cpu(),
            )
            assert torch.equal(
                d_side.scale_views[main_layer][dst_b].view(torch.uint8).cpu(),
                p_side.scale_views[main_layer][src_b].view(torch.uint8).cpu(),
            )
        for untouched_b in (1, 3):
            assert d_side.kv_caches[main_layer][untouched_b].eq(0xEE).all()
            assert (
                d_side.scale_views[main_layer][untouched_b]
                .view(torch.uint8)
                .eq(0xEE)
                .all()
            )

        # Verify idx_layer blocks (r=2 kernel blocks per scheduler block!):
        # Scheduler block b owns kernel rows [2*b, 2*b+1]!
        # P[1]->D[3] means kernel rows [2,3]->[6,7]; P[4]->D[1] means kernel rows [8,9]->[2,3]!
        for src_b, dst_b in ((1, 3), (4, 1)):
            assert torch.equal(
                d_side.kv_caches[idx_layer][2 * dst_b : 2 * dst_b + 2].cpu(),
                p_side.kv_caches[idx_layer][2 * src_b : 2 * src_b + 2].cpu(),
            )
            assert torch.equal(
                d_side.scale_views[idx_layer][2 * dst_b : 2 * dst_b + 2]
                .view(torch.uint8)
                .cpu(),
                p_side.scale_views[idx_layer][2 * src_b : 2 * src_b + 2]
                .view(torch.uint8)
                .cpu(),
            )
        for untouched_b in (0, 2):
            assert (
                d_side.kv_caches[idx_layer][2 * untouched_b : 2 * untouched_b + 2]
                .eq(0xEE)
                .all()
            )
            assert (
                d_side.scale_views[idx_layer][2 * untouched_b : 2 * untouched_b + 2]
                .view(torch.uint8)
                .eq(0xEE)
                .all()
            )

        # Verify skipped SWA layer on D remained completely untouched (0xEE)!
        assert d_side.kv_caches[swa_layer].eq(0xEE).all()
    finally:
        for side in (d_side, p_side):
            if side is None:
                continue
            w = side.worker
            for handles in w.dst_xfer_side_handles.values():
                for h in handles.values():
                    w.nixl_wrapper.release_dlist_handle(h)
            for h in w.src_xfer_handles_by_block_size.values():
                w.nixl_wrapper.release_dlist_handle(h)
            for reg in w._registered_descs:
                w.nixl_wrapper.deregister_memory(reg, backends=["UCX"])
