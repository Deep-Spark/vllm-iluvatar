# SPDX-License-Identifier: Apache-2.0
"""Unit tests for IluNixlPushConnector GLM-5.3-Flash and DeepSeek-V4.1 support."""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KpoolTailSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1 import skip_layers
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_push.base_scheduler import (
    IluNixlPushBaseConnectorScheduler,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_push.base_worker import (
    IluNixlPushWorkerMixin,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_push.connector import (
    IluNixlPushConnector,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_common.dsv41 import (
    Dsv41Layout,
    compute_prefix_replay_alignment,
    legal_decode_cut_layers,
    resolve_dsv41_layout,
    resolve_quantized_kv_format,
    validate_decode_pp_split,
    validate_dsv41_peer_handshake,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_common.semantic_pairing import (
    build_mamba_hma_transfer_routes,
    pair_hma_transfer_regions,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_push.metadata import (
    ChunkPushReqMeta,
    HmaTransferRoute,
    IluNixlPushAgentMetadata,
    IluNixlPushConnectorMetadata,
    IluNixlPushTransferRegion,
    RegionPairing,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_push.push_scheduler import (
    IluNixlPushConnectorScheduler,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_push.push_worker import (
    IluNixlPushConnectorWorker,
    _CHUNK_PROGRESS_PREFIX,
)


def _region(
    region_id: int,
    block_len: int,
    identities: list[str],
    *,
    num_blocks: int = 8,
    kernel_blocks: int = 1,
    block_tokens: int = 128,
    identity_groups: dict[str, list[int]] | None = None,
) -> IluNixlPushTransferRegion:
    groups: set[int] = set()
    kinds: set[str] = set()
    layers: set[int] = set()
    id_groups: dict[str, list[int]] = defaultdict(list)
    for ident in identities:
        parts = ident.split(":", 2)
        layers.add(int(parts[0]))
        kinds.add(parts[1])
        if len(parts) == 3 and parts[2].startswith("group"):
            gid = int(parts[2].removeprefix("group"))
            groups.add(gid)
            base_ident = f"{parts[0]}:{parts[1]}"
            if gid not in id_groups[base_ident]:
                id_groups[base_ident].append(gid)
    if identity_groups:
        for k, v in identity_groups.items():
            id_groups[k] = list(v)
            groups.update(v)
    return IluNixlPushTransferRegion(
        region_id=region_id,
        block_len=block_len,
        shared_by=[],
        layer_indices=sorted(layers),
        group_ids=sorted(groups),
        cache_kinds=sorted(kinds),
        identities=list(identities),
        block_stride=block_len,
        kernel_blocks=kernel_blocks,
        block_tokens=block_tokens,
        num_blocks=num_blocks,
        identity_groups=dict(id_groups),
    )


def _v41_hf_config():
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
        model_config=SimpleNamespace(
            model="deepseek-v4.1",
            dtype=torch.bfloat16,
            hf_text_config=hf,
            hf_config=hf,
            get_total_num_kv_heads=lambda: 1,
            get_head_size=lambda: 512,
            get_total_num_hidden_layers=lambda: 40,
        ),
        cache_config=SimpleNamespace(
            block_size=128,
            cache_dtype="bfloat16",
            swa_bounded_replay=swa_bounded_replay,
            mamba_cache_mode="none",
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
            kv_connector="IluNixlPushConnector",
            engine_id=f"eng-{role}",
            kv_buffer_device="cuda",
            kv_connector_extra_config={},
            get_from_extra_config=lambda _k, default=None: default,
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=None,
    )


# =====================================================================
# GLM-5.3-Flash Push Tests
# =====================================================================


def test_glm5_flash_cache_kind_and_group_free_identity():
    assert (
        IluNixlPushWorkerMixin._cache_kind_from_layer_name(
            "model.layers.3.self_attn.mla_attn"
        )
        == "main_mla"
    )
    assert (
        IluNixlPushWorkerMixin._cache_kind_from_layer_name(
            "model.layers.3.self_attn.indexer.k_cache"
        )
        == "indexer_k_cache"
    )
    assert (
        IluNixlPushWorkerMixin._cache_kind_from_layer_name(
            "model.layers.3.self_attn.indexer.tail_cache"
        )
        == "kpool_tail_cache"
    )
    assert (
        IluNixlPushWorkerMixin._cache_kind_from_layer_name(
            "model.layers.0.self_attn"
        )
        == "mamba_cache"
    )
    assert (
        IluNixlPushWorkerMixin._cache_kind_from_layer_name(
            "model.layers.1.linear_attn"
        )
        == "mamba_cache"
    )
    assert (
        IluNixlPushWorkerMixin._cache_kind_from_layer_name(
            "model.layers.2.attn.kv_cache_scales"
        )
        == "main_mla_scales"
    )
    assert (
        IluNixlPushWorkerMixin._cache_kind_from_layer_name(
            "model.layers.2.attn.indexer.k_cache.kv_cache_scales"
        )
        == "indexer_k_cache_scales"
    )
    assert (
        IluNixlPushWorkerMixin._transfer_identity("model.layers.5.linear_attn", 6)
        == "5:mamba_cache"
    )


def test_glm5_flash_kpool_tail_preserved_under_clip_rules():
    mla_spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
    )
    tail_spec = KpoolTailSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=16,
    )
    tail_uniform = UniformTypeKVCacheSpecs.from_specs(
        {"model.layers.3.self_attn.indexer.tail_cache": tail_spec}
    )
    assert tail_uniform is not None
    groups = [
        KVCacheGroupSpec(["model.layers.3.self_attn.mla_attn"], mla_spec),
        KVCacheGroupSpec(
            ["model.layers.3.self_attn.indexer.tail_cache"], tail_uniform
        ),
    ]
    kv_cfg = SimpleNamespace(
        kv_cache_groups=groups,
        transfer_groups=tuple(groups),
        transfer_group_ids=(0, 1),
        select_transfer_block_ids=lambda ids: tuple(ids),
    )
    sched = object.__new__(IluNixlPushBaseConnectorScheduler)
    sched.kv_cache_config = kv_cfg
    sched._is_hma_required = True
    sched.blocks_per_sw = [0, 2]
    sched._ssm_spec_blocks = [None, None]
    sched.dsv41_layout = None
    sched._init_ilu_transfer_clip_rules()

    # KpoolTailSpec must NOT be clipped to [] on GLM-5.3-Flash.
    clipped = sched.get_exchange_clipped_blocks(([10, 11], [7]))
    assert clipped == ([10, 11], [7])


# =====================================================================
# DeepSeek-V4.1 Push Tests
# =====================================================================


def test_dsv41_layout_pp_cut_handshake_and_replay_alignment(monkeypatch):
    monkeypatch.delenv("VLLM_ILU_PD_PREFILL_FULL_DEPTH", raising=False)
    monkeypatch.delenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", raising=False)
    monkeypatch.delenv("VLLM_ILU_DS41_INDEXER_KV_LAYOUT", raising=False)
    monkeypatch.delenv("VLLM_ILU_V41_REMOTE_REPLAY_ALIGN2", raising=False)

    p_cfg = _v41_vllm_config(role="kv_producer", swa_bounded_replay=True)
    p_layout = resolve_dsv41_layout(p_cfg)
    assert p_layout is not None
    assert p_layout.kv_layer_end == 21

    d_cfg = _v41_vllm_config(role="kv_consumer", pp_size=8, swa_bounded_replay=True)
    d_layout = resolve_dsv41_layout(d_cfg)
    assert d_layout is not None
    assert d_layout.kv_layer_end is None
    assert legal_decode_cut_layers(d_layout) == (1, 2, 8, 14, 20)

    with pytest.raises(ValueError, match="cannot cut inside a KV-sharing group"):
        validate_decode_pp_split(d_cfg, d_layout, pp_split="5,5,5,5,5,5,5,5")

    with pytest.raises(RuntimeError, match="swa_bounded_replay mismatch"):
        validate_dsv41_peer_handshake(
            d_layout,
            remote_engine_id="p-1",
            remote_swa_bounded_replay=False,
            remote_kv_source_layers=d_layout.kv_source_layers,
            remote_index_source_layers=d_layout.index_source_layers,
            remote_compress_ratios=d_layout.compress_ratios,
        )

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
    align = compute_prefix_replay_alignment(
        d_cfg, kv_cfg, is_consumer=True, has_mamba=False
    )
    assert align == 2

    conn = object.__new__(IluNixlPushConnector)
    conn.connector_scheduler = SimpleNamespace(prefix_replay_alignment=2)
    assert conn.prefix_replay_alignment == 2


def test_dsv41_packed_multi_ratio_registration_and_paired_scales(monkeypatch):
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

    num_blocks = 3
    main_layer = "model.layers.2.attn"
    idx_layer = "model.layers.2.attn.indexer.k_cache"
    swa_skip = "model.layers.2.attn.swa_cache"
    monkeypatch.setattr(skip_layers, "_SKIP_LAYERS", {swa_skip})

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
    layer_list = [main_layer, idx_layer, swa_skip]
    total_size = num_blocks * page * len(layer_list)
    tensors = [
        SimpleNamespace(
            size=total_size,
            layers=[lname],
            offset=i * num_blocks * page,
            layer_stride=num_blocks * page,
            block_stride=page,
        )
        for i, lname in enumerate(layer_list)
    ]
    groups = [
        SimpleNamespace(layer_names=[main_layer], kv_cache_spec=main_spec),
        SimpleNamespace(layer_names=[idx_layer], kv_cache_spec=idx_spec),
        SimpleNamespace(layer_names=[swa_skip], kv_cache_spec=swa_spec),
    ]
    kv_cfg = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=groups,
        transfer_groups=tuple(groups),
        kv_cache_tensors=tensors,
    )

    payload_buf = torch.zeros(total_size, dtype=torch.uint8)
    kv_caches = {
        # r=1 on main MLA
        main_layer: torch.as_strided(
            payload_buf,
            size=(num_blocks, I8G64.rows_per_page(512), I8G64.payload_bytes(512)),
            stride=(page, I8G64.payload_bytes(512), 1),
            storage_offset=0,
        ),
        # r=2 on split indexer
        idx_layer: torch.as_strided(
            payload_buf,
            size=(num_blocks * 2, 256, 128),
            stride=(page // 2, 128, 1),
            storage_offset=num_blocks * page,
        ),
        swa_skip: torch.as_strided(
            payload_buf,
            size=(num_blocks, 64, 584),
            stride=(page, 584, 1),
            storage_offset=2 * num_blocks * page,
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
    ].view(num_blocks * 2, ip.rows_per_manager_block // 2)

    vllm_cfg = _v41_vllm_config(role="kv_producer")
    vllm_cfg.compilation_config.static_forward_context = {
        main_layer: SimpleNamespace(kv_cache_scales=main_scale_view),
        idx_layer: SimpleNamespace(kv_cache_scales=idx_scale_view),
    }

    w = object.__new__(IluNixlPushConnectorWorker)
    w.vllm_config = vllm_cfg
    w.kv_transfer_config = vllm_cfg.kv_transfer_config
    w.dsv41_layout = resolve_dsv41_layout(vllm_cfg)
    w._host_staging_producer = False
    w._pool = None
    w._wire_plan = None
    w._hetero_cfg = SimpleNamespace(enabled=False)
    w.use_host_buffer = False
    w._logical_num_blocks = num_blocks
    w._scheduler_block_size = 256
    w._physical_blocks_per_logical_kv_block = 2  # verify reset to 1
    w.num_blocks = num_blocks * 2
    w.block_size = 128
    w.kv_cache_config = kv_cfg
    w.engine_id = "producer"
    w.tp_rank = 0
    w.world_size = 1
    w.pp_rank = 0
    w.pp_world_size = 1
    w.pp_split = "40"
    w.use_mla = True
    w._has_mamba = False
    w.model_config = vllm_cfg.model_config
    w.attn_backends = []
    w.backend_name = "FLASHMLA_SPARSE"
    w.kv_cache_layout = "LBNHC"
    w._mamba_ssm_size = (0, 0)
    w._group_spec_types = (MLAAttentionSpec, MLAAttentionSpec, SlidingWindowMLASpec)
    w.nixl_memory_type = "DRAM"
    w.nixl_backends = ["UCX"]
    w._registered_descs = []
    w.region_mem_types = []
    w.dst_region_mem_types = {}
    w.dst_region_num_blocks = {}
    w.dst_region_group_ids = {}
    w.dst_uses_region_group_mapping = {}
    w.kv_caches_base_addr = {"producer": {}}
    w.dst_num_blocks = {}
    w.src_xfer_handles_by_block_size = {}
    w._push_writer_thread = MagicMock()  # suppress daemon thread spawn in unit test
    w._stack_descs = lambda addrs, length, dev: np.column_stack(
        [addrs, np.full(len(addrs), length), np.full(len(addrs), dev)]
    )
    w.nixl_wrapper = MagicMock()
    w.nixl_wrapper.get_agent_metadata.return_value = b"agent"
    w.nixl_wrapper.prep_xfer_dlist.return_value = 77

    w.register_kv_caches(kv_caches)

    assert w.num_blocks == num_blocks
    assert w._physical_blocks_per_logical_kv_block == 1
    assert swa_skip not in w.layer_region_ids
    assert w._skipped_layers == [swa_skip]
    assert len(w.kv_caches_base_addr["producer"][0]) == 3
    assert w.num_regions == 4
    by_kind = {r.cache_kinds[0]: r for r in w.transfer_regions}
    assert set(by_kind) == {
        "main_mla",
        "indexer_k_cache",
        "main_mla_scales",
        "indexer_k_cache_scales",
    }
    assert by_kind["main_mla"].kernel_blocks == 1
    assert by_kind["indexer_k_cache"].kernel_blocks == 2
    assert by_kind["main_mla_scales"].kernel_blocks == 1
    assert by_kind["indexer_k_cache_scales"].kernel_blocks == 2
    assert w._local_desc_cumsum.tolist() == [0, 3, 6, 9]
    assert w.xfer_handshake_metadata.main_kv_format == "i8g64"
    assert w.xfer_handshake_metadata.indexer_kv_layout == "split"


def test_dsv41_empty_producer_rank_handshakes_and_completes_notif_only(monkeypatch):
    """Verify V4.1 PP stage with all layers skipped (e.g. stage [3,6) in 3,3,3,3,2,2,2,22)."""
    swa_only = "model.layers.3.attn.swa_cache"
    monkeypatch.setattr(skip_layers, "_SKIP_LAYERS", {swa_only})
    monkeypatch.delenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", raising=False)
    monkeypatch.delenv("VLLM_ILU_DS41_INDEXER_KV_LAYOUT", raising=False)

    vllm_cfg = _v41_vllm_config(role="kv_producer", pp_size=8, tp_size=1)
    kv_cfg = SimpleNamespace(
        num_blocks=8,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[swa_only],
                kv_cache_spec=SlidingWindowMLASpec(
                    block_size=64,
                    num_kv_heads=1,
                    head_size=512,
                    dtype=torch.uint8,
                    sliding_window=128,
                    bounded_replay=True,
                ),
            )
        ],
        transfer_groups=(),
        kv_cache_tensors=[],
    )

    p_worker = object.__new__(IluNixlPushConnectorWorker)
    p_worker.vllm_config = vllm_cfg
    p_worker.kv_transfer_config = vllm_cfg.kv_transfer_config
    p_worker.dsv41_layout = resolve_dsv41_layout(vllm_cfg)
    p_worker._host_staging_producer = False
    p_worker._pool = None
    p_worker._wire_plan = None
    p_worker._hetero_cfg = SimpleNamespace(enabled=False)
    p_worker.use_host_buffer = False
    p_worker._logical_num_blocks = 8
    p_worker._scheduler_block_size = 128
    p_worker._physical_blocks_per_logical_kv_block = 1
    p_worker.num_blocks = 8
    p_worker.block_size = 128
    p_worker.kv_cache_config = kv_cfg
    p_worker.engine_id = "producer-pp8"
    p_worker.tp_rank = 0
    p_worker.world_size = 1
    p_worker.pp_rank = 1
    p_worker.pp_world_size = 8
    p_worker.pp_split = "3,3,3,3,2,2,2,22"
    p_worker.use_mla = True
    p_worker._has_mamba = False
    p_worker._is_hma_required = True
    p_worker._engine_ttl = 0.0
    p_worker._handshake_lock = threading.RLock()
    p_worker.model_config = vllm_cfg.model_config
    p_worker.attn_backends = []
    p_worker.backend_name = "FLASHMLA_SPARSE"
    p_worker.kv_cache_layout = "LBNHC"
    p_worker._mamba_ssm_size = (0, 0)
    p_worker._group_spec_types = ()
    p_worker.nixl_memory_type = "DRAM"
    p_worker.kv_caches_base_addr = {"producer-pp8": {}}
    p_worker.dst_num_blocks = {}
    p_worker.dst_region_mem_types = {}
    p_worker._remote_agents = defaultdict(dict)
    p_worker._remote_pp_size_by_engine = {}
    p_worker._remote_pp_split_by_engine = {}
    p_worker.local_region_ids_by_group = {}
    p_worker.remote_region_ids_by_group = {}
    p_worker.tp_mappings = {}
    p_worker._engine_last_active = {}
    p_worker._writer_done_sending = queue.Queue()
    p_worker._push_writer_thread = MagicMock()
    notifs = []
    p_worker.nixl_wrapper = MagicMock()
    p_worker.nixl_wrapper.get_agent_metadata.return_value = b"empty-agent"
    p_worker.nixl_wrapper.add_remote_agent.return_value = "d_agent_0"
    p_worker.nixl_wrapper.send_notif.side_effect = (
        lambda agent, notif_msg=None: notifs.append((agent, notif_msg))
    )

    p_worker.register_kv_caches({swa_only: torch.zeros((8, 64, 16), dtype=torch.uint8)})
    assert p_worker._empty_producer_rank is True
    assert p_worker.num_regions == 0
    assert p_worker.block_len_per_layer == []

    # D-side _add_notif_only_remote_agent must accept empty block_lens without IndexError!
    import msgspec

    empty_meta = msgspec.msgpack.decode(
        p_worker.xfer_handshake_metadata.agent_metadata_bytes,
        type=IluNixlPushAgentMetadata,
    )
    d_worker = object.__new__(IluNixlPushConnectorWorker)
    d_worker.block_len_per_layer = [65536]
    d_worker.transfer_topo = MagicMock()
    d_worker.nixl_wrapper = MagicMock()
    d_worker.nixl_wrapper.add_remote_agent.return_value = "p_stage1"
    assert d_worker._add_notif_only_remote_agent(empty_meta, 1) == "p_stage1"

    # P stage 1 handshakes with D and sends notify-only on final chunk, releasing send lease!
    d_agent_meta = IluNixlPushAgentMetadata(
        engine_id="decode",
        agent_metadata=b"d",
        kv_caches_base_addr=[1000],
        device_id=0,
        num_blocks=8,
        block_lens=[65536],
        block_strides=[65536],
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
        transfer_regions=[_region(0, 65536, ["2:main_mla"])],
        virtual_packed=True,
        swa_bounded_replay=True,
        kv_source_layers=list(p_worker.dsv41_layout.kv_source_layers),
        index_source_layers=list(p_worker.dsv41_layout.index_source_layers),
        compress_ratios=list(p_worker.dsv41_layout.compress_ratios),
    )
    p_worker.add_remote_agent(d_agent_meta, 0, 1, 0, 1)
    p_worker._remote_agents["decode"][0] = "d_agent_0"

    reg_progress = {
        "request_id": f"{_CHUNK_PROGRESS_PREFIX}0:req-1",
        "decode_engine_id": "decode",
        "decode_host": "127.0.0.1",
        "decode_port": 5601,
        "decode_tp_size": 1,
        "decode_pp_size": 1,
        "decode_pp_split": "40",
        "local_block_ids": ([4, 5],),
    }
    p_worker._do_start_push_kv("req-1", (), reg_progress)
    assert notifs == []
    assert p_worker._writer_done_sending.empty()

    reg_final = dict(reg_progress, request_id="req-1")
    p_worker._do_start_push_kv("req-1", (), reg_final)
    # P0-4: predictable empty V4.1 stage (no kv_source in [3,6) of 3,3,3,3,2,2,2,22)
    # is excluded from D's effective_pp_producers (4 instead of 8), so it finishes
    # locally with zero dummy NIXL notifications!
    assert notifs == []
    assert p_worker._writer_done_sending.get_nowait() == ("req-1", False)
    assert p_worker._effective_pp_producers_for_remote("decode", 0) == 4


def test_native_nixl_vram_v41_push_write_end_to_end(monkeypatch):
    """End-to-end VRAM NIXL WRITE test for V4.1 multi-ratio + paired scales + middle block slice."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/Iluvatar GPU required for native NIXL VRAM smoke test")
    nixl_api = pytest.importorskip("nixl._api")
    import time
    import uuid
    import msgspec
    from vllm.distributed.nixl_utils import NixlWrapper
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

    def _build_side(role: str, num_blocks: int, include_forwarder: bool, seed: int):
        layer_list = [main_layer, idx_layer, swa_layer]
        if include_forwarder:
            layer_list.insert(0, fwd_layer)
        total_bytes = num_blocks * page * len(layer_list)
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
            payload_buf = ((idx_t + (idx_t // page) * 31 + seed) % 251).to(
                torch.uint8
            )
        else:
            payload_buf = torch.full(
                (total_bytes,), 0xEE, dtype=torch.uint8, device=device
            )

        slot_of = {lname: i for i, lname in enumerate(layer_list)}
        kv_caches = {}
        if include_forwarder:
            kv_caches[fwd_layer] = torch.as_strided(
                payload_buf,
                size=(
                    num_blocks,
                    I8G64.rows_per_page(512),
                    I8G64.payload_bytes(512),
                ),
                stride=(page, I8G64.payload_bytes(512), 1),
                storage_offset=slot_of[fwd_layer] * num_blocks * page,
            )
        kv_caches[main_layer] = torch.as_strided(
            payload_buf,
            size=(
                num_blocks,
                I8G64.rows_per_page(512),
                I8G64.payload_bytes(512),
            ),
            stride=(page, I8G64.payload_bytes(512), 1),
            storage_offset=slot_of[main_layer] * num_blocks * page,
        )
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
            main_scale_buf = ((s_idx + 53) % 251).to(torch.uint8).view(torch.float16)
        else:
            main_scale_buf = torch.empty(
                main_scale_elems, dtype=torch.float16, device=device
            )
            main_scale_buf.view(torch.uint8).fill_(0xEE)

        scale_views = {}
        for lname in ([fwd_layer, main_layer] if include_forwarder else [main_layer]):
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
            idx_scale_buf = ((is_idx + 97) % 251).to(torch.uint8).view(torch.float32)
        else:
            idx_scale_buf = torch.empty(
                idx_scale_elems, dtype=torch.float32, device=device
            )
            idx_scale_buf.view(torch.uint8).fill_(0xEE)
        ip = idx_scale_layout.placements[idx_layer]
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
        w = object.__new__(IluNixlPushConnectorWorker)
        w.vllm_config = vllm_cfg
        w.kv_transfer_config = vllm_cfg.kv_transfer_config
        w.dsv41_layout = resolve_dsv41_layout(vllm_cfg)
        w._host_staging_producer = False
        w._pool = None
        w._wire_plan = None
        w._hetero_cfg = SimpleNamespace(enabled=False)
        w.use_host_buffer = False
        w._mixed_mem_types = False
        w._empty_producer_rank = False
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
        w.pp_rank = 0
        w.pp_world_size = 2 if include_forwarder else 1
        w.pp_split = "20,20" if include_forwarder else "40"
        w.use_mla = True
        w._has_mamba = False
        w._is_hma_required = True
        w.model_config = vllm_cfg.model_config
        w.attn_backends = []
        w.backend_name = "FLASHMLA_SPARSE"
        w.kv_cache_layout = "LBNHC"
        w._mamba_ssm_size = (0, 0)
        w._group_spec_types = tuple(type(g.kv_cache_spec) for g in groups)
        w.nixl_memory_type = "VRAM"
        w.nixl_backends = ["UCX"]
        w.nixl_wrapper = NixlWrapper(engine_id, agent_cfg)
        w._registered_descs = []
        w.region_mem_types = []
        w.dst_region_mem_types = {}
        w.dst_region_num_blocks = {}
        w.dst_region_group_ids = {}
        w.dst_uses_region_group_mapping = {}
        w.kv_caches_base_addr = {engine_id: {}}
        w.dst_num_blocks = {}
        w.src_xfer_handles_by_block_size = {}
        w.src_xfer_handles_by_tp_ratio = {}
        w.dst_xfer_side_handles = defaultdict(dict)
        w.tp_mappings = {}
        w._remote_agents = defaultdict(dict)
        w._remote_pp_size_by_engine = {}
        w._remote_pp_split_by_engine = {}
        w.local_region_ids_by_group = {}
        w.remote_region_ids_by_group = {}
        w.region_pairings = {}
        w._remote_desc_cumsum = {}
        w._inactive_remote_keys = {}
        w._engine_last_active = {}
        w._engine_ttl = 0.0
        w._handshake_lock = threading.RLock()
        w._sending_transfers = defaultdict(list)
        w._send_failures = set()
        w._sending_transfers_lock = threading.Lock()
        w._writer_done_sending = queue.Queue()
        w._push_writer_thread = MagicMock()
        torch.cuda.synchronize()
        w.register_kv_caches(kv_caches)
        return w, kv_caches, scale_views

    p_worker, p_kv, p_scales = _build_side("kv_producer", 6, True, 17)
    d_worker, d_kv, d_scales = _build_side("kv_consumer", 5, False, 0)
    try:
        p_meta = msgspec.msgpack.decode(
            p_worker.xfer_handshake_metadata.agent_metadata_bytes,
            type=IluNixlPushAgentMetadata,
        )
        d_worker.nixl_wrapper.add_remote_agent(p_meta.agent_metadata)
        d_meta = msgspec.msgpack.decode(
            d_worker.xfer_handshake_metadata.agent_metadata_bytes,
            type=IluNixlPushAgentMetadata,
        )
        d_agent = p_worker.add_remote_agent(
            d_meta, remote_tp_rank=0, remote_tp_size=1, remote_pp_rank=0, remote_pp_size=1
        )
        p_worker._remote_agents[d_worker.engine_id][0] = d_agent

        # P prompt blocks: main=[1,2,3,4,5], idx=[5,4,3,2,1]; D prefix hit=1, replay=2
        # -> D asks for 2 middle blocks starting at index 1 into D blocks [3,1] & [0,4].
        reg_data = {
            "request_id": "d-e2e-1",
            "decode_engine_id": d_worker.engine_id,
            "decode_host": "127.0.0.1",
            "decode_port": 0,
            "decode_tp_size": 1,
            "decode_pp_size": 1,
            "decode_pp_split": "40",
            "local_block_ids": ([3, 1], [0, 4], []),
            "remote_block_starts": [1, 1, -1],
        }
        p_worker._do_start_push_kv(
            "p-e2e-1",
            ([0], [1, 2, 3, 4, 5], [5, 4, 3, 2, 1], []),
            reg_data,
        )

        handles = p_worker._sending_transfers["p-e2e-1"]
        assert len(handles) == 1
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            state = p_worker.nixl_wrapper.check_xfer_state(handles[0])
            if state == "DONE":
                break
            assert state == "PROC", f"unexpected NIXL state: {state}"
            time.sleep(0.005)
        else:
            pytest.fail("NIXL WRITE transfer timed out")
        torch.cuda.synchronize()

        # Verify main payload + FP16 scales: P [2, 3] -> D [3, 1]
        assert torch.equal(d_kv[main_layer][3], p_kv[main_layer][2])
        assert torch.equal(d_kv[main_layer][1], p_kv[main_layer][3])
        assert torch.equal(
            d_scales[main_layer][3].view(torch.uint8),
            p_scales[main_layer][2].view(torch.uint8),
        )
        assert torch.equal(
            d_scales[main_layer][1].view(torch.uint8),
            p_scales[main_layer][3].view(torch.uint8),
        )

        # Verify r=2 split indexer payload + FP32 scales: P sched [4, 3] -> D sched [0, 4]
        assert torch.equal(d_kv[idx_layer][0:2], p_kv[idx_layer][8:10])
        assert torch.equal(d_kv[idx_layer][8:10], p_kv[idx_layer][6:8])
        assert torch.equal(
            d_scales[idx_layer][0:2].view(torch.uint8),
            p_scales[idx_layer][8:10].view(torch.uint8),
        )
        assert torch.equal(
            d_scales[idx_layer][8:10].view(torch.uint8),
            p_scales[idx_layer][6:8].view(torch.uint8),
        )

        # Untouched D blocks and skipped SWA stay sentinel 0xEE!
        assert torch.all(d_kv[main_layer][0] == 0xEE)
        assert torch.all(d_kv[swa_layer] == 0xEE)
    finally:
        for worker in (p_worker, d_worker):
            for handles in worker._sending_transfers.values():
                for h in handles:
                    worker.nixl_wrapper.release_xfer_handle(h)
            for engine_handles in worker.dst_xfer_side_handles.values():
                for h in engine_handles.values():
                    worker.nixl_wrapper.release_dlist_handle(h)
            for h in worker.src_xfer_handles_by_block_size.values():
                worker.nixl_wrapper.release_dlist_handle(h)
            for desc in worker._registered_descs:
                worker.nixl_wrapper.deregister_memory(desc, backends=["UCX"])


def test_native_nixl_vram_glm5_flash_push_write_end_to_end(monkeypatch):
    """End-to-end VRAM NIXL WRITE test for GLM-5.3-Flash P(TP=1,7g) -> D(TP=2,6g) hetero PP+TP!"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/Iluvatar GPU required for native NIXL VRAM smoke test")
    nixl_api = pytest.importorskip("nixl._api")
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    import vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils as ssm_conv_mod

    monkeypatch.setattr(ssm_conv_mod, "is_conv_state_dim_first", lambda: True)
    import time
    import uuid
    import msgspec
    from vllm.distributed.nixl_utils import NixlWrapper
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import KVCacheLayout

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    mla_page = 4096
    idx_page = 1024
    num_blocks = 4

    def _mamba_spec_for_tp(tp: int) -> MambaSpec:
        return MambaSpec(
            shapes=((16 // tp, 4), (4 // tp, 2, 16)),
            dtypes=(torch.float16, torch.float32),
            block_size=128,
            page_size_padded=mla_page,
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        )

    mla_spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=32, dtype=torch.uint8
    )
    idx_spec = MLAAttentionSpec(
        block_size=128,
        tokens_per_state=4,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.uint8,
    )
    tail_spec = KpoolTailSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=32,
        head_size_v=0,
        dtype=torch.uint8,
        sliding_window=16,
        page_size_padded=idx_page,
    )

    p_total = num_blocks * (mla_page + idx_page)
    p_buf = torch.arange(p_total, dtype=torch.int64, device=device)
    p_buf = ((p_buf * 7 + 13) % 251).to(torch.uint8)
    p_mla_name = "model.layers.3.self_attn.mla_attn"
    p_idx_name = "model.layers.3.self_attn.indexer.k_cache"
    p_tail_name = "model.layers.3.self_attn.indexer.tail_cache"
    p_mambas = [f"model.layers.{i}.linear_attn" for i in (0, 1, 2, 4, 5)]

    p_kv = {
        p_mla_name: torch.as_strided(
            p_buf,
            size=(num_blocks, 1, 128, 32),
            stride=(mla_page, mla_page, 32, 1),
            storage_offset=0,
        ),
        p_idx_name: torch.as_strided(
            p_buf,
            size=(num_blocks, 1, 32, 32),
            stride=(idx_page, idx_page, 32, 1),
            storage_offset=num_blocks * mla_page,
        ),
        p_tail_name: torch.as_strided(
            p_buf,
            size=(num_blocks, 1, 32, 32),
            stride=(idx_page, idx_page, 32, 1),
            storage_offset=num_blocks * mla_page,
        ),
    }
    for m_name in p_mambas:
        p_kv[m_name] = torch.as_strided(
            p_buf, size=(num_blocks, mla_page), stride=(mla_page, 1), storage_offset=0
        )

    p_mamba_spec = _mamba_spec_for_tp(1)
    p_attn_uniform = UniformTypeKVCacheSpecs.from_specs(
        {p_mla_name: mla_spec, p_idx_name: idx_spec}
    )
    p_tail_uniform = UniformTypeKVCacheSpecs.from_specs({p_tail_name: tail_spec})
    assert p_attn_uniform is not None and p_tail_uniform is not None
    p_groups = [
        KVCacheGroupSpec([p_mla_name, p_idx_name], p_attn_uniform),
        KVCacheGroupSpec([p_tail_name], p_tail_uniform),
        *[KVCacheGroupSpec([m_name], p_mamba_spec) for m_name in p_mambas],
    ]
    p_cfg = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            model="glm5-flash",
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                model_type="glm5_next", num_hidden_layers=45
            ),
            hf_config=SimpleNamespace(model_type="glm5_next", num_hidden_layers=45),
            use_mla=True,
            get_total_num_kv_heads=lambda: 1,
            get_head_size=lambda: 32,
            get_total_num_hidden_layers=lambda: 45,
        ),
        cache_config=SimpleNamespace(
            block_size=128,
            cache_dtype="bfloat16",
            mamba_cache_mode="align",
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=8,
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_index=0,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_producer",
            kv_connector="IluNixlPushConnector",
            engine_id="glm-p",
            kv_buffer_device="cuda",
            get_from_extra_config=lambda _k, default=None: default,
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=None,
    )
    p_kv_cfg = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=p_groups,
        transfer_groups=tuple(p_groups),
        transfer_group_index_by_layer={
            lname: gidx
            for gidx, grp in enumerate(p_groups)
            for lname in grp.layer_names
        },
        kv_cache_tensors=[],
    )

    def _make_glm_worker(
        vllm_cfg,
        kv_cfg,
        tp_rank: int,
        tp_size: int,
        pp_rank: int,
        pp_size: int,
        pp_split: str,
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
            derive_mamba_conv_split,
        )

        token = uuid.uuid4().hex[:8]
        eid = f"{vllm_cfg.kv_transfer_config.engine_id}-{token}"
        agent_cfg = nixl_api.nixl_agent_config(backends=["UCX"], num_threads=1)
        w = object.__new__(IluNixlPushConnectorWorker)
        w.vllm_config = vllm_cfg
        w.kv_transfer_config = vllm_cfg.kv_transfer_config
        w.kv_buffer_device = "cuda"
        w.dsv41_layout = None
        w._host_staging_producer = False
        w._pool = None
        w._wire_plan = None
        w._hetero_cfg = SimpleNamespace(enabled=False)
        w.use_host_buffer = False
        w.host_xfer_buffers = {}
        w._mixed_mem_types = False
        w._empty_producer_rank = False
        w._virtual_packed_regions = False
        w.packed_block_stride = 0
        w._bypass_upstream_pp_hma_guard = pp_size > 1
        w._supports_pp_hma = False
        w._is_csa_linear = False
        w._ple_group_index = None
        w._ple_region_index = None
        w._has_packed_cache = False
        w._logical_num_blocks = num_blocks
        w._scheduler_block_size = 128
        w._physical_blocks_per_logical_kv_block = 1
        w.num_blocks = num_blocks
        w.block_size = 128
        w.kv_cache_config = kv_cfg
        w.engine_id = eid
        w.tp_rank = tp_rank
        w.world_size = tp_size
        w.transfer_tp_rank = tp_rank
        w.transfer_tp_size = tp_size
        w.dcp_size = 1
        w.pcp_size = 1
        w.pp_rank = pp_rank
        w.pp_size = pp_size
        w.pp_world_size = pp_size
        w.pp_split = pp_split
        w.use_mla = True
        w._has_mamba = True
        w._is_hma_required = True
        w.model_config = vllm_cfg.model_config
        w.attn_backends = []
        w.backend_name = "FLASHMLA_SPARSE"
        w.kv_cache_layout = "LBNHC"
        w.host_buffer_kv_cache_layout = "LBNHC"
        w._layer_specs = IluNixlPushWorkerMixin._layer_specs_for_config(kv_cfg)
        m_spec = next(s for s in w._layer_specs.values() if isinstance(s, MambaSpec))
        w._conv_decomp = derive_mamba_conv_split(m_spec, tp_size)
        w._mamba_ssm_size = w._conv_decomp.ssm_sizes
        w._group_spec_types = tuple(
            MLAAttentionSpec
            if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
            and any(
                isinstance(s, MLAAttentionSpec)
                for s in g.kv_cache_spec.kv_cache_specs.values()
            )
            else (
                KpoolTailSpec
                if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
                else type(g.kv_cache_spec)
            )
            for g in kv_cfg.transfer_groups
        )
        w.nixl_memory_type = "VRAM"
        w.nixl_backends = ["UCX"]
        w.nixl_wrapper = NixlWrapper(eid, agent_cfg)
        w._registered_descs = []
        w.block_len_per_layer = []
        w.block_stride_per_layer = []
        w.region_group_ids = []
        w.region_names = []
        w.region_num_blocks = []
        w.region_mem_types = []
        w._region_is_mla = []
        w._ssm_region_indices = []
        w._scratch_region_indices = []
        w.region_members = []
        w._transfer_layer_names = ()
        w._transfer_layer_region_indices = ()
        w._transfer_layer_group_ids = ()
        w.dst_region_mem_types = {}
        w.dst_region_num_blocks = {}
        w.dst_region_group_ids = {}
        w.dst_uses_region_group_mapping = {}
        w.kv_caches_base_addr = defaultdict(dict)
        w.dst_num_blocks = {}
        w.src_xfer_handles_by_block_size = {}
        w.src_blocks_data_by_block_size = {}
        w.src_xfer_handles_by_tp_ratio = {}
        w.dst_xfer_side_handles = defaultdict(dict)
        w.tp_mappings = {}
        w._remote_agents = defaultdict(dict)
        w._remote_pp_size_by_engine = {}
        w._remote_pp_split_by_engine = {}
        w.local_region_ids_by_group = {}
        w.remote_region_ids_by_group = {}
        w.region_pairings = {}
        w._mamba_hma_routes = {}
        w._remote_fa_num_regions = {}
        w._selected_ssm_remote_region_ids = {}
        w._remote_desc_cumsum = {}
        w._inactive_remote_keys = {}
        w._engine_last_active = {}
        w._engine_ttl = 0.0
        w._handshake_lock = threading.RLock()
        w._handshake_futures = {}
        w._sending_transfers = defaultdict(list)
        w._send_failures = set()
        w._sending_transfers_lock = threading.Lock()
        w._writer_done_sending = queue.Queue()
        w._push_writer_thread = MagicMock()
        return w

    d_mla_names = [
        "model.layers.3.self_attn.mla_attn",
        "model.layers.7.self_attn.mla_attn",
    ]
    d_idx_names = [
        "model.layers.3.self_attn.indexer.k_cache",
        "model.layers.7.self_attn.indexer.k_cache",
    ]
    d_tail_names = [
        "model.layers.3.self_attn.indexer.tail_cache",
        "model.layers.7.self_attn.indexer.tail_cache",
    ]
    d_mamba_groups_names = [
        ["model.layers.0.linear_attn", "model.layers.5.linear_attn"],
        ["model.layers.1.linear_attn", "model.layers.6.linear_attn"],
        ["model.layers.2.linear_attn", "model.layers.8.linear_attn"],
        ["model.layers.4.linear_attn", "model.layers.9.linear_attn"],
    ]
    d_mamba_spec = _mamba_spec_for_tp(2)
    d_attn_uniform = UniformTypeKVCacheSpecs.from_specs(
        {
            d_mla_names[0]: mla_spec,
            d_idx_names[0]: idx_spec,
            d_mla_names[1]: mla_spec,
            d_idx_names[1]: idx_spec,
        }
    )
    d_tail_uniform = UniformTypeKVCacheSpecs.from_specs(
        {d_tail_names[0]: tail_spec, d_tail_names[1]: tail_spec}
    )
    assert d_attn_uniform is not None and d_tail_uniform is not None
    d_groups = [
        KVCacheGroupSpec([*d_mla_names, *d_idx_names], d_attn_uniform),
        KVCacheGroupSpec(d_tail_names, d_tail_uniform),
        *[KVCacheGroupSpec(names, d_mamba_spec) for names in d_mamba_groups_names],
    ]
    d_kv_cfg = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=d_groups,
        transfer_groups=tuple(d_groups),
        transfer_group_index_by_layer={
            lname: gidx
            for gidx, grp in enumerate(d_groups)
            for lname in grp.layer_names
        },
        kv_cache_tensors=[],
    )

    d_total = num_blocks * (2 * mla_page + 2 * idx_page)
    d_workers = []
    d_kvs = []

    p_worker = _make_glm_worker(p_cfg, p_kv_cfg, 0, 1, 0, 8, "6,6,6,6,6,6,5,4")
    torch.cuda.synchronize()
    p_worker.register_kv_caches(p_kv)

    d_engine_id = f"glm-d-{uuid.uuid4().hex[:8]}"
    for d_tp_rank in range(2):
        d_cfg = SimpleNamespace(
            use_v2_model_runner=True,
            model_config=p_cfg.model_config,
            cache_config=p_cfg.cache_config,
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=2,
                tensor_parallel_size=2,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                data_parallel_index=0,
            ),
            scheduler_config=p_cfg.scheduler_config,
            kv_transfer_config=SimpleNamespace(
                kv_role="kv_consumer",
                kv_connector="IluNixlPushConnector",
                engine_id=d_engine_id,
                kv_buffer_device="cuda",
                get_from_extra_config=lambda _k, default=None: default,
            ),
            compilation_config=p_cfg.compilation_config,
            speculative_config=None,
        )
        d_buf = torch.full((d_total,), 0xDD, dtype=torch.uint8, device=device)
        d_kv = {}
        for i, mla_n in enumerate(d_mla_names):
            off = i * num_blocks * mla_page
            d_kv[mla_n] = torch.as_strided(
                d_buf,
                size=(num_blocks, 1, 128, 32),
                stride=(mla_page, mla_page, 32, 1),
                storage_offset=off,
            )
            for grp_names in d_mamba_groups_names:
                d_kv[grp_names[i]] = torch.as_strided(
                    d_buf,
                    size=(num_blocks, mla_page),
                    stride=(mla_page, 1),
                    storage_offset=off,
                )
        idx_base = 2 * num_blocks * mla_page
        for i, idx_n in enumerate(d_idx_names):
            off = idx_base + i * num_blocks * idx_page
            d_kv[idx_n] = torch.as_strided(
                d_buf,
                size=(num_blocks, 1, 32, 32),
                stride=(idx_page, idx_page, 32, 1),
                storage_offset=off,
            )
            d_kv[d_tail_names[i]] = torch.as_strided(
                d_buf,
                size=(num_blocks, 1, 32, 32),
                stride=(idx_page, idx_page, 32, 1),
                storage_offset=off,
            )
        dw = _make_glm_worker(d_cfg, d_kv_cfg, d_tp_rank, 2, 0, 2, "23,22")
        dw.nixl_wrapper = NixlWrapper(
            f"{d_engine_id}-r{d_tp_rank}",
            nixl_api.nixl_agent_config(backends=["UCX"], num_threads=1),
        )
        dw.engine_id = d_engine_id
        torch.cuda.synchronize()
        dw.register_kv_caches(d_kv)
        d_workers.append(dw)
        d_kvs.append(d_kv)

    try:
        for d_tp_rank, dw in enumerate(d_workers):
            d_meta = msgspec.msgpack.decode(
                dw.xfer_handshake_metadata.agent_metadata_bytes,
                type=IluNixlPushAgentMetadata,
            )
            agent = p_worker.add_remote_agent(
                d_meta,
                remote_tp_rank=d_tp_rank,
                remote_tp_size=2,
                remote_pp_rank=0,
                remote_pp_size=2,
            )
            p_worker._remote_agents[d_engine_id][d_tp_rank] = agent

        reg_data = {
            "request_id": "glm-d-req-1",
            "decode_engine_id": d_engine_id,
            "decode_host": "127.0.0.1",
            "decode_port": 0,
            "decode_tp_size": 2,
            "decode_pp_size": 2,
            "decode_pp_split": "23,22",
            "local_block_ids": ([2], [], [1], [], [], []),
        }
        p_worker._do_start_push_kv(
            "glm-p-req-1",
            ([1], [], [], [], [], [], [3]),
            reg_data,
        )

        handles = p_worker._sending_transfers["glm-p-req-1"]
        assert len(handles) == 2
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            states = [p_worker.nixl_wrapper.check_xfer_state(h) for h in handles]
            if all(s == "DONE" for s in states):
                break
            time.sleep(0.005)
        else:
            pytest.fail("GLM-5.3-Flash NIXL WRITE transfer timed out")
        torch.cuda.synchronize()

        for d_tp_rank in range(2):
            assert torch.equal(
                d_kvs[d_tp_rank][d_mla_names[0]][2],
                p_kv[p_mla_name][1],
            )
            assert torch.equal(
                d_kvs[d_tp_rank][d_idx_names[0]][2],
                p_kv[p_idx_name][1],
            )
            p_m5_blk3 = p_kv["model.layers.5.linear_attn"][3]
            d_m5_blk1 = d_kvs[d_tp_rank]["model.layers.5.linear_attn"][1]
            assert torch.equal(
                d_m5_blk1[0:16],
                p_m5_blk3[d_tp_rank * 16 : (d_tp_rank + 1) * 16],
            )
            assert torch.equal(
                d_m5_blk1[16:32],
                p_m5_blk3[32 + d_tp_rank * 16 : 32 + (d_tp_rank + 1) * 16],
            )
            assert torch.equal(
                d_m5_blk1[32:64],
                p_m5_blk3[64 + d_tp_rank * 32 : 64 + (d_tp_rank + 1) * 32],
            )
            assert torch.equal(
                d_m5_blk1[64:320],
                p_m5_blk3[128 + d_tp_rank * 256 : 128 + (d_tp_rank + 1) * 256],
            )
            assert torch.all(d_m5_blk1[320:] == 0xDD)
    finally:
        for worker in [p_worker, *d_workers]:
            for handles in worker._sending_transfers.values():
                for h in handles:
                    worker.nixl_wrapper.release_xfer_handle(h)
            for engine_handles in worker.dst_xfer_side_handles.values():
                for h in engine_handles.values():
                    worker.nixl_wrapper.release_dlist_handle(h)
            for split_handles in worker.src_xfer_handles_by_tp_ratio.values():
                for h in split_handles:
                    worker.nixl_wrapper.release_dlist_handle(h)
            for h in worker.src_xfer_handles_by_block_size.values():
                worker.nixl_wrapper.release_dlist_handle(h)
            for desc in worker._registered_descs:
                worker.nixl_wrapper.deregister_memory(desc, backends=["UCX"])


# =====================================================================
# PR #180 & PR #263 Regression Tests for IluNixlPushConnector
# =====================================================================


def test_pr180_remote_engine_replacement_tracks_and_cleans_up_after_drain():
    """Ensure restarted remote engines at the same (host, port) are tracked and cleaned up in push mode."""
    w = object.__new__(IluNixlPushConnectorWorker)
    w._handshake_lock = threading.RLock()
    w._handshake_futures = {}
    w._push_engine_by_address = {}
    w._engine_by_address = {}
    w._remote_agents = {
        "d-old": {0: "agent-d-old-0"},
        "d-new": {0: "agent-d-new-0"},
    }
    w._engine_last_active = {"d-old": 1.0, "d-new": 2.0}
    w._engine_clock_offset = {"d-old": 0.0, "d-new": 0.0}
    w.dst_xfer_side_handles = {"d-old": {0: 701}, "d-new": {0: 702}}
    w.kv_caches_base_addr = {"d-old": {0: [100]}, "d-new": {0: [200]}}
    w.dst_num_blocks = {"d-old": 8, "d-new": 8}
    w.dst_region_num_blocks = {"d-old": [8], "d-new": [8]}
    w.dst_region_group_ids = {"d-old": [0], "d-new": [0]}
    w.dst_uses_region_group_mapping = {"d-old": False, "d-new": False}
    w.dst_region_mem_types = {"d-old": ["VRAM"], "d-new": ["VRAM"]}
    w.tp_mappings = {"d-old": MagicMock(), "d-new": MagicMock()}
    w.region_pairings = {"d-old": {0: []}, "d-new": {0: []}}
    w._mamba_hma_routes = {"d-old": {0: []}, "d-new": {0: []}}
    w._remote_desc_cumsum = {"d-old": {0: np.zeros(1)}, "d-new": {0: np.zeros(1)}}
    w.local_region_ids_by_group = {"d-old": {}, "d-new": {}}
    w.remote_region_ids_by_group = {"d-old": {}, "d-new": {}}
    w._remote_pp_size_by_engine = {"d-old": 1, "d-new": 1}
    w._remote_pp_split_by_engine = {"d-old": "40", "d-new": "40"}
    w.transfer_topo = MagicMock()
    w._recving_metadata = {}
    w._reg_send_inbox = queue.Queue()
    w._deferred_push_inbox = queue.Queue()
    w._sending_transfers_lock = threading.Lock()
    w._inflight_send_engines = {"req-inflight": "d-old"}
    w._pending_d_registrations = {}
    w._chunk_lock = threading.Lock()
    w._chunk_pending = {}
    w._chunk_plans = {}
    w._chunk_registrations = {}
    w._push_writer_thread = None
    w.nixl_wrapper = MagicMock()

    # Simulate handshake with d-old followed by restart at same [redacted:private-ip]:5601 -> d-new
    w._track_remote_engine_replacement("d-old", "10.0.0.2", 5601)
    w._track_remote_engine_replacement("d-new", "10.0.0.2", 5601)
    assert w._push_engine_by_address[("10.0.0.2", 5601, False)] == "d-new"

    # While d-old still has an in-flight WRITE, cleanup must hold back!
    w._cleanup_replaced_remote_engines()
    assert "d-old" in w._remote_agents
    w.nixl_wrapper.release_dlist_handle.assert_not_called()

    # Once in-flight WRITE finishes, d-old is torn down while d-new stays intact!
    w._inflight_send_engines.clear()
    w._cleanup_replaced_remote_engines()

    assert "d-old" not in w._remote_agents
    assert "d-old" not in w.region_pairings
    assert "d-old" not in w._mamba_hma_routes
    assert "d-old" not in w._remote_desc_cumsum
    assert "d-new" in w._remote_agents
    w.nixl_wrapper.release_dlist_handle.assert_called_once_with(701)
    w.nixl_wrapper.remove_remote_agent.assert_called_once_with("agent-d-old-0")


def test_pr263_pp_last_stage_layer_range_extends_for_extra_mtp_layers():
    """Verify _pp_layer_range, _ordered_local_layers, and _target_pp_ranks extend last PP rank using layer_region_ids."""
    layer_regs = {
        "model.layers.40.linear_attn": [0],
        "model.layers.43.self_attn.mla_attn": [0],
        # GLM-5.3-Flash MTP layer at index 45 (>= sum(pp_split) == 45, no 'mtp' in name!)
        "model.layers.45.self_attn.mla_attn": [2],
        "model.layers.45.self_attn.indexer.k_cache": [3],
    }
    pp_split = "5,6,5,7,5,7,5,5"  # sum = 45

    assert IluNixlPushWorkerMixin._pp_layer_range(pp_split, 7, layer_regs) == (
        40,
        46,
    )
    # Non-final stages must NOT extend beyond their partition slice.
    assert IluNixlPushWorkerMixin._pp_layer_range(pp_split, 0, layer_regs) == (
        0,
        5,
    )

    w = object.__new__(IluNixlPushConnectorWorker)
    w.pp_split = pp_split
    w.pp_rank = 7
    w.layer_region_ids = layer_regs
    w.kv_cache_config = SimpleNamespace(kv_cache_groups=())
    w.vllm_config = SimpleNamespace(speculative_config=None)

    assert w._pp_local_layer_range() == (40, 46)
    ordered = w._ordered_local_layers()
    assert "model.layers.45.self_attn.mla_attn" in ordered
    assert "model.layers.45.self_attn.indexer.k_cache" in ordered

    # Handshake target PP ranks when D is PP=2 ("23,22" -> sum=45) also includes D stage 1.
    assert w._target_pp_ranks(pp_split, 7, "23,22", layer_regs) == [1]


# =====================================================================
# P0 / P1 / P2 Architectural Optimization Tests
# =====================================================================


def test_p0_async_chunk_handshake_never_blocks_writer():
    """P0-1: _activate_pending_chunks must attach a non-blocking callback instead of fut.result()."""
    from concurrent.futures import Future

    w = object.__new__(IluNixlPushConnectorWorker)
    w._chunk_lock = threading.Lock()
    w._sending_transfers_lock = threading.Lock()
    w._chunk_pending = {
        ("req-async", 0): ChunkPushReqMeta(
            local_block_ids=([1, 2],),
            decode_engine_id="d-slow",
            prefill_tokens=256,
            block_offsets=(0,),
            chunk_index=0,
            is_last_chunk=True,
        )
    }
    w._chunk_plans = {}
    w._chunk_expected_layers = {("req-async", 0): ("model.layers.0.attn",)}
    w._chunk_open = {"req-async"}
    w._chunk_registrations = {}
    w._pending_d_registrations = {
        "req-async": {
            "request_id": "req-async",
            "decode_engine_id": "d-slow",
            "decode_host": "10.0.0.9",
            "decode_port": 5601,
            "decode_tp_size": 2,
            "decode_pp_size": 1,
            "decode_pp_split": "40",
            "local_block_ids": ([10, 11],),
        }
    }
    w._remote_agents = {}
    w.pp_split = "40"
    w.pp_rank = 0
    w.layer_region_ids = {"model.layers.0.attn": [0]}
    w._push_writer_wake = threading.Event()
    w._ensure_req_indices()
    w._pending_reg_index.add("req-async")
    w._chunk_inc_pending_locked("req-async")

    slow_fut: Future = Future()
    # Trap fut.result(): if _activate_pending_chunks tried to block on it, raise AssertionError!
    orig_result = slow_fut.result
    slow_fut.result = lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("fut.result() must never be called on writer thread!")
    )
    w._ensure_handshake = lambda *a, **kw: slow_fut

    # First pass returns immediately without blocking while handshake is still in-flight!
    w._activate_pending_chunks()
    assert ("req-async", 0) in w._chunk_pending
    assert ("req-async", 0) not in w._chunk_plans
    assert not w._push_writer_wake.is_set()

    # Completing handshake triggers wake callback and next pass commits plan!
    slow_fut.result = orig_result
    w._remote_agents["d-slow"] = {0: "agent-0"}
    slow_fut.set_result({0: "agent-0"})
    assert w._push_writer_wake.is_set()

    w._activate_pending_chunks()
    assert ("req-async", 0) not in w._chunk_pending
    assert ("req-async", 0) in w._chunk_plans


def test_p0_leader_push_reg_and_rle_wire_compression():
    """P0-2: Only D TP leader sends PUSH_REG, and block IDs use compact RLE on the wire."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        PUSH_REG_NOTIF_PREFIX,
    )
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_common.types import (
        decode_push_reg_wire,
        encode_push_reg_wire,
    )
    import msgspec

    contiguous_blocks = (list(range(100, 612)), [900, 901, 905])
    reg_payload = {
        "request_id": "req-rle",
        "decode_engine_id": "d-eng",
        "decode_host": "10.0.0.2",
        "decode_port": 5601,
        "decode_tp_size": 8,
        "local_block_ids": contiguous_blocks,
        "remote_engine_id": "p-eng",
        "remote_host": "10.0.0.1",
        "remote_port": 5600,
        "remote_tp_size": 1,
        "remote_pp_size": 8,
        "remote_pp_split": "5,6,5,7,5,7,5,5",
    }
    raw_bytes = msgspec.msgpack.encode(reg_payload)
    rle_bytes = encode_push_reg_wire(reg_payload)
    assert len(rle_bytes) < len(raw_bytes) // 4
    decoded = decode_push_reg_wire(rle_bytes)
    assert decoded["local_block_ids"] == contiguous_blocks

    # Verify D=TP8 -> P=TP1: only tp_rank==0 sends PUSH_REG; tp_rank 1..7 skip!
    sent_by_rank = {}
    for d_tp_rank in range(8):
        dw = object.__new__(IluNixlPushConnectorWorker)
        dw.world_size = 8
        dw.tp_rank = d_tp_rank
        dw.pp_world_size = 1
        dw.pp_rank = 0
        dw.pp_split = "45"
        dw._handshake_lock = threading.RLock()
        dw._remote_agents = {"p-eng": {p: f"p_agent_{p}" for p in range(8)}}
        dw.nixl_wrapper = SimpleNamespace(
            send_notif=lambda agent, notif_msg=None, r=d_tp_rank: sent_by_rank.setdefault(
                r, []
            ).append((agent, notif_msg))
        )
        dw._ensure_handshake = lambda *a, **kw: None
        dw._send_registration_to_p("req-rle", reg_payload)

    assert list(sent_by_rank.keys()) == [0]
    assert len(sent_by_rank[0]) == 8
    assert sent_by_rank[0][0][1].startswith(PUSH_REG_NOTIF_PREFIX)


def test_p0_write_once_sealed_chunks_and_p1_hybrid_mamba_split_cadence():
    """P0-3 & P1-1: Intermediate chunks only emit sealed whole blocks, and GLM-5.3-Flash streams MLA per chunk while piggybacking Mamba+Tail on the last chunk."""
    mla_spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
    )
    tail_spec = KpoolTailSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=16,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    mamba_spec = MambaSpec(
        shapes=((16, 4), (4, 2, 16)),
        dtypes=(torch.float16, torch.float32),
        block_size=128,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    groups = [
        KVCacheGroupSpec(["model.layers.3.self_attn.mla_attn"], mla_spec),
        KVCacheGroupSpec(["model.layers.3.self_attn.indexer.tail_cache"], tail_spec),
        KVCacheGroupSpec(["model.layers.0.linear_attn"], mamba_spec),
    ]
    vllm_cfg = _v41_vllm_config(role="kv_producer")
    vllm_cfg.model_config.hf_text_config.model_type = "glm5_next"
    vllm_cfg.num_prefill_lookahead_tokens = 0
    kv_cfg = SimpleNamespace(
        num_blocks=32,
        kv_cache_groups=groups,
        transfer_groups=tuple(groups),
        transfer_group_ids=(0, 1, 2),
        select_transfer_block_ids=lambda ids: tuple(ids),
        has_mamba_layers=True,
    )

    sched = IluNixlPushConnectorScheduler(vllm_cfg, "glm-p", kv_cfg)
    req = SimpleNamespace(
        request_id="req-glm-stream",
        prompt_token_ids=list(range(321)),  # backoff 1 -> prefill_tokens = 320 (2.5 blocks of 128!)
        num_prompt_tokens=321,
        _all_token_ids=list(range(321)),
        max_tokens=16,
        kv_transfer_params={
            "do_remote_decode": True,
            "remote_engine_id": "glm-d",
        },
    )
    sched.on_new_request(req)
    assert req.num_prompt_tokens == 320
    sched._maybe_note_chunk_candidate(req)
    assert "req-glm-stream" in sched._chunk_candidates

    # Chunk 0: 192 tokens scheduled -> seals block 0 ([0..128)), leaves block 1 ([128..192)) unsealed!
    meta0 = IluNixlPushConnectorMetadata()
    sched._maybe_emit_chunk(
        meta0,
        "req-glm-stream",
        num_scheduled_tokens=192,
        num_computed_tokens=0,
        new_block_ids=([10, 11], [7], [30, 31]),
    )
    c0 = meta0.reqs_to_send_chunked["req-glm-stream"]
    assert c0.is_last_chunk is False
    # Write-Once: ONLY sealed block [10] emitted for MLA; Tail & Mamba are [] with offset -1!
    assert c0.local_block_ids == ([10], [], [])
    assert c0.block_offsets == (0, -1, -1)

    # Chunk 1 (final): remaining 128 tokens (total 320 -> 3 MLA blocks [10, 11, 12])!
    meta1 = IluNixlPushConnectorMetadata()
    sched._maybe_emit_chunk(
        meta1,
        "req-glm-stream",
        num_scheduled_tokens=128,
        num_computed_tokens=192,
        new_block_ids=([12], [], [32]),
    )
    c1 = meta1.reqs_to_send_chunked["req-glm-stream"]
    assert c1.is_last_chunk is True
    # Block [10] is NEVER re-written! Chunk 1 writes [11, 12] at offset 1, plus Tail [7] and final Mamba slot [32]!
    assert c1.local_block_ids == ([11, 12], [7], [32])
    assert c1.block_offsets == (1, -1, -1)


def test_p0_v2_bitset_dedup_and_fast_err_circuit_breaker():
    """P0-4: Duplicate V2 notifications from the same (src_pp, src_tp) are ignored, and :ERR fails D immediately without waiting 480s."""
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl_common.types import (
        format_push_notif,
    )

    dw = object.__new__(IluNixlPushConnectorWorker)
    dw.engine_id = "decode"
    dw.world_size = 1
    dw.tp_rank = 0
    dw.kv_transfer_config = SimpleNamespace(kv_role="kv_consumer")
    dw.transfer_topo = SimpleNamespace(tp_ratio=lambda _: 1)
    dw._pending_completion_notifs = queue.Queue()
    dw.consumer_notification_counts_by_req = defaultdict(int)
    dw._recv_notif_bitsets = defaultdict(set)
    dw._recving_metadata = {
        "req-dup": SimpleNamespace(local_block_ids=([1],), remote=None),
        "req-err": SimpleNamespace(local_block_ids=([2],), remote=None),
    }
    dw._recving_transfers = {}
    dw._failed_recv_reqs = queue.Queue()
    dw.pre_recvs = set()
    dw._failed_pre_recvs = set()
    dw._ensure_req_indices()
    dw._recv_meta_index.add("req-dup")
    dw._recv_meta_index.add("req-err")

    # 1. Duplicate V2 notif from (pp=0, tp=0) when expecting 2 stages must NOT prematurely complete!
    n_stage0 = format_push_notif("req-dup", 1, 2, src_pp=0, src_tp=0, status="OK")
    n_stage1 = format_push_notif("req-dup", 1, 2, src_pp=1, src_tp=0, status="OK")
    dw._pending_completion_notifs.put(n_stage0)
    dw._pending_completion_notifs.put(n_stage0)  # duplicate!
    dw._get_new_notifs()
    assert "req-dup" not in dw._recving_transfers
    assert dw.get_push_observability_stats()["duplicate_notifs_ignored"] == 1

    dw._pending_completion_notifs.put(n_stage1)
    dw._get_new_notifs()
    assert "req-dup" in dw._recving_transfers

    # 2. Fast :ERR from stage 0 immediately enqueues req-err into _failed_recv_reqs!
    n_err = format_push_notif("req-err", 1, 4, src_pp=0, src_tp=0, status="ERR")
    dw._pending_completion_notifs.put(n_err)
    dw._get_new_notifs()
    assert dw._failed_recv_reqs.get_nowait() == "req-err"


def test_p1_single_owner_handshake_executes_nixl_ops_on_writer_thread():
    """P1-2: Initiator thread runs ZMQ only, while add_remote_agent & prep_xfer_dlist execute on nixl-push-writer."""
    w = object.__new__(IluNixlPushConnectorWorker)
    w.engine_id = "prefill"
    w.tp_rank = 0
    w.world_size = 1
    w.pp_rank = 0
    w.pp_world_size = 1
    w.pp_split = "40"
    w.use_host_buffer = True
    w._cuda_device_id = 0
    w.device_id = 0
    w._push_writer_stop = threading.Event()
    w._push_writer_wake = threading.Event()
    w._handshake_lock = threading.RLock()
    w._handshake_futures = {}
    w._remote_agents = {}
    w._engine_last_active = {}
    w._push_engine_by_address = {}
    w._handshake_install_inbox = queue.Queue()
    w._hb_send_inbox = queue.Queue()
    w._raw_notif_send_inbox = queue.Queue()
    w._reg_send_inbox = queue.Queue()
    w._deferred_push_inbox = queue.Queue()
    w._finished_blocks_inbox = queue.Queue()
    w._evict_finished_inbox = queue.Queue()
    w._evict_engine_inbox = queue.Queue()
    w._pending_completion_notifs = queue.Queue()
    w._chunk_inbox = queue.Queue()
    w._chunk_deferred = []
    w._chunk_pending = {}
    w._chunk_plans = {}
    w._chunk_lock = threading.Lock()
    w._push_finished_blocks = {}
    w._pending_d_registrations = {}
    w._sending_transfers = defaultdict(list)
    w._sending_transfers_lock = threading.Lock()
    w.block_len_per_layer = [4096]
    w.transfer_topo = MagicMock()
    w._ensure_req_indices()

    caller_threads: list[str] = []

    def _fake_add_remote_agent(agent_bytes):
        caller_threads.append(threading.current_thread().name)
        return "installed-on-writer"

    w.nixl_wrapper = SimpleNamespace(
        add_remote_agent=_fake_add_remote_agent,
        get_new_notifs=lambda: {},
    )
    fake_meta = IluNixlPushAgentMetadata(
        engine_id="d-peer",
        agent_metadata=b"meta",
        kv_caches_base_addr=[1000],
        device_id=0,
        num_blocks=4,
        block_lens=[4096],
        block_strides=[4096],
        kv_cache_layout="LBNHC",
        block_size=128,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASHMLA_SPARSE",
        physical_blocks_per_logical_kv_block=1,
    )
    w._fetch_and_validate_handshake = lambda *a, **kw: [(0, 0, 0, fake_meta)]

    w._push_writer_thread = None
    w._start_push_writer()
    try:
        res: dict = {}

        def _initiator_job():
            res["agents"] = w._nixl_handshake(
                "127.0.0.1", 5601, 1, "d-peer", notif_agents_only=True
            )

        initiator = threading.Thread(
            target=_initiator_job, name="vllm-nixl-handshake-initiator-0"
        )
        initiator.start()
        initiator.join(timeout=3.0)
        assert not initiator.is_alive()
        assert res["agents"] == {0: "installed-on-writer"}
        assert caller_threads == ["nixl-push-writer"]
    finally:
        w._push_writer_stop.set()
        w._push_writer_wake.set()
        w._push_writer_thread.join(timeout=2.0)





def _lease_report_worker():
    """Minimal P-side worker state for the send-completion/lease tests."""
    w = object.__new__(IluNixlPushConnectorWorker)
    w._sending_transfers = defaultdict(list)
    w._send_failures = set()
    w._sending_transfers_lock = threading.Lock()
    w._writer_done_sending = queue.Queue()
    w._send_done_pending = {}
    w._send_done_pending_lock = threading.Lock()
    w._lease_renewals = {}
    w._kv_lease_duration_cfg = 30
    w._max_lease_renewals = 12
    w._lease_extension = 20
    w._obs_stats = None
    w._chunk_lock = threading.Lock()
    w._chunk_plans = {}
    w._chunk_pending = {}
    w._chunk_expected_layers = {}
    w._chunk_current = set()
    w._chunk_meta = {}
    w._chunk_open = set()
    w._chunk_registrations = {}
    w._chunk_deferred = []
    w._chunk_inbox = queue.Queue()
    w._pending_d_registrations = {}
    w._push_finished_blocks = {}
    w._reqs_to_send = {}
    w._reqs_to_process = set()
    w._recving_transfers = {}
    w._recving_metadata = {}
    w._recv_failures = set()
    w._failed_recv_reqs = queue.Queue()
    w._recv_meta_index = None
    w._evict_finished_inbox = queue.Queue()
    w._push_writer_wake = threading.Event()
    w._pending_completion_notifs = queue.Queue()
    w.pre_recvs = set()
    w.transfer_topo = SimpleNamespace()
    w.kv_transfer_config = SimpleNamespace(kv_role="kv_producer")
    w.consumer_notification_counts_by_req = defaultdict(int)
    w._reqs_to_process = set()
    w._ensure_req_indices()

    def _no_new_notifs() -> set[str]:
        return set()

    w._get_new_notifs = _no_new_notifs
    w._cleanup_replaced_remote_engines = lambda: None
    w._sync_device_after_direct_recv = lambda recving: None
    w.expected_consumer_notifications_by_req = {}
    w.xfer_stats = SimpleNamespace(
        record_kv_expired_req=lambda: None,
        record_transfer=lambda res: None,
    )
    w._link_pre_recvs = lambda: None
    return w


def test_send_completion_that_overtakes_lease_is_buffered_not_dropped():
    """A WRITE that finishes before ``request_finished`` must still be reported.

    Even when ``req_id`` is already in ``_reqs_to_process`` (populated on
    prefill step 0 while the request is still ``RUNNING``), it must NOT be
    reported until ``_reqs_to_send`` is registered, otherwise
    ``Scheduler._free_blocks`` trips ``assert request.is_finished()``.
    """
    w = _lease_report_worker()
    w._reqs_to_process.add("req-early")
    w._writer_done_sending.put(("req-early", False))

    first = w.get_transfer_results()
    assert "req-early" not in first.finished_sending
    assert "req-early" in w._reqs_to_process
    assert "req-early" in w._send_done_pending
    assert w.get_push_observability_stats()["completed_early_buffered"] == 1

    # The scheduler finalizes the request and registers the lease on the next step.
    w._reqs_to_send["req-early"] = time.perf_counter() + 30.0
    second = w.get_transfer_results()
    assert second.finished_sending == {"req-early"}
    assert w._send_done_pending == {}
    assert "req-early" not in w._reqs_to_send
    assert "req-early" not in w._reqs_to_process


def test_intermediate_chunk_completion_does_not_buffer_premature_request_done():
    """Chunk 0 finishing while ``_chunk_open`` holds must not fake final done."""
    w = _lease_report_worker()
    w._stock_pop_done_transfers_supports_kind = False
    w.nixl_wrapper = SimpleNamespace(
        check_xfer_state=lambda h: "DONE",
        get_xfer_telemetry=lambda h: None,
        release_xfer_handle=lambda h: None,
    )
    w._sending_transfers["req-multi"] = [101]
    w._mark_chunk_open_locked("req-multi")

    done_ids, failed_ids = w._pop_done_transfers(w._sending_transfers)
    assert done_ids == set()
    assert failed_ids == set()
    assert "req-multi" not in w._sending_transfers
    assert w._send_done_pending == {}


def test_expired_lease_is_never_retained_and_evicts_stuck_chunks():
    """Expired leases must always emit ``finished_sending`` and clean chunk state."""
    w = _lease_report_worker()
    w._reqs_to_process.add("req-stuck")
    w._reqs_to_send["req-stuck"] = time.perf_counter() - 1.0
    w._mark_chunk_open_locked("req-stuck")
    w._chunk_pending[("req-stuck", 0)] = ChunkPushReqMeta(
        local_block_ids=([1, 2],),
        decode_engine_id="decode",
        prefill_tokens=128,
        block_offsets=(0,),
        chunk_index=0,
        is_last_chunk=True,
    )
    w._chunk_inc_pending_locked("req-stuck")

    res = w.get_transfer_results()
    assert res.finished_sending == {"req-stuck"}
    assert "req-stuck" not in w._reqs_to_send
    evicted_id = w._evict_finished_inbox.get_nowait()
    assert evicted_id == "req-stuck"

    w._evict_finished_request(evicted_id)
    assert "req-stuck" not in w._chunk_open
    assert ("req-stuck", 0) not in w._chunk_pending
    assert not w._chunk_should_retain("req-stuck")


def test_opt_nixl_proxy_protocol_alignment(monkeypatch):
    """Verify node_info teardown version, string port coercion, and disconnect ACK."""
    import msgspec
    import zmq
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        HeartbeatInfo as StockHeartbeatInfo,
    )
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    sched = object.__new__(IluNixlPushConnectorScheduler)
    sched.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=8,
            tensor_parallel_size=1,
            data_parallel_rank=3,
        ),
        kv_transfer_config=SimpleNamespace(kv_role="kv_consumer"),
    )
    sched.engine_id = "eng-d3"
    sched.side_channel_host = "10.112.2.52"
    sched.side_channel_port = 8043
    sched.block_size = 64
    sched._pp_split = ""
    sched._init_proxy_registration()
    sched._reqs_need_recv = {}
    sched._remote_block_starts = {}
    sched._reqs_need_send = {}
    sched._reqs_in_batch = set()
    sched._reqs_not_processed = set()
    sched._heartbeat_by_engine = {}
    sched._heartbeat_req_engine = {}
    sched._last_heartbeat_time = 0.0
    sched._heartbeat_interval = 5
    sched.use_host_buffer = False
    sched._push_registration_deadlines = {}
    sched._push_pending_registrations = {}
    sched._newly_finished_push_blocks = {}
    sched._chunk_candidates = {}

    monkeypatch.setenv("VLLM_ILU_NIXL_SERVER_ADDR", "10.112.2.52:8001")
    monkeypatch.delenv("VLLM_ILU_NIXL_PROXY_ADDR", raising=False)
    node_info = sched._proxy_node_info()
    assert node_info["teardown_protocol_version"] == 1
    assert node_info["dp_rank"] == 3

    # Pre-seed proxy cache so lookup succeeds without network, while Router
    # passes ``remote_port`` as a string on the request.
    sched._proxy_meta_cache["eng-p1"] = {
        "tp_size": 1,
        "pp_size": 8,
        "pp_split": "3,3,3,3,2,3,2,21",
        "side_channel_addr": "10.112.2.51:8020",
    }
    req = SimpleNamespace(
        request_id="chatcmpl-test-1",
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_engine_id": "eng-p1",
            "remote_host": "10.112.2.51",
            "remote_port": "8020",
            "transfer_id": "chatcmpl-test-1",
        },
    )
    sched.on_new_request(req)
    assert req.kv_transfer_params["remote_port"] == 8020
    assert isinstance(req.kv_transfer_params["remote_port"], int)
    hb = sched._heartbeat_by_engine["eng-p1"]
    raw_hb = MsgpackEncoder().encode(hb)
    decoded_hb = MsgpackDecoder(StockHeartbeatInfo).decode(raw_hb)
    assert decoded_hb.port == 8020

    # Verify side-channel listener answers Router's 2-tuple disconnect message.
    ready = threading.Event()
    stop = threading.Event()
    port = 39455
    t = threading.Thread(
        target=sched._nixl_handshake_listener,
        args=({}, ready, stop, "127.0.0.1", port),
        daemon=True,
    )
    t.start()
    assert ready.wait(timeout=2.0)
    try:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.connect(f"tcp://127.0.0.1:{port}")
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.send(msgspec.msgpack.encode(("disconnect", ["eng-p1", 99])))
        reply = sock.recv()
        assert reply == b"teardown_complete"
        sock.close()
        ctx.destroy(linger=0)
    finally:
        stop.set()
        t.join(timeout=2.0)

    meta = sched.build_connector_meta(SimpleNamespace())
    assert meta.disconnected_engines == ("eng-p1",)
    assert "eng-p1" not in sched._proxy_meta_cache
    assert "eng-p1" not in sched._heartbeat_by_engine


def test_send_completion_waits_for_last_chunk_before_reporting():
    """Blocks stay pinned while chunks remain, but the completion survives."""
    w = _lease_report_worker()
    w._reqs_to_send["req-chunk"] = time.perf_counter() + 30.0
    w._chunk_open.add("req-chunk")
    w._writer_done_sending.put(("req-chunk", False))

    assert "req-chunk" not in w.get_transfer_results().finished_sending
    assert "req-chunk" in w._send_done_pending

    # Last chunk submitted and completed -> the buffered completion fires.
    w._chunk_open.discard("req-chunk")
    assert w.get_transfer_results().finished_sending == {"req-chunk"}
    assert w._send_done_pending == {}


def test_send_completion_waits_for_inflight_write_after_last_chunk():
    """``_chunk_open`` clears on submit, so an in-flight WRITE still gates it."""
    w = _lease_report_worker()
    w._reqs_to_send["req-inflight"] = time.perf_counter() + 30.0
    w._sending_transfers["req-inflight"] = [42]
    w._writer_done_sending.put(("req-inflight", False))

    assert "req-inflight" not in w.get_transfer_results().finished_sending
    assert "req-inflight" in w._send_done_pending

    w._sending_transfers.pop("req-inflight")
    assert w.get_transfer_results().finished_sending == {"req-inflight"}


def test_heartbeat_renewals_are_capped_so_blocks_eventually_free():
    """A decoder that keeps heartbeating must not pin producer KV forever."""
    w = _lease_report_worker()
    w._reqs_to_send["req-pin"] = time.perf_counter() + 30.0

    for _ in range(w._max_lease_renewals):
        w._handle_heartbeat("req-pin")
    assert w._reqs_to_send["req-pin"] > time.perf_counter()

    w._handle_heartbeat("req-pin")  # over the cap -> force expiry
    assert w._reqs_to_send["req-pin"] <= time.perf_counter()
