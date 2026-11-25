# SPDX-License-Identifier: Apache-2.0
"""Consolidated unit & mocked-store tests for Iluvatar MooncakeStoreConnector.

Covers:
  1. Config precedence, env/JSON overrides, and 16 MiB slab allocation.
  2. LookupKey protocol codec & per-engine IPC socket isolation.
  3. Deterministic topology namespace & paired DSv4.1 (`i4g32`/`i8g64`, split INT8
     indexer, SWA exclusion, forwarder skip_layers) + TP-sharded/Mamba layouts.
  4. KVConnectorFactory registration, MultiConnector + OptNixlConnector +
     prefix_state compatibility, and PP handshake fanout.
  5. 2P cross-Prefill DRAM offload & bit-exact restore (async & sync load),
     BlockPool pinning, KV events, stats, and reset_cache.
  6. DeepSeek-V4.1-Flash PP=8 non-uniform sharing (SWA-only stage 0 + paired
     i4g32 Ratio-2/Ratio-1 + split INT8 indexer + FP16/FP32 scales + quorum).
  7. `-200` store pressure recovery, preemption handling, and load error tracking.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorStats,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.outputs import KVConnectorOutput

from vllm_iluvatar.distributed.kv_transfer import apply_kv_transfer_registrations
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.mooncake_store import (
    IluMooncakeStoreConnector,
    MooncakeStoreConnector,
    MooncakeStoreKVEvents,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.mooncake_store.config import (
    MOONCAKE_SLAB_SIZE_BYTES,
    IluMooncakeStoreConfig,
    parse_byte_size,
    resolve_pool_dram_bytes,
    setup_ilu_mooncake_store,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.mooncake_store.layout import (
    STORE_GROUP_PP_OWNERS_ATTR,
    GroupStoreLayout,
    KeyMetadata,
    LBHNCStoreLayout,
    LBNHCStoreLayout,
    _resolve_dsv41_quant_format,
    compute_store_topology_namespace,
    plan_group_store_regions,
    resolve_group_pp_owners,
    resolve_store_tp_size,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.mooncake_store.metrics import (
    MooncakeStoreConnectorStats,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.mooncake_store.protocol import (
    MooncakeLookupResult,
    TailKeyBoundary,
    decode_lookup_response,
    encode_lookup_response,
    get_zmq_rpc_path_lookup,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.skip_layers import (
    _SKIP_LAYERS,
    add_skip_layers,
)
from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    PAYLOAD_PAGE_BYTES,
    CompressedKVState,
    plan_scale_layout,
)
from vllm_iluvatar.models.deepseek_v41.indexer_kv import (
    SPLIT_INDEXER_FORMAT,
    IndexerScaleState,
    plan_indexer_scales,
)
from vllm_iluvatar.v1.core.kv_cache_utils import make_get_kv_cache_configs
from vllm_iluvatar.v1.worker.prefill_state.configuration import prefix_transfer_config
from vllm_iluvatar.v1.worker.prefill_state.runtime import attach_prefix_state


# ---------------------------------------------------------------------------
# Shared Mocks & Factories
# ---------------------------------------------------------------------------


class FakeReplicateConfig:
    def __init__(self) -> None:
        self.replica_num = 1
        self.nof_replica_num = 0
        self.preferred_segment = ""
        self.prefer_alloc_in_same_node = False
        self.group_ids: list[str] = []


class SharedDRAMCluster:
    """Thread-safe in-memory simulation of Mooncake master + DRAM segment pool."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.objects: dict[str, bytes] = {}
        self.setup_calls: list[dict[str, Any]] = []
        self.fail_get_keys: set[str] = set()
        self.fail_put_code: int | None = None


class FakeMooncakeClient:
    def __init__(self, cluster: SharedDRAMCluster) -> None:
        self.cluster = cluster
        self.registered: dict[int, int] = {}
        self.closed = False

    def setup(
        self,
        local_hostname: str,
        metadata_server: str,
        global_segment_size: int,
        local_buffer_size: int,
        protocol: str,
        rdma_devices: str,
        master_server_addr: str,
        **kwargs: Any,
    ) -> int:
        with self.cluster.lock:
            self.cluster.setup_calls.append(
                {
                    "local_hostname": local_hostname,
                    "metadata_server": metadata_server,
                    "global_segment_size": global_segment_size,
                    "local_buffer_size": local_buffer_size,
                    "protocol": protocol,
                    "rdma_devices": rdma_devices,
                    "master_server_addr": master_server_addr,
                    **kwargs,
                }
            )
        return 0

    def register_buffer(self, buffer_ptr: int, size: int) -> int:
        self.registered[int(buffer_ptr)] = int(size)
        return 0

    def unregister_buffer(self, buffer_ptr: int) -> int:
        self.registered.pop(int(buffer_ptr), None)
        return 0

    def _is_registered(self, addr: int, size: int) -> bool:
        return any(
            base <= addr and addr + size <= base + r_len
            for base, r_len in self.registered.items()
        )

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        with self.cluster.lock:
            return [1 if k in self.cluster.objects else 0 for k in keys]

    def batch_put_from_multi_buffers(
        self,
        keys: list[str],
        all_addrs: list[list[int]],
        all_sizes: list[list[int]],
        config: FakeReplicateConfig | None = None,
    ) -> list[int]:
        if config is not None and config.group_ids:
            assert len(config.group_ids) == len(keys)
        with self.cluster.lock:
            if self.cluster.fail_put_code is not None:
                return [self.cluster.fail_put_code] * len(keys)
            out: list[int] = []
            for key, addrs, sizes in zip(keys, all_addrs, all_sizes, strict=True):
                buf = bytearray()
                for addr, sz in zip(addrs, sizes, strict=True):
                    if not self._is_registered(addr, sz):
                        out.append(-1)
                        break
                    buf.extend(ctypes.string_at(addr, sz))
                else:
                    self.cluster.objects[key] = bytes(buf)
                    out.append(0)
            return out

    def batch_get_into_multi_buffers(
        self,
        keys: list[str],
        all_addrs: list[list[int]],
        all_sizes: list[list[int]],
        prefer_alloc_in_same_node: bool = False,
    ) -> list[int]:
        del prefer_alloc_in_same_node
        with self.cluster.lock:
            out: list[int] = []
            for key, addrs, sizes in zip(keys, all_addrs, all_sizes, strict=True):
                blob = self.cluster.objects.get(key)
                total = sum(sizes)
                if (
                    key in self.cluster.fail_get_keys
                    or blob is None
                    or len(blob) != total
                    or not all(self._is_registered(a, s) for a, s in zip(addrs, sizes, strict=True))
                ):
                    out.append(-1)
                    continue
                cursor = 0
                for addr, sz in zip(addrs, sizes, strict=True):
                    ctypes.memmove(addr, blob[cursor : cursor + sz], sz)
                    cursor += sz
                out.append(total)
            return out

    def remove_all(self, force: bool = False) -> int:
        del force
        with self.cluster.lock:
            n = len(self.cluster.objects)
            self.cluster.objects.clear()
            return n

    def close(self) -> int:
        self.closed = True
        return 0


def _install_fake_mooncake(
    monkeypatch: pytest.MonkeyPatch, cluster: SharedDRAMCluster, tmp_path: Any
) -> list[FakeMooncakeClient]:
    clients: list[FakeMooncakeClient] = []

    def _factory() -> FakeMooncakeClient:
        c = FakeMooncakeClient(cluster)
        clients.append(c)
        return c

    mc_pkg = ModuleType("mooncake")
    mc_store = ModuleType("mooncake.store")
    mc_store.MooncakeDistributedStore = _factory  # type: ignore[attr-defined]
    mc_store.ReplicateConfig = FakeReplicateConfig  # type: ignore[attr-defined]
    mc_pkg.store = mc_store  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", mc_pkg)
    monkeypatch.setitem(sys.modules, "mooncake.store", mc_store)
    monkeypatch.setattr("vllm.envs.VLLM_RPC_BASE_PATH", str(tmp_path), raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return clients


def _make_hashes(tokens: list[int], hash_block_size: int) -> list[BlockHash]:
    hashes: list[BlockHash] = []
    acc = b""
    for start in range(0, len(tokens) // hash_block_size * hash_block_size, hash_block_size):
        acc = hashlib.sha256(acc + repr(tokens[start : start + hash_block_size]).encode()).digest()[:16]
        hashes.append(BlockHash(acc))
    return hashes


def _mock_req(req_id: str, tokens: list[int], hashes: list[BlockHash]) -> Any:
    return SimpleNamespace(
        request_id=req_id,
        num_tokens=len(tokens),
        prompt_token_ids=tokens,
        all_token_ids=tokens,
        block_hashes=hashes,
        num_computed_tokens=0,
    )


def _mock_blocks(group_blocks: tuple[list[int], ...]) -> Any:
    return SimpleNamespace(
        get_block_ids=lambda group_ids=None: tuple(
            group_blocks[g] for g in (group_ids or range(len(group_blocks)))
        )
    )


def _sched_output(
    *,
    new_reqs: list[NewRequestData] | None = None,
    num_scheduled_tokens: dict[str, int] | None = None,
    finished_req_ids: set[str] | None = None,
    preempted_req_ids: set[str] | None = None,
) -> SchedulerOutput:
    scheduled = num_scheduled_tokens or {}
    out = SchedulerOutput(
        scheduled_new_reqs=new_reqs or [],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        ),
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=finished_req_ids or set(),
        free_encoder_mm_hashes=[],
    )
    if preempted_req_ids:
        out.preempted_req_ids = preempted_req_ids  # type: ignore[attr-defined]
    return out


def _new_req_data(
    req_id: str, tokens: list[int], block_ids: tuple[list[int], ...], num_computed: int = 0
) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=tokens,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=block_ids,
        num_computed_tokens=num_computed,
        lora_request=None,
    )


def _make_vllm_cfg(
    engine_id: str,
    *,
    model: str = "/models/Qwen3-8B",
    use_mla: bool = False,
    num_kv_heads: int = 2,
    block_size: int = 16,
    prefix_match_unit: int | None = None,
    rank: int = 0,
    pp_size: int = 1,
    extra_config: dict[str, Any] | None = None,
) -> Any:
    return SimpleNamespace(
        model_config=SimpleNamespace(model=model, use_mla=use_mla, get_total_num_kv_heads=lambda: num_kv_heads),
        parallel_config=SimpleNamespace(
            rank=rank,
            local_rank=rank,
            world_size=pp_size,
            tensor_parallel_size=1,
            pipeline_parallel_size=pp_size,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_index=0,
        ),
        cache_config=SimpleNamespace(
            block_size=block_size,
            prefix_match_unit=prefix_match_unit,
            cache_dtype="auto",
            prefix_caching_hash_algo="sha256",
            enable_prefix_caching=True,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=None,
        attention_config=None,
        kv_events_config=SimpleNamespace(enable_kv_cache_events=True),
        kv_transfer_config=SimpleNamespace(
            kv_connector="IluMooncakeStoreConnector",
            kv_role="kv_both",
            engine_id=engine_id,
            is_kv_producer=True,
            kv_connector_extra_config=extra_config
            or {
                "master_server_address": "127.0.0.1:50051",
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "rdma",
                "device_name": "mlx5_2",
                "cpu_bytes_to_use": "64MiB",
            },
        ),
    )


def _build_standard_setup(
    engine_id: str, *, num_blocks: int = 16, block_size: int = 16, load_async: bool = True
) -> tuple[Any, KVCacheConfig, dict[str, torch.Tensor]]:
    spec = FullAttentionSpec(block_size=block_size, num_kv_heads=2, head_size=32, dtype=torch.float16)
    layers = ["model.layers.0.attn", "model.layers.1.attn"]
    page = spec.page_size_bytes
    stride = len(layers) * page
    total = num_blocks * stride
    kv_cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=total, layers=layers, layer_stride=page, block_stride=stride, offset=0)],
        kv_cache_groups=[KVCacheGroupSpec(layers, spec)],
    )
    kv_cfg.kv_cache_layout = KVCacheLayout.LBHNC
    vllm_cfg = _make_vllm_cfg(
        engine_id,
        block_size=block_size,
        extra_config={
            "master_server_address": "127.0.0.1:50051",
            "metadata_server": "http://127.0.0.1:18080/metadata",
            "protocol": "rdma",
            "device_name": "mlx5_2",
            "cpu_bytes_to_use": "64MiB",
            "load_async": load_async,
        },
    )
    backing = torch.zeros(total // 2, dtype=torch.float16)
    kv_caches = {
        layer: torch.as_strided(
            backing, (num_blocks, 2, block_size, 32), (stride // 2, block_size * 32, 32, 1), idx * (page // 2)
        )
        for idx, layer in enumerate(layers)
    }
    return vllm_cfg, kv_cfg, kv_caches


def _make_dsv41_specs(fmt_name: str = "i4g32", tokens_per_state: int = 2, idx_block_size: int = 1024):
    fmt = _resolve_dsv41_quant_format(fmt_name)
    rows, payload_b = fmt.rows_per_page(512), fmt.payload_bytes(512)
    swa = SlidingWindowMLASpec(
        bounded_replay=True, block_size=64, sliding_window=128, num_kv_heads=1, head_size=512, head_size_v=0, dtype=torch.bfloat16
    )
    main = MLAAttentionSpec(
        block_size=rows * tokens_per_state,
        tokens_per_state=tokens_per_state,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=payload_b,
        cache_dtype_str=fmt_name,
        storage_block_size=rows * tokens_per_state,
    )
    idx = MLAAttentionSpec(
        block_size=idx_block_size,
        tokens_per_state=tokens_per_state,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        state_content_bytes=128,
        cache_dtype_str=SPLIT_INDEXER_FORMAT,
    )
    return fmt, rows, payload_b, swa, main, idx


def _alloc_dsv41_paired_kv(
    kv_cfg: KVCacheConfig,
    fmt: Any,
    rows: int,
    payload_b: int,
    idx_spec: MLAAttentionSpec,
    main_layers: list[str],
    idx_layers: list[str],
    stride: int,
    *,
    seed: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], CompressedKVState, IndexerScaleState]:
    num_blocks = kv_cfg.num_blocks
    idx_page = idx_spec.page_size_bytes
    n_layers = len(main_layers)
    raw = torch.zeros(num_blocks * stride, dtype=torch.uint8)
    if seed is not None:
        raw.random_(0, 255, generator=torch.Generator().manual_seed(seed))
    fwd_ctx: dict[str, Any] = {
        n: SimpleNamespace(compressed_kv_format=fmt, kv_cache_scales=None) for n in main_layers
    }
    for n in idx_layers:
        fwd_ctx[n] = SimpleNamespace(kv_cache_scales=None)
    m_state = CompressedKVState(plan_scale_layout(kv_cfg, fmt), fmt, "cpu", fwd_ctx)
    i_state = IndexerScaleState(plan_indexer_scales(kv_cfg), "cpu", fwd_ctx)
    if seed is not None:
        m_state.scale_storage.uniform_(-2.0, 2.0)
        i_state.storage.uniform_(0.1, 5.0)
    kv_caches = {
        n: torch.as_strided(raw, (num_blocks, rows, payload_b), (stride, payload_b, 1), pos * PAYLOAD_PAGE_BYTES)
        for pos, n in enumerate(main_layers)
    }
    for pos, n in enumerate(idx_layers):
        kv_caches[n] = torch.as_strided(
            raw,
            (num_blocks, idx_spec.num_states, 128),
            (stride, 128, 1),
            n_layers * PAYLOAD_PAGE_BYTES + pos * idx_page,
        )
    return kv_caches, fwd_ctx, m_state, i_state


# ---------------------------------------------------------------------------
# 1. Config, Protocol, Namespace & Layouts
# ---------------------------------------------------------------------------


def test_config_precedence_and_slab_allocation(tmp_path, monkeypatch) -> None:
    for key in (
        "MOONCAKE_CONFIG_PATH",
        "MOONCAKE_MASTER",
        "MOONCAKE_METADATA_SERVER",
        "MOONCAKE_PROTOCOL",
        "MOONCAKE_DEVICE_NAME",
        "MOONCAKE_GLOBAL_SEGMENT_SIZE",
        "MOONCAKE_LOCAL_BUFFER_SIZE",
        "MOONCAKE_POOL_DRAM_BYTES",
    ):
        monkeypatch.delenv(key, raising=False)

    cfg_file = tmp_path / "mooncake.json"
    cfg_file.write_text(
        json.dumps(
            {
                "metadata_server": "http://10.0.0.1:18080/metadata",
                "master_server_address": "10.0.0.1:50051",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "global_segment_size": "8GB",
                "local_buffer_size": "0",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MOONCAKE_DEVICE_NAME", "mlx5_2")
    cfg = IluMooncakeStoreConfig.load(
        {
            "config_path": str(cfg_file),
            "mooncake_config": {"global_segment_size": "16GiB"},
            "cpu_bytes_to_use": "112GB",
        }
    )
    assert cfg.master_server_address == "10.0.0.1:50051"
    assert cfg.metadata_server == "http://10.0.0.1:18080/metadata"
    assert cfg.protocol == "rdma"
    assert cfg.device_name == "mlx5_2"
    assert cfg.global_segment_size == 16 * 1024**3
    assert cfg.local_buffer_size == 0

    pool_bytes = resolve_pool_dram_bytes({"cpu_bytes_to_use": "112GB"})
    assert pool_bytes == 112 * 1024**3
    per_rank = cfg.resolve_rank_segment_size(
        participating_ranks=7, contribute_memory=True, pool_dram_bytes=pool_bytes
    )
    assert per_rank == 16 * 1024**3 and per_rank % MOONCAKE_SLAB_SIZE_BYTES == 0
    assert cfg.resolve_rank_segment_size(participating_ranks=7, contribute_memory=False, pool_dram_bytes=pool_bytes) == 0

    recorded: list[tuple[Any, ...]] = []
    setup_ilu_mooncake_store(
        SimpleNamespace(setup=lambda *a, **_k: recorded.append(a) or 0),
        cfg,
        "192.168.1.37",
        global_segment_size=per_rank,
    )
    assert recorded[0] == (
        "192.168.1.37",
        "http://10.0.0.1:18080/metadata",
        16 * 1024**3,
        0,
        "rdma",
        "mlx5_2",
        "10.0.0.1:50051",
    )
    assert parse_byte_size("512MiB") == 512 * 1024**2


def test_protocol_codec_and_ipc_socket_isolation() -> None:
    original = MooncakeLookupResult(
        hit_length=384,
        tail_key_boundaries=(TailKeyBoundary(group_id=0, num_tokens=384), TailKeyBoundary(group_id=1, num_tokens=400)),
    )
    assert decode_lookup_response(encode_lookup_response(original)) == original

    mk = lambda eid: SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0),
        kv_transfer_config=SimpleNamespace(engine_id=eid, kv_connector_extra_config={}),
    )
    p1_sock = get_zmq_rpc_path_lookup(mk("prefill-node-p1"))
    assert p1_sock == get_zmq_rpc_path_lookup(mk("prefill-node-p1"))
    assert p1_sock != get_zmq_rpc_path_lookup(mk("prefill-node-p2"))
    assert len(p1_sock.removeprefix("ipc://")) < 100


def test_deterministic_topology_namespace(monkeypatch) -> None:
    spec = MLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=512, dtype=torch.uint8)
    kv_cfg = KVCacheConfig(num_blocks=16, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["model.layers.2.attn"], spec)])
    kv_cfg.kv_cache_layout = "LBHNC"
    mk_cfg = lambda pp=8, tp=1: SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )

    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "2,3,3,3,3,2,3,21")
    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "i4g32")
    monkeypatch.setenv("VLLM_ILU_DS41_INDEXER_KV_LAYOUT", "split")
    monkeypatch.setenv("VLLM_ILU_DS4_INDEXER_KV_DTYPE", "int8")

    ns_p1 = compute_store_topology_namespace(mk_cfg(), kv_cfg, hash_block_size=64)
    assert ns_p1 == compute_store_topology_namespace(mk_cfg(), kv_cfg, hash_block_size=64)

    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "i8g64")
    assert compute_store_topology_namespace(mk_cfg(), kv_cfg, hash_block_size=64) != ns_p1

    assert resolve_store_tp_size({"enable_store_tp_lcm": True, "prefill_tp_sizes": [2, 4]}) == 4
    shared_extra = LBHNCStoreLayout.shared_namespace(store_tp_size=4, pp_size=1)
    assert compute_store_topology_namespace(
        mk_cfg(pp=1, tp=2), kv_cfg, hash_block_size=64, extra_namespace=shared_extra
    ) == compute_store_topology_namespace(
        mk_cfg(pp=1, tp=4), kv_cfg, hash_block_size=64, extra_namespace=shared_extra
    )


@pytest.mark.parametrize("fmt_name,groups", [("i4g32", 16), ("i8g64", 8)])
def test_dsv41_paired_layout_and_skip_layers(fmt_name: str, groups: int) -> None:
    fmt, rows, payload_b, swa_spec, main_spec, idx_spec = _make_dsv41_specs(fmt_name)
    assert not swa_spec.prefix_cacheable

    num_blocks = 4
    main_layers = ["model.layers.2.attn", "model.layers.5.attn"]  # 2=source, 5=PP forwarder
    idx_layers = ["model.layers.2.attn.indexer.k_cache", "model.layers.5.attn.indexer.k_cache"]
    idx_page = idx_spec.page_size_bytes
    stride = 2 * PAYLOAD_PAGE_BYTES + 2 * idx_page
    kv_cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=num_blocks * stride, layers=main_layers, layer_stride=PAYLOAD_PAGE_BYTES, block_stride=stride, offset=0),
            KVCacheTensor(size=num_blocks * stride, layers=idx_layers, layer_stride=idx_page, block_stride=stride, offset=2 * PAYLOAD_PAGE_BYTES),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.0.swa"], swa_spec),
            KVCacheGroupSpec(main_layers, main_spec),
            KVCacheGroupSpec(idx_layers, idx_spec),
        ],
    )
    assert tuple(kv_cfg.prefix_cacheable_group_ids) == (1, 2)
    kv_caches, forward_ctx, main_state, _idx_state = _alloc_dsv41_paired_kv(
        kv_cfg, fmt, rows, payload_b, idx_spec, main_layers, idx_layers, stride
    )

    saved_skip = set(_SKIP_LAYERS)
    try:
        add_skip_layers("model.layers.5.attn", "model.layers.5.attn.indexer.k_cache")
        per_group_regions, registrations = plan_group_store_regions(
            kv_cfg, kv_cfg.prefix_cacheable_group_ids, kv_caches, forward_context=forward_ctx, coalesce=True, honor_skip_layers=False
        )
        assert len(registrations) == 3
        main_regions, idx_regions = per_group_regions
        assert sum(r.page_bytes for r in main_regions) == 2 * PAYLOAD_PAGE_BYTES + 2 * rows * groups * 2
        assert sum(r.page_bytes for r in idx_regions) == 2 * idx_page + 2 * idx_spec.num_states * 4

        layout = GroupStoreLayout(KeyMetadata("dsv41", 0, 0, 0, 1, group_id=0), block_size=main_spec.block_size, hash_block_size=64)
        layout.bind_regions(main_regions)
        addrs, sizes, bids = layout.prepare_values(
            [(0, main_spec.block_size), (main_spec.block_size, 2 * main_spec.block_size)], [1, 3], [0, 0]
        )
        assert bids == [1, 3] and len(addrs) == 2 and all(sum(s) == layout.bytes_per_block for s in sizes)
    finally:
        _SKIP_LAYERS.clear()
        _SKIP_LAYERS.update(saved_skip)
        main_state.release()


def test_nonuniform_pp_owners_tp_sharding_and_mamba() -> None:
    swa_spec = SlidingWindowMLASpec(
        bounded_replay=True, block_size=64, sliding_window=128, num_kv_heads=1, head_size=512, head_size_v=0, dtype=torch.bfloat16
    )
    main_spec = MLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=512, dtype=torch.uint8)
    worker_configs = [
        KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(s, swa_spec), KVCacheGroupSpec(g1, main_spec), KVCacheGroupSpec(g2, main_spec)],
        )
        for s, g1, g2 in [
            (["l0.swa"], [], []),
            (["l2.swa"], ["l2.attn"], []),
            (["l5.swa"], ["l5.attn"], []),
            (["l8.swa"], [], ["l8.attn"]),
        ]
    ]
    vllm_cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=4, tensor_parallel_size=1),
        kv_transfer_config=SimpleNamespace(
            kv_connector="MultiConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "OptNixlConnector", "kv_role": "kv_producer"},
                    {"kv_connector": "IluMooncakeStoreConnector", "kv_role": "kv_both"},
                ]
            },
        ),
    )
    out = make_get_kv_cache_configs(lambda *_a: worker_configs)(vllm_cfg, [{}] * 4, [1] * 4)
    for cfg in out:
        assert getattr(cfg, STORE_GROUP_PP_OWNERS_ATTR) == ((0, 1, 2, 3), (1, 2), (3,))
    assert resolve_group_pp_owners(vllm_cfg, out[0], out[0].prefix_cacheable_group_ids) == ((1, 2), (3,))

    meta = KeyMetadata("dense", tp_rank=1, pcp_rank=0, dcp_rank=0, pp_rank=0)
    lbhnc = LBHNCStoreLayout(meta, 16, 16, local_tp_size=2, store_tp_size=4, tp_rank=1, num_kv_heads=8)
    lbhnc.register_kv_caches([torch.zeros((4, 4, 16, 64), dtype=torch.float16)], num_blocks=4)
    addrs, sizes, _ = lbhnc.prepare_values([(0, 16), (0, 16)], [2], list(lbhnc.local_shard_ids))
    assert len(addrs) == 2 and sizes == [[4096], [4096]]

    lbnhc = LBNHCStoreLayout(meta, 4, 4, local_tp_size=2, store_tp_size=4, tp_rank=0, num_kv_heads=8)
    lbnhc.register_kv_caches(
        [torch.zeros((4, 4, 4, 32), dtype=torch.float16).permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)], num_blocks=4
    )
    addrs_n, sizes_n, _ = lbnhc.prepare_values([(0, 4)], [1], [0])
    assert len(addrs_n[0]) == 4 and sum(sizes_n[0]) == 4 * 2 * 32 * 2

    mamba_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "m0": MambaSpec(shapes=((4, 8), (2, 4, 4)), dtypes=(torch.float32, torch.float16), block_size=16, mamba_cache_mode="align")
        },
    )
    m_cfg = KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["m0"], mamba_spec)])
    m_regions, m_regs = plan_group_store_regions(
        m_cfg, [0], {"m0": [torch.zeros((4, 8, 4), dtype=torch.float32).transpose(1, 2), torch.zeros((4, 2, 4, 4), dtype=torch.float16)]}
    )
    assert len(m_regs) == 2 and sum(r.page_bytes for r in m_regions[0]) == 4 * 8 * 4 + 2 * 4 * 4 * 2
    assert BlockHash(b"\xab" * 16).hex() == "ab" * 16


# ---------------------------------------------------------------------------
# 2. Registration, MultiConnector, Prefix-State & End-to-End Connector Lifecycle
# ---------------------------------------------------------------------------


def test_factory_registration_and_multiconnector_compat() -> None:
    apply_kv_transfer_registrations()
    apply_kv_transfer_registrations()  # idempotent
    cls_std = KVConnectorFactory.get_connector_class_by_name("MooncakeStoreConnector")
    cls_ilu = KVConnectorFactory.get_connector_class_by_name("IluMooncakeStoreConnector")
    assert cls_std is MooncakeStoreConnector and cls_ilu is IluMooncakeStoreConnector
    assert cls_std.requires_piecewise_for_cudagraph({}) is False
    assert cls_std.get_required_kvcache_layout(SimpleNamespace(attention_config=None, model_config=SimpleNamespace(use_mla=False))) == "LBHNC"  # type: ignore[arg-type]
    assert cls_std.get_required_kvcache_layout(SimpleNamespace(attention_config=None, model_config=SimpleNamespace(use_mla=True))) is None  # type: ignore[arg-type]
    assert cls_std.get_required_kvcache_layout(SimpleNamespace(attention_config=SimpleNamespace(hisparse_config=object()), model_config=SimpleNamespace(use_mla=True))) == "BLHNC"  # type: ignore[arg-type]

    conn = object.__new__(IluMooncakeStoreConnector)
    assert conn.requires_kv_delivery is False
    conn.set_xfer_handshake_metadata({0: MagicMock()})
    conn.set_xfer_handshake_metadata_pp_aware({(0, 0): MagicMock(), (7, 0): MagicMock()})

    seen: dict[str, Any] = {}
    fake_nixl = SimpleNamespace(
        set_xfer_handshake_metadata_pp_aware=lambda meta: seen.setdefault("meta", meta),
        attach_prefix_state=lambda st: seen.setdefault("state", st),
    )
    multi = object.__new__(MultiConnector)
    multi._connectors = [fake_nixl, conn]
    payload = {(0, 0): object(), (7, 0): object()}
    multi.set_xfer_handshake_metadata_pp_aware(payload)  # type: ignore[arg-type]
    assert seen["meta"] is payload

    multi_vllm_cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="MultiConnector",
            engine_id="prefill-p1",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "OptNixlConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {}},
                    {"kv_connector": "IluMooncakeStoreConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use": "64GB"}},
                ]
            },
        )
    )
    resolved = prefix_transfer_config(multi_vllm_cfg)
    assert resolved is not None and resolved.kv_connector == "OptNixlConnector"

    sentinel = object()
    attach_prefix_state(SimpleNamespace(sub_connectors=[fake_nixl, conn]), sentinel)
    assert seen["state"] is sentinel

    s1 = MooncakeStoreConnectorStats()
    s1.record_operation("save_put", 0.005, 4, num_bytes=16384)
    multi_stats = MultiConnector.build_kv_connector_stats({"IluMooncakeStoreConnector": s1.data})
    assert isinstance(multi_stats, MultiKVConnectorStats)
    assert multi_stats.reduce()["IluMooncakeStoreConnector"]["save_put_total_bytes"] == 16384


@pytest.mark.parametrize("load_async", [True, False])
def test_2p_cross_prefill_dram_offload_and_restore(tmp_path, monkeypatch, load_async: bool) -> None:
    cluster = SharedDRAMCluster()
    clients = _install_fake_mooncake(monkeypatch, cluster, tmp_path)

    cfg_p1, kv_p1_cfg, p1_caches = _build_standard_setup("single-node-p1", load_async=load_async)
    cfg_p2, kv_p2_cfg, p2_caches = _build_standard_setup("single-node-p2", load_async=load_async)
    p1_worker = IluMooncakeStoreConnector(cfg_p1, KVConnectorRole.WORKER, kv_p1_cfg)
    p1_sched = IluMooncakeStoreConnector(cfg_p1, KVConnectorRole.SCHEDULER, kv_p1_cfg)
    p2_worker = IluMooncakeStoreConnector(cfg_p2, KVConnectorRole.WORKER, kv_p2_cfg)
    p2_sched = IluMooncakeStoreConnector(cfg_p2, KVConnectorRole.SCHEDULER, kv_p2_cfg)

    try:
        pool_p1 = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=16)
        p1_sched.bind_gpu_block_pool(pool_p1)
        p1_worker.register_kv_caches(p1_caches)
        p2_worker.register_kv_caches(p2_caches)

        tokens_p1 = list(range(100, 164))
        hashes_p1 = _make_hashes(tokens_p1, 16)
        req_p1 = _mock_req("req-p1-1", tokens_p1, hashes_p1)
        assert p1_sched.get_num_new_matched_tokens(req_p1, 0) == (0, False)
        p1_sched.update_state_after_alloc(req_p1, _mock_blocks(([1, 2, 3, 4],)), 0)

        for layer_idx, tensor in enumerate(p1_caches.values()):
            for b in (1, 2, 3, 4):
                tensor[b].fill_(float(layer_idx * 100 + b * 7 + 1.5))

        meta_p1 = p1_sched.build_connector_meta(
            _sched_output(new_reqs=[_new_req_data("req-p1-1", tokens_p1, ([1, 2, 3, 4],))], num_scheduled_tokens={"req-p1-1": 64})
        )
        assert p1_sched.has_pending_push_work() is True
        assert all(pool_p1.blocks[b].ref_cnt == 1 for b in (1, 2, 3, 4))

        p1_worker.bind_connector_metadata(meta_p1)
        p1_worker.wait_for_save()
        assert p1_worker.connector_worker and p1_worker.connector_worker.kv_send_thread
        p1_worker.connector_worker.kv_send_thread.request_queue.join()
        worker_meta = p1_worker.build_connector_worker_meta()
        events = p1_worker.get_kv_connector_kv_cache_events()
        p1_worker.clear_connector_metadata()

        p1_sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=worker_meta, kv_cache_events=events))
        assert not p1_sched.has_pending_push_work()
        assert all(pool_p1.blocks[b].ref_cnt == 0 for b in (1, 2, 3, 4))
        assert len(list(p1_sched.take_events())) == 4 and len(cluster.objects) == 4

        # Exact 64-token prompt clamped to 48; 80-token prompt hits all 64 prefix tokens
        assert p2_sched.get_num_new_matched_tokens(_mock_req("req-p2-exact", tokens_p1, hashes_p1), 0) == (48, load_async)
        tokens_p2 = tokens_p1 + list(range(500, 516))
        req_p2 = _mock_req("req-p2-hit", tokens_p2, _make_hashes(tokens_p2, 16))
        assert p2_sched.get_num_new_matched_tokens(req_p2, 0) == (64, load_async)

        dst_blocks = [9, 7, 11, 5]
        p2_sched.update_state_after_alloc(req_p2, _mock_blocks((dst_blocks + [12],)), 64)
        meta_p2 = p2_sched.build_connector_meta(
            _sched_output(
                new_reqs=None if load_async else [_new_req_data("req-p2-hit", tokens_p2, (dst_blocks + [12],), 64)],
                num_scheduled_tokens=None if load_async else {"req-p2-hit": 16},
                finished_req_ids={"req-p2-exact"},
            )
        )
        p2_worker.bind_connector_metadata(meta_p2)
        p2_worker.start_load_kv(SimpleNamespace())  # type: ignore[arg-type]
        assert p2_worker.connector_worker
        p2_worker.connector_worker.recv_request_queue.join()
        xfer = p2_worker.get_transfer_results(set())
        assert (xfer.finished_recving == {"req-p2-hit"}) if load_async else (not xfer.finished_recving)
        assert not xfer.failed_recving
        p2_worker.clear_connector_metadata()

        for lname in p1_caches:
            for src_b, dst_b in zip([1, 2, 3, 4], dst_blocks, strict=True):
                assert torch.equal(p1_caches[lname][src_b], p2_caches[lname][dst_b])

        stats = p1_worker.get_kv_connector_stats()
        assert isinstance(stats, MooncakeStoreConnectorStats) and stats.reduce()["save_put_total_keys"] == 4

        assert p1_sched.reset_cache() is True and len(cluster.objects) == 0
        assert p2_sched.get_num_new_matched_tokens(req_p2, 0) == (0, False)
    finally:
        p1_sched.shutdown()
        p2_sched.shutdown()
        p1_worker.shutdown()
        p2_worker.shutdown()
        assert all(c.closed and not c.registered for c in clients)


def test_dsv41_pp8_nonuniform_2p_paired_kv_sharing(tmp_path, monkeypatch) -> None:
    """DeepSeek-V4.1-Flash PP=8 (2,3,3,3,3,2,3,21): Stage 0 is SWA-only (0B DRAM,
    runs LookupKeyServer); Stages 1..4 own Ratio-2, Stages 5..7 own Ratio-1 (padded stride)."""
    cluster = SharedDRAMCluster()
    _install_fake_mooncake(monkeypatch, cluster, tmp_path)
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "2,3,3,3,3,2,3,21")
    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "i4g32")
    monkeypatch.setenv("VLLM_ILU_DS41_INDEXER_KV_LAYOUT", "split")
    monkeypatch.setenv("VLLM_ILU_DS4_INDEXER_KV_DTYPE", "int8")

    fmt, rows, payload_b, swa_spec, r2_main, r2_idx = _make_dsv41_specs("i4g32", tokens_per_state=2, idx_block_size=512)
    _, _, _, _, r1_main, r1_idx = _make_dsv41_specs("i4g32", tokens_per_state=1, idx_block_size=256)
    pp_size, num_blocks = 8, 6
    owners = (tuple(range(pp_size)), (1, 2, 3, 4), (1, 2, 3, 4), (5, 6, 7), (5, 6, 7))
    extra = {
        "master_server_address": "127.0.0.1:50051",
        "protocol": "rdma",
        "device_name": "mlx5_2",
        "cpu_bytes_to_use": str(7 * 16 * 1024**2),
    }

    def _build_engine(engine_id: str, fill_seed: int | None):
        workers, states, stage_data = [], [], []
        mk_cfg = lambda rank: _make_vllm_cfg(
            engine_id,
            model="/models/DeepSeek-V4.1-Flash",
            use_mla=True,
            num_kv_heads=1,
            block_size=64,
            prefix_match_unit=64,
            rank=rank,
            pp_size=pp_size,
            extra_config=extra,
        )
        sched_kv_cfg = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["l0.swa"], swa_spec),
                KVCacheGroupSpec(["l2.attn"], r2_main),
                KVCacheGroupSpec(["l2.attn.indexer.k_cache"], r2_idx),
                KVCacheGroupSpec(["l10.attn"], r1_main),
                KVCacheGroupSpec(["l10.attn.indexer.k_cache"], r1_idx),
            ],
        )
        setattr(sched_kv_cfg, STORE_GROUP_PP_OWNERS_ATTR, owners)

        for stage in range(pp_size):
            vcfg = mk_cfg(stage)
            if stage == 0:
                w_kv_cfg = KVCacheConfig(
                    num_blocks=num_blocks,
                    kv_cache_tensors=[],
                    kv_cache_groups=[
                        KVCacheGroupSpec(["model.layers.0.swa"], swa_spec),
                        KVCacheGroupSpec([], r2_main),
                        KVCacheGroupSpec([], r2_idx),
                        KVCacheGroupSpec([], r1_main),
                        KVCacheGroupSpec([], r1_idx),
                    ],
                )
                setattr(w_kv_cfg, STORE_GROUP_PP_OWNERS_ATTR, owners)
                w = IluMooncakeStoreConnector(vcfg, KVConnectorRole.WORKER, w_kv_cfg)
                w.register_kv_caches({"model.layers.0.swa": torch.zeros((num_blocks, 64, 512), dtype=torch.bfloat16)})
                workers.append(w)
                stage_data.append(None)
                continue

            is_r2 = stage <= 4
            m_spec, i_spec = (r2_main, r2_idx) if is_r2 else (r1_main, r1_idx)
            m_name, i_name = f"model.layers.{stage * 2}.attn", f"model.layers.{stage * 2}.attn.indexer.k_cache"
            i_page = i_spec.page_size_bytes
            stride = ((PAYLOAD_PAGE_BYTES + i_page + PAYLOAD_PAGE_BYTES - 1) // PAYLOAD_PAGE_BYTES) * PAYLOAD_PAGE_BYTES
            groups = [KVCacheGroupSpec([f"model.layers.{stage * 2}.swa"], swa_spec)]
            if is_r2:
                groups.extend([KVCacheGroupSpec([m_name], m_spec), KVCacheGroupSpec([i_name], i_spec), KVCacheGroupSpec([], r1_main), KVCacheGroupSpec([], r1_idx)])
            else:
                groups.extend([KVCacheGroupSpec([], r2_main), KVCacheGroupSpec([], r2_idx), KVCacheGroupSpec([m_name], m_spec), KVCacheGroupSpec([i_name], i_spec)])

            w_kv_cfg = KVCacheConfig(
                num_blocks=num_blocks,
                kv_cache_tensors=[
                    KVCacheTensor(size=num_blocks * stride, layers=[m_name], layer_stride=PAYLOAD_PAGE_BYTES, block_stride=stride, offset=0),
                    KVCacheTensor(size=num_blocks * stride, layers=[i_name], layer_stride=i_page, block_stride=stride, offset=PAYLOAD_PAGE_BYTES),
                ],
                kv_cache_groups=groups,
            )
            setattr(w_kv_cfg, STORE_GROUP_PP_OWNERS_ATTR, owners)
            kv_caches, fwd_ctx, m_state, i_state = _alloc_dsv41_paired_kv(
                w_kv_cfg, fmt, rows, payload_b, i_spec, [m_name], [i_name], stride,
                seed=(fill_seed + stage) if fill_seed is not None else None,
            )
            vcfg.compilation_config.static_forward_context = fwd_ctx
            states.extend([m_state, i_state])
            w = IluMooncakeStoreConnector(vcfg, KVConnectorRole.WORKER, w_kv_cfg)
            w.register_kv_caches(kv_caches)
            workers.append(w)
            stage_data.append((kv_caches[m_name], kv_caches[i_name], fwd_ctx[m_name].kv_cache_scales, fwd_ctx[i_name].kv_cache_scales))

        return IluMooncakeStoreConnector(mk_cfg(0), KVConnectorRole.SCHEDULER, sched_kv_cfg), workers, stage_data, states

    p1_sched, p1_workers, p1_data, p1_states = _build_engine("dsv41-p1", fill_seed=42)
    p2_sched, p2_workers, p2_data, p2_states = _build_engine("dsv41-p2", fill_seed=None)

    try:
        assert cluster.setup_calls[0]["global_segment_size"] == 0
        assert all(c["global_segment_size"] == 16 * 1024**2 for c in cluster.setup_calls[1:pp_size])

        pool = BlockPool(num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=64)
        p1_sched.bind_gpu_block_pool(pool)

        tokens_512 = list(range(512))
        req1 = _mock_req("dsv41-req1", tokens_512, _make_hashes(tokens_512, 64))
        assert p1_sched.get_num_new_matched_tokens(req1, 0) == (0, False)
        p1_all_blocks = ([1] * 8, [2], [2], [1, 3], [1, 3])
        p1_sched.update_state_after_alloc(req1, _mock_blocks(p1_all_blocks), 0)

        meta1 = p1_sched.build_connector_meta(
            _sched_output(new_reqs=[_new_req_data("dsv41-req1", tokens_512, p1_all_blocks)], num_scheduled_tokens={"dsv41-req1": 512})
        )
        agg_wm = None
        agg_ev: MooncakeStoreKVEvents | None = None
        for w in p1_workers:
            w.bind_connector_metadata(meta1)
            w.wait_for_save()
            assert w.connector_worker and w.connector_worker.kv_send_thread
            w.connector_worker.kv_send_thread.request_queue.join()
            wm = w.build_connector_worker_meta()
            assert wm is not None
            agg_wm = wm if agg_wm is None else agg_wm.aggregate(wm)
            ev = w.get_kv_connector_kv_cache_events()
            assert ev is not None
            if agg_ev is None:
                agg_ev = ev
            else:
                agg_ev.add_events(ev.get_all_events())
                agg_ev.increment_workers(1)
            w.clear_connector_metadata()

        p1_sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=agg_wm, kv_cache_events=agg_ev))
        assert not p1_sched.has_pending_push_work()
        assert all(pool.blocks[b].ref_cnt == 0 for b in (1, 2, 3))
        assert len(list(p1_sched.take_events())) == 6  # 1 r2_main + 1 r2_idx + 2 r1_main + 2 r1_idx

        tokens_640 = tokens_512 + list(range(1000, 1128))
        req2 = _mock_req("dsv41-req2", tokens_640, _make_hashes(tokens_640, 64))
        assert p2_sched.get_num_new_matched_tokens(req2, 0) == (512, True)
        p2_sched.update_state_after_alloc(req2, _mock_blocks(([1] * 8, [4], [4], [5, 2], [5, 2])), 512)
        meta2 = p2_sched.build_connector_meta(_sched_output())

        for w in p2_workers:
            w.bind_connector_metadata(meta2)
            w.start_load_kv(SimpleNamespace())  # type: ignore[arg-type]
            assert w.connector_worker
            w.connector_worker.recv_request_queue.join()
            res = w.get_transfer_results(set())
            assert res.finished_recving == {"dsv41-req2"} and not res.failed_recving
            w.clear_connector_metadata()

        for stage in range(1, pp_size):
            m1, i1, ms1, is1 = p1_data[stage]
            m2, i2, ms2, is2 = p2_data[stage]
            for src_b, dst_b in (((2, 4),) if stage <= 4 else ((1, 5), (3, 2))):
                assert torch.equal(m2[dst_b], m1[src_b]) and torch.equal(ms2[dst_b], ms1[src_b])
                assert torch.equal(i2[dst_b], i1[src_b]) and torch.equal(is2[dst_b], is1[src_b])
    finally:
        p1_sched.shutdown()
        p2_sched.shutdown()
        for w in (*p1_workers, *p2_workers):
            w.shutdown()
        for st in (*p1_states, *p2_states):
            st.release()


def test_pressure_recovery_preemption_and_load_errors(tmp_path, monkeypatch) -> None:
    cluster = SharedDRAMCluster()
    _install_fake_mooncake(monkeypatch, cluster, tmp_path)
    vllm_cfg, kv_cfg, caches = _build_standard_setup("err-engine")
    worker = IluMooncakeStoreConnector(vllm_cfg, KVConnectorRole.WORKER, kv_cfg)
    sched = IluMooncakeStoreConnector(vllm_cfg, KVConnectorRole.SCHEDULER, kv_cfg)

    try:
        worker.register_kv_caches(caches)
        tokens = list(range(32))
        sched.update_state_after_alloc(_mock_req("req-pressure", tokens, _make_hashes(tokens, 16)), _mock_blocks(([1, 2],)), 0)

        cluster.fail_put_code = -200
        meta_save = sched.build_connector_meta(
            _sched_output(new_reqs=[_new_req_data("req-pressure", tokens, ([1, 2],))], num_scheduled_tokens={"req-pressure": 32})
        )
        worker.bind_connector_metadata(meta_save)
        worker.wait_for_save()
        assert worker.connector_worker and worker.connector_worker.kv_send_thread
        worker.connector_worker.kv_send_thread.request_queue.join()
        assert worker.connector_worker.kv_send_thread._store_pressure_active is True
        worker.clear_connector_metadata()

        cluster.fail_put_code = None
        tokens2 = list(range(48))
        hashes2 = _make_hashes(tokens2, 16)
        sched.update_state_after_alloc(_mock_req("req-ok", tokens2, hashes2), _mock_blocks(([3, 4, 5],)), 0)
        meta_save2 = sched.build_connector_meta(
            _sched_output(
                new_reqs=[_new_req_data("req-ok", tokens2, ([3, 4, 5],))],
                num_scheduled_tokens={"req-ok": 48},
                preempted_req_ids={"req-pressure"},
            )
        )
        worker.bind_connector_metadata(meta_save2)
        worker.handle_preemptions(meta_save2)
        worker.wait_for_save()
        worker.connector_worker.kv_send_thread.request_queue.join()
        assert worker.connector_worker.kv_send_thread._store_pressure_active is False
        worker.clear_connector_metadata()

        cluster.fail_get_keys.add(next(iter(cluster.objects)))
        req_load = _mock_req("req-load-err", tokens2, hashes2)
        assert sched.get_num_new_matched_tokens(req_load, 0)[0] == 32
        sched.update_state_after_alloc(req_load, _mock_blocks(([6, 7],)), 32)
        worker.bind_connector_metadata(sched.build_connector_meta(_sched_output()))
        worker.start_load_kv(SimpleNamespace())  # type: ignore[arg-type]
        worker.connector_worker.recv_request_queue.join()
        res = worker.get_transfer_results(set())
        worker.clear_connector_metadata()

        assert res.finished_recving == {"req-load-err"}
        assert worker.get_block_ids_with_load_errors() & {6, 7}

        stats = worker.get_kv_connector_stats()
        assert stats is not None and not stats.is_empty()
        reduced = stats.reduce()
        assert (reduced["save_put_count"], reduced["save_put_failed_keys"], reduced["load_get_failed_keys"]) == (2, 2, 1)
    finally:
        sched.shutdown()
        worker.shutdown()
