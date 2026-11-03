# SPDX-License-Identifier: Apache-2.0
"""CPU behavior tests for PP group storage, admission and transfer lifetimes."""

from types import SimpleNamespace as NS

import pytest
import torch
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCacheTensor,
    CanonicalKVCaches,
    LookupResult,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics as Metrics
from vllm_iluvatar.v1.worker.grouped_cpu_offloading import (
    CPUGroupAllocation,
    GroupCPUOffloadingManager,
    compact_worker_cache,
    plan_cpu_groups,
)


def key(value, group):
    return make_offload_key(value.to_bytes(32, "big"), group)


def two_groups():
    return (
        CPUGroupAllocation(3, 1, (8, 0), 1),
        CPUGroupAllocation(7, 1, (0, 16), 1),
    )


def test_pp8_budget_counts_only_required_group_planes():
    groups = []
    rows = [[0] * 16 for _ in range(8)]
    for i, rank in enumerate([1, 2, 3, 4, 5, 6, 7, 7]):
        ratio = 1 if i == 7 else 2
        for kind, page_tokens, size in [(0, 256, 73728), (1, 512, 67584)]:
            idx = 2 * i + kind
            groups.append(NS(group_id=idx + 5, tokens_per_block=page_tokens * ratio))
            rows[rank][idx] = size
    allocations = plan_cpu_groups(groups, rows, 1, 256 * 2**30)
    coverage = {g.num_chunks * g.tokens_per_chunk for g in allocations}
    assert len(coverage) == 1
    tokens = coverage.pop()
    assert tokens >= 145_437_000
    used = sum(g.num_chunks * sum(g.bytes_per_worker) for g in allocations)
    assert used == tokens * 1890
    assert used <= 256 * 2**30 < used + 1024 * 1890
    assert all(g.bytes_per_worker[0] == 0 for g in allocations)


@pytest.mark.parametrize(
    "rows,tokens,budget",
    [
        ([[0]], 1, 100),
        ([[-1]], 1, 100),
        ([[8, 8]], 1, 100),
        ([[8]], 0, 100),
        ([[8]], 1, 7),
    ],
)
def test_invalid_or_insufficient_group_budget_fails(rows, tokens, budget):
    with pytest.raises(ValueError):
        plan_cpu_groups([NS(group_id=3, tokens_per_block=tokens)], rows, 1, budget)


def test_worker_layout_duplicates_aliases_only_for_live_groups():
    tensors = [
        CanonicalKVCacheTensor(torch.zeros(2, n, dtype=torch.int8), n)
        for n in [64, 8, 2]
    ]
    caches = CanonicalKVCaches(
        tensors,
        [
            [CanonicalKVCacheRef(0, 64), CanonicalKVCacheRef(1, 8)],
            [CanonicalKVCacheRef(0, 64), CanonicalKVCacheRef(2, 2)],
        ],
    )
    allocations = (
        CPUGroupAllocation(3, 2, (72,), 4),
        CPUGroupAllocation(7, 4, (66,), 2),
    )
    compact, counts = compact_worker_cache(caches, allocations, 0, 1)
    assert counts == [4, 4, 2, 2]
    assert [[r.tensor_idx for r in refs] for refs in compact.group_data_refs] == [
        [0, 1],
        [2, 3],
    ]
    assert compact.tensors[0].tensor.data_ptr() == compact.tensors[2].tensor.data_ptr()
    assert sum(t.page_size_bytes * n for t, n in zip(compact.tensors, counts)) == 420
    with pytest.raises(ValueError, match="bytes"):
        compact_worker_cache(
            caches, (CPUGroupAllocation(3, 2, (73,), 4), allocations[1]), 0, 1
        )


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_group_slots_are_isolated_and_active_reads_block_eviction(policy):
    manager = GroupCPUOffloadingManager(two_groups(), cache_policy=policy)
    req = ReqContext("first")
    manager.on_new_request(req)
    keys = [key(1, 7), key(1, 3)]
    pending = manager.prepare_store(keys, req)
    assert pending.keys_to_store == keys
    assert pending.store_spec.block_ids.tolist() == [0, 0]
    assert all(manager.lookup(k, req) == LookupResult.HIT_PENDING for k in keys)
    manager.complete_store(keys, req)
    assert all(manager.lookup(k, req) == LookupResult.HIT for k in keys)
    assert manager.prepare_load(keys, req).block_ids.tolist() == [0, 0]
    newer = [key(2, 7), key(2, 3)]
    assert manager.prepare_store(newer, ReqContext("blocked")) is None
    manager.complete_load(keys, req)
    replacement = manager.prepare_store(newer, ReqContext("replace"))
    assert set(replacement.evicted_keys) == set(keys)
    manager.complete_store(newer, req)
    assert all(manager.lookup(k, req) == LookupResult.HIT for k in newer)
    assert all(manager.lookup(k, req) == LookupResult.MISS for k in keys)


def test_partial_store_failure_rolls_back_prepared_groups():
    manager = GroupCPUOffloadingManager(two_groups())
    ctx = ReqContext("request")
    old = key(1, 7)
    manager.prepare_store([old], ctx)
    first, blocked = key(2, 3), key(2, 7)
    assert manager.prepare_store([first, blocked], ctx) is None
    assert manager.lookup(first, ctx) == LookupResult.MISS
    manager.complete_store([old], ctx)
    prepared = manager.prepare_store([first, blocked], ctx)
    assert prepared.store_spec.block_ids.tolist() == [0, 0]
    manager.complete_store(prepared.keys_to_store, ctx, success=False)
    assert all(manager.lookup(k, ctx) == LookupResult.MISS for k in [first, blocked])
    assert manager.prepare_store([first, blocked], ctx) is not None


def test_finish_before_completion_and_reset_reuse_request_context():
    manager = GroupCPUOffloadingManager(two_groups(), enable_events=True)
    ctx = ReqContext("request")
    keys = [key(1, 3), key(1, 7)]
    manager.on_new_request(ctx)
    manager.prepare_store(keys, ctx)
    manager.on_request_finished(ctx)
    manager.complete_store(keys, ctx)
    assert all(manager.lookup(k, ctx) == LookupResult.HIT for k in keys)
    assert {k for event in manager.take_events() for k in event.keys} == set(keys)
    manager.reset_cache()
    assert all(manager.lookup(k, ctx) == LookupResult.MISS for k in keys)
    again = manager.prepare_store(keys, ctx)
    assert again.store_spec.block_ids.tolist() == [0, 0]
    manager.complete_store(keys, ctx)
    manager.on_request_finished(ctx)


def test_usage_is_weighted_by_reserved_bytes():
    manager = GroupCPUOffloadingManager(two_groups())
    ctx = ReqContext("request")
    keys = [key(1, 3), key(1, 7)]
    manager.prepare_store(keys[:1], ctx)
    stats = manager.get_stats().reduce()
    assert stats[Metrics.CPU_CACHE_USAGE_PERC] == pytest.approx(1 / 3)
    manager.complete_store(keys[:1], ctx)
    manager.prepare_store(keys[1:], ctx)
    manager.complete_store(keys[1:], ctx)
    manager.prepare_load(keys, ctx)
    assert manager.get_stats().reduce()[Metrics.CPU_CACHE_USAGE_PERC] == 1
    manager.complete_load(keys, ctx)
    assert manager.get_stats().reduce()[Metrics.CPU_CACHE_USAGE_PERC] == 0


def test_admission_and_contexts_are_independent_for_each_group():
    manager = GroupCPUOffloadingManager(two_groups(), store_threshold=2)
    keys = [key(1, 3), key(1, 7)]
    first = ReqContext("first")
    manager.on_new_request(first)
    assert manager.prepare_store(keys, first).keys_to_store == []
    manager.on_request_finished(first)
    second = ReqContext("second")
    manager.on_new_request(second)
    prepared = manager.prepare_store(keys, second)
    assert prepared.keys_to_store == keys
    manager.complete_store(keys, second)
    manager.on_request_finished(second)
    assert all(manager.lookup(k, second) == LookupResult.HIT for k in keys)


def test_unknown_group_is_rejected_before_partial_admission():
    manager = GroupCPUOffloadingManager(two_groups())
    ctx = ReqContext("request")
    known, unknown = key(1, 3), key(1, 100)
    assert manager.lookup(unknown, ctx) == LookupResult.MISS
    with pytest.raises(ValueError, match="group"):
        manager.prepare_store([known, unknown], ctx)
    assert manager.lookup(known, ctx) == LookupResult.MISS


@pytest.mark.parametrize("counts", [[], [0], [-1], [1, 2]])
def test_plugin_worker_rejects_wrong_per_plane_capacity(counts):
    from vllm_iluvatar.v1.worker.grouped_cpu_offloading_worker import (
        GroupCPUOffloadingWorker,
    )

    caches = CanonicalKVCaches(
        [CanonicalKVCacheTensor(torch.zeros(1, 8, dtype=torch.int8), 8)], [[]]
    )
    with pytest.raises(ValueError, match="capacities"):
        GroupCPUOffloadingWorker(caches, 1, cpu_chunks_per_tensor=counts)


@pytest.mark.parametrize("tiering", [False, True])
def test_spec_selects_compact_cpu_and_preserves_flat_tiering(monkeypatch, tiering):
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
    from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec
    from vllm_iluvatar.ilu_patch.general_patches import patch_pp_cpu_offloading as patch

    cls = TieringOffloadingSpec if tiering else CPUOffloadingSpec
    spec = object.__new__(cls)
    spec.extra_config = {"cpu_bytes_to_use": 8192}
    spec.blocks_per_chunk = 1
    config = NS(
        canonical_layout=False,
        parallel=NS(world_size=2, rank=0),
        groups=[NS(group_id=3, tokens_per_block=1), NS(group_id=7, tokens_per_block=1)],
    )
    setattr(config, patch._GEOMETRY_ATTR, (8, 8))
    setattr(config, patch._GROUP_GEOMETRY_ATTR, ((8, 0), (0, 8)))
    monkeypatch.setattr(patch, "_ORIGINAL_SPEC_INIT", lambda *_: None)
    monkeypatch.delenv("VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED", raising=False)
    patch._ilu_cpu_offloading_spec_init(spec, config)
    if tiering:
        assert not hasattr(spec, "_ilu_group_allocations")
        assert spec.num_chunks == 2
    else:
        assert (
            sum(g.num_chunks * g.bytes_per_chunk for g in spec._ilu_group_allocations)
            == 8192
        )
        assert {g.num_chunks for g in spec._ilu_group_allocations} == {512}


@pytest.mark.parametrize("blocks_per_chunk", [0, -1])
def test_plugin_worker_rejects_nonpositive_chunk_geometry(blocks_per_chunk):
    from vllm_iluvatar.v1.worker.grouped_cpu_offloading_worker import (
        GroupCPUOffloadingWorker,
    )

    caches = CanonicalKVCaches(
        [CanonicalKVCacheTensor(torch.zeros(1, 8, dtype=torch.int8), 8)], [[]]
    )
    with pytest.raises(ValueError, match="positive"):
        GroupCPUOffloadingWorker(caches, blocks_per_chunk, [1])
