# SPDX-License-Identifier: Apache-2.0
"""Group-owned CPU arenas for nonuniform pipeline-parallel KV offloading."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import lcm

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    LookupResult,
    Medium,
    OffloadingManager,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec, CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


@dataclass(frozen=True)
class CPUGroupAllocation:
    group_id: int
    tokens_per_chunk: int
    bytes_per_worker: tuple[int, ...]
    num_chunks: int

    @property
    def bytes_per_chunk(self) -> int:
        return sum(self.bytes_per_worker)


def plan_cpu_groups(groups, bytes_by_worker, blocks_per_chunk, budget):
    """Balance complete prefix coverage without exceeding the CPU byte budget."""
    if not groups or not bytes_by_worker or blocks_per_chunk <= 0:
        raise ValueError("CPU group geometry must be nonempty and positive")
    if any(len(row) != len(groups) for row in bytes_by_worker):
        raise ValueError("CPU group geometry does not match selected groups")
    if any(value < 0 for row in bytes_by_worker for value in row):
        raise ValueError("CPU group bytes cannot be negative")
    ids = [g.group_id for g in groups]
    tokens = [g.tokens_per_block * blocks_per_chunk for g in groups]
    costs = [
        tuple(row[i] * blocks_per_chunk for row in bytes_by_worker)
        for i in range(len(groups))
    ]
    if len(set(ids)) != len(ids) or min(tokens) <= 0 or any(sum(c) <= 0 for c in costs):
        raise ValueError("CPU group IDs, token widths or byte costs are invalid")
    quantum = lcm(*tokens)
    quantum_bytes = sum(sum(c) * (quantum // t) for c, t in zip(costs, tokens))
    units = int(budget) // quantum_bytes
    if units <= 0:
        raise ValueError("CPU budget cannot hold one complete group alignment")
    return tuple(
        CPUGroupAllocation(gid, t, c, units * (quantum // t))
        for gid, t, c in zip(ids, tokens, costs)
    )


def compact_worker_cache(kv_caches, allocations, rank, blocks_per_chunk):
    """Give each group separate CPU planes while preserving GPU aliases."""
    if len(kv_caches.group_data_refs) != len(allocations):
        raise ValueError("CPU worker group count does not match arena plan")
    tensors, groups, counts = [], [], []
    for allocation, refs in zip(allocations, kv_caches.group_data_refs):
        actual = sum(ref.page_size_bytes for ref in refs) * blocks_per_chunk
        if actual != allocation.bytes_per_worker[rank]:
            raise ValueError("CPU worker group bytes do not match arena plan")
        remapped = []
        for ref in refs:
            tensor = kv_caches.tensors[ref.tensor_idx]
            if tensor.page_size_bytes != ref.page_size_bytes:
                raise ValueError("Direct CPU pages must have matching byte widths")
            remapped.append(CanonicalKVCacheRef(len(tensors), ref.page_size_bytes))
            tensors.append(tensor)
            counts.append(allocation.num_chunks)
        groups.append(remapped)
    return CanonicalKVCaches(tensors, groups), counts


@dataclass
class _GroupRequestState:
    owner: object
    contexts: dict[int, ReqContext] = field(default_factory=dict)


class GroupCPUOffloadingManager(OffloadingManager):
    """Reuse native cache policies in independent group-local slot spaces."""

    def __init__(self, allocations, **kwargs):
        self.medium = Medium.CPU
        self.allocations = {g.group_id: g for g in allocations}
        if len(self.allocations) != len(allocations) or not allocations:
            raise ValueError("CPU group allocation IDs must be unique")
        if any(g.num_chunks <= 0 or g.bytes_per_chunk <= 0 for g in allocations):
            raise ValueError("CPU group capacities and bytes must be positive")
        total_chunks = sum(g.num_chunks for g in allocations)
        tracker_budget = kwargs.pop("max_tracker_size", 64_000)
        self.managers = {
            g.group_id: CPUOffloadingManager(
                num_chunks=g.num_chunks,
                max_tracker_size=tracker_budget * g.num_chunks // total_chunks,
                **kwargs,
            )
            for g in allocations
        }
        self.allocated_bytes = sum(
            g.num_chunks * g.bytes_per_chunk for g in allocations
        )

    def _context(self, req, group_id):
        state = req.get_state(_GroupRequestState)
        if state is None or state.owner is not self:
            state = _GroupRequestState(self)
            req.set_state(state)
        if group_id not in state.contexts:
            child = ReqContext(
                req.req_id, req.kv_transfer_params, req.kv_hints, req.load_tier_filter
            )
            # Positions arrive incrementally; policies read the same live map.
            child._offload_key_positions = req._offload_key_positions
            state.contexts[group_id] = child
            self.managers[group_id].on_new_request(child)
        return state.contexts[group_id]

    def _group_keys(self, keys):
        groups = {}
        for key in keys:
            gid = get_offload_group_idx(key)
            if gid not in self.managers:
                raise ValueError(f"Unknown CPU offload group: {gid}")
            groups.setdefault(gid, []).append(key)
        return groups

    def on_new_request(self, req_context):
        return RequestOffloadingContext()

    def lookup(self, key, req_context):
        gid = get_offload_group_idx(key)
        manager = self.managers.get(gid)
        if manager is None:
            return LookupResult.MISS
        return manager.lookup(key, self._context(req_context, gid))

    def prepare_load(self, keys, req_context):
        keys = list(keys)
        grouped = self._group_keys(keys)
        prepared, locations = [], {}
        try:
            for gid, group_keys in grouped.items():
                child = self._context(req_context, gid)
                spec = self.managers[gid].prepare_load(group_keys, child)
                prepared.append((gid, group_keys, child))
                locations.update(zip(group_keys, spec.block_ids))
        except Exception:
            for gid, group_keys, child in prepared:
                self.managers[gid].complete_load(group_keys, child)
            raise
        return CPULoadStoreSpec([locations[k] for k in keys])

    def complete_load(self, keys, req_context):
        for gid, group_keys in self._group_keys(keys).items():
            self.managers[gid].complete_load(
                group_keys, self._context(req_context, gid)
            )

    def touch(self, keys, req_context):
        for gid, group_keys in self._group_keys(keys).items():
            self.managers[gid].touch(group_keys, self._context(req_context, gid))

    def _rollback_stores(self, prepared):
        for gid, output, child in prepared:
            self.managers[gid].complete_store(
                output.keys_to_store, child, success=False
            )

    def prepare_store(self, keys, req_context):
        keys = list(keys)
        grouped = self._group_keys(keys)
        prepared, locations, evicted = [], {}, []
        try:
            for gid, group_keys in grouped.items():
                child = self._context(req_context, gid)
                output = self.managers[gid].prepare_store(group_keys, child)
                if output is None:
                    self._rollback_stores(prepared)
                    return None
                prepared.append((gid, output, child))
                locations.update(zip(output.keys_to_store, output.store_spec.block_ids))
                evicted.extend(output.evicted_keys)
        except Exception:
            self._rollback_stores(prepared)
            raise
        stored = [k for k in keys if k in locations]
        return PrepareStoreOutput(
            stored, CPULoadStoreSpec([locations[k] for k in stored]), evicted
        )

    def complete_store(self, keys, req_context, success=True):
        for gid, group_keys in self._group_keys(keys).items():
            self.managers[gid].complete_store(
                group_keys, self._context(req_context, gid), success=success
            )

    def on_request_finished(self, req_context):
        state = req_context.get_state(_GroupRequestState)
        if state is not None and state.owner is self:
            for gid, child in state.contexts.items():
                self.managers[gid].on_request_finished(child)

    def take_events(self):
        for manager in self.managers.values():
            yield from manager.take_events()

    def reset_cache(self):
        for manager in self.managers.values():
            manager.reset_cache()

    def get_stats(self):
        stats = OffloadingConnectorStats()
        gauges = [
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC,
            CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC,
            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC,
        ]
        weighted = dict.fromkeys(gauges, 0.0)
        for gid, manager in self.managers.items():
            child_stats = manager.get_stats()
            values = child_stats.reduce()
            allocation = self.allocations[gid]
            weight = allocation.num_chunks * allocation.bytes_per_chunk
            for gauge in gauges:
                weighted[gauge] += values.get(gauge, 0.0) * weight
            stats.aggregate(child_stats)
        for gauge, value in weighted.items():
            stats.set_gauge(gauge, value / self.allocated_bytes)
        return stats
