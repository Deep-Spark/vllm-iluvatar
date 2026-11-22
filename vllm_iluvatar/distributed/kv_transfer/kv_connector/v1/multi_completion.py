# SPDX-License-Identifier: Apache-2.0
"""Keep distinct Multi completion sources through worker/rank barriers.

The original counters consume each child and worker once per pending barrier.
Only completion sets and output shells are copied; device events, futures,
stats and child metadata are not deep copied.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from functools import wraps
import os
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiKVConnectorWorkerMetadata,
)

_MARK = "_ilu_multi_completion_sources"
_DISTINCT_CHILDREN = "_ilu_multi_distinct_child_sends"
_DISTINCT_WORKERS = "_ilu_multi_distinct_worker_sends"
_MP_STORE_BARRIER = "_ilu_multi_mp_store_terminal_completion"


def _merge(target: dict[str, set[int]], source: dict[str, set[int]]) -> None:
    for request_id, children in source.items():
        target.setdefault(request_id, set()).update(children)


def _snapshot(source: dict[str, set[int]]) -> dict[str, set[int]]:
    return {request_id: set(children) for request_id, children in source.items()}


@dataclass
class MultiCompletionWorkerMetadata(MultiKVConnectorWorkerMetadata):
    """Small source labels, aggregated with the original per-child metadata."""

    send_sources: dict[str, set[int]] = field(default_factory=dict)
    recv_sources: dict[str, set[int]] = field(default_factory=dict)

    def aggregate(self, other: Any) -> MultiCompletionWorkerMetadata:
        if not isinstance(other, MultiCompletionWorkerMetadata):
            raise TypeError("Multi completion metadata versions differ")
        base = super().aggregate(other)
        sends, recvs = _snapshot(self.send_sources), _snapshot(self.recv_sources)
        _merge(sends, other.send_sources)
        _merge(recvs, other.recv_sources)
        return MultiCompletionWorkerMetadata(base.metadata, sends, recvs)


def _state(owner: Any, name: str) -> dict[str, set[int]]:
    value = owner.__dict__.get(name)
    if value is None:
        value = {}
        owner.__dict__[name] = value
    return value


def _copy_set(value: Any) -> set[Any] | None:
    return None if value is None else set(value)


def _early_sends(child: Any, sends: set[str]) -> set[str]:
    # These are the pinned OptNixl scheduler's existing two release halves.
    scheduler = getattr(child, "connector_scheduler", None)
    pending = getattr(scheduler, "_pending_worker_sends", None)
    awaiting = getattr(scheduler, "_sends_awaiting_release", None)
    if isinstance(pending, set) and isinstance(awaiting, set):
        return (sends & pending) - awaiting
    return set()


def _mp_eager_child(connectors: list[Any]) -> tuple[int, Any] | None:
    if len(connectors) != 2:
        return None
    names = [type(child).__name__ for child in connectors]
    if sorted(names) != ["LMCacheMPConnector", "OptNixlConnector"]:
        return None
    index = names.index("LMCacheMPConnector")
    connector = connectors[index]
    if (getattr(connector, "_can_store", False) is not True
            or getattr(connector, "lazy_offload", None) is not False):
        return None
    return index, connector


def _terminal_fence_profile(connector: Any, config: Any) -> bool:
    # Deferred finalization can call get_finished before the last wait_for_save.
    # Keep the initial device fence limited to the explicit synchronous profile.
    # The rank aggregator holds RID failures until all receives complete. A
    # legacy flat block-id failure can reach the scheduler before that fence.
    if (getattr(connector, "_report_request_errors", False) is not True
            or config is None or getattr(config, "speculative_config", None) is not None):
        return False
    scheduler = getattr(config, "scheduler_config", None)
    if getattr(scheduler, "async_scheduling", None) is not False:
        return False
    child_config = getattr(getattr(connector, "_vllm_config", None), "kv_transfer_config", None)
    extra = getattr(child_config, "kv_connector_extra_config", {}) or {}
    mode = extra.get("lmcache.mp.mp_transfer_mode", os.getenv("LMCACHE_MP_TRANSFER_MODE"))
    return (mode == "lmcache_driven"
            and extra.get("lmcache.mp.transfer_intermediate_tensors", False) is False
            and os.getenv("LMCACHE_ILUVATAR_TERMINAL_TRANSFERS") == "1"
            and os.getenv("VLLM_ILUVATAR_TERMINAL_KV_OWNERSHIP", "1") == "1")


def _mp_store_completion_gate(connectors: list[Any]) -> int | None:
    match = _mp_eager_child(connectors)
    if match is None:
        return None
    index, connector = match
    adapter = getattr(connector, "worker_adapter", None)
    enable = getattr(adapter, "_ilu_enable_terminal_transfer_fence", None)
    if getattr(adapter, "lazy_offload", None) is not False:
        raise RuntimeError("Unsupported non-lazy LMCache MP completion contract")
    if enable is None or not enable(generation_aware=True):
        raise RuntimeError("MP worker rejected the configured terminal transfer fence")
    return index


def install_multi_completion_sources(cls: type) -> None:
    """Install three Multi-only adapters; standalone/single-child stay intact."""
    if getattr(cls.update_connector_output, _MARK, False):
        return
    original_results = cls.get_transfer_results
    original_meta = cls.build_connector_worker_meta
    original_update = cls.update_connector_output
    original_init = cls.__init__
    original_new = getattr(cls, "on_new_request", None)
    original_build = getattr(cls, "build_connector_meta", None)
    original_bind = getattr(cls, "bind_connector_metadata", None)
    original_finish = getattr(cls, "_aggregate_request_finished", None)

    @wraps(original_init)
    def init(self: Any, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        match = _mp_eager_child(getattr(self, "_connectors", []))
        if match is None:
            return
        mp_index, child = match
        nixl = self._connectors[1 - mp_index]
        if not _terminal_fence_profile(child, getattr(self, "_vllm_config", None)):
            return
        scheduler_adapter = getattr(child, "scheduler_adapter", None)
        native = (nixl.connector_scheduler if scheduler_adapter is not None
                  else nixl.connector_worker)
        if native.is_consumer:
            return
        if scheduler_adapter is not None:
            enable = getattr(child, "_ilu_enable_terminal_session_fence", None)
            if enable is None:
                raise RuntimeError("Terminal transfer fencing requires the matching LMCache plugin")
            enable(generation_aware=True)
            self._ilu_nixl_admissions = {}
            self._ilu_nixl_retirements = set()
            self._ilu_nixl_generation_sequence = 0
            self._ilu_nixl_generation_authority = native
        elif not callable(getattr(getattr(child, "worker_adapter", None),
                                  "_ilu_enable_terminal_transfer_fence", None)):
            raise RuntimeError("Terminal transfer fencing requires the matching LMCache plugin")
        else:
            native._ilu_multi_send_generations = {}
            self._ilu_nixl_worker = native
            self._ilu_nixl_generation_sequence = -1
        self._ilu_terminal_fence_profile = True

    def on_new_request(self: Any, request: Any) -> Any:
        result = original_new(self, request)
        admissions = getattr(self, "_ilu_nixl_admissions", None)
        if admissions is not None:
            params = request.kv_transfer_params
            generation = None
            if params and params.get("do_remote_decode"):
                generation = self._ilu_nixl_generation_authority._request_generation(
                    request.request_id, params
                )
            # Every admission invalidates any retained receipt for the same ID,
            # including requests that do not use the NIXL data path at all.
            admissions[request.request_id] = generation
        return result

    def build_connector_meta(self: Any, scheduler_output: Any) -> Any:
        result = original_build(self, scheduler_output)
        admissions = getattr(self, "_ilu_nixl_admissions", None)
        if admissions is not None:
            self._ilu_nixl_generation_sequence += 1
            result._ilu_send_generation_delta = (
                self._ilu_nixl_generation_sequence, admissions,
                self._ilu_nixl_retirements
            )
            self._ilu_nixl_admissions = {}
            self._ilu_nixl_retirements = set()
        return result

    def bind_connector_metadata(self: Any, metadata: Any) -> Any:
        worker = getattr(self, "_ilu_nixl_worker", None)
        if worker is not None:
            delta = getattr(metadata, "_ilu_send_generation_delta", None)
            if delta is None:
                raise RuntimeError("Missing Multi send-generation metadata")
            sequence, admissions, retirements = delta
            if sequence < self._ilu_nixl_generation_sequence:
                raise RuntimeError("Out-of-order Multi send-generation metadata")
            if sequence == self._ilu_nixl_generation_sequence:
                return original_bind(self, metadata)
            self._ilu_nixl_generation_sequence = sequence
            expected = worker._ilu_multi_send_generations
            pending = _state(self, "_ilu_multi_worker_send_sources")
            completed = self.__dict__.setdefault("_ilu_multi_terminal_completed", set())
            # Apply invalidation before the original bind installs this step's
            # async-save counts. A retirement and fresh admission can share an
            # output; the new generation is authoritative in that case.
            for request_id in retirements | admissions.keys():
                expected.pop(request_id, None)
                pending.pop(request_id, None)
                self._extra_async_saves.pop(request_id, None)
                completed.discard(request_id)
            expected.update(admissions)
            worker._ilu_replay_unbound_send_completions(admissions)
        return original_bind(self, metadata)

    def aggregate_request_finished(self: Any, request: Any, finish: Any) -> Any:
        result = original_finish(self, request, finish)
        retirements = getattr(self, "_ilu_nixl_retirements", None)
        if retirements is not None and not result[0]:
            retirements.add(request.request_id)
        return result

    @wraps(original_results)
    def get_transfer_results(self: Any, finished_req_ids: set[str]) -> Any:
        if len(self._connectors) <= 1:
            return original_results(self, finished_req_ids)
        ready = self.__dict__.get("_ilu_multi_ready_sources")
        if ready is not None and any(ready):
            raise RuntimeError("Multi completion sources were not packed before the next drain")
        results = KVConnectorTransferResults()
        pending = _state(self, "_ilu_multi_worker_send_sources")
        store_gate = (_mp_store_completion_gate(self._connectors)
                      if getattr(self, "_ilu_terminal_fence_profile", False) else None)
        if store_gate is not None:
            completed = self.__dict__.setdefault("_ilu_multi_terminal_completed", set())
            for request_id in completed.intersection(self._extra_async_saves):
                del self._extra_async_saves[request_id]
        ready_sends: dict[str, set[int]] = {}
        ready_recvs: dict[str, set[int]] = {}
        for index, connector in enumerate(self._connectors):
            child_results = connector.get_transfer_results(finished_req_ids)
            results.finished_recving.update(child_results.finished_recving)
            results.failed_recving.update(child_results.failed_recving)
            for request_id in child_results.finished_recving:
                ready_recvs.setdefault(request_id, set()).add(index)
            for request_id in child_results.finished_sending:
                if store_gate is not None and request_id in completed:
                    # This worker's complete group was already reported. Still
                    # drain child metadata, receive failures and other results;
                    # only this duplicate/stale send credit is suppressed.
                    continue
                # Capture before the original extra-save counter can swallow it.
                sources = pending.setdefault(request_id, set())
                if index in sources:
                    # A terminal retry is still the same child's completion.
                    # extra_async_saves can already be absent while one child
                    # remains outstanding; pending survives until that exit.
                    continue
                sources.add(index)
                extra_pending = self._extra_async_saves.get(request_id)
                if store_gate is not None:
                    if not (extra_pending is None or
                            (type(extra_pending) is int and extra_pending == 1)):
                        raise RuntimeError("Invalid two-child MP async-save count")
                    # Keep the declared cardinality until the whole group is
                    # ready. A Nixl completion that predates terminal metadata
                    # must not give the parent an irreversible rank credit.
                    if (store_gate not in sources
                            or len(sources) < (extra_pending or 0) + 1):
                        continue
                    self._extra_async_saves.pop(request_id, None)
                    completed.add(request_id)
                    results.finished_sending.add(request_id)
                    ready_sends.setdefault(request_id, set()).update(
                        pending.pop(request_id)
                    )
                    continue
                if extra_pending is None:
                    results.finished_sending.add(request_id)
                    ready_sends.setdefault(request_id, set()).update(
                        pending.pop(request_id)
                    )
                else:
                    assert extra_pending > 0
                    if extra_pending == 1:
                        del self._extra_async_saves[request_id]
                    else:
                        self._extra_async_saves[request_id] = extra_pending - 1
        self._ilu_multi_ready_sources = (ready_sends, ready_recvs)
        return results

    @wraps(original_meta)
    def build_connector_worker_meta(self: Any) -> Any:
        base = original_meta(self)
        if len(self._connectors) <= 1:
            return base
        sends, recvs = self.__dict__.pop("_ilu_multi_ready_sources", ({}, {}))
        if base is None and not sends and not recvs:
            return None
        metadata = base.metadata if base is not None else (None,) * len(self._connectors)
        return MultiCompletionWorkerMetadata(metadata, sends, recvs)

    @wraps(original_update)
    def update_connector_output(self: Any, output: Any) -> None:
        if len(self._connectors) <= 1:
            return original_update(self, output)
        meta = output.kv_connector_worker_meta
        if meta is not None and not isinstance(meta, MultiKVConnectorWorkerMetadata):
            raise TypeError("Expected Multi worker metadata")
        child_meta = meta.metadata if meta is not None else (None,) * len(self._connectors)
        if len(child_meta) != len(self._connectors):
            raise ValueError("Multi child metadata length differs")
        pending_sends = _state(self, "_ilu_multi_scheduler_send_sources")
        pending_recvs = _state(self, "_ilu_multi_scheduler_recv_sources")
        if isinstance(meta, MultiCompletionWorkerMetadata):
            # KVOutputAggregator retains its rank counters across calls, but
            # not worker metadata. Keep source labels until its actual exit.
            _merge(pending_sends, meta.send_sources)
            _merge(pending_recvs, meta.recv_sources)
        original_sends = set(output.finished_sending or ())
        original_recvs = set(output.finished_recving or ())
        for request_id in original_sends:
            if not pending_sends.get(request_id):
                raise ValueError("Missing Multi send completion source")
        for request_id in original_recvs:
            if not pending_recvs.get(request_id):
                raise ValueError("Missing Multi receive completion source")
        size = len(self._connectors)
        for pending in (pending_sends, pending_recvs):
            if any(type(index) is not int or not 0 <= index < size
                   for sources in pending.values() for index in sources):
                raise ValueError("Invalid Multi completion child index")

        approved_sends: set[str] = set()
        approved_recvs: set[str] = set()
        original_failed_recvs = set(output.failed_recving or ())
        failed_recvs = original_failed_recvs.copy()
        invalid_blocks = set(output.invalid_block_ids or ())
        early: set[str] = set()
        for index, connector in enumerate(self._connectors):
            sends = {rid for rid in original_sends if index in pending_sends[rid]}
            recvs = {rid for rid in original_recvs if index in pending_recvs[rid]}
            early.update(_early_sends(connector, sends))
            view = copy(output)
            view.finished_sending = sends if output.finished_sending is not None else None
            view.finished_recving = recvs if output.finished_recving is not None else None
            view.failed_recving = original_failed_recvs & recvs
            view.invalid_block_ids = _copy_set(output.invalid_block_ids)
            view.kv_connector_worker_meta = child_meta[index]
            connector.update_connector_output(view)
            approved_sends.update(view.finished_sending or ())
            approved_recvs.update(view.finished_recving or ())
            failed_recvs.update(view.failed_recving or ())
            invalid_blocks.update(view.invalid_block_ids or ())

        # Stale removes only its source contribution. A real current Nixl early
        # completion still vetoes release; a sibling must not add it back.
        output.finished_sending = (approved_sends - early) or None
        retirements = getattr(self, "_ilu_nixl_retirements", None)
        if retirements is not None:
            retirements.update(output.finished_sending or ())
        output.finished_recving = approved_recvs or None
        output.failed_recving = failed_recvs
        output.invalid_block_ids = invalid_blocks
        # Do not swap/mutate the parent's child metadata or copy its resources.
        for request_id in original_sends:
            pending_sends.pop(request_id, None)
        for request_id in original_recvs:
            pending_recvs.pop(request_id, None)

    for method in (get_transfer_results, build_connector_worker_meta, update_connector_output):
        setattr(method, _MARK, True)
    setattr(get_transfer_results, _DISTINCT_CHILDREN, True)
    setattr(get_transfer_results, _MP_STORE_BARRIER, True)
    cls.__init__ = init
    cls.get_transfer_results = get_transfer_results
    cls.build_connector_worker_meta = build_connector_worker_meta
    cls.update_connector_output = update_connector_output
    if original_new is not None:
        cls.on_new_request = wraps(original_new)(on_new_request)
    if original_build is not None:
        cls.build_connector_meta = wraps(original_build)(build_connector_meta)
    if original_bind is not None:
        cls.bind_connector_metadata = wraps(original_bind)(bind_connector_metadata)
    if original_finish is not None:
        cls._aggregate_request_finished = wraps(original_finish)(aggregate_request_finished)


def install_multi_rank_completion_dedup(cls: type) -> None:
    """Count distinct worker positions within an unfinished Multi send barrier.

    The pinned executor returns a fixed-position worker list, including empty
    positions. State belongs to this aggregator instance only. The original
    expected-count updates and send/receive counters remain authoritative.
    """
    if getattr(cls.aggregate, _DISTINCT_WORKERS, False):
        return
    original = cls.aggregate
    state_name = "_ilu_multi_send_worker_positions"

    @wraps(original)
    def aggregate(self: Any, outputs: list[Any], output_rank: int = 0) -> Any:
        # Preserve the original None return/assert paths without recording a
        # worker whose output the original method has not consumed.
        if not outputs[output_rank] or any(output is None for output in outputs):
            return original(self, outputs, output_rank)
        if not any(
            output.kv_connector_output
            and isinstance(
                output.kv_connector_output.kv_connector_worker_meta,
                MultiCompletionWorkerMetadata,
            )
            for output in outputs
        ):
            return original(self, outputs, output_rank)

        seen = _snapshot(self.__dict__.get(state_name, {}))
        filtered = []
        for position, output in enumerate(outputs):
            shell = copy(output)
            kv_output = output.kv_connector_output
            if kv_output is not None:
                view = copy(kv_output)
                shell.kv_connector_output = view
                if isinstance(
                    kv_output.kv_connector_worker_meta,
                    MultiCompletionWorkerMetadata,
                ) and kv_output.finished_sending:
                    distinct = set()
                    for request_id in kv_output.finished_sending:
                        positions = seen.setdefault(request_id, set())
                        if position not in positions:
                            positions.add(position)
                            distinct.add(request_id)
                    view.finished_sending = distinct or None
            filtered.append(shell)

        result = original(self, filtered, output_rank)
        # A new barrier's first output must be remembered too. Only the
        # original counter decides whether that barrier has finished; never
        # replace it with len(outputs) or the latest default expected count.
        self.__dict__[state_name] = {
            request_id: positions
            for request_id, positions in seen.items()
            if request_id in self._send_remaining_count
        }
        return result

    setattr(aggregate, _DISTINCT_WORKERS, True)
    cls.aggregate = aggregate
