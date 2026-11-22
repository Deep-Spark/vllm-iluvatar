# SPDX-License-Identifier: Apache-2.0
"""Use the product NIXL sender methods and Multi handoff without CUDA."""
from __future__ import annotations

import ast
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import importlib.util
from pathlib import Path
import queue
import sys
import time
from types import MethodType, ModuleType, SimpleNamespace

import pytest


ROOT = (Path(__file__).resolve().parents[2]
        / "vllm_iluvatar/distributed/kv_transfer/kv_connector/v1")


@dataclass
class Results:
    finished_sending: set = field(default_factory=set)
    finished_recving: set = field(default_factory=set)
    failed_recving: set = field(default_factory=set)
    invalid_block_ids: set = field(default_factory=set)
    kv_connector_worker_meta: object = None


@dataclass
class WorkerMetadata:
    metadata: tuple


class BaseMulti:
    def __init__(self, connectors):
        self._connectors = connectors
        self._extra_async_saves = {}
        self._vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=False),
            speculative_config=None,
        )

    def on_new_request(self, request):
        pass

    def build_connector_meta(self, step):
        return SimpleNamespace(extra_async_saves=step)

    def bind_connector_metadata(self, metadata):
        self._extra_async_saves.update(metadata.extra_async_saves)

    def _aggregate_request_finished(self, request, finish):
        return finish(request)

    def get_transfer_results(self, ids):
        return Results()

    def build_connector_worker_meta(self):
        return None

    update_connector_output = get_transfer_results


class OptNixlConnector:
    def __init__(self, native):
        self.connector_scheduler = native
        self.connector_worker = native

    def get_transfer_results(self, ids):
        return self.connector_worker.get_transfer_results()

    def build_connector_worker_meta(self):
        return None


class LMCacheMPConnector:
    _can_store = True
    _report_request_errors = True
    lazy_offload = False
    _vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={})
    )

    def __init__(self, role):
        self.reports = set()
        self.enabled = []
        self.scheduler_adapter = object() if role == "scheduler" else None
        self.worker_adapter = SimpleNamespace(
            lazy_offload=False,
            _ilu_enable_terminal_transfer_fence=self._enable,
        )

    def _enable(self, *, generation_aware):
        self.enabled.append(generation_aware)
        return generation_aware

    _ilu_enable_terminal_session_fence = _enable

    def build_connector_worker_meta(self):
        return None

    def get_transfer_results(self, _):
        reports, self.reports = self.reports, set()
        return Results(reports)


def load_actual_methods(file, class_name, names, namespace):
    parsed = ast.parse(file.read_text(encoding="utf-8"))
    scope = dict(namespace)
    cls = next(x for x in parsed.body
               if isinstance(x, ast.ClassDef) and x.name == class_name)
    if names is None:
        selected = [cls]
    else:
        selected = [x for x in cls.body
                    if isinstance(x, ast.FunctionDef) and x.name in names]
    future = ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[future, *selected], type_ignores=[])), str(file), "exec"), scope)
    return scope


@pytest.fixture
def mesh(monkeypatch):
    for name, attrs in (
        ("vllm.distributed.kv_transfer.kv_connector.v1.base",
         {"KVConnectorTransferResults": Results}),
        ("vllm.distributed.kv_transfer.kv_connector.v1.multi_connector",
         {"MultiKVConnectorWorkerMetadata": WorkerMetadata}),
    ):
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    name = "actual_vp_generation_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "multi_completion.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    lifecycle = ROOT / "opt_nixl/lifecycle.py"
    definitions = {"dataclass": dataclass, "field": field, "time": time,
                   "OrderedDict": OrderedDict, "Enum": Enum, "__name__": __name__}
    for name in ("RequestOwnershipState", "RequestOwnershipTombstone",
                 "RequestTombstoneStore", "CompletionBatch"):
        definitions[name] = load_actual_methods(
            lifecycle, name, None, definitions,
        )[name]
    batch = definitions["CompletionBatch"]
    errors = []
    names = ("_send_state_for_generation", "_finish_send", "_put_completion",
             "_apply_start_load", "_cancel_send_request", "get_transfer_results",
             "_ilu_replay_unbound_send_completions", "_ilu_filter_send_completions")
    scope = load_actual_methods(
        ROOT / "opt_nixl/worker.py", "NixlConnectorWorker", names,
        {**definitions, "queue": queue,
         "logger": SimpleNamespace(warning=lambda *a: None,
                                   error=lambda *a: errors.append(a)),
         "KVConnectorTransferResults": Results, "SendRequestState": SimpleNamespace},
    )
    native = type("ActualNativeSender", (), {name: scope[name] for name in names})()
    native.is_consumer = False
    native._completion_outbox = queue.SimpleQueue()
    native._send_requests = {}
    native._send_transfers = {}
    native._send_failures = {}
    native._invalid_block_ids = set()
    native.remote_sessions = {}
    native.enable_permute_local_kv = False
    native.use_quant_to_fp8 = False
    native._accepting_work = True
    native._send_tombstones = definitions["RequestTombstoneStore"](60, 16)

    worker_type = type("WrappedMulti", (BaseMulti,), {})
    module.install_multi_completion_sources(worker_type)
    monkeypatch.setenv("LMCACHE_ILUVATAR_TERMINAL_TRANSFERS", "1")
    monkeypatch.setenv("LMCACHE_MP_TRANSFER_MODE", "lmcache_driven")
    monkeypatch.setenv("VLLM_ILUVATAR_TERMINAL_KV_OWNERSHIP", "1")
    native_scheduler = SimpleNamespace(
        is_consumer=False,
        _request_generation=lambda rid, params: params["transfer_id"],
    )
    lm_worker = LMCacheMPConnector("worker")
    lm_scheduler = LMCacheMPConnector("scheduler")
    worker = worker_type([OptNixlConnector(native), lm_worker])
    scheduler = worker_type([OptNixlConnector(native_scheduler), lm_scheduler])
    assert lm_scheduler.enabled == [True]
    return SimpleNamespace(worker=worker, scheduler=scheduler, lm=lm_worker,
                           native=native, worker_type=worker_type, batch=batch,
                           errors=errors, ownership=definitions["RequestOwnershipState"])


def request(rid, generation):
    return SimpleNamespace(
        request_id=rid,
        kv_transfer_params=({"do_remote_decode": True, "transfer_id": generation}
                            if generation is not None else None),
    )


def bind(mesh, extra=None):
    metadata = mesh.scheduler.build_connector_meta(extra or {})
    mesh.worker.bind_connector_metadata(metadata)
    return metadata


def poll(mesh):
    result = mesh.worker.get_transfer_results(set())
    mesh.worker.build_connector_worker_meta()
    return result.finished_sending


def native_done(mesh, rid, generation):
    mesh.native._send_requests[rid] = SimpleNamespace(
        generation=generation, metadata=None, started=True,
    )
    assert mesh.native._finish_send(rid)


def test_current_nixl_vote_survives_early_completion_and_rebinding(mesh):
    mesh.scheduler.on_new_request(request("R", "actual"))
    admission = bind(mesh)
    native_done(mesh, "R", "actual")
    assert not poll(mesh)
    mesh.worker.bind_connector_metadata(admission)
    assert mesh.worker._ilu_multi_worker_send_sources == {"R": {0}}
    ending = bind(mesh, {"R": 1})
    mesh.lm.reports = {"R"}
    assert poll(mesh) == {"R"}
    mesh.worker.bind_connector_metadata(ending)
    mesh.native._send_state_for_generation("R", "actual")
    mesh.lm.reports = {"R"}
    assert not poll(mesh)
    assert not mesh.worker._extra_async_saves


def test_stale_tombstone_and_queued_actual_vote_cannot_finish_new_generation(mesh):
    mesh.scheduler.on_new_request(request("R", "old"))
    bind(mesh)
    # Old completion can still be queued at the first new-generation drain.
    native_done(mesh, "R", "old")
    mesh.scheduler.on_new_request(request("R", "new"))
    bind(mesh, {"R": 1})
    assert mesh.native._send_state_for_generation("R", "old") is None
    mesh.lm.reports = {"R"}
    assert not poll(mesh)
    assert mesh.worker._ilu_multi_worker_send_sources == {"R": {1}}
    native_done(mesh, "R", "new")
    assert poll(mesh) == {"R"}


def test_pending_sources_from_an_earlier_generation_are_discarded_on_admission(mesh):
    mesh.scheduler.on_new_request(request("R", "old"))
    bind(mesh)
    native_done(mesh, "R", "old")
    assert not poll(mesh)
    mesh.scheduler.on_new_request(request("R", "new"))
    bind(mesh, {"R": 1})
    mesh.lm.reports = {"R"}
    assert not poll(mesh)
    native_done(mesh, "R", "new")
    assert poll(mesh) == {"R"}


def test_current_admission_without_nixl_ignores_old_native_retry(mesh):
    mesh.scheduler.on_new_request(request("R", None))
    bind(mesh)
    mesh.native._send_tombstones.record("R", "old", mesh.ownership.CANCELED)
    assert mesh.native._send_state_for_generation("R", "old") is None
    mesh.lm.reports = {"R"}
    assert poll(mesh) == {"R"}
    assert not mesh.worker._ilu_multi_worker_send_sources


def test_native_off_path_does_not_require_a_generation_or_lm_hook(mesh, monkeypatch):
    monkeypatch.setenv("LMCACHE_ILUVATAR_TERMINAL_TRANSFERS", "0")
    del mesh.native._ilu_multi_send_generations
    off = mesh.worker_type([OptNixlConnector(mesh.native)])
    assert not hasattr(off, "_ilu_terminal_fence_profile")
    mesh.native._put_completion(mesh.batch(done_sending={"legacy"}))
    assert mesh.native.get_transfer_results().finished_sending == {"legacy"}


def test_no_async_request_retirement_discards_staged_nixl_state(mesh):
    r = request("R", "one")
    mesh.scheduler.on_new_request(r)
    bind(mesh)
    assert mesh.native._ilu_multi_send_generations == {"R": "one"}
    native_done(mesh, "R", "one")
    assert not poll(mesh)
    assert mesh.worker._ilu_multi_worker_send_sources
    mesh.scheduler._aggregate_request_finished(r, lambda _: (False, None))
    bind(mesh)
    assert not mesh.worker._ilu_multi_worker_send_sources
    assert not mesh.native._ilu_multi_send_generations
    assert mesh.native._send_state_for_generation("R", "one") is None
    assert not poll(mesh)


def test_completed_source_history_is_retired_after_each_authoritative_step(mesh):
    for ix in range(128):
        rid = f"R{ix}"
        r = request(rid, None)
        mesh.scheduler.on_new_request(r)
        bind(mesh)
        mesh.lm.reports = {rid}
        assert poll(mesh) == {rid}
        # This fixture injects the parent’s retirement decision. The worker
        # must discard it only when the following scheduler metadata arrives.
        mesh.scheduler._ilu_nixl_retirements.add(rid)
        bind(mesh)
    assert not mesh.native._ilu_multi_send_generations
    assert not mesh.worker._ilu_multi_terminal_completed


def test_untagged_native_votes_cannot_satisfy_generation_safe_barrier(mesh):
    mesh.scheduler.on_new_request(request("R", "new"))
    bind(mesh, {"R": 1})
    mesh.native._put_completion(mesh.batch(done_sending={"R"}))
    mesh.lm.reports = {"R"}
    assert not poll(mesh)
    native_done(mesh, "R", "new")
    assert poll(mesh) == {"R"}


def early_cancel(mesh, rid, generation):
    mesh.native._send_requests[rid] = SimpleNamespace(
        generation=generation, metadata=None, started=False, ready_layers=set(),
    )
    mesh.native._cancel_send_request(rid, generation, "peer canceled")
    assert mesh.native._send_tombstones.get(rid, generation) == mesh.ownership.CANCELED


def test_preadmission_cancel_reaches_real_nixl_scheduler_without_final_send(mesh):
    early_cancel(mesh, "R", "wire")
    assert not poll(mesh)
    # Retry before admission retains one exact proof.
    mesh.native._cancel_send_request("R", "wire", "peer retried")
    assert not poll(mesh)
    assert len(mesh.native._ilu_unbound_send_completions) == 1

    req = request("R", "wire")
    mesh.scheduler.on_new_request(req)
    bind(mesh)
    assert not len(mesh.native._ilu_unbound_send_completions)

    # The producer completed a partial chunk and an LM store, then canceled
    # before publishing final NIXL send metadata. Only LM gates local free.
    mesh.lm.reports = {"R"}
    result = mesh.worker.get_transfer_results({"R"})
    result.kv_connector_worker_meta = mesh.worker.build_connector_worker_meta()
    assert result.kv_connector_worker_meta.send_sources == {"R": {0, 1}}

    replies = []
    update = load_actual_methods(
        ROOT / "opt_nixl/scheduler.py", "NixlConnectorScheduler",
        {"update_connector_output"},
        {"suppress": suppress, "RequestOwnershipState": mesh.ownership,
         "OWNERSHIP_RESULT_TERMINAL": "terminal",
         "logger": SimpleNamespace(warning=lambda *a: None)},
    )["update_connector_output"]
    nx, lm = mesh.scheduler._connectors
    nx_state = nx.connector_scheduler
    nx_state.prefix_state = None
    nx_state._scheduler_failed_recvs = set()
    nx_state._live_recv_reqs = set()
    nx_state._ownership_generations = {"R": "wire"}
    nx_state._ownership_states = {}
    nx_state._pending_worker_sends = set()
    nx_state._sends_awaiting_release = set()
    nx_state._ownership_terminal_events = SimpleNamespace(put_nowait=replies.append)
    nx_state._async_loop = SimpleNamespace(
        call_soon_threadsafe=lambda f, arg: f(arg),
    )
    nx.update_connector_output = MethodType(update, nx_state)
    lm.update_connector_output = lambda output: None
    mesh.scheduler.update_connector_output(result)
    assert replies == [("R", "wire", "terminal")]
    assert result.finished_sending == {"R"}
    # Original local ownership remains unchanged because nixl had no final send.
    assert not nx_state._pending_worker_sends
    mesh.native._cancel_send_request("R", "wire", "peer retried")
    assert not poll(mesh)
    assert not len(mesh.native._ilu_unbound_send_completions)


def test_unbound_wrong_generation_and_untagged_receipts_do_not_count(mesh):
    early_cancel(mesh, "R", "prior")
    mesh.native._put_completion(mesh.batch(done_sending={"R"}))
    assert not poll(mesh)
    mesh.scheduler.on_new_request(request("R", "wire"))
    bind(mesh, {"R": 1})
    mesh.lm.reports = {"R"}
    assert not poll(mesh)
    assert mesh.worker._ilu_multi_worker_send_sources == {"R": {1}}
    native_done(mesh, "R", "wire")
    assert poll(mesh) == {"R"}


def test_preadmission_replay_and_real_handoff_count_the_same_vote_once(mesh):
    early_cancel(mesh, "R", "wire")
    assert not poll(mesh)
    mesh.scheduler.on_new_request(request("R", "wire"))
    bind(mesh, {"R": 1})
    # Native generation replay via actual device-side send metadata is still
    # available. It must deduplicate with the previously buffered receipt.
    mesh.native._apply_start_load(
        {}, {"R": SimpleNamespace(generation="wire")}, {}, set(), {}, {},
    )
    assert not poll(mesh)
    assert mesh.worker._ilu_multi_worker_send_sources == {"R": {0}}
    mesh.lm.reports = {"R"}
    assert poll(mesh) == {"R"}
    assert not poll(mesh)


def test_unbound_receipts_are_bounded_and_overflow_withholds_credit(mesh):
    mesh.native._send_tombstones.limit = 1
    early_cancel(mesh, "R", "first")
    assert not poll(mesh)
    # Already queued true native receipts can arrive in the same drain even if
    # the engine-side bound is occupied. Keep prior proofs and log the pressure.
    mesh.native._put_completion(mesh.batch(
        done_sending={"S"}, send_generations={("S", "overflow")},
    ))
    assert not poll(mesh)
    assert len(mesh.native._ilu_unbound_send_completions) == 1
    assert len(mesh.errors) == 1
    mesh.scheduler.on_new_request(request("S", "overflow"))
    bind(mesh, {"S": 1})
    mesh.lm.reports = {"S"}
    assert not poll(mesh)
    assert mesh.worker._ilu_multi_worker_send_sources == {"S": {1}}
    mesh.scheduler.on_new_request(request("R", "first"))
    bind(mesh, {"R": 1})
    mesh.lm.reports = {"R"}
    assert poll(mesh) == {"R"}
    assert not len(mesh.native._ilu_unbound_send_completions)


def test_native_generation_requires_a_fresh_transfer_identity_on_rid_reuse():
    scope = load_actual_methods(
        ROOT / "opt_nixl/scheduler.py", "NixlConnectorScheduler",
        {"_request_generation"}, {"hashlib": hashlib},
    )
    generation = scope["_request_generation"]
    # A manual PD client that falls back to RID, or supplies the same wire
    # identity twice, cannot ask the receiver to distinguish those votes.
    assert generation("R", {}) == generation("R", {})
    assert generation("R", {"transfer_id": "reused"}) == generation(
        "R", {"transfer_id": "reused"}
    )
    assert generation("R", {"transfer_id": "first"}) != generation(
        "R", {"transfer_id": "second"}
    )
