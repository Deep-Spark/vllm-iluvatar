# SPDX-License-Identifier: Apache-2.0
"""CPU tests for letting in-flight async loads pass a refused async load."""

from __future__ import annotations

import inspect
import textwrap
from types import SimpleNamespace

import pytest
from vllm.v1.request import RequestStatus

from vllm_iluvatar.ilu_patch.general_patches import patch_sched_async_load_hol as patch


def _schedule_source() -> str:
    from vllm.v1.core.sched.scheduler import Scheduler

    current = Scheduler.schedule
    source = getattr(current, "_ilu_pp_spec_schedule_source", None)
    return source or textwrap.dedent(inspect.getsource(current))


def test_patch_schedule_source_applies_every_hunk_once():
    patched = patch.patch_schedule_source(_schedule_source())
    for _old, new in patch._REPLACEMENTS:
        assert patched.count(new) == 1
    compile(patched, "<schedule>", "exec")


def test_patch_schedule_source_is_idempotent():
    patched = patch.patch_schedule_source(_schedule_source())
    assert patch.patch_schedule_source(patched) == patched


def test_patch_schedule_source_fails_on_drift():
    with pytest.raises(RuntimeError, match="re-sync the patch"):
        patch.patch_schedule_source("def schedule(self):\n    return None\n")


def test_has_finished_inflight_load():
    loading = SimpleNamespace(
        request_id="a", status=RequestStatus.WAITING_FOR_REMOTE_KVS
    )
    prefilling = SimpleNamespace(request_id="b", status=RequestStatus.RUNNING)
    scheduler = SimpleNamespace(
        _inflight_prefills=[loading, prefilling], finished_recving_kv_req_ids=set()
    )
    assert not patch.has_finished_inflight_load(scheduler)
    scheduler.finished_recving_kv_req_ids = {"b"}
    assert not patch.has_finished_inflight_load(scheduler)
    scheduler.finished_recving_kv_req_ids = {"a"}
    assert patch.has_finished_inflight_load(scheduler)


def test_note_refused_async_load_is_rate_limited(monkeypatch):
    logged = []
    monkeypatch.setattr(patch.logger, "info", lambda *a: logged.append(a))
    scheduler = SimpleNamespace()
    request = SimpleNamespace(request_id="big")
    patch.note_refused_async_load(scheduler, request, 7)
    patch.note_refused_async_load(scheduler, request, 7)
    assert len(logged) == 1
    assert logged[0][1:] == ("big", 7)


def test_register_respects_disable_env(monkeypatch):
    called = []
    monkeypatch.setenv("VLLM_ILU_SCHED_ASYNC_LOAD_HOL", "0")
    monkeypatch.setattr(patch.p, "register_once", lambda *a, **k: called.append(a))
    patch.register()
    assert called == []
