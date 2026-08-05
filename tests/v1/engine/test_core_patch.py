# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar DP EngineCore wrappers."""

from __future__ import annotations

import inspect
import queue
from types import SimpleNamespace

import torch.distributed as dist
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import DPEngineCoreProc

from vllm_iluvatar.ilu_patch import general_patch
from vllm_iluvatar.ilu_patch.general_patches import patch_core
from vllm_iluvatar.v1.engine import core


def _request(*, is_prefill_chunk: bool):
    return SimpleNamespace(is_prefill_chunk=is_prefill_chunk)


def test_community_method_fingerprints_match():
    assert general_patch.source_sha(
        inspect.getsource(DPEngineCoreProc._should_throttle_prefills)
    ) == general_patch.source_sha(patch_core.SHOULD_THROTTLE_PREFILLS)
    assert general_patch.source_sha(
        inspect.getsource(DPEngineCoreProc._process_engine_step)
    ) == general_patch.source_sha(patch_core.PROCESS_ENGINE_STEP)
    assert general_patch.source_sha(
        inspect.getsource(DPEngineCoreProc._handle_client_request)
    ) == general_patch.source_sha(patch_core.HANDLE_CLIENT_REQUEST)


def _engine(
    *,
    waiting=(),
    running=(),
    sync=True,
    grace_s=0.0,
    max_defer_steps=8,
    initialized=True,
    engines_running=True,
):
    scheduler = SimpleNamespace(
        waiting=list(waiting),
        skipped_waiting=[],
        running=list(running),
        has_unfinished_requests=lambda: bool(
            scheduler.waiting or scheduler.running
        ),
    )
    engine = SimpleNamespace(
        scheduler=scheduler,
        engines_running=engines_running,
        dp_group=object(),
        dp_store=object(),
        dp_size=4,
        input_queue=queue.Queue(),
        _process_input_queue=lambda: None,
        _handle_client_request=lambda *_: None,
    )
    if initialized:
        engine._ilu_dp_prefill_sync_state = core.DPPrefillSyncState(
            enabled=sync,
            max_replacement_wait_s=grace_s,
            max_defer_steps=max_defer_steps,
            sync_group=engine.dp_group if sync else None,
        )
    return engine


def test_local_prefill_state_distinguishes_decode():
    scheduler = SimpleNamespace(
        waiting=[object()],
        skipped_waiting=[],
        running=[_request(is_prefill_chunk=False)],
    )
    assert core._local_prefill_state(scheduler) == (True, True)


def test_local_prefill_state_includes_running_chunks_but_not_blocked_waiting():
    scheduler = SimpleNamespace(
        waiting=[],
        skipped_waiting=[object()],
        running=[_request(is_prefill_chunk=True)],
    )
    assert core._local_prefill_state(scheduler) == (True, False)

    scheduler.running.clear()
    assert core._local_prefill_state(scheduler) == (False, False)


def test_prefill_sync_state_is_initialized_once_from_env(monkeypatch):
    import vllm.envs as envs

    engine = _engine(initialized=False)
    cloned_group = object()
    seen_tensors = []

    monkeypatch.setattr(envs, "VLLM_ILU_DP_ENABLE_SYNC_PREFILLS", True)
    monkeypatch.setattr(
        envs,
        "VLLM_ILU_DP_PREFILL_MAX_DEFER_STEPS",
        3,
    )
    monkeypatch.setattr(envs, "VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS", 7.0)
    monkeypatch.setattr(
        core,
        "_clone_dp_process_group",
        lambda group, store: cloned_group,
    )

    def all_reduce(state, op, group):
        assert group is cloned_group
        seen_tensors.append(state)
        state.zero_()

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    wrapped = core._make_process_engine_step(lambda _self: False)

    assert wrapped(engine) is False
    assert wrapped(engine) is False
    prefill_sync_state = engine._ilu_dp_prefill_sync_state
    assert prefill_sync_state.max_defer_steps == 3
    assert prefill_sync_state.max_replacement_wait_s == 0.007
    assert prefill_sync_state.sync_group is cloned_group
    assert seen_tensors[0] is seen_tensors[1]


def test_partial_prefill_is_deferred_while_peers_decode(monkeypatch):
    engine = _engine(waiting=[object()], engines_running=False)
    called = False

    def orig(_self):
        nonlocal called
        called = True
        return True

    def all_reduce(state, op, group):
        state[0] = 1
        state[1] = 3

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    wrapped = core._make_process_engine_step(orig)

    assert wrapped(engine) is False
    assert called is False
    assert engine.engines_running is True
    assert engine._ilu_dp_prefill_sync_state.force_throttle is True


def test_running_prefill_chunk_is_deferred_while_peers_decode(monkeypatch):
    engine = _engine(running=[_request(is_prefill_chunk=True)])

    def all_reduce(state, op, group):
        state[0] = 1
        state[1] = 3

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    wrapped = core._make_process_engine_step(lambda _self: True)

    assert wrapped(engine) is False
    assert engine._ilu_dp_prefill_sync_state.force_throttle is True


def test_all_ready_ranks_release_prefill_together(monkeypatch):
    engine = _engine(waiting=[object()])

    def all_reduce(state, op, group):
        state[0] = 4
        state[1] = 0

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    wrapped = core._make_process_engine_step(lambda _self: True)

    assert wrapped(engine) is True
    assert engine._ilu_dp_prefill_sync_state.force_throttle is False


def test_partial_prefill_is_released_after_bounded_decode_wait(monkeypatch):
    engine = _engine(waiting=[object()], max_defer_steps=2)
    calls = 0

    def orig(_self):
        nonlocal calls
        calls += 1
        return True

    def all_reduce(state, op, group):
        state[0] = 1
        state[1] = 3

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    wrapped = core._make_process_engine_step(orig)

    assert wrapped(engine) is False
    assert wrapped(engine) is False
    assert wrapped(engine) is True
    assert calls == 1
    assert engine._ilu_dp_prefill_sync_state.force_throttle is False


def test_throttle_wrapper_preserves_upstream_decision():
    wrapped = core._make_should_throttle_prefills(lambda _self: True)
    engine = _engine(initialized=True)
    assert wrapped(engine) is True

    engine._ilu_dp_prefill_sync_state.force_throttle = True
    assert wrapped(engine) is True


def test_idle_engine_skips_prefill_collective(monkeypatch):
    engine = _engine(engines_running=False)

    def unexpected_all_reduce(*args, **kwargs):
        raise AssertionError("idle rank entered prefill collective")

    monkeypatch.setattr(dist, "all_reduce", unexpected_all_reduce)
    wrapped = core._make_process_engine_step(lambda _self: False)

    assert wrapped(engine) is False
    assert engine.engines_running is False


def test_sync_prefill_start_wave_does_not_exclude_target():
    captured = []

    def orig(_self, request_type, request):
        captured.append((request_type, request))

    engine = _engine()
    wrapped = core._make_handle_client_request(orig)
    wrapped(
        engine,
        EngineCoreRequestType.START_DP_WAVE,
        (7, 2),
    )

    assert captured == [(EngineCoreRequestType.START_DP_WAVE, (7, None))]


def test_disabled_sync_preserves_start_wave_exclusion():
    captured = []

    def orig(_self, request_type, request):
        captured.append((request_type, request))

    engine = _engine(sync=False)
    wrapped = core._make_handle_client_request(orig)
    wrapped(
        engine,
        EngineCoreRequestType.START_DP_WAVE,
        (7, 2),
    )

    assert captured == [(EngineCoreRequestType.START_DP_WAVE, (7, 2))]
