# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar DP EngineCore wrappers."""

from __future__ import annotations

import inspect
import queue
from collections import deque
from types import SimpleNamespace

import pytest
import torch.distributed as dist
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import DPEngineCoreProc, EngineCore
from vllm.v1.request import RequestStatus

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
        has_unfinished_requests=lambda: bool(scheduler.waiting or scheduler.running),
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


def test_local_prefill_state_includes_ready_remote_replay():
    request = SimpleNamespace(
        request_id="replay",
        status=RequestStatus.WAITING_FOR_REMOTE_KVS,
        replay_start=0,
    )
    scheduler = SimpleNamespace(
        waiting=[],
        skipped_waiting=[request],
        running=[],
        finished_recving_kv_req_ids={"replay"},
    )
    assert core._local_prefill_state(scheduler, sync_ready_replay=True) == (
        True,
        False,
    )

    scheduler.finished_recving_kv_req_ids.clear()
    assert core._local_prefill_state(scheduler, sync_ready_replay=True) == (
        False,
        False,
    )

    request.status = RequestStatus.WAITING
    request.replay_start = 13_372
    assert core._local_prefill_state(scheduler, sync_ready_replay=True) == (
        True,
        False,
    )


def test_local_prefill_state_uses_community_waiting_queues():
    request = SimpleNamespace(
        request_id="replay",
        status=RequestStatus.WAITING_FOR_REMOTE_KVS,
        replay_start=0,
    )
    scheduler = SimpleNamespace(
        waiting=[],
        kv_holding_waiting=[request],
        running=[],
        finished_recving_kv_req_ids={"replay"},
    )
    assert core._local_prefill_state(scheduler, sync_ready_replay=True) == (
        True,
        False,
    )
    scheduler.finished_recving_kv_req_ids.clear()
    assert core._local_prefill_state(scheduler, sync_ready_replay=True) == (
        False,
        False,
    )


def test_prefill_sync_state_is_initialized_once_from_env(monkeypatch):
    import vllm.envs as envs

    engine = _engine(initialized=False)
    cloned_group = object()
    seen_tensors = []

    monkeypatch.setattr(envs, "VLLM_ILU_DP_ENABLE_SYNC_PREFILLS", True)
    monkeypatch.setattr(envs, "VLLM_ILU_DP_SYNC_READY_REPLAY", True)
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
    assert prefill_sync_state.sync_ready_replay is True
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


class _PPArrivals:
    def __init__(self, *items):
        self.items = deque(items)
        self.timeouts = []

    def get(self, timeout):
        self.timeouts.append(timeout)
        if not self.items:
            raise queue.Empty
        return self.items.popleft()


def _pp_request(name):
    return SimpleNamespace(
        request_id=name,
        num_tokens=85909,
        num_prompt_tokens=85909,
        num_computed_tokens=79040,
        resumable=False,
    )


def _pp_engine():
    engine = SimpleNamespace(
        batch_queue=deque(),
        batch_queue_size=8,
        is_pooling_model=False,
        scheduler=SimpleNamespace(
            running=[_pp_request("a")],
            waiting=[],
            max_num_scheduled_tokens=2048,
        ),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1),
            scheduler_config=SimpleNamespace(async_scheduling=False),
            speculative_config=None,
            kv_transfer_config=None,
        ),
        input_queue=_PPArrivals(
            ("add", _pp_request("b")),
            ("add", _pp_request("c")),
        ),
        scheduled=[],
        _ilu_pp_initial_fill=True,
        _ilu_pp_coalesce_remaining_s=0.2,
    )
    engine._handle_client_request = lambda kind, request: (
        engine.scheduler.running.append(request)
    )
    engine._process_input_queue = lambda: None
    return engine


def _pp_original_step(engine):
    left = 2048
    for request in engine.scheduler.running:
        take = min(left, max(0, request.num_tokens - request.num_computed_tokens))
        request.num_computed_tokens += take
        left -= take
    engine.scheduled.append(2048 - left)
    engine.batch_queue.appendleft(2048 - left)
    if len(engine.batch_queue) == 8:
        engine.batch_queue.pop()
        return {}, True
    return None, left < 2048


def test_pp_community_method_fingerprint_matches():
    assert (
        general_patch.source_sha(
            inspect.getsource(inspect.unwrap(EngineCore.step_with_batch_queue))
        )
        == patch_core.PP_BATCH_STEP_SOURCE_SHA
    )


def test_pp_arrivals_fill_partial_batches_before_first_result():
    engine = _pp_engine()
    wrapped = core._make_pp_prefill_coalesce(_pp_original_step, 0.2)
    for _ in range(8):
        wrapped(engine)
    assert engine.scheduled == [2048] * 8
    assert len(engine.input_queue.timeouts) == 2
    assert engine._ilu_pp_initial_fill is False


def test_pp_disabled_preserves_short_and_empty_batches():
    engine = _pp_engine()
    wrapped = core._make_pp_prefill_coalesce(_pp_original_step, 0)
    for _ in range(8):
        wrapped(engine)
    assert engine.scheduled == [2048, 2048, 2048, 725, 0, 0, 0, 0]
    assert engine.input_queue.timeouts == []


def test_pp_new_idle_wave_resets_budget_and_prior_state():
    engine = _pp_engine()
    engine._ilu_pp_initial_fill = False
    engine._ilu_pp_coalesce_remaining_s = 0
    engine._ilu_pp_coalesce_last = (("a", 85184),)
    wrapped = core._make_pp_prefill_coalesce(_pp_original_step, 0.2)
    for _ in range(8):
        wrapped(engine)
    assert engine.scheduled == [2048] * 8
    assert len(engine.input_queue.timeouts) == 2


def test_pp_wait_budget_is_shared_across_gaps(monkeypatch):
    engine = _pp_engine()
    engine.batch_queue.extend([1, 1])
    engine.scheduler.running[0].num_computed_tokens = 85909
    wrapped = core._make_pp_prefill_coalesce(lambda _: (None, False), 0.2)
    times = iter([0, 0.15, 0.16, 0.22])
    monkeypatch.setattr(core.time, "monotonic", lambda: next(times))
    wrapped(engine)
    engine.scheduler.running[-1].num_computed_tokens = 85909
    wrapped(engine)
    assert engine.input_queue.timeouts == pytest.approx([0.2, 0.05])
    assert engine._ilu_pp_coalesce_remaining_s == 0
    engine.scheduler.running[-1].num_computed_tokens = 85909
    wrapped(engine)
    assert len(engine.input_queue.timeouts) == 2


def test_pp_unchanged_state_waits_only_once():
    engine = _pp_engine()
    engine.input_queue = _PPArrivals()
    engine.batch_queue.extend([1, 1])
    engine.scheduler.running[0].num_computed_tokens = 85909
    wrapped = core._make_pp_prefill_coalesce(lambda _: (None, False), 0.2)
    wrapped(engine)
    wrapped(engine)
    assert len(engine.input_queue.timeouts) == 1


@pytest.mark.parametrize(
    "skip",
    [
        "decode",
        "after_result",
        "dp",
        "async",
        "kv",
        "spec",
        "pool",
        "resumable",
        "in_process",
    ],
)
def test_pp_other_execution_paths_delegate_without_wait(skip):
    engine = _pp_engine()
    engine.batch_queue.extend([1, 1])
    engine.scheduler.running[0].num_computed_tokens = 85909
    if skip == "decode":
        engine.scheduler.running[0].num_tokens += 1
    elif skip == "after_result":
        engine._ilu_pp_initial_fill = False
    elif skip == "dp":
        engine.vllm_config.parallel_config.data_parallel_size = 2
    elif skip == "async":
        engine.vllm_config.scheduler_config.async_scheduling = True
    elif skip == "kv":
        engine.vllm_config.kv_transfer_config = object()
    elif skip == "spec":
        engine.vllm_config.speculative_config = object()
    elif skip == "pool":
        engine.is_pooling_model = True
    elif skip == "resumable":
        engine.scheduler.running[0].resumable = True
    else:
        del engine.input_queue
    expected = (None, False)
    assert core._make_pp_prefill_coalesce(lambda _: expected, 0.2)(engine) is expected
    if hasattr(engine, "input_queue"):
        assert engine.input_queue.timeouts == []
