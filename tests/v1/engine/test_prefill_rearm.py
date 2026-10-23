# SPDX-License-Identifier: Apache-2.0
"""Admission liveness and chunk-continuation contract for periodic coalescing."""

from vllm_iluvatar.v1.engine.core import DPPrefillSyncState


def policy(limit=8, rearm=True):
    return DPPrefillSyncState(
        enabled=True,
        max_replacement_wait_s=0,
        max_defer_steps=limit,
        sync_group=None,
        rearm_after_admission=rearm,
    )


def test_continuous_arrivals_form_bounded_batches():
    state = policy()
    admissions = [step for step in range(1, 46) if not state.update(1, 7, 16)]
    assert admissions == [9, 18, 27, 36, 45]
    assert state.sync_tensor.numel() == 3


def test_disabled_policy_preserves_original_fail_open_behavior():
    state = policy(rearm=False)
    admissions = [step for step in range(1, 20) if not state.update(1, 7, 16)]
    assert admissions == list(range(9, 20))
    assert state.sync_tensor.numel() == 2


def test_running_chunked_prefill_is_never_delayed():
    state = policy()
    for _ in range(200):
        assert not state.update(1, 7, 16, continuing_ranks=1)
    # After the chunked request completes, fresh arrivals coalesce again.
    assert state.update(1, 7, 16)


def test_empty_and_all_ready_reset_admission_window():
    state = policy()
    for _ in range(7):
        assert state.update(1, 7, 16)
    assert not state.update(0, 7, 16)
    for _ in range(7):
        assert state.update(1, 7, 16)
    assert not state.update(16, 7, 16)
    assert state.update(1, 7, 16)


def test_without_active_decode_startup_wait_stays_short():
    state = policy(limit=32)
    assert state.update(1, 0, 16)
    assert state.update(1, 0, 16)
    assert not state.update(1, 0, 16)


def test_community_cadence_does_not_multiply_wait():
    state = policy()
    admissions = []
    for step in range(1, 49):
        community_allows = step % 16 == 0
        forced = state.update(1, 7, 16, admission_allowed=community_allows)
        if not forced and community_allows:
            admissions.append(step)
    assert admissions == [16, 32, 48]


def test_zero_delay_never_blocks():
    state = policy(limit=0)
    assert not any(state.update(1, 7, 16) for _ in range(50))


def test_continuation_vote_reaches_engine_admission(monkeypatch):
    from types import SimpleNamespace

    import torch.distributed as dist

    from vllm_iluvatar.v1.engine import core

    scheduler = SimpleNamespace(
        waiting=[],
        skipped_waiting=[],
        running=[SimpleNamespace(is_prefill_chunk=True)],
        has_unfinished_requests=lambda: True,
    )
    state = policy()
    engine = SimpleNamespace(
        scheduler=scheduler,
        engines_running=True,
        dp_size=16,
        _ilu_dp_prefill_sync_state=state,
    )
    calls = []

    def collective(tensor, **kwargs):
        assert tensor.numel() == 3
        assert tensor[2].item() == 1
        tensor[0], tensor[1], tensor[2] = 1, 7, 1

    monkeypatch.setattr(dist, "all_reduce", collective)
    wrapper = core._make_process_engine_step(
        lambda e: calls.append(e) or True,
        community_should_throttle=lambda e: False,
    )
    assert wrapper(engine)
    assert calls == [engine]
    assert not state.force_throttle
