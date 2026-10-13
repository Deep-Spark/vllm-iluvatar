# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the PP+spec schedule bonus-slot / negative-budget guards."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_iluvatar.ilu_patch.general_patches import patch_pp_spec_schedule as patch


def test_patched_schedule_source_widens_idle_check():
    source = """\
def schedule(self, throttle_prefills=False):
    while True:
        if num_new_tokens == 0:
            continue
        if num_new_tokens == 0:
            break
"""
    patched = patch._patched_schedule_source(source)
    assert "if num_new_tokens == 0:" not in patched
    assert patched.count("if num_new_tokens <= 0:") == 2
    assert "num_new_tokens += self.num_sampled_tokens_per_step" not in patched


def test_equal_computed_with_inflight_drops_drafts():
    """In-flight sample already covers this position; do not rewind."""
    req = SimpleNamespace(
        num_tokens=10,
        num_output_placeholders=0,
        num_computed_tokens=10,
        num_in_flight_tokens=6,
        spec_token_ids=[1, 2, 3, 4, 5],
        is_finished=lambda: False,
    )
    scheduler = SimpleNamespace(requests={"r0": req})
    drafts = SimpleNamespace(req_ids=["r0"])
    patch._align_computed_for_drafts(scheduler, drafts)
    assert req.num_computed_tokens == 10
    assert req.spec_token_ids == []


def test_equal_computed_rolls_back_one_for_bonus_row():
    """Idle computed == tokens: restore tokens-1 so schedule yields drafts+bonus."""
    req = SimpleNamespace(
        num_tokens=10,
        num_output_placeholders=0,
        num_computed_tokens=10,
        num_in_flight_tokens=0,
        spec_token_ids=[1, 2, 3, 4, 5],
        is_finished=lambda: False,
    )
    scheduler = SimpleNamespace(requests={"r0": req})
    drafts = SimpleNamespace(req_ids=["r0"])
    patch._align_computed_for_drafts(scheduler, drafts)
    assert req.num_computed_tokens == 9
    assert req.spec_token_ids == [1, 2, 3, 4, 5]


def test_drop_drafts_when_pp_ahead():
    req = SimpleNamespace(
        num_tokens=10,
        num_output_placeholders=0,
        num_computed_tokens=15,
        num_in_flight_tokens=0,
        spec_token_ids=[1, 2, 3, 4, 5],
        is_finished=lambda: False,
    )
    scheduler = SimpleNamespace(requests={"r0": req})
    drafts = SimpleNamespace(req_ids=["r0"])
    patch._align_computed_for_drafts(scheduler, drafts)
    assert req.spec_token_ids == []
    assert req.num_computed_tokens == 15


def test_behind_computed_unchanged():
    req = SimpleNamespace(
        num_tokens=10,
        num_output_placeholders=0,
        num_computed_tokens=9,
        num_in_flight_tokens=0,
        spec_token_ids=[1, 2, 3, 4, 5],
        is_finished=lambda: False,
    )
    scheduler = SimpleNamespace(requests={"r0": req})
    drafts = SimpleNamespace(req_ids=["r0"])
    patch._align_computed_for_drafts(scheduler, drafts)
    assert req.num_computed_tokens == 9
    assert req.spec_token_ids == [1, 2, 3, 4, 5]


def test_schedule_patch_installs_on_community_scheduler():
    from vllm.v1.core.sched.scheduler import Scheduler

    patch.register()
    patch._apply()

    assert getattr(Scheduler.schedule, "_ilu_pp_spec_schedule_patched", False)
    source = Scheduler.schedule._ilu_pp_spec_schedule_source
    assert "if num_new_tokens <= 0:" in source
    assert "if num_new_tokens == 0:" not in source
    assert "num_new_tokens += self.num_sampled_tokens_per_step" not in source
    assert "_ilu_align_prefix_replay_ext_tokens" in source
    assert "ext_tokens -= ext_tokens % self.block_size" not in source
    assert getattr(
        Scheduler.update_draft_token_ids, "_ilu_pp_spec_draft_patched", False
    )
