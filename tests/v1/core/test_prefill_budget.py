# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for producer chunk sharing and scheduler composition."""

from types import SimpleNamespace as NS

import pytest

from vllm_iluvatar.ilu_patch.general_patches import patch_prefill_budget as patch
from vllm_iluvatar.v1.core.sched import prefill_supply as supply


class Queue(list):
    def peek_request(self):
        return self[0]

    def remove_request(self, request):
        self.remove(request)

    def prepend_request(self, request):
        self.insert(0, request)


def request(rid, total, done=0, status="WAITING", prefix="seen"):
    return NS(
        request_id=rid,
        num_tokens=total,
        num_tokens_with_spec=total,
        num_prompt_tokens=total,
        num_computed_tokens=done,
        num_output_placeholders=0,
        status=NS(name=status),
        block_hashes=[prefix] * 500,
        has_encoder_inputs=False,
    )


def scheduler(running=(), waiting=(), skipped=(), cached=0):
    reads = []

    def get_computed_blocks(req):
        reads.append(req.request_id)
        return None, cached, 0

    result = NS(
        vllm_config=NS(
            kv_transfer_config=NS(is_kv_producer=True, is_kv_consumer=False)
        ),
        running=list(running),
        waiting=Queue(waiting),
        skipped_waiting=Queue(skipped),
        hash_block_size=256,
        policy=NS(name="FCFS"),
        _pause_state=NS(name="UNPAUSED"),
        lora_config=None,
        finished_recving_kv_req_ids=set(),
        prefix_replay_tokens=128,
        kv_cache_manager=NS(
            enable_kv_cache_events=False, get_computed_blocks=get_computed_blocks
        ),
    )

    def promote(req):
        assert req.request_id in result.finished_recving_kv_req_ids
        req.status = NS(name="WAITING")
        result.finished_recving_kv_req_ids.remove(req.request_id)
        return True

    result._try_promote_blocked_waiting_request = promote
    return result, reads


@pytest.fixture(autouse=True)
def producer_partition(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "2,3,3,3,3,2,3,21")


def test_conserves_budget_and_keeps_long_progress():
    plan = supply.BudgetPlan(2048, {"short"}, 512)
    assert plan.limit("long", 10000) == 1536
    plan.consume("long", 1536)
    assert plan.limit("another_long", 10000) == 0
    assert plan.limit("short", 2000) == 512
    plan.consume("short", 512)
    assert plan.limit("short", 10) == 0
    assert plan.long_used + plan.short_used == 2048
    assert supply.BudgetPlan(2048, {"short"}, 10000).long_budget == 1024


@pytest.mark.parametrize("budget", [512, 2048, 4096])
def test_bounded_multichunk_work_can_bypass_a_long_prefill(budget):
    medium = request("medium", 3 * budget, prefix="unseen")
    sched, _ = scheduler([request("long", 100 * budget)], [medium])
    plan = supply.plan(sched, budget)
    assert plan is not None and "medium" in plan.short_ids
    assert plan.long_budget >= budget // 2
    assert plan.limit("long", 100 * budget) + plan.limit("medium", 3 * budget) == budget


def test_mostly_cached_multichunk_request_uses_real_residency():
    medium = request("medium", 80000)
    sched, reads = scheduler([request("long", 800000)], [medium], cached=64000)
    supply.state(sched).seen["seen"] = None
    plan = supply.plan(sched, 4096)
    assert reads == ["medium"]
    assert plan is not None and "medium" in plan.short_ids
    assert medium.num_computed_tokens == 0


def test_all_bounded_running_work_keeps_full_normal_budget():
    sched, _ = scheduler([request("a", 6000), request("b", 9000)])
    assert supply.plan(sched, 4096) is None


def test_uncached_work_beyond_the_bound_is_not_promoted():
    req = request("large", 4 * 4096 + 1, prefix="unseen")
    sched, _ = scheduler([request("long", 800000)], [req])
    assert supply.plan(sched, 4096) is None
    assert sched.waiting == [req]


@pytest.mark.parametrize("total", [512, 10000])
def test_single_class_uses_original_scheduler(total):
    sched, _ = scheduler([request("r", total)])
    assert supply.plan(sched, 2048) is None


@pytest.mark.parametrize("cached,selected", [(99000, True), (100, False)])
def test_seen_hint_requires_resident_coverage(cached, selected):
    cold = request("cold", 110000, prefix="unseen")
    warm = request("warm", 100000)
    sched, reads = scheduler([request("long", 10000)], [cold, warm], cached=cached)
    supply.state(sched).seen["seen"] = None
    plan = supply.plan(sched, 2048)
    assert (plan is not None) == selected
    assert sched.waiting == ([warm, cold] if selected else [cold, warm])
    assert reads == ["warm"]
    assert warm.num_computed_tokens == 0


def test_common_prefix_does_not_exhaust_probe_budget():
    cold = [request(str(i), 100000) for i in range(12)]
    for req in cold:
        req.block_hashes = ["seen"] * 128 + ["unseen"] * 372
    warm = request("warm", 100000)
    sched, reads = scheduler([request("long", 10000)], cold + [warm], cached=99000)
    supply.state(sched).seen["seen"] = None
    assert supply.plan(sched, 2048) is not None
    assert reads == ["warm"]
    assert sched.waiting == [warm] + cold


def test_remote_completion_fence_is_required():
    loaded = request("loaded", 100000, 99000, "WAITING_FOR_REMOTE_KVS")
    sched, _ = scheduler([request("long", 10000)], skipped=[loaded])
    assert supply.plan(sched, 2048) is None
    assert loaded.status.name == "WAITING_FOR_REMOTE_KVS"
    sched.finished_recving_kv_req_ids.add("loaded")
    plan = supply.plan(sched, 2048)
    assert plan is not None and "loaded" in plan.short_ids
    assert loaded.status.name == "WAITING"


def test_event_enabled_cache_is_not_probed():
    sched, reads = scheduler(
        [request("long", 10000)], [request("warm", 100000)], cached=99000
    )
    supply.state(sched).seen["seen"] = None
    sched.kv_cache_manager.enable_kv_cache_events = True
    assert supply.plan(sched, 2048) is None
    assert reads == []


@pytest.mark.parametrize(
    "mode",
    ["no_partition", "consumer", "kv_both", "no_kv", "paused", "priority", "lora"],
)
def test_other_schedulers_unchanged(mode, monkeypatch):
    sched, _ = scheduler([request("long", 10000)], [request("short", 512)])
    if mode == "no_partition":
        monkeypatch.delenv("VLLM_PP_LAYER_PARTITION")
    elif mode in ("consumer", "kv_both"):
        sched.vllm_config.kv_transfer_config = NS(
            is_kv_producer=mode == "kv_both", is_kv_consumer=True
        )
    elif mode == "no_kv":
        sched.vllm_config.kv_transfer_config = None
    elif mode == "paused":
        sched._pause_state.name = "PAUSED_NEW"
    elif mode == "priority":
        sched.policy.name = "PRIORITY"
    else:
        sched.lora_config = object()
    assert supply.plan(sched, 2048) is None
    assert not hasattr(sched, "_p_supply_state")
    if mode in ("no_partition", "consumer", "kv_both", "no_kv"):
        supply.finish(sched, request("finished", 1000, 1000, "FINISHED_STOPPED"))
        assert not hasattr(sched, "_p_supply_state")


def test_probe_limit():
    sched, reads = scheduler(
        [request("long", 10000)],
        [request(str(i), 100000) for i in range(10)],
        cached=100,
    )
    supply.state(sched).seen["seen"] = None
    assert supply.plan(sched, 2048) is None
    assert len(reads) == 4


def test_consumer_registration_does_not_wrap(monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION")
    monkeypatch.setattr(
        patch.p,
        "register_once",
        lambda *args, **kwargs: pytest.fail("D must not install the patch"),
    )
    patch.register()


def test_real_schedule_composes_and_is_idempotent(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler

    # Restore methods after the test; no scheduler instance or GPU is needed.
    monkeypatch.setattr(Scheduler, "schedule", Scheduler.schedule)
    monkeypatch.setattr(Scheduler, "_free_request", Scheduler._free_request)
    patch._apply()
    schedule, free = Scheduler.schedule, Scheduler._free_request
    source = schedule._ilu_pp_spec_schedule_source
    assert schedule._ilu_pp_spec_schedule_patched
    assert "_ilu_align_prefix_replay_ext_tokens" in source
    assert "if num_new_tokens <= 0:" in source
    assert source.count("_ilu_prefill_supply.plan(") == 1
    assert source.count("supply_plan.limit(") == 1
    assert source.count("supply_plan.consume(") == 1
    patch._apply()
    assert Scheduler.schedule is schedule and Scheduler._free_request is free
    with pytest.raises(RuntimeError, match="re-sync"):
        patch.patch_schedule_source("def schedule(self): pass")
