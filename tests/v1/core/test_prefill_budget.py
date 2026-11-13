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


def scheduler(running=(), waiting=(), skipped=(), cached=0, pp=8):
    reads = []

    def get_computed_blocks(req):
        reads.append(req.request_id)
        return None, cached, 0

    result = NS(
        vllm_config=NS(
            kv_transfer_config=NS(is_kv_producer=True, is_kv_consumer=False),
            parallel_config=NS(pipeline_parallel_size=pp),
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


@pytest.mark.parametrize("pp,selected", [(1, False), (2, False), (4, False),
                                         (8, True), (16, True)])
def test_latency_window_tracks_pipeline_depth(pp, selected):
    medium = request("medium", 5 * 2048, prefix="unseen")
    sched, _ = scheduler([request("long", 1_000_000)], [medium], pp=pp)
    plan = supply.plan(sched, 2048)
    assert (plan is not None) is selected
    if selected:
        assert "medium" in plan.short_ids and plan.long_budget >= 1024


def test_cache_alignment_tail_fits_pipeline_window():
    medium = request("medium", 80030)
    sched, reads = scheduler([request("long", 800000)], [medium], cached=63616)
    supply.state(sched).seen["seen"] = None
    plan = supply.plan(sched, 4096)
    assert reads == ["medium"]
    assert plan is not None and "medium" in plan.short_ids
    assert medium.num_computed_tokens == 0


def test_all_bounded_running_work_keeps_full_normal_budget():
    sched, _ = scheduler([request("a", 6000), request("b", 9000)])
    assert supply.plan(sched, 4096) is None


def test_uncached_work_beyond_the_bound_is_not_promoted():
    req = request("large", 40000, prefix="unseen")
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
    sched, reads = scheduler([request("long", 100000)], [cold, warm], cached=cached)
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
    sched, reads = scheduler([request("long", 100000)], cold + [warm], cached=99000)
    supply.state(sched).seen["seen"] = None
    assert supply.plan(sched, 2048) is not None
    assert reads == ["warm"]
    assert sched.waiting == [warm] + cold


def test_remote_completion_fence_is_required():
    loaded = request("loaded", 100000, 99000, "WAITING_FOR_REMOTE_KVS")
    sched, _ = scheduler([request("long", 100000)], skipped=[loaded])
    assert supply.plan(sched, 2048) is None
    assert loaded.status.name == "WAITING_FOR_REMOTE_KVS"
    sched.finished_recving_kv_req_ids.add("loaded")
    plan = supply.plan(sched, 2048)
    assert plan is not None and "loaded" in plan.short_ids
    assert loaded.status.name == "WAITING"


class LMCacheMPConnector:
    """Answers None for requests whose lookup has not resolved yet."""

    def __init__(self, resolved=(), hits=None):
        self.resolved = set(resolved)
        self.hits = dict(hits or {})
        self.calls = []
        self.allocs = []

    def get_num_new_matched_tokens(self, req, num_computed_tokens):
        self.calls.append((req.request_id, num_computed_tokens))
        if req.request_id in self.hits:
            ext = max(0, self.hits[req.request_id] - num_computed_tokens)
            return ext, ext > 0
        if req.request_id in self.resolved:
            return 0, False
        return None, True

    def update_state_after_alloc(self, req, blocks, num_external_tokens):
        self.allocs.append((req.request_id, num_external_tokens))


class OffloadingConnector:
    """A synchronous connector: every lookup resolves on the first call."""

    def __init__(self, hits=None):
        self.hits = dict(hits or {})
        self.calls = []
        self.allocs = []

    def get_num_new_matched_tokens(self, req, num_computed_tokens):
        self.calls.append((req.request_id, num_computed_tokens))
        ext = max(0, self.hits.get(req.request_id, 0) - num_computed_tokens)
        return ext, ext > 0

    def update_state_after_alloc(self, req, blocks, num_external_tokens):
        self.allocs.append((req.request_id, num_external_tokens))


class OptNixlConnector:
    """Producer-side NIXL connector stub: never loads external prefix tokens."""

    def get_num_new_matched_tokens(self, req, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, req, blocks, num_external_tokens):
        pass


class MultiConnector:
    def __init__(self, *children):
        self.sub_connectors = list(children)

    def get_num_new_matched_tokens(self, req, num_computed_tokens):
        chosen = (0, False)
        for child in self.sub_connectors:
            toks, load_async = child.get_num_new_matched_tokens(
                req, num_computed_tokens
            )
            if toks is None:
                return None, False
            if chosen[0] == 0 and toks > 0:
                chosen = (toks, load_async)
        return chosen

    def update_state_after_alloc(self, req, blocks, num_external_tokens):
        for child in self.sub_connectors:
            if hasattr(child, "update_state_after_alloc"):
                child.update_state_after_alloc(req, blocks, num_external_tokens)


def with_connector(sched, connector, local_hit=0, block_size=64):
    sched.connector = connector
    sched.block_size = block_size
    sched._get_local_prefix_cache_hit = lambda req: (None, local_hit, 0, False)
    return sched


def test_lookup_ahead_is_off_by_default():
    a, b = request("a", 800, prefix="unseen"), request("b", 8000, prefix="unseen")
    connector = LMCacheMPConnector(resolved={"b"})
    sched, _ = scheduler([request("long", 800000)], [b, a])
    with_connector(sched, connector)
    # Without LOOKUP_AHEAD, only the shortest candidate ("a") is probed; since
    # "a" is unresolved, it is not promoted so long prefill keeps full budget.
    assert supply.plan(sched, 4096) is None
    assert sched.waiting == [b, a]
    assert connector.calls == [("a", 0)]
    assert supply.state(sched).submitted == set()

    # Once "a" resolves, it is promoted and isolates the waiting loop.
    connector.resolved.add("a")
    plan = supply.plan(sched, 4096)
    assert plan is not None and "a" in plan.short_ids
    assert plan.isolate_waiting is True
    assert plan.promoted_waiting_id == "a"
    assert sched.waiting == [a, b]


def test_lookup_ahead_promotes_first_resolved(monkeypatch):
    monkeypatch.setenv("VLLM_ILU_PREFILL_LOOKUP_AHEAD", "4")
    a, b = request("a", 800, prefix="unseen"), request("b", 8000, prefix="unseen")
    connector = LMCacheMPConnector(resolved={"b"})
    sched, _ = scheduler([request("long", 800000)], [a, b])
    with_connector(sched, connector, local_hit=130)
    plan = supply.plan(sched, 4096)
    assert plan is not None and "b" in plan.short_ids and "a" not in plan.short_ids
    assert plan.isolate_waiting is True
    assert plan.promoted_waiting_id == "b"
    assert sched.waiting == [b, a]
    # Block-aligned local hit, as the waiting loop passes it.
    assert connector.calls == [("a", 128), ("b", 128)]
    assert a.num_computed_tokens == b.num_computed_tokens == 0


def test_lookup_ahead_defers_when_unresolved(monkeypatch):
    monkeypatch.setenv("VLLM_ILU_PREFILL_LOOKUP_AHEAD", "1")
    a, b = request("a", 800, prefix="unseen"), request("b", 8000, prefix="unseen")
    connector = LMCacheMPConnector(resolved={"b"})
    sched, _ = scheduler([request("long", 800000)], [b, a])
    with_connector(sched, connector)
    assert supply.plan(sched, 4096) is None
    assert sched.waiting == [b, a]
    assert connector.calls == [("a", 0)]

    # If running already has bounded short work, keep its share without
    # promoting the unresolved waiting candidate.
    running_short = request("s_run", 1024, status="RUNNING")
    sched2, _ = scheduler([request("long", 800000), running_short], [b, a])
    with_connector(sched2, LMCacheMPConnector(resolved={"b"}))
    plan = supply.plan(sched2, 4096)
    assert plan is not None
    assert plan.short_ids == {"s_run"}
    assert plan.promoted_waiting_id is None
    assert plan.isolate_waiting is True
    assert sched2.waiting == [b, a]


def test_lookup_submit_and_candidate_probe_do_not_double_query(monkeypatch):
    monkeypatch.setenv("VLLM_ILU_PREFILL_LOOKUP_SUBMIT", "4")
    long_w = request("long_w", 800000, prefix="unseen")
    short_w = request("short_w", 800, prefix="unseen")
    connector = LMCacheMPConnector()
    sched, _ = scheduler([request("long", 800000)], [long_w, short_w])
    with_connector(sched, connector)
    assert supply.plan(sched, 4096) is None
    assert connector.calls == [("long_w", 0), ("short_w", 0)]


def test_lookup_ahead_without_connector(monkeypatch):
    monkeypatch.setenv("VLLM_ILU_PREFILL_LOOKUP_AHEAD", "4")
    a = request("a", 800, prefix="unseen")
    sched, _ = scheduler([request("long", 800000)], [a])
    sched.connector = None
    plan = supply.plan(sched, 4096)
    assert plan is not None and "a" in plan.short_ids
    assert plan.isolate_waiting is False


def test_lookup_submit_is_off_by_default():
    sched, _ = scheduler([request("long", 800000)], [request("w", 100000)])
    connector = LMCacheMPConnector()
    with_connector(sched, connector)
    supply.plan(sched, 4096)
    assert connector.calls == []


def test_lookup_submit_sends_each_waiting_request_once(monkeypatch):
    monkeypatch.setenv("VLLM_ILU_PREFILL_LOOKUP_SUBMIT", "2")
    reqs = [request(str(i), 100000 * (i + 1), prefix="unseen") for i in range(3)]
    sched, _ = scheduler([request("long", 800000)], reqs)
    connector = LMCacheMPConnector()
    with_connector(sched, connector)
    supply.plan(sched, 4096)
    assert [rid for rid, _ in connector.calls] == ["0", "1"]
    supply.plan(sched, 4096)
    assert [rid for rid, _ in connector.calls] == ["0", "1", "2"]
    supply.plan(sched, 4096)
    assert len(connector.calls) == 3


def test_lookup_submit_runs_without_long_prefill(monkeypatch):
    monkeypatch.setenv("VLLM_ILU_PREFILL_LOOKUP_SUBMIT", "8")
    sched, _ = scheduler(waiting=[request("w", 100000)])
    connector = LMCacheMPConnector()
    with_connector(sched, connector)
    assert supply.plan(sched, 4096) is None
    assert connector.calls == [("w", 0)]


def test_lookup_submit_skips_requests_that_need_no_lookup(monkeypatch):
    monkeypatch.setenv("VLLM_ILU_PREFILL_LOOKUP_SUBMIT", "8")
    partial = request("partial", 100000, done=4096)
    remote = request("remote", 100000, status="WAITING_FOR_REMOTE_KVS")
    mm = request("mm", 100000)
    mm.has_encoder_inputs = True
    sched, _ = scheduler(waiting=[partial, remote, mm])
    connector = LMCacheMPConnector()
    with_connector(sched, connector)
    supply.plan(sched, 4096)
    assert connector.calls == []


def test_lookup_submit_forgets_requests_that_left(monkeypatch):
    monkeypatch.setenv("VLLM_ILU_PREFILL_LOOKUP_SUBMIT", "8")
    w = request("w", 100000)
    sched, _ = scheduler(waiting=[w])
    connector = LMCacheMPConnector()
    with_connector(sched, connector)
    supply.plan(sched, 4096)
    sched.waiting.remove(w)
    supply.plan(sched, 4096)
    assert supply.state(sched).submitted == set()


@pytest.mark.parametrize(
    "env", ["VLLM_ILU_PREFILL_LOOKUP_AHEAD", "VLLM_ILU_PREFILL_LOOKUP_SUBMIT"]
)
def test_lookahead_skips_cpu_offload(env, monkeypatch):
    monkeypatch.setenv(env, "8")
    offload = OffloadingConnector()
    short = request("a", 800, prefix="unseen")
    sched, _ = scheduler([request("long", 800000)], [short])
    with_connector(sched, MultiConnector(offload))
    plan = supply.plan(sched, 4096)
    assert plan is not None and "a" in plan.short_ids
    assert offload.calls == []


def test_lookahead_finds_lmcache_inside_multiconnector(monkeypatch):
    monkeypatch.setenv("VLLM_ILU_PREFILL_LOOKUP_SUBMIT", "8")
    lmcache, offload = LMCacheMPConnector(), OffloadingConnector()
    sched, _ = scheduler(waiting=[request("w", 100000)])
    with_connector(sched, MultiConnector(offload, lmcache))
    supply.plan(sched, 4096)
    assert lmcache.calls == [("w", 0)]


def test_event_enabled_cache_is_not_probed():
    sched, reads = scheduler(
        [request("long", 100000)], [request("warm", 100000)], cached=99000
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
    sched, _ = scheduler([request("long", 100000)], [request("short", 512)])
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
        [request("long", 100000)],
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
    assert source.count("supply_plan.limit(") == 3
    assert source.count("supply_plan.consume(") == 2
    assert source.count("supply_plan.should_skip_waiting(") == 1
    assert patch.patch_schedule_source(source) == source
    patch._apply()
    assert Scheduler.schedule is schedule and Scheduler._free_request is free
    with pytest.raises(RuntimeError, match="re-sync"):
        patch.patch_schedule_source("def schedule(self): pass")


def test_prefill_budget_and_async_load_hol_compose_in_both_orders(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_iluvatar.ilu_patch.general_patches import (
        patch_pp_spec_schedule,
        patch_sched_async_load_hol,
    )

    patch_pp_spec_schedule.register()
    if not getattr(Scheduler.schedule, "_ilu_pp_spec_schedule_patched", False):
        patch_pp_spec_schedule._apply()
    base_schedule = Scheduler.schedule
    base_free = Scheduler._free_request

    monkeypatch.setattr(Scheduler, "schedule", base_schedule)
    monkeypatch.setattr(Scheduler, "_free_request", base_free)
    patch._apply()
    patch_sched_async_load_hol._apply()
    sched_ab = Scheduler.schedule
    source_ab = sched_ab._ilu_pp_spec_schedule_source
    assert getattr(sched_ab, "_ilu_pp_spec_schedule_patched", False)
    assert getattr(sched_ab, "_ilu_prefill_budget_patched", False)
    assert getattr(sched_ab, "_ilu_async_load_hol_patched", False)

    # Reverse order must produce the exact same rewritten schedule source.
    Scheduler.schedule = base_schedule
    Scheduler._free_request = base_free
    patch_sched_async_load_hol._apply()
    patch._apply()
    sched_ba = Scheduler.schedule
    source_ba = sched_ba._ilu_pp_spec_schedule_source
    assert getattr(sched_ba, "_ilu_pp_spec_schedule_patched", False)
    assert getattr(sched_ba, "_ilu_prefill_budget_patched", False)
    assert getattr(sched_ba, "_ilu_async_load_hol_patched", False)
    assert source_ab == source_ba

    # Re-applying both again is a strict no-op.
    patch._apply()
    patch_sched_async_load_hol._apply()
    assert Scheduler.schedule is sched_ba


def test_connector_detection_nested_and_rebind():
    class LMCacheMPConnectorUpstream:
        def get_num_new_matched_tokens(self, req, num_computed_tokens):
            return 0, False

    sched, _ = scheduler([request("long", 800000)], [request("a", 800, prefix="unseen")])
    with_connector(sched, MultiConnector(OptNixlConnector(), OffloadingConnector()))
    plan_offload = supply.plan(sched, 4096)
    assert plan_offload is not None and plan_offload.isolate_waiting is False

    # Rebinding connector on the same scheduler invalidates the cached lookup.
    sched.waiting = Queue([request("a", 800, prefix="unseen")])
    nested = NS(_connectors=[OptNixlConnector(), MultiConnector(LMCacheMPConnectorUpstream())])
    nested.get_num_new_matched_tokens = lambda req, n: (0, False)
    with_connector(sched, nested)
    plan_lmcache = supply.plan(sched, 4096)
    assert plan_lmcache is not None and plan_lmcache.isolate_waiting is True


class _FakeBlocks:
    def __init__(self, ids=(1,)):
        self._ids = (list(ids),)

    def get_block_ids(self):
        return self._ids


class _LiveRequest(NS):
    __hash__ = object.__hash__
    __eq__ = object.__eq__


def _live_request(rid, total, done=0, status=None, prefix="unseen"):
    from vllm.v1.request import RequestStatus

    st = status if status is not None else RequestStatus.WAITING
    return _LiveRequest(
        request_id=rid,
        prompt_token_ids=[],
        _all_token_ids=[],
        all_token_ids=range(total),
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        lora_request=None,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        replay_start=0,
        num_tokens=total,
        num_tokens_with_spec=total,
        num_prompt_tokens=total,
        num_computed_tokens=done,
        num_output_placeholders=0,
        num_in_flight_tokens=0,
        max_tokens=16,
        next_decode_eligible_step=0,
        is_prefill_chunk=done < total,
        has_encoder_inputs=False,
        spec_token_ids=[],
        num_stale_output_tokens=0,
        drop_stale_output=False,
        skip_reading_prefix_cache=False,
        shared_prefix_boundary=0,
        prefill_stats=None,
        num_preemptions=0,
        use_structured_output=False,
        status=st,
        block_hashes=[prefix] * 500,
        kv_transfer_params=None,
        is_finished=lambda: False,
    )


def _live_scheduler(
    monkeypatch,
    *,
    running=(),
    waiting=(),
    skipped=(),
    connector=None,
    budget=4096,
    pp=8,
    local_hits=None,
    reserved_blocks=0,
    refuse_async_loads=(),
):
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
    from vllm.v1.core.sched.scheduler import PauseState, Scheduler

    from vllm_iluvatar.ilu_patch.general_patches import patch_sched_async_load_hol

    monkeypatch.setattr(Scheduler, "schedule", Scheduler.schedule)
    monkeypatch.setattr(Scheduler, "_free_request", Scheduler._free_request)
    patch._apply()
    patch_sched_async_load_hol._apply()

    hits = dict(local_hits or {})
    refused = set(refuse_async_loads)
    empty_blocks = _FakeBlocks(())

    def get_computed_blocks(req):
        return empty_blocks, hits.get(req.request_id, 0), 0

    def get_computed_blocks_for_connector(req):
        return empty_blocks, hits.get(req.request_id, 0), 0, False

    def allocate_slots(req, num_tokens, **kwargs):
        if kwargs.get("delay_cache_blocks") and req.request_id in refused:
            return None
        return _FakeBlocks((1,))

    sched = Scheduler.__new__(Scheduler)
    sched.current_step = 0
    sched.max_num_scheduled_tokens = budget
    sched.vllm_config = NS(
        speculative_config=None,
        kv_transfer_config=NS(is_kv_producer=True, is_kv_consumer=False),
        parallel_config=NS(pipeline_parallel_size=pp),
    )
    sched.scheduler_config = NS(
        max_num_batched_tokens=budget,
        long_prefill_token_threshold=0,
        enable_chunked_prefill=True,
    )
    sched._pause_state = PauseState.UNPAUSED
    sched.max_num_encoder_input_tokens = 0
    sched.prefill_capacity_bound = False
    sched.policy = SchedulingPolicy.FCFS
    sched.lora_config = None
    sched.ec_connector = None
    sched.aux_output_connector = None
    sched.connector = connector
    sched.connector_prefix_cache_stats = None
    sched.max_model_len = 2_000_000
    sched.num_sampled_tokens_per_step = 1
    sched.need_mamba_block_aligned_split = False
    sched.num_prefill_lookahead = 0
    sched.num_lookahead_tokens = 0
    sched.num_waiting_for_streaming_input = 0
    sched.max_num_active_reqs = 64
    sched.max_num_running_reqs = 64
    sched.block_size = 64
    sched.hash_block_size = 256
    sched.prefix_replay_tokens = 128
    sched.num_spec_tokens = 0
    sched.dynamic_sd_lookup = None
    sched.is_encoder_decoder = False
    sched.scheduler_reserve_full_isl = False
    sched._inflight_prefills = set()
    sched.finished_recving_kv_req_ids = set()
    sched.failed_recving_kv_req_ids = set()
    sched.needs_kv_cache_zeroing = False
    sched._skip_zero_block_ids = set()
    sched.log_stats = False
    sched.kv_cache_config = NS(kv_cache_groups=[NS()])
    sched.use_v2_model_runner = True
    sched.model_uses_mrope = False
    sched.sched_step_seq = 0
    sched.defer_block_free = False
    sched.reset_preempted_req_ids = set()
    sched.finished_req_ids = set()
    sched.encoder_cache_manager = NS(
        get_freed_mm_hashes=lambda: [],
        get_manager_metadata=lambda: None,
    )
    sched.kv_cache_manager = NS(
        enable_kv_cache_events=False,
        empty_kv_cache_blocks=empty_blocks,
        new_step_starts=lambda: None,
        get_computed_blocks=get_computed_blocks,
        get_computed_blocks_for_connector=get_computed_blocks_for_connector,
        truncate_computed_blocks=lambda blocks, n: blocks,
        allocate_slots=allocate_slots,
        get_blocks=lambda rid: _FakeBlocks((1,)),
        get_block_ids=lambda rid: ([1],),
        cache_blocks=lambda req, n: None,
        record_prefix_cache_stats=lambda req, n: None,
        get_num_common_prefix_blocks=lambda rid: [0],
        take_boundary_state_offloads=lambda: {},
        take_kv_cache_block_copies=lambda: ([], []),
    )
    sched._get_new_block_ids_to_zero = lambda: None
    sched._inflight_prefill_reserved_blocks = lambda: reserved_blocks
    sched._spec_decode_step_blocks = lambda: 0
    sched._build_kv_connector_meta = lambda conn, out: None
    sched._make_cached_request_data = lambda *a, **k: NS()
    sched._get_local_prefix_cache_hit = (
        lambda req: (empty_blocks, hits.get(req.request_id, 0), 0, False)
    )

    sched.running = list(running)
    sched.waiting = FCFSRequestQueue()
    for r in waiting:
        sched.waiting.add_request(r)
    sched.skipped_waiting = FCFSRequestQueue()
    for r in skipped:
        sched.skipped_waiting.add_request(r)
    sched.requests = {
        r.request_id: r for r in [*running, *waiting, *skipped]
    }
    return sched


def test_schedule_e2e_cpu_offload_behavior_unchanged(monkeypatch):
    from vllm.v1.request import RequestStatus

    long_1 = _live_request("long_1", 800_000, done=4096, status=RequestStatus.RUNNING)
    long_2 = _live_request("long_2", 800_000)
    short_1 = _live_request("short_1", 8000)
    long_3 = _live_request("long_3", 800_000)
    conn = MultiConnector(OptNixlConnector(), OffloadingConnector())

    sched = _live_scheduler(
        monkeypatch,
        running=[long_1],
        waiting=[long_2, short_1, long_3],
        connector=conn,
        budget=4096,
    )
    out = sched.schedule()
    assert out.num_scheduled_tokens == {"long_1": 2048, "short_1": 2048}
    assert [r.request_id for r in sched.running] == ["long_1", "short_1"]
    assert [r.request_id for r in sched.waiting] == ["long_2", "long_3"]


@pytest.mark.parametrize("ahead", ["0", "4"])
def test_schedule_e2e_lmcache_unresolved_short_never_leaks_to_800k_long(
    monkeypatch, ahead
):
    from vllm.v1.request import RequestStatus

    monkeypatch.setenv("VLLM_ILU_PREFILL_LOOKUP_AHEAD", ahead)
    monkeypatch.setenv("VLLM_ILU_PREFILL_LOOKUP_SUBMIT", "16")

    long_1 = _live_request("long_1", 800_000, done=4096, status=RequestStatus.RUNNING)
    long_2 = _live_request("long_2", 800_000)
    long_3 = _live_request("long_3", 800_000)
    short_1 = _live_request("short_1", 8000)
    # Queued 800k requests already resolved to (0, False) via early lookup submit,
    # while short_1 just arrived and its lookup is still in flight (None, True).
    lmcache = LMCacheMPConnector(resolved={"long_2", "long_3"})
    conn = MultiConnector(OptNixlConnector(), lmcache)

    sched = _live_scheduler(
        monkeypatch,
        running=[long_1],
        waiting=[long_2, short_1, long_3],
        connector=conn,
        budget=4096,
    )
    out1 = sched.schedule()
    # Step 1: short_1 is unresolved -> long_1 retains full 4096 budget, and
    # neither long_2 nor long_3 leaks into running!
    assert out1.num_scheduled_tokens == {"long_1": 4096}
    assert [r.request_id for r in sched.running] == ["long_1"]
    assert [r.request_id for r in sched.waiting] == ["long_2", "short_1", "long_3"]
    assert ("short_1", 0) in lmcache.calls

    # Step 2: short_1 resolves -> promoted and scheduled for 2048; long_2/3 stay queued.
    lmcache.resolved.add("short_1")
    out2 = sched.schedule()
    assert out2.num_scheduled_tokens == {"long_1": 2048, "short_1": 2048}
    assert [r.request_id for r in sched.running] == ["long_1", "short_1"]
    assert [r.request_id for r in sched.waiting] == ["long_2", "long_3"]


def test_schedule_e2e_lmcache_async_kv_load_short_never_leaks_to_800k_long(monkeypatch):
    from vllm.v1.request import RequestStatus

    long_1 = _live_request("long_1", 800_000, done=4096, status=RequestStatus.RUNNING)
    long_2 = _live_request("long_2", 800_000)
    long_3 = _live_request("long_3", 800_000)
    short_1 = _live_request("short_1", 8000)
    # short_1 hits 6144 tokens in LMCache (load_kv_async=True -> num_new_tokens=0),
    # while long_2 and long_3 resolved to (0, False).
    lmcache = LMCacheMPConnector(
        resolved={"long_2", "long_3"}, hits={"short_1": 6144}
    )
    conn = MultiConnector(OptNixlConnector(), lmcache)

    sched = _live_scheduler(
        monkeypatch,
        running=[long_1],
        waiting=[long_2, short_1, long_3],
        connector=conn,
        budget=4096,
    )
    out1 = sched.schedule()
    # Step 1: short_1 enters WAITING_FOR_REMOTE_KVS consuming 0 compute budget.
    # The remaining 2048 short budget MUST NOT leak to long_2 or long_3!
    assert out1.num_scheduled_tokens == {"long_1": 2048}
    assert short_1.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert [r.request_id for r in sched.running] == ["long_1"]
    assert [r.request_id for r in sched.skipped_waiting] == ["short_1"]
    assert [r.request_id for r in sched.waiting] == ["long_2", "long_3"]

    # Step 2: while short_1 DMA is in flight, long_1 gets full 4096 budget.
    out2 = sched.schedule()
    assert out2.num_scheduled_tokens == {"long_1": 4096}
    assert [r.request_id for r in sched.running] == ["long_1"]
    assert [r.request_id for r in sched.waiting] == ["long_2", "long_3"]

    # Step 3: short_1 finishes async KV recv (6144 - 128 SWA replay = 6016,
    # leaving 1984 tokens) and runs to completion without admitting long_2.
    sched.finished_recving_kv_req_ids.add("short_1")
    out3 = sched.schedule()
    assert out3.num_scheduled_tokens == {"long_1": 2112, "short_1": 1984}
    assert [r.request_id for r in sched.running] == ["long_1", "short_1"]
    assert [r.request_id for r in sched.waiting] == ["long_2", "long_3"]


def test_schedule_e2e_lmcache_partial_short_consumption_never_leaks_to_long(
    monkeypatch,
):
    from vllm.v1.request import RequestStatus

    # Case A: short_1 (8000) gets short_budget=2048 in plan(), then hits 7168
    # local APC tokens in WAITING loop (7168 - 128 replay = 7040 -> 960 new
    # tokens < 2048). Remaining 1088 tokens must not leak to long_2!
    long_1 = _live_request("long_1", 800_000, done=4096, status=RequestStatus.RUNNING)
    long_2 = _live_request("long_2", 800_000)
    short_1 = _live_request("short_1", 8000)
    lmcache = LMCacheMPConnector(resolved={"short_1", "long_2"})
    sched = _live_scheduler(
        monkeypatch,
        running=[long_1],
        waiting=[long_2, short_1],
        connector=MultiConnector(OptNixlConnector(), lmcache),
        budget=4096,
        local_hits={"short_1": 7168},
    )
    out = sched.schedule()
    assert out.num_scheduled_tokens == {"long_1": 2048, "short_1": 960}
    assert [r.request_id for r in sched.running] == ["long_1", "short_1"]
    assert [r.request_id for r in sched.waiting] == ["long_2"]

    # Case B: running already has short_run (skipped due to decode cadence),
    # while waiting has resolved long_2 and unresolved short_wait.
    long_a = _live_request("long_a", 800_000, done=4096, status=RequestStatus.RUNNING)
    short_run = _live_request("short_run", 4096, done=3072, status=RequestStatus.RUNNING)
    short_run.next_decode_eligible_step = 99
    long_b = _live_request("long_b", 800_000)
    short_wait = _live_request("short_wait", 800)
    sched2 = _live_scheduler(
        monkeypatch,
        running=[long_a, short_run],
        waiting=[long_b, short_wait],
        connector=MultiConnector(
            OptNixlConnector(), LMCacheMPConnector(resolved={"long_b"})
        ),
        budget=4096,
    )
    out2 = sched2.schedule()
    assert out2.num_scheduled_tokens == {"long_a": 3072}
    assert [r.request_id for r in sched2.running] == ["long_a", "short_run"]
    assert [r.request_id for r in sched2.waiting] == ["long_b", "short_wait"]


def test_schedule_e2e_lmcache_skipped_waiting_long_does_not_block_promoted_short(
    monkeypatch,
):
    from vllm.v1.request import RequestStatus

    long_1 = _live_request("long_1", 800_000, done=4096, status=RequestStatus.RUNNING)
    # A long request in skipped_waiting whose remote KV load already finished
    # (remaining = 160k > short_limit), plus an in-progress remote KV load.
    long_loaded = _live_request(
        "long_loaded", 800_000, done=640_000, status=RequestStatus.WAITING
    )
    still_loading = _live_request(
        "still_loading", 200_000, done=160_000, status=RequestStatus.WAITING_FOR_REMOTE_KVS
    )
    long_2 = _live_request("long_2", 800_000)
    short_1 = _live_request("short_1", 1500)

    lmcache = LMCacheMPConnector(resolved={"short_1", "long_2"})
    sched = _live_scheduler(
        monkeypatch,
        running=[long_1],
        skipped=[long_loaded, still_loading],
        waiting=[long_2, short_1],
        connector=MultiConnector(OptNixlConnector(), lmcache),
        budget=4096,
    )
    out = sched.schedule()
    # short_1 in waiting is reached past skipped_waiting, scheduled for 1500,
    # long_2 is not admitted, and FCFS order in both queues is preserved!
    assert out.num_scheduled_tokens == {"long_1": 2596, "short_1": 1500}
    assert [r.request_id for r in sched.running] == ["long_1", "short_1"]
    assert [r.request_id for r in sched.skipped_waiting] == [
        "long_loaded",
        "still_loading",
    ]
    assert [r.request_id for r in sched.waiting] == ["long_2"]


def test_schedule_e2e_lmcache_composes_with_async_load_hol_refusal(monkeypatch):
    from vllm.v1.request import RequestStatus

    long_1 = _live_request("long_1", 800_000, done=4096, status=RequestStatus.RUNNING)
    short_hit = _live_request("short_hit", 8000)
    in_progress = [
        _live_request(
            f"loading_{i}",
            100_000,
            done=64_000,
            status=RequestStatus.WAITING_FOR_REMOTE_KVS,
        )
        for i in range(64)
    ]
    # Placed past plan()'s 64-entry scan window so only the waiting loop's
    # HOL bypass + BudgetPlan.should_skip_waiting reaches and promotes it.
    inflight_done = _live_request(
        "inflight_done",
        400_000,
        done=320_000,
        status=RequestStatus.WAITING_FOR_REMOTE_KVS,
    )
    long_2 = _live_request("long_2", 800_000)

    lmcache = LMCacheMPConnector(
        resolved={"long_2"}, hits={"short_hit": 6144}
    )
    sched = _live_scheduler(
        monkeypatch,
        running=[long_1],
        skipped=[short_hit, *in_progress, inflight_done],
        waiting=[long_2],
        connector=MultiConnector(OptNixlConnector(), lmcache),
        budget=4096,
        reserved_blocks=128,
        refuse_async_loads={"short_hit"},
    )
    sched._inflight_prefills.update([*in_progress, inflight_done])
    sched.finished_recving_kv_req_ids.add("inflight_done")

    out = sched.schedule()
    assert out.num_scheduled_tokens == {"long_1": 2048}
    assert inflight_done.status == RequestStatus.WAITING
    assert "inflight_done" not in sched.finished_recving_kv_req_ids
    assert [r.request_id for r in sched.running] == ["long_1"]
    assert [r.request_id for r in sched.waiting] == ["long_2"]
