# SPDX-License-Identifier: Apache-2.0
"""Share a PP producer's chunk budget between long and bounded prefills.

Long running prefills retain at least half of the step budget. At most one
waiting short request is promoted per step; normal KV admission still applies.
Only FCFS pure KV producers with an explicit PP partition use this policy.

Select short work using actual resident cache coverage / confirmed remote completion,
not the seen-prefix hint (the hint only bounds expensive cache probes).

LMCache asynchronous-lookup connectors are handled by ``prefill_supply_lmcache``
so synchronous connectors (e.g. CPU offload) and no-connector runs keep their
original single-step promotion and waiting-loop behavior unchanged.
"""

import collections
import os

from vllm_iluvatar.v1.core.sched import prefill_supply_lmcache

_MIN_SHORT_WORK_CHUNKS = 4


class SupplyState:
    def __init__(self, block_size):
        self.hash_block_size = block_size
        self.seen = collections.OrderedDict()
        # Used by prefill_supply_lmcache when an async-lookup connector is bound.
        self.submitted = set()
        self.ultra_ids = set()
        self.resolved_hits = {}
        self.first_seen = {}
        self.b_cap_skips = 0
        self.a_promotions = 0
        self.a_starve_promotions = 0

    def key(self, request, budget, replay):
        # Only a prefix covering the required boundary can make this request
        # short. A common 32K prefix alone says nothing about a 100K prompt.
        need = max(1, request.num_tokens + replay - budget)
        index = (need + self.hash_block_size - 1) // self.hash_block_size - 1
        h = request.block_hashes
        return h[index] if len(h) > index else None

    def finish(self, request):
        # Hint only: residency is still checked through the original manager.
        for key in request.block_hashes:
            self.seen[key] = None
            self.seen.move_to_end(key)
        while len(self.seen) > 262144:
            self.seen.popitem(last=False)
        rid = getattr(request, "request_id", None)
        if rid is not None:
            self.ultra_ids.discard(rid)
            self.submitted.discard(rid)
            self.resolved_hits.pop(rid, None)
            self.first_seen.pop(rid, None)


class BudgetPlan:
    isolate_waiting = False

    def __init__(self, budget, short_ids, short_work):
        self.short_ids = set(short_ids)
        # The oldest long-running prefill retains >= half of the step budget.
        self.short_budget = min(budget // 2, short_work)
        self.long_budget = budget - self.short_budget
        self.short_used = self.long_used = 0

    def limit(self, rid, tokens):
        available = (
            self.short_budget - self.short_used
            if rid in self.short_ids
            else self.long_budget - self.long_used
        )
        return min(tokens, max(0, available))

    def consume(self, rid, tokens):
        if rid in self.short_ids:
            self.short_used += tokens
        else:
            self.long_used += tokens

    def should_skip_waiting(self, scheduler, rid):
        return False

    def can_admit_waiting(self, scheduler, request, num_computed_tokens, load_kv_async):
        return True

    def on_admit_waiting(self, scheduler, request, num_computed_tokens, load_kv_async):
        return None


def state(scheduler):
    if not hasattr(scheduler, "_p_supply_state"):
        scheduler._p_supply_state = SupplyState(scheduler.hash_block_size)
    return scheduler._p_supply_state


def remaining(request):
    return max(
        0,
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens,
    )


def short_work_limit(scheduler, budget):
    # A bounded request may span one pipeline-filling wave of chunk work.
    # Preserve the existing floor for shallow pipelines.
    short_chunks = max(
        _MIN_SHORT_WORK_CHUNKS,
        scheduler.vllm_config.parallel_config.pipeline_parallel_size,
    )
    return short_chunks * budget


def enabled(scheduler):
    config = scheduler.vllm_config.kv_transfer_config
    return bool(
        os.environ.get("VLLM_PP_LAYER_PARTITION")
        and config is not None
        and config.is_kv_producer
        and not config.is_kv_consumer
    )


def plan(scheduler, budget):
    """Called after new_step_starts, before either allocation loop."""
    if (
        not enabled(scheduler)
        or budget <= 0
        or scheduler.policy.name != "FCFS"
        or scheduler._pause_state.name != "UNPAUSED"
        or scheduler.lora_config is not None
    ):
        return None
    if prefill_supply_lmcache.has_lmcache_connector(scheduler):
        return prefill_supply_lmcache.plan(scheduler, budget, state(scheduler))

    st = state(scheduler)
    short_limit = short_work_limit(scheduler, budget)
    short = {
        r.request_id: remaining(r)
        for r in scheduler.running
        if 0 < remaining(r) <= short_limit
    }
    long_exists = any(remaining(r) > short_limit for r in scheduler.running)
    if not long_exists:
        return None

    candidates = []
    probes = 0
    for queue in (scheduler.skipped_waiting, scheduler.waiting):
        for n, request in enumerate(list(queue)):
            if n >= 64:
                break
            status = request.status.name
            if (
                status == "WAITING_FOR_REMOTE_KVS"
                and request.request_id in scheduler.finished_recving_kv_req_ids
            ):
                # Retain the original completion fence and cache bookkeeping.
                scheduler._try_promote_blocked_waiting_request(request)
                status = request.status.name
            if status != "WAITING" or request.has_encoder_inputs:
                continue
            if request.num_computed_tokens:
                count = remaining(request)
            elif request.num_tokens <= budget:
                count = request.num_tokens
            elif (
                probes < 4
                and st.key(request, short_limit, scheduler.prefix_replay_tokens) in st.seen
                and not scheduler.kv_cache_manager.enable_kv_cache_events
            ):
                # Read-only coverage probe. Do not retain block views or mutate
                # num_computed_tokens. Normal admission revalidates after any
                # running allocations, and decides whether host loading is needed.
                probes += 1
                _, cached, _ = scheduler.kv_cache_manager.get_computed_blocks(request)
                count = request.num_tokens - max(
                    0, cached - scheduler.prefix_replay_tokens
                )
            elif request.num_tokens <= short_limit:
                count = request.num_tokens
            else:
                continue
            if 0 < count <= short_limit:
                candidates.append((count, n, queue, request))
    # One bounded promotion per step. Retain original order of all other requests.
    if candidates:
        count, _, queue, request = min(candidates, key=lambda x: (x[0], x[1]))
        if queue.peek_request() is not request:
            queue.remove_request(request)
            queue.prepend_request(request)
        short[request.request_id] = count
    if not short:
        return None
    return BudgetPlan(budget, short, sum(short.values()))


def finish(scheduler, request):
    if enabled(scheduler):
        if (
            request.num_computed_tokens >= request.num_prompt_tokens
            and request.status.name in ("FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED")
        ):
            state(scheduler).finish(request)
        elif hasattr(scheduler, "_p_supply_state"):
            st = scheduler._p_supply_state
            rid = getattr(request, "request_id", None)
            if rid is not None:
                st.ultra_ids.discard(rid)
                st.submitted.discard(rid)
                st.resolved_hits.pop(rid, None)
                st.first_seen.pop(rid, None)
