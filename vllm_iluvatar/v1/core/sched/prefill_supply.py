# SPDX-License-Identifier: Apache-2.0
"""Share a PP producer's chunk budget between long and bounded prefills.

Long running prefills retain at least half of the step budget. At most one
waiting short request is promoted per step; normal KV admission still applies.
Only FCFS pure KV producers with an explicit PP partition use this policy.

Select short work using actual resident cache coverage / confirmed remote completion,
not the seen-prefix hint (the hint only bounds expensive cache probes).

VLLM_ILU_PREFILL_LOOKUP_AHEAD=K (default 0, off) drives the connector lookups
of the K shortest candidates each step and promotes the first resolved one, so
an asynchronous lookup does not hold the short share for several steps.
VLLM_ILU_PREFILL_LOOKUP_SUBMIT=N (default 0, off) submits, each step, the
connector lookup of up to N waiting requests not yet looked up, long or short,
so the lookup is in flight before the waiting loop first visits the request.
Both apply only when the connector includes LMCacheMPConnector.
"""

import collections
import os

_MIN_SHORT_WORK_CHUNKS = 4


class SupplyState:
    def __init__(self, block_size):
        self.hash_block_size = block_size
        self.seen = collections.OrderedDict()
        # Waiting requests whose connector lookup was submitted ahead.
        self.submitted = set()

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


class BudgetPlan:
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
    st = state(scheduler)
    _submit_lookups(scheduler, st)
    # A bounded request may span one pipeline-filling wave of chunk work.
    # Preserve the existing floor for shallow pipelines.
    short_chunks = max(
        _MIN_SHORT_WORK_CHUNKS,
        scheduler.vllm_config.parallel_config.pipeline_parallel_size,
    )
    short_limit = short_chunks * budget
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
        candidates.sort(key=lambda x: (x[0], x[1]))
        count, _, queue, request = (
            _first_resolved(scheduler, candidates) or candidates[0]
        )
        if queue.peek_request() is not request:
            queue.remove_request(request)
            queue.prepend_request(request)
        short[request.request_id] = count
    if not short:
        return None
    return BudgetPlan(budget, short, sum(short.values()))


def _env_count(name):
    try:
        return max(0, int(os.environ.get(name, "0")))
    except ValueError:
        return 0


def _lookup_ahead():
    return _env_count("VLLM_ILU_PREFILL_LOOKUP_AHEAD")


_ASYNC_LOOKUP_CONNECTORS = frozenset({"LMCacheMPConnector"})


def _async_lookup_connector(scheduler):
    """The scheduler's connector if it carries an LMCache MP lookup, else None.

    Only that connector answers lookups asynchronously; a synchronous one
    (CPU offload) resolves on the first visit, so lookahead buys it nothing.
    """
    cached = scheduler.__dict__.get("_p_async_lookup_connector", False)
    if cached is not False:
        return cached
    connector = getattr(scheduler, "connector", None)
    children = getattr(connector, "sub_connectors", None) or [connector]
    found = any(
        cls.__name__ in _ASYNC_LOOKUP_CONNECTORS
        for child in children
        if child is not None
        for cls in type(child).__mro__
    )
    result = connector if found else None
    scheduler.__dict__["_p_async_lookup_connector"] = result
    return result


def _lookup_hit(scheduler, connector, request):
    """The connector call the waiting loop makes on a visit."""
    _, local, _, _ = scheduler._get_local_prefix_cache_hit(request)
    local -= local % scheduler.block_size
    ext_tokens, _ = connector.get_num_new_matched_tokens(request, local)
    return ext_tokens


def _submit_lookups(scheduler, st):
    """Submit the connector lookup of waiting requests before they are visited.

    The waiting loop reaches a request only while step budget is left, and its
    first visit is the one that submits the lookup, so an asynchronous lookup
    always costs that request at least one more visit. With a connector that
    finishes lookups in the background (LMCache #5960 callbacks), one early
    call per request is enough for the result to be ready by admission.
    """
    limit = _env_count("VLLM_ILU_PREFILL_LOOKUP_SUBMIT")
    if limit <= 0:
        return
    connector = _async_lookup_connector(scheduler)
    if connector is None:
        return
    if scheduler.kv_cache_manager.enable_kv_cache_events:
        return
    waiting = [*scheduler.skipped_waiting, *scheduler.waiting]
    st.submitted.intersection_update(r.request_id for r in waiting)
    for request in waiting:
        if limit <= 0:
            break
        if (
            request.request_id in st.submitted
            or request.status.name != "WAITING"
            or request.num_computed_tokens
            or request.has_encoder_inputs
        ):
            continue
        st.submitted.add(request.request_id)
        _lookup_hit(scheduler, connector, request)
        limit -= 1


def _first_resolved(scheduler, candidates):
    """Return the first candidate whose connector lookup has resolved.

    An asynchronous connector (LMCache MP) answers None until its lookup
    resolves, which takes several scheduler visits. A promoted request still
    waiting on it leaves the short share idle and keeps the slot for those
    steps. Driving the leading candidates' lookups every step lets them
    resolve in parallel; the first resolved one takes the slot. The calls are
    the ones the waiting loop would make on a visit.
    """
    limit = _lookup_ahead()
    if limit <= 0:
        return None
    connector = _async_lookup_connector(scheduler)
    if connector is None:
        return None
    if scheduler.kv_cache_manager.enable_kv_cache_events:
        return None
    for candidate in candidates[:limit]:
        request = candidate[3]
        if request.num_computed_tokens:
            return candidate
        if _lookup_hit(scheduler, connector, request) is not None:
            return candidate
    return None


def finish(scheduler, request):
    if (
        enabled(scheduler)
        and request.num_computed_tokens >= request.num_prompt_tokens
        and request.status.name in ("FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED")
    ):
        state(scheduler).finish(request)
