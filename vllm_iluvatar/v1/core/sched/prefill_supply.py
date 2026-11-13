# SPDX-License-Identifier: Apache-2.0
"""Share a PP producer's chunk budget between long and bounded prefills.

Long running prefills retain at least half of the step budget. At most one
waiting short request is promoted per step; normal KV admission still applies.
Only FCFS pure KV producers with an explicit PP partition use this policy.

Select short work using actual resident cache coverage / confirmed remote completion,
not the seen-prefix hint (the hint only bounds expensive cache probes).

VLLM_ILU_PREFILL_LOOKUP_AHEAD=K (default 0, off) drives the connector lookups
of up to K shortest candidates each step in parallel and promotes the first
resolved one; when 0, only the shortest candidate's lookup is driven. With
LMCacheMPConnector, unresolved candidates are never promoted ahead of lookup
completion, and the waiting loop is isolated to each request's BudgetPlan bucket
so unused short budget cannot leak into unpromoted long prefills.
VLLM_ILU_PREFILL_LOOKUP_SUBMIT=N (default 0, off) submits, each step, the
connector lookup of up to N waiting requests not yet looked up, long or short,
so the lookup is in flight before the waiting loop first visits the request.
Synchronous connectors (e.g. CPU offload) and no-connector runs keep the
original single-step promotion and waiting-loop behavior unchanged.
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
    def __init__(
        self,
        budget,
        short_ids,
        short_work,
        *,
        isolate_waiting=False,
        promoted_waiting_id=None,
    ):
        self.short_ids = set(short_ids)
        # The oldest long-running prefill retains >= half of the step budget.
        self.short_budget = min(budget // 2, short_work)
        self.long_budget = budget - self.short_budget
        self.short_used = self.long_used = 0
        self.isolate_waiting = bool(isolate_waiting)
        self.promoted_waiting_id = promoted_waiting_id

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
        target = self.promoted_waiting_id
        if target is not None and rid != target:
            if (
                scheduler.waiting
                and scheduler.waiting.peek_request().request_id == target
            ) or any(r.request_id == target for r in scheduler.skipped_waiting):
                return True
        done = getattr(scheduler, "finished_recving_kv_req_ids", None)
        if done and scheduler.skipped_waiting:
            return any(
                r.status.name == "WAITING_FOR_REMOTE_KVS" and r.request_id in done
                for r in scheduler.skipped_waiting
            )
        return False


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
    probed = _submit_lookups(scheduler, st)
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
    promoted_waiting_id = None
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        selected = _select_candidate(scheduler, st, candidates, probed)
        if selected is not None:
            count, _, queue, request = selected
            if queue.peek_request() is not request:
                queue.remove_request(request)
                queue.prepend_request(request)
            short[request.request_id] = count
            promoted_waiting_id = request.request_id
    if not short:
        return None
    return BudgetPlan(
        budget,
        short,
        sum(short.values()),
        isolate_waiting=_async_lookup_connector(scheduler) is not None,
        promoted_waiting_id=promoted_waiting_id,
    )


def _env_count(name):
    try:
        return max(0, int(os.environ.get(name, "0")))
    except ValueError:
        return 0


def _lookup_ahead():
    return _env_count("VLLM_ILU_PREFILL_LOOKUP_AHEAD")


_ASYNC_LOOKUP_CONNECTORS = frozenset(
    {"LMCacheMPConnector", "LMCacheMPConnectorUpstream"}
)


def _async_lookup_connector(scheduler):
    """The scheduler's connector if it carries an LMCache MP lookup, else None.

    Only that connector answers lookups asynchronously; a synchronous one
    (CPU offload) resolves on the first visit, so lookahead and waiting-loop
    budget isolation leave synchronous paths untouched.
    """
    connector = getattr(scheduler, "connector", None)
    cached = scheduler.__dict__.get("_p_async_lookup_connector")
    if isinstance(cached, tuple) and cached[0] is connector:
        return cached[1]
    seen = set()
    stack = [connector]
    found = False
    while stack:
        child = stack.pop()
        if child is None or id(child) in seen:
            continue
        seen.add(id(child))
        if any(
            cls.__name__ in _ASYNC_LOOKUP_CONNECTORS for cls in type(child).__mro__
        ):
            found = True
            break
        children = (
            getattr(child, "sub_connectors", None)
            or getattr(child, "_connectors", None)
        )
        if children:
            stack.extend(children)
    result = connector if found else None
    scheduler.__dict__["_p_async_lookup_connector"] = (connector, result)
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
    probed = {}
    limit = _env_count("VLLM_ILU_PREFILL_LOOKUP_SUBMIT")
    if limit <= 0:
        return probed
    connector = _async_lookup_connector(scheduler)
    if connector is None:
        return probed
    if scheduler.kv_cache_manager.enable_kv_cache_events:
        return probed
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
        probed[request.request_id] = _lookup_hit(scheduler, connector, request)
        limit -= 1
    return probed


def _select_candidate(scheduler, st, candidates, probed):
    """Select at most one waiting short candidate to promote this step.

    Synchronous connectors (CPU offload) and no-connector runs resolve on the
    first waiting-loop visit, so the shortest candidate is promoted directly.
    An asynchronous connector (LMCache MP) answers None until its lookup
    resolves; promoting an unresolved request throttles long prefills and
    leaves unused short budget in the waiting loop. Drive at least the shortest
    candidate's lookup (or VLLM_ILU_PREFILL_LOOKUP_AHEAD candidates in
    parallel) and promote only once a candidate's lookup has resolved.
    """
    connector = _async_lookup_connector(scheduler)
    if connector is None or scheduler.kv_cache_manager.enable_kv_cache_events:
        return candidates[0]
    track_submitted = _env_count("VLLM_ILU_PREFILL_LOOKUP_SUBMIT") > 0
    limit = max(1, _lookup_ahead())
    for candidate in candidates[:limit]:
        request = candidate[3]
        if request.num_computed_tokens or getattr(
            request, "skip_reading_prefix_cache", False
        ):
            return candidate
        rid = request.request_id
        if rid in probed:
            hit = probed[rid]
        else:
            if track_submitted:
                st.submitted.add(rid)
            hit = _lookup_hit(scheduler, connector, request)
        if hit is not None:
            return candidate
    return None


def finish(scheduler, request):
    if (
        enabled(scheduler)
        and request.num_computed_tokens >= request.num_prompt_tokens
        and request.status.name in ("FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED")
    ):
        state(scheduler).finish(request)
