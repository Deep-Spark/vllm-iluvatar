# SPDX-License-Identifier: Apache-2.0
"""LMCache-specific producer prefill scheduling extensions.

Houses all asynchronous-lookup connector scheduling logic parallel to
``prefill_supply`` so synchronous connectors (CPU offload) and no-connector
deployments never execute LMCache-specific code:

- ``VLLM_ILU_PREFILL_LOOKUP_AHEAD=K`` (default 0): drives up to K candidate
  lookups in parallel and promotes only resolved work, isolating the waiting
  loop so unused short budget never leaks into unpromoted long prefills.
- ``VLLM_ILU_PREFILL_LOOKUP_SUBMIT=N`` (default 0): submits up to N waiting
  connector lookups per step ahead of admission.
- Strategy B — conditional ultra-long concurrency cap:
  ``VLLM_ILU_PREFILL_LONG_CAP_TOKENS`` (default 0, off; e.g. 256000): caps
  concurrent ultra-long prefills in ``running`` / in-flight loads to
  ``VLLM_ILU_PREFILL_LONG_CAP_MAX`` (default 1) when non-ultra work is present,
  and relaxes to ``VLLM_ILU_PREFILL_LONG_CAP_IDLE_MAX`` (default 2; 0 = no cap)
  when no non-ultra pressure exists. An active ultra-long request releases its
  slot once its remaining work drops to ``<= VLLM_ILU_PREFILL_LONG_RELEASE_TOKENS``
  (default 0 -> ``short_limit``).
- Strategy A — cache-aware waiting short-first + smooth linear aging:
  ``VLLM_ILU_PREFILL_SHORT_FIRST=1`` (default 0, off):
  1. Strict fast-lane isolation (Opt 1): while a ``> short_limit`` prefill is
     running, the 50% bounded short lane is strictly reserved for ``<= short_limit``
     work; ``> short_limit`` requests never borrow the short lane and instead
     contend at main-lane handover steps.
  2. Smooth linear aging + FCFS starvation ceiling (Opt 2): normal candidates
     are ordered by ``max(0, effective_remaining - AGING_RATE * wait_sec)``
     (``VLLM_ILU_PREFILL_AGING_RATE``, default 2500 tok/s); requests waiting
     ``>= VLLM_ILU_PREFILL_STARVE_SEC`` (default 120s; 0 = off) promote to
     ``starve_tier=0`` ordered oldest-arrival-first (``arrival_time``), while
     still respecting Strategy B's ultra-long concurrency cap.
"""

from __future__ import annotations

import collections
import os
import time

_ASYNC_LOOKUP_CONNECTORS = frozenset(
    {"LMCacheMPConnector", "LMCacheMPConnectorUpstream"}
)
_MIN_SHORT_WORK_CHUNKS = 4


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _lookup_ahead() -> int:
    return _env_int("VLLM_ILU_PREFILL_LOOKUP_AHEAD", 0)


def _remaining(request) -> int:
    return max(
        0,
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens,
    )


def _short_work_limit(scheduler, budget: int) -> int:
    short_chunks = max(
        _MIN_SHORT_WORK_CHUNKS,
        scheduler.vllm_config.parallel_config.pipeline_parallel_size,
    )
    return short_chunks * budget


def _find_lmcache_connector(scheduler):
    """Return ``(top_connector, lmcache_child)`` if LMCache MP is present."""
    connector = getattr(scheduler, "connector", None)
    cached = scheduler.__dict__.get("_p_async_lookup_connector")
    if isinstance(cached, tuple) and len(cached) == 3 and cached[0] is connector:
        return cached[1], cached[2]
    seen = set()
    stack = [connector]
    lmc_child = None
    while stack:
        child = stack.pop()
        if child is None or id(child) in seen:
            continue
        seen.add(id(child))
        if any(
            cls.__name__ in _ASYNC_LOOKUP_CONNECTORS for cls in type(child).__mro__
        ):
            lmc_child = child
            break
        children = (
            getattr(child, "sub_connectors", None)
            or getattr(child, "_connectors", None)
        )
        if children:
            stack.extend(children)
    top = connector if lmc_child is not None else None
    scheduler.__dict__["_p_async_lookup_connector"] = (connector, top, lmc_child)
    return top, lmc_child


def has_lmcache_connector(scheduler) -> bool:
    top, _ = _find_lmcache_connector(scheduler)
    return top is not None


class LMCacheBudgetPlan:
    def __init__(
        self,
        budget: int,
        short_ids,
        short_work: int,
        *,
        budget_split: bool = True,
        isolate_waiting: bool = True,
        promoted_waiting_id: str | None = None,
        ordered_ids: list[str] | None = None,
        strict_ordered: bool = False,
        st=None,
        cap_tokens: int = 0,
        cap_max: int = 1,
        cap_idle_max: int = 2,
        release_tokens: int = 0,
        active_ultra_ids: set[str] | None = None,
        has_short_pressure: bool = False,
    ):
        self.budget_split = bool(budget_split)
        self.short_ids = set(short_ids)
        if self.budget_split:
            # The oldest long-running prefill retains >= half of the step budget.
            self.short_budget = min(budget // 2, short_work)
            self.long_budget = budget - self.short_budget
        else:
            self.short_budget = budget
            self.long_budget = budget
        self.short_used = self.long_used = 0
        self.isolate_waiting = bool(isolate_waiting)
        self.promoted_waiting_id = promoted_waiting_id
        self.ordered_queue = collections.deque(ordered_ids or ())
        self.strict_ordered = bool(strict_ordered)
        self.st = st
        self.cap_tokens = int(cap_tokens)
        self.cap_max = int(cap_max)
        self.cap_idle_max = int(cap_idle_max)
        self.release_tokens = int(release_tokens)
        self.active_ultra_ids = set(active_ultra_ids or ())
        self.has_short_pressure = bool(has_short_pressure)
        self._last_reject_reason: str | None = None

    def limit(self, rid: str, tokens: int) -> int:
        if not self.budget_split:
            return tokens
        available = (
            self.short_budget - self.short_used
            if rid in self.short_ids
            else self.long_budget - self.long_used
        )
        return min(tokens, max(0, available))

    def consume(self, rid: str, tokens: int) -> None:
        if not self.budget_split:
            return
        if rid in self.short_ids:
            self.short_used += tokens
        else:
            self.long_used += tokens

    @staticmethod
    def _in_waiting_queues(scheduler, rid: str) -> bool:
        return any(r.request_id == rid for r in scheduler.skipped_waiting) or any(
            r.request_id == rid for r in scheduler.waiting
        )

    def _current_target(self, scheduler) -> str | None:
        while self.ordered_queue:
            cand = self.ordered_queue[0]
            if self._in_waiting_queues(scheduler, cand):
                return cand
            self.ordered_queue.popleft()
        return None

    def _effective_cap(self) -> int | None:
        if self.cap_tokens <= 0:
            return None
        if self.has_short_pressure:
            return self.cap_max
        if self.cap_idle_max > 0:
            return self.cap_idle_max
        return None

    def _quick_eff_rem(self, scheduler, request) -> int:
        if request.num_computed_tokens:
            return _remaining(request)
        if self.st is not None and request.request_id in self.st.resolved_hits:
            hit = self.st.resolved_hits[request.request_id]
            replay = getattr(scheduler, "prefix_replay_tokens", 0)
            return max(1, request.num_tokens - max(0, hit - replay))
        return request.num_tokens

    def can_admit_waiting(
        self, scheduler, request, num_computed_tokens: int, load_kv_async: bool
    ) -> bool:
        self._last_reject_reason = None
        rid = request.request_id

        # Strategy A ordered sequence check (when transition-step ordering is active).
        if not self.budget_split:
            target = self._current_target(scheduler)
            if target is not None:
                if rid != target:
                    self._last_reject_reason = "ordered"
                    return False
            elif self.strict_ordered:
                # An unresolved shorter request sits ahead of remaining candidates
                # while this step already scheduled higher-priority work.
                self._last_reject_reason = "strict_ordered"
                return False

        # Strategy B conditional ultra-long concurrency cap.
        # Never block a request that already holds in-flight reserved KV blocks.
        inflight = getattr(scheduler, "_inflight_prefills", ())
        if self.cap_tokens > 0 and request not in inflight:
            rem = max(0, request.num_tokens - num_computed_tokens)
            if rem >= self.cap_tokens:
                cap = self._effective_cap()
                if cap is not None and len(self.active_ultra_ids) >= cap:
                    if self.st is not None:
                        self.st.b_cap_skips += 1
                    self._last_reject_reason = "b_cap"
                    return False

        return True

    def on_admit_waiting(
        self, scheduler, request, num_computed_tokens: int, load_kv_async: bool
    ) -> None:
        rid = request.request_id
        if self.ordered_queue and self.ordered_queue[0] == rid:
            self.ordered_queue.popleft()
        if self.cap_tokens > 0:
            rem = max(0, request.num_tokens - num_computed_tokens)
            is_ultra = rem >= self.cap_tokens or (
                self.st is not None and rid in self.st.ultra_ids
            )
            if is_ultra:
                if self.st is not None:
                    self.st.ultra_ids.add(rid)
                if rem > self.release_tokens:
                    self.active_ultra_ids.add(rid)

    def should_skip_waiting(self, scheduler, rid: str) -> bool:
        # 1. Baseline promoted short target.
        target = self.promoted_waiting_id
        if target is not None and rid != target:
            if (
                scheduler.waiting
                and scheduler.waiting.peek_request().request_id == target
            ) or any(r.request_id == target for r in scheduler.skipped_waiting):
                return True

        # 2. Strategy A ordered target waiting behind skipped_waiting items.
        if not self.budget_split:
            ord_target = self._current_target(scheduler)
            if ord_target is not None and rid != ord_target:
                return True
            if self._last_reject_reason == "strict_ordered":
                done = getattr(scheduler, "finished_recving_kv_req_ids", None)
                if done and scheduler.skipped_waiting:
                    return any(
                        r.status.name == "WAITING_FOR_REMOTE_KVS"
                        and r.request_id in done
                        for r in scheduler.skipped_waiting
                    )
                return False

        # 3. Finished async KV loads in skipped_waiting must be reached to promote.
        done = getattr(scheduler, "finished_recving_kv_req_ids", None)
        if done and scheduler.skipped_waiting:
            if any(
                r.status.name == "WAITING_FOR_REMOTE_KVS" and r.request_id in done
                for r in scheduler.skipped_waiting
            ):
                return True

        # 4. Strategy B rejected an ultra-long candidate: skip it only if there is
        #    a non-ultra-long (or already in-flight) request behind it to try.
        if self._last_reject_reason == "b_cap" and not self.isolate_waiting:
            inflight = getattr(scheduler, "_inflight_prefills", ())
            for q in (scheduler.skipped_waiting, scheduler.waiting):
                for other in q:
                    if other.request_id == rid:
                        continue
                    st_name = other.status.name
                    if st_name not in ("WAITING", "PREEMPTED"):
                        continue
                    if other in inflight:
                        return True
                    if self._quick_eff_rem(scheduler, other) < self.cap_tokens:
                        return True
            return False

        return False


def _lookup_hit(scheduler, connector, request, st=None):
    """The connector call the waiting loop makes on a visit."""
    _, local, _, _ = scheduler._get_local_prefix_cache_hit(request)
    local -= local % scheduler.block_size
    ext_tokens, _ = connector.get_num_new_matched_tokens(request, local)
    if ext_tokens is not None and st is not None:
        st.resolved_hits[request.request_id] = local + ext_tokens
    return ext_tokens


def _passive_adapter_hit(lmc_child, rid: str, st) -> int | None:
    """Read an in-flight LMCache lookup result without mutating tracker state."""
    if rid in st.resolved_hits:
        return st.resolved_hits[rid]
    adapter = getattr(lmc_child, "scheduler_adapter", None)
    lookups = getattr(adapter, "_lookups", None)
    if not isinstance(lookups, dict) or rid not in lookups:
        return None
    check = getattr(adapter, "check_lookup_result", None)
    if not callable(check):
        return None
    hit = check(rid)
    if hit is not None:
        st.resolved_hits[rid] = int(hit)
    return hit


def _priority_key(
    eff_rem: int,
    request,
    st,
    now: float,
    starve_sec: float,
    aging_rate: float,
) -> tuple[int, float, float, str]:
    """Compute cache-aware short-first priority key with smooth aging & FCFS bailout.

    - ``tier == 0`` (waited >= ``starve_sec``): ordered strictly oldest-first by
      ``arrival_time`` so aged ultra-long requests cannot be jumped anymore by
      younger aged medium requests under heavy steady-state queues.
    - ``tier == 1`` (normal): ordered by ``max(0, eff_rem - aging_rate * wait_sec)``,
      smoothly advancing waiting requests without a binary threshold cliff.
    """
    rid_str = str(request.request_id)
    raw_arr = getattr(request, "arrival_time", None)
    if raw_arr is None:
        arr = st.first_seen.setdefault(request.request_id, now)
    else:
        arr = float(raw_arr)
    wait_sec = max(0.0, now - arr)
    if starve_sec > 0.0 and wait_sec >= starve_sec:
        return (0, arr, float(eff_rem), rid_str)
    aged_score = max(0.0, float(eff_rem) - aging_rate * wait_sec)
    return (1, aged_score, arr, rid_str)


def _submit_lookups(
    scheduler,
    connector,
    st,
    *,
    short_first: bool = False,
    now: float = 0.0,
    starve_sec: float = 120.0,
    aging_rate: float = 2500.0,
) -> dict[str, int | None]:
    """Submit the connector lookup of waiting requests before they are visited."""
    probed: dict[str, int | None] = {}
    limit = _env_int("VLLM_ILU_PREFILL_LOOKUP_SUBMIT", 0)
    if limit <= 0 or connector is None:
        return probed
    if scheduler.kv_cache_manager.enable_kv_cache_events:
        return probed
    waiting = [*scheduler.skipped_waiting, *scheduler.waiting]
    st.submitted.intersection_update(r.request_id for r in waiting)
    eligible = [
        r
        for r in waiting
        if r.request_id not in st.submitted
        and r.status.name == "WAITING"
        and not r.num_computed_tokens
        and not r.has_encoder_inputs
    ]
    if short_first and len(eligible) > 1:
        eligible.sort(
            key=lambda r: _priority_key(
                _remaining(r), r, st, now, starve_sec, aging_rate
            )
        )
    for request in eligible[:limit]:
        st.submitted.add(request.request_id)
        probed[request.request_id] = _lookup_hit(scheduler, connector, request, st)
    return probed


def _select_candidate(scheduler, connector, st, candidates, probed):
    """Select at most one waiting candidate to promote into the short lane."""
    if connector is None or scheduler.kv_cache_manager.enable_kv_cache_events:
        return candidates[0]
    track_submitted = _env_int("VLLM_ILU_PREFILL_LOOKUP_SUBMIT", 0) > 0
    limit = max(1, _lookup_ahead())
    for candidate in candidates[:limit]:
        request = candidate[-1]
        if request.num_computed_tokens or getattr(
            request, "skip_reading_prefix_cache", False
        ):
            return candidate
        rid = request.request_id
        if rid in probed:
            hit = probed[rid]
        elif rid in st.resolved_hits:
            hit = st.resolved_hits[rid]
        else:
            if track_submitted:
                st.submitted.add(rid)
            hit = _lookup_hit(scheduler, connector, request, st)
        if hit is not None:
            return candidate
    return None


def _prune_state(scheduler, st) -> None:
    inflight = getattr(scheduler, "_inflight_prefills", ())
    active_rids = {
        r.request_id
        for r in (
            *scheduler.running,
            *scheduler.waiting,
            *scheduler.skipped_waiting,
            *inflight,
        )
    }
    if st.ultra_ids:
        st.ultra_ids.intersection_update(active_rids)
    if st.resolved_hits:
        for k in [k for k in st.resolved_hits if k not in active_rids]:
            st.resolved_hits.pop(k, None)
    if st.first_seen:
        for k in [k for k in st.first_seen if k not in active_rids]:
            st.first_seen.pop(k, None)


def _baseline_plan(scheduler, connector, budget: int, st):
    """Exact 0130d65b LMCache behavior when both Strategy A and Strategy B are off."""
    probed = _submit_lookups(scheduler, connector, st)
    short_limit = _short_work_limit(scheduler, budget)
    short = {
        r.request_id: _remaining(r)
        for r in scheduler.running
        if 0 < _remaining(r) <= short_limit
    }
    long_exists = any(_remaining(r) > short_limit for r in scheduler.running)
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
                scheduler._try_promote_blocked_waiting_request(request)
                status = request.status.name
            if status != "WAITING" or request.has_encoder_inputs:
                continue
            if request.num_computed_tokens:
                count = _remaining(request)
            elif request.num_tokens <= budget:
                count = request.num_tokens
            elif (
                probes < 4
                and st.key(request, short_limit, scheduler.prefix_replay_tokens)
                in st.seen
                and not scheduler.kv_cache_manager.enable_kv_cache_events
            ):
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

    promoted_waiting_id = None
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        selected = _select_candidate(scheduler, connector, st, candidates, probed)
        if selected is not None:
            count, _, queue, request = selected
            if queue.peek_request() is not request:
                queue.remove_request(request)
                queue.prepend_request(request)
            short[request.request_id] = count
            promoted_waiting_id = request.request_id
    if not short:
        return None
    return LMCacheBudgetPlan(
        budget,
        short,
        sum(short.values()),
        budget_split=True,
        isolate_waiting=True,
        promoted_waiting_id=promoted_waiting_id,
        st=st,
    )


def _estimate_candidate(
    scheduler, lmc_child, st, request, short_limit: int, probes: int
) -> tuple[int, bool, bool, int]:
    """Return ``(eff_rem, resolved_hint, async_hint, probes)``."""
    rid = request.request_id
    replay = getattr(scheduler, "prefix_replay_tokens", 0)
    inflight = getattr(scheduler, "_inflight_prefills", ())

    if request.num_computed_tokens or getattr(
        request, "skip_reading_prefix_cache", False
    ):
        return _remaining(request), True, False, probes

    ext_hit = _passive_adapter_hit(lmc_child, rid, st)
    local_cached = 0
    if (
        probes < 4
        and st.key(request, short_limit, replay) in st.seen
        and not scheduler.kv_cache_manager.enable_kv_cache_events
    ):
        probes += 1
        _, local_cached, _ = scheduler.kv_cache_manager.get_computed_blocks(request)

    best_hit = max(local_cached, ext_hit or 0)
    if best_hit > 0:
        eff_rem = max(1, request.num_tokens - max(0, best_hit - replay))
    else:
        eff_rem = request.num_tokens

    resolved = ext_hit is not None or request in inflight
    async_hint = bool(ext_hit is not None and ext_hit > local_cached and ext_hit > replay)
    return eff_rem, resolved, async_hint, probes


def plan(scheduler, budget: int, st):
    """Entry point for LMCache producer prefill scheduling."""
    connector, lmc_child = _find_lmcache_connector(scheduler)
    if connector is None:
        return None

    cap_tokens = _env_int("VLLM_ILU_PREFILL_LONG_CAP_TOKENS", 0)
    short_first = _env_int("VLLM_ILU_PREFILL_SHORT_FIRST", 0) > 0
    if cap_tokens <= 0 and not short_first:
        return _baseline_plan(scheduler, connector, budget, st)

    _prune_state(scheduler, st)
    now = time.time()
    starve_sec = _env_float("VLLM_ILU_PREFILL_STARVE_SEC", 120.0)
    aging_rate = _env_float("VLLM_ILU_PREFILL_AGING_RATE", 2500.0)
    probed = _submit_lookups(
        scheduler,
        connector,
        st,
        short_first=short_first,
        now=now,
        starve_sec=starve_sec,
        aging_rate=aging_rate,
    )

    short_limit = _short_work_limit(scheduler, budget)
    cap_max = max(1, _env_int("VLLM_ILU_PREFILL_LONG_CAP_MAX", 1))
    cap_idle_max = _env_int("VLLM_ILU_PREFILL_LONG_CAP_IDLE_MAX", 2)
    release_tokens = _env_int("VLLM_ILU_PREFILL_LONG_RELEASE_TOKENS", 0) or short_limit

    inflight = getattr(scheduler, "_inflight_prefills", ())
    active_ultra_ids: set[str] = set()
    if cap_tokens > 0:
        for r in (*scheduler.running, *inflight):
            rem = _remaining(r)
            if (r.request_id in st.ultra_ids or rem >= cap_tokens) and rem > release_tokens:
                active_ultra_ids.add(r.request_id)
                st.ultra_ids.add(r.request_id)

    has_short_pressure = False
    if cap_tokens > 0:
        has_short_pressure = any(
            0 < _remaining(r) < cap_tokens and r.request_id not in st.ultra_ids
            for r in scheduler.running
        )

    # Scan waiting queues: promote finished remote KV loads and inspect candidates.
    scanned = []
    probes = 0
    seq_idx = 0
    for q_idx, queue in enumerate((scheduler.skipped_waiting, scheduler.waiting)):
        for n, request in enumerate(list(queue)):
            if n >= 128:
                break
            status = request.status.name
            if (
                status == "WAITING_FOR_REMOTE_KVS"
                and request.request_id in scheduler.finished_recving_kv_req_ids
            ):
                scheduler._try_promote_blocked_waiting_request(request)
                status = request.status.name
            if status != "WAITING" or request.has_encoder_inputs:
                if (
                    cap_tokens > 0
                    and not has_short_pressure
                    and status == "WAITING_FOR_REMOTE_KVS"
                    and request.request_id not in st.ultra_ids
                    and _remaining(request) < cap_tokens
                ):
                    has_short_pressure = True
                continue

            eff_rem, resolved_hint, async_hint, probes = _estimate_candidate(
                scheduler, lmc_child, st, request, short_limit, probes
            )
            if request.request_id in probed and probed[request.request_id] is not None:
                resolved_hint = True
            if (
                cap_tokens > 0
                and not has_short_pressure
                and eff_rem < cap_tokens
                and request.request_id not in st.ultra_ids
            ):
                has_short_pressure = True

            pkey = _priority_key(eff_rem, request, st, now, starve_sec, aging_rate)
            scanned.append(
                (
                    pkey,
                    eff_rem,
                    seq_idx,
                    q_idx,
                    queue,
                    resolved_hint,
                    async_hint,
                    request,
                )
            )
            seq_idx += 1

    eff_cap: int | None = None
    if cap_tokens > 0:
        if has_short_pressure:
            eff_cap = cap_max
        elif cap_idle_max > 0:
            eff_cap = cap_idle_max

    def _allowed_by_b(request, eff_rem: int, cur_ultra_count: int) -> bool:
        if cap_tokens <= 0 or request in inflight:
            return True
        if eff_rem < cap_tokens:
            return True
        if eff_cap is None:
            return True
        return cur_ultra_count < eff_cap

    # Optimization 1: Strict fast-lane isolation.
    # Only requests with remaining <= short_limit enter the 50% bounded short lane.
    # Medium/long requests (> short_limit) NEVER borrow short_budget while another
    # > short_limit prefill is running; they take over at Regime 2 handover steps.
    short = {
        r.request_id: _remaining(r)
        for r in scheduler.running
        if 0 < _remaining(r) <= short_limit
    }
    long_exists = any(_remaining(r) > short_limit for r in scheduler.running)

    # Regime 1: running already holds > short_limit prefill work -> share chunk budget
    # exclusively with <= short_limit requests.
    if long_exists:
        lane_candidates = []
        for item in scanned:
            pkey, eff_rem, s_idx, _q_idx, queue, _res, _async, req = item
            if eff_rem <= 0 or eff_rem > short_limit:
                continue
            if not _allowed_by_b(req, eff_rem, len(active_ultra_ids)):
                continue
            sort_key = pkey if short_first else (eff_rem, s_idx)
            lane_candidates.append((sort_key, eff_rem, pkey[0], queue, req))

        promoted_waiting_id = None
        if lane_candidates:
            lane_candidates.sort(key=lambda x: x[0])
            selected = _select_candidate(
                scheduler, connector, st, lane_candidates, probed
            )
            if selected is not None:
                _key, eff_rem, tier, queue, req = selected
                if queue.peek_request() is not req:
                    queue.remove_request(req)
                    queue.prepend_request(req)
                short[req.request_id] = eff_rem
                promoted_waiting_id = req.request_id
                if short_first:
                    st.a_promotions += 1
                    if tier == 0:
                        st.a_starve_promotions += 1

        if short:
            return LMCacheBudgetPlan(
                budget,
                short,
                sum(short.values()),
                budget_split=True,
                isolate_waiting=True,
                promoted_waiting_id=promoted_waiting_id,
                st=st,
                cap_tokens=cap_tokens,
                cap_max=cap_max,
                cap_idle_max=cap_idle_max,
                release_tokens=release_tokens,
                active_ultra_ids=active_ultra_ids,
                has_short_pressure=has_short_pressure,
            )

        if cap_tokens <= 0 and not short_first:
            return None

    # Regime 2: transition / main-lane handover step (running empty or <= short_limit).
    running_demand = sum(_remaining(r) for r in scheduler.running)
    ordered_ids: list[str] = []
    strict_ordered = False

    if short_first and scanned and running_demand < budget:
        avail_budget = budget - running_demand
        sorted_items = sorted(scanned, key=lambda x: x[0])
        track_submitted = _env_int("VLLM_ILU_PREFILL_LOOKUP_SUBMIT", 0) > 0
        ahead_budget = max(1, _lookup_ahead())
        ahead_used = 0
        cum_compute = 0
        sim_ultra = len(active_ultra_ids)
        sim_long_admits = 0
        chosen_by_queue: tuple[list, list] = ([], [])
        seen_waiting_queue = False
        max_pack = max(
            1, getattr(scheduler, "max_num_active_reqs", 32) - len(scheduler.running)
        )

        for item in sorted_items:
            if len(ordered_ids) >= max_pack or cum_compute >= avail_budget:
                break
            pkey, eff_rem, _s_idx, q_idx, queue, res_hint, async_hint, req = item
            if not _allowed_by_b(req, eff_rem, sim_ultra):
                continue
            # Avoid burst-admitting >2 main-lane (>short_limit) async loads in one
            # handover step, which would bloat #44560 reserved_blocks.
            if eff_rem > short_limit and sim_long_admits >= 2:
                continue

            # Check/drive lookup resolution.
            is_resolved = res_hint
            is_async = async_hint
            if not is_resolved:
                if connector is None or scheduler.kv_cache_manager.enable_kv_cache_events:
                    is_resolved = True
                elif ahead_used < ahead_budget:
                    rid = req.request_id
                    if rid in probed:
                        hit = probed[rid]
                    else:
                        if track_submitted:
                            st.submitted.add(rid)
                        hit = _lookup_hit(scheduler, connector, req, st)
                    ahead_used += 1
                    if hit is not None:
                        is_resolved = True
                        replay = getattr(scheduler, "prefix_replay_tokens", 0)
                        eff_rem = max(1, req.num_tokens - max(0, hit - replay))
                        is_async = hit > replay
                        if not _allowed_by_b(req, eff_rem, sim_ultra):
                            continue
                        if eff_rem > short_limit and sim_long_admits >= 2:
                            continue

            if not is_resolved:
                # If we already scheduled real compute this step, stop before
                # letting a longer resolved request leapfrog an unresolved shorter one.
                if cum_compute > 0:
                    strict_ordered = True
                    break
                continue

            # scheduler visits skipped_waiting (q_idx=0) before waiting (q_idx=1);
            # do not append a skipped_waiting item after a waiting item in the same step.
            if q_idx == 0 and seen_waiting_queue:
                strict_ordered = True
                break
            if q_idx == 1:
                seen_waiting_queue = True

            ordered_ids.append(req.request_id)
            chosen_by_queue[q_idx].append(req)
            st.a_promotions += 1
            if pkey[0] == 0:
                st.a_starve_promotions += 1
            if eff_rem > short_limit:
                sim_long_admits += 1
            if (
                cap_tokens > 0
                and req not in inflight
                and eff_rem >= cap_tokens
                and eff_rem > release_tokens
            ):
                sim_ultra += 1
            if not is_async:
                cum_compute += eff_rem

        # Place chosen candidates at the front of their respective queues in exact order.
        for q_idx, queue in enumerate((scheduler.skipped_waiting, scheduler.waiting)):
            chosen = chosen_by_queue[q_idx]
            if not chosen:
                continue
            for req in reversed(chosen):
                queue.remove_request(req)
                queue.prepend_request(req)

    return LMCacheBudgetPlan(
        budget,
        (),
        0,
        budget_split=False,
        isolate_waiting=False,
        ordered_ids=ordered_ids,
        strict_ordered=strict_ordered,
        st=st,
        cap_tokens=cap_tokens,
        cap_max=cap_max,
        cap_idle_max=cap_idle_max,
        release_tokens=release_tokens,
        active_ultra_ids=active_ultra_ids,
        has_short_pressure=has_short_pressure,
    )
