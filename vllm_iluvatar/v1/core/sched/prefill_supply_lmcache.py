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
- Strategy B — two-dimensional concurrency control (compute vs in-flight DMA):
  1. Compute concurrency cap: ``VLLM_ILU_PREFILL_LONG_CAP_TOKENS`` (default 0,
     off; e.g. 256000) caps concurrent ultra-long *compute* prefills
     (``compute_rem >= cap_tokens``) to ``VLLM_ILU_PREFILL_LONG_CAP_MAX``
     (default 1) under non-ultra pressure, relaxing to
     ``VLLM_ILU_PREFILL_LONG_CAP_IDLE_MAX`` (default 2; 0 = no cap) when idle,
     and releasing the slot once remaining compute drops to
     ``<= VLLM_ILU_PREFILL_LONG_RELEASE_TOKENS`` (default 0 -> ``short_limit``).
  2. Transient large async-load DMA gate: while ``LONG_CAP_TOKENS > 0``, large
     LMCache async KV loads (``async_load_toks >= VLLM_ILU_PREFILL_ASYNC_LOAD_CAP_TOKENS``,
     default 131072) currently in ``WAITING_FOR_REMOTE_KVS`` are bounded to
     ``VLLM_ILU_PREFILL_ASYNC_LOAD_CAP_MAX`` (default 1; idle 2) so duplicate
     same-prefix requests wait 1–2 steps to share the freshly cached GPU blocks
     instead of multiplying HBM & ``#44560`` reservations, and release the DMA
     gate immediately once the transfer completes!
- Strategy A — tier-aware waiting short-first + zero-bubble fast lane + smooth aging:
  ``VLLM_ILU_PREFILL_SHORT_FIRST=1`` (default 0, off):
  1. Distinguishes zero-copy GPU APC hits from LMCache H2D DMA hits via
     ``sched_cost = compute_rem + (1024 + ASYNC_LOAD_WEIGHT * async_load_toks)``
     (``VLLM_ILU_PREFILL_ASYNC_LOAD_WEIGHT``, default 0.05) and eliminates the
     ``probes < 4`` local-cache blind spot by caching ``local_hit`` from early
     lookup submissions.
  2. Zero-bubble Regime 1 fast lane: initiating an async KV load
     (``num_new_tokens = 0``) in Regime 1 reserves ``0`` compute tokens from the
     running long prefill's budget (and can co-schedule alongside a ready local
     short request in the same step).
  3. Smooth linear aging (``VLLM_ILU_PREFILL_AGING_RATE``, default 2500 tok/s) +
     oldest-first FCFS anti-starvation ceiling at ``VLLM_ILU_PREFILL_STARVE_SEC``
     (default 120s).
"""

from __future__ import annotations

import collections
import os
import time

_ASYNC_LOOKUP_CONNECTORS = frozenset(
    {"LMCacheMPConnector", "LMCacheMPConnectorUpstream"}
)
_MIN_SHORT_WORK_CHUNKS = 4
_ASYNC_STEP_PENALTY_TOKENS = 1024
_DEFAULT_ASYNC_LOAD_CAP_TOKENS = 131072


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
        promoted_waiting_ids: tuple[str, ...] | list[str] = (),
        async_init_ids: set[str] | tuple[str, ...] = (),
        ordered_ids: list[str] | None = None,
        strict_ordered: bool = False,
        st=None,
        cap_tokens: int = 0,
        cap_max: int = 1,
        cap_idle_max: int = 2,
        release_tokens: int = 0,
        active_ultra_ids: set[str] | None = None,
        has_short_pressure: bool = False,
        async_cap_tokens: int = 0,
        async_cap_max: int = 1,
        async_cap_idle_max: int = 2,
        active_large_dma_ids: set[str] | None = None,
    ):
        self.budget_split = bool(budget_split)
        self.short_ids = set(short_ids)
        self.async_init_ids = set(async_init_ids)
        if self.budget_split:
            # The oldest long-running prefill retains >= half of the step budget.
            self.short_budget = min(budget // 2, short_work)
            self.long_budget = budget - self.short_budget
        else:
            self.short_budget = budget
            self.long_budget = budget
        self.short_used = self.long_used = 0
        self.isolate_waiting = bool(isolate_waiting)
        self._promoted_queue = collections.deque(promoted_waiting_ids)
        self.promoted_waiting_id = (
            self._promoted_queue[0] if self._promoted_queue else None
        )
        self.ordered_queue = collections.deque(ordered_ids or ())
        self.strict_ordered = bool(strict_ordered)
        self.st = st
        self.cap_tokens = int(cap_tokens)
        self.cap_max = int(cap_max)
        self.cap_idle_max = int(cap_idle_max)
        self.release_tokens = int(release_tokens)
        self.active_ultra_ids = set(active_ultra_ids or ())
        self.has_short_pressure = bool(has_short_pressure)
        self.async_cap_tokens = int(async_cap_tokens)
        self.async_cap_max = int(async_cap_max)
        self.async_cap_idle_max = int(async_cap_idle_max)
        self.active_large_dma_ids = set(active_large_dma_ids or ())
        self._last_reject_reason: str | None = None

    def has_pending_async_init(self) -> bool:
        return bool(self.async_init_ids)

    def limit(self, rid: str, tokens: int) -> int:
        if not self.budget_split:
            return tokens
        if rid in self.async_init_ids:
            # Allow passing the pre-lookup isolate_waiting gate even when
            # token_budget == 0 (long prefill took full 4096); if lookup
            # unexpectedly downgraded to local compute, can_admit_waiting
            # defers it cleanly without consuming compute budget.
            return 1
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

    def _current_promoted(self, scheduler) -> str | None:
        while self._promoted_queue:
            cand = self._promoted_queue[0]
            if self._in_waiting_queues(scheduler, cand):
                self.promoted_waiting_id = cand
                return cand
            self._promoted_queue.popleft()
        self.promoted_waiting_id = None
        return None

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

    def _effective_async_cap(self) -> int | None:
        if self.async_cap_tokens <= 0:
            return None
        if self.has_short_pressure:
            return self.async_cap_max
        if self.async_cap_idle_max > 0:
            return self.async_cap_idle_max
        return None

    def _quick_eff_rem(self, scheduler, request) -> int:
        if request.num_computed_tokens:
            return _remaining(request)
        if self.st is not None:
            replay = getattr(scheduler, "prefix_replay_tokens", 0)
            loc = self.st.local_hits.get(request.request_id, 0)
            tot = self.st.resolved_hits.get(request.request_id, 0)
            best = max(loc, tot)
            if best > 0:
                return max(1, request.num_tokens - max(0, best - replay))
        return request.num_tokens

    def _quick_async_load(self, request) -> int:
        if request.num_computed_tokens or self.st is None:
            return 0
        loc = self.st.local_hits.get(request.request_id, 0)
        tot = self.st.resolved_hits.get(request.request_id)
        if tot is None:
            return 0
        return max(0, tot - loc)

    def can_admit_waiting(
        self, scheduler, request, num_computed_tokens: int, load_kv_async: bool
    ) -> bool:
        self._last_reject_reason = None
        rid = request.request_id

        # If a request was co-promoted purely to kick off a 0-compute async DMA
        # in Regime 1, do not admit it for synchronous compute if its hit shrank;
        # and when token_budget == 0, only async-init requests may enter.
        if rid in self.async_init_ids and not load_kv_async:
            self.async_init_ids.discard(rid)
            if self._promoted_queue and self._promoted_queue[0] == rid:
                self._promoted_queue.popleft()
            self._last_reject_reason = "async_init_downgraded"
            return False
        if self.async_init_ids and rid not in self.async_init_ids and rid not in self.short_ids:
            self._last_reject_reason = "awaiting_async_init"
            return False

        # Strategy A ordered sequence check (when transition-step ordering is active).
        if not self.budget_split:
            target = self._current_target(scheduler)
            if target is not None:
                if rid != target:
                    self._last_reject_reason = "ordered"
                    return False
            elif self.strict_ordered:
                self._last_reject_reason = "strict_ordered"
                return False

        # Never block a request that already holds in-flight reserved KV blocks.
        inflight = getattr(scheduler, "_inflight_prefills", ())
        if request not in inflight:
            # 1. Strategy B1: ultra-long compute concurrency cap.
            if self.cap_tokens > 0:
                rem = max(0, request.num_tokens - num_computed_tokens)
                if rem >= self.cap_tokens:
                    cap = self._effective_cap()
                    if cap is not None and len(self.active_ultra_ids) >= cap:
                        if self.st is not None:
                            self.st.b_cap_skips += 1
                        self._last_reject_reason = "b_cap"
                        return False

            # 2. Strategy B2: transient large async-load DMA gate.
            # Prevents multiple large LMCache async loads (e.g. duplicate 392k
            # prefix hits before the first transfer caches its blocks) from
            # multiplying HBM allocations & #44560 reservations simultaneously.
            if self.async_cap_tokens > 0 and load_kv_async:
                loc = (
                    self.st.local_hits.get(rid, 0)
                    if self.st is not None
                    else 0
                )
                ext_load = max(0, num_computed_tokens - loc)
                if ext_load >= self.async_cap_tokens:
                    acap = self._effective_async_cap()
                    if acap is not None and len(self.active_large_dma_ids) >= acap:
                        if self.st is not None:
                            self.st.b_async_cap_skips += 1
                        self._last_reject_reason = "b_async_cap"
                        return False

        return True

    def on_admit_waiting(
        self, scheduler, request, num_computed_tokens: int, load_kv_async: bool
    ) -> None:
        rid = request.request_id
        self.async_init_ids.discard(rid)
        if self._promoted_queue and self._promoted_queue[0] == rid:
            self._promoted_queue.popleft()
            self.promoted_waiting_id = (
                self._promoted_queue[0] if self._promoted_queue else None
            )
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
        if self.async_cap_tokens > 0 and load_kv_async:
            loc = self.st.local_hits.get(rid, 0) if self.st is not None else 0
            if max(0, num_computed_tokens - loc) >= self.async_cap_tokens:
                self.active_large_dma_ids.add(rid)

    def should_skip_waiting(self, scheduler, rid: str) -> bool:
        # 1. Promoted short / async-init targets in Regime 1.
        target = self._current_promoted(scheduler)
        if target is not None and rid != target:
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

        # 4. Strategy B rejected an ultra-long compute or large async-load candidate:
        #    skip it if there is an eligible request behind it to try.
        if (
            self._last_reject_reason in ("b_cap", "b_async_cap")
            and not self.isolate_waiting
        ):
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
                    eff_ok = (
                        self.cap_tokens <= 0
                        or self._quick_eff_rem(scheduler, other) < self.cap_tokens
                        or (
                            self._effective_cap() is not None
                            and len(self.active_ultra_ids) < self._effective_cap()
                        )
                    )
                    async_ok = (
                        self.async_cap_tokens <= 0
                        or self._quick_async_load(other) < self.async_cap_tokens
                        or (
                            self._effective_async_cap() is not None
                            and len(self.active_large_dma_ids)
                            < self._effective_async_cap()
                        )
                    )
                    if eff_ok and async_ok:
                        return True
            return False

        return False


def _lookup_hit(scheduler, connector, request, st=None):
    """The connector call the waiting loop makes on a visit."""
    _, local, _, _ = scheduler._get_local_prefix_cache_hit(request)
    local -= local % scheduler.block_size
    if st is not None:
        st.local_hits[request.request_id] = local
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


def _sched_cost(
    compute_rem: int, async_load_toks: int, async_weight: float
) -> float:
    if async_load_toks > 0:
        return (
            float(compute_rem)
            + _ASYNC_STEP_PENALTY_TOKENS
            + async_weight * float(async_load_toks)
        )
    return float(compute_rem)


def _priority_key(
    sched_cost: float,
    is_async_load: bool,
    request,
    st,
    now: float,
    starve_sec: float,
    aging_rate: float,
) -> tuple[int, float, int, float, str]:
    """Compute tier-aware short-first priority key with smooth aging & FCFS bailout.

    - ``tier == 0`` (waited >= ``starve_sec``): ordered strictly oldest-first by
      ``arrival_time`` so aged ultra-long requests cannot be jumped anymore by
      younger aged medium requests under heavy steady-state queues.
    - ``tier == 1`` (normal): ordered by ``max(0, sched_cost - aging_rate * wait_sec)``,
      breaking ties in favor of locally ready work (``is_async_load=0``) before
      async DMA loads (``is_async_load=1``), then ``arrival_time``.
    """
    rid_str = str(request.request_id)
    raw_arr = getattr(request, "arrival_time", None)
    if raw_arr is None:
        arr = st.first_seen.setdefault(request.request_id, now)
    else:
        arr = float(raw_arr)
    wait_sec = max(0.0, now - arr)
    async_flag = 1 if is_async_load else 0
    if starve_sec > 0.0 and wait_sec >= starve_sec:
        return (0, arr, async_flag, sched_cost, rid_str)
    aged_score = max(0.0, sched_cost - aging_rate * wait_sec)
    return (1, aged_score, async_flag, arr, rid_str)


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
                float(_remaining(r)), False, r, st, now, starve_sec, aging_rate
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
    if st.local_hits:
        for k in [k for k in st.local_hits if k not in active_rids]:
            st.local_hits.pop(k, None)
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
        promoted_waiting_ids=(promoted_waiting_id,) if promoted_waiting_id else (),
        st=st,
    )


def _estimate_candidate(
    scheduler, lmc_child, st, request, short_limit: int, probes: int
) -> tuple[int, int, bool, bool, int]:
    """Return ``(compute_rem, async_load_toks, resolved_hint, async_hint, probes)``."""
    rid = request.request_id
    replay = getattr(scheduler, "prefix_replay_tokens", 0)
    inflight = getattr(scheduler, "_inflight_prefills", ())

    # Already completed remote KV load (or preempted with computed tokens):
    # blocks are resident in GPU HBM now -> 0 async load!
    if request.num_computed_tokens or getattr(
        request, "skip_reading_prefix_cache", False
    ):
        return _remaining(request), 0, True, False, probes

    total_ext_hit = _passive_adapter_hit(lmc_child, rid, st)

    # Reuse exact local hit recorded during early _lookup_hit submission first
    # (eliminates the 4-probe blind spot for up to LOOKUP_SUBMIT=16 requests).
    if rid in st.local_hits:
        local_cached = st.local_hits[rid]
    elif (
        probes < 4
        and (
            total_ext_hit is not None
            or st.key(request, short_limit, replay) in st.seen
        )
        and not scheduler.kv_cache_manager.enable_kv_cache_events
    ):
        probes += 1
        if hasattr(scheduler, "_get_local_prefix_cache_hit"):
            _, local_cached, _, _ = scheduler._get_local_prefix_cache_hit(request)
        else:
            _, local_cached, _ = scheduler.kv_cache_manager.get_computed_blocks(request)
        local_cached -= local_cached % getattr(scheduler, "block_size", 64)
        st.local_hits[rid] = local_cached
    else:
        local_cached = 0

    best_hit = max(local_cached, total_ext_hit or 0)
    if best_hit > 0:
        compute_rem = max(1, request.num_tokens - max(0, best_hit - replay))
    else:
        compute_rem = request.num_tokens

    raw_async = (
        max(0, total_ext_hit - local_cached) if total_ext_hit is not None else 0
    )
    async_hint = bool(raw_async > 0 and (local_cached + raw_async) > replay)
    async_load_toks = raw_async if async_hint else 0
    resolved = total_ext_hit is not None or request in inflight
    return compute_rem, async_load_toks, resolved, async_hint, probes


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
    async_weight = _env_float("VLLM_ILU_PREFILL_ASYNC_LOAD_WEIGHT", 0.05)
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

    # Strategy B2 transient large async-load DMA gate (active whenever B is on).
    async_cap_tokens = (
        _env_int(
            "VLLM_ILU_PREFILL_ASYNC_LOAD_CAP_TOKENS", _DEFAULT_ASYNC_LOAD_CAP_TOKENS
        )
        if cap_tokens > 0
        else 0
    )
    async_cap_max = max(1, _env_int("VLLM_ILU_PREFILL_ASYNC_LOAD_CAP_MAX", 1))
    async_cap_idle_max = _env_int("VLLM_ILU_PREFILL_ASYNC_LOAD_CAP_IDLE_MAX", 2)

    inflight = getattr(scheduler, "_inflight_prefills", ())

    # Promote finished WAITING_FOR_REMOTE_KVS requests within the 128-entry scan
    # window up-front so active DMA / ultra-long slot counts reflect transfers
    # that already completed this step (leaving entries >128 to the waiting loop's
    # HOL bypass path). When any remote KV load completes, newly cached GPU
    # blocks enter the local prefix hash table, so clear cached local_hits<total
    # entries so sibling same-prefix requests refresh their local GPU hit!
    done_recvs = getattr(scheduler, "finished_recving_kv_req_ids", ())
    if done_recvs:
        if st.local_hits:
            st.local_hits.clear()
        for n_sk, request in enumerate(list(scheduler.skipped_waiting)):
            if n_sk >= 128:
                break
            if (
                request.status.name == "WAITING_FOR_REMOTE_KVS"
                and request.request_id in done_recvs
            ):
                scheduler._try_promote_blocked_waiting_request(request)

    active_ultra_ids: set[str] = set()
    active_large_dma_ids: set[str] = set()
    if cap_tokens > 0:
        for r in (*scheduler.running, *inflight):
            rem = _remaining(r)
            if (
                r.request_id in st.ultra_ids or rem >= cap_tokens
            ) and rem > release_tokens:
                active_ultra_ids.add(r.request_id)
                st.ultra_ids.add(r.request_id)
            if (
                async_cap_tokens > 0
                and r.status.name == "WAITING_FOR_REMOTE_KVS"
                and (r.num_computed_tokens - st.local_hits.get(r.request_id, 0))
                >= async_cap_tokens
            ):
                active_large_dma_ids.add(r.request_id)

    has_short_pressure = False
    if cap_tokens > 0:
        has_short_pressure = any(
            0 < _remaining(r) < cap_tokens and r.request_id not in st.ultra_ids
            for r in scheduler.running
        )

    # Scan waiting queues and inspect candidates.
    scanned = []
    probes = 0
    seq_idx = 0
    for q_idx, queue in enumerate((scheduler.skipped_waiting, scheduler.waiting)):
        for n, request in enumerate(list(queue)):
            if n >= 128:
                break
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

            compute_rem, async_load_toks, resolved_hint, async_hint, probes = (
                _estimate_candidate(
                    scheduler, lmc_child, st, request, short_limit, probes
                )
            )
            if request.request_id in probed and probed[request.request_id] is not None:
                resolved_hint = True
            if (
                cap_tokens > 0
                and not has_short_pressure
                and compute_rem < cap_tokens
                and request.request_id not in st.ultra_ids
            ):
                has_short_pressure = True

            cost = _sched_cost(
                compute_rem,
                async_load_toks if short_first else 0,
                async_weight if short_first else 0.0,
            )
            pkey = _priority_key(
                cost, async_hint, request, st, now, starve_sec, aging_rate
            )
            scanned.append(
                (
                    pkey,
                    compute_rem,
                    async_load_toks,
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
    eff_async_cap: int | None = None
    if cap_tokens > 0:
        if has_short_pressure:
            eff_cap = cap_max
            eff_async_cap = async_cap_max
        else:
            if cap_idle_max > 0:
                eff_cap = cap_idle_max
            if async_cap_idle_max > 0:
                eff_async_cap = async_cap_idle_max

    def _allowed_by_b(
        request, compute_rem: int, async_load_toks: int, cur_ultra: int, cur_dma: int
    ) -> bool:
        if cap_tokens <= 0 or request in inflight:
            return True
        if compute_rem >= cap_tokens and eff_cap is not None and cur_ultra >= eff_cap:
            return False
        if (
            async_cap_tokens > 0
            and async_load_toks >= async_cap_tokens
            and eff_async_cap is not None
            and cur_dma >= eff_async_cap
        ):
            return False
        return True

    def _resolve_one_lane_candidate(candidates, cur_ultra: int, cur_dma: int):
        """Resolve at most LOOKUP_AHEAD candidates and return the first B-eligible one."""
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        if connector is None or scheduler.kv_cache_manager.enable_kv_cache_events:
            return candidates[0]
        track_sub = _env_int("VLLM_ILU_PREFILL_LOOKUP_SUBMIT", 0) > 0
        limit = max(1, _lookup_ahead())
        replay = getattr(scheduler, "prefix_replay_tokens", 0)
        for cand in candidates[:limit]:
            _k, c_rem, a_load, tier, q_idx, queue, is_async, req = cand
            if req.num_computed_tokens or getattr(
                req, "skip_reading_prefix_cache", False
            ):
                if _allowed_by_b(req, c_rem, 0, cur_ultra, cur_dma):
                    return (_k, c_rem, 0, tier, q_idx, queue, False, req)
                continue
            rid = req.request_id
            if rid in probed and probed[rid] is not None:
                ext = probed[rid]
                loc = st.local_hits.get(rid, 0)
                tot = loc + ext
            elif rid in st.resolved_hits:
                tot = st.resolved_hits[rid]
                loc = st.local_hits.get(rid, 0)
            else:
                if track_sub:
                    st.submitted.add(rid)
                ext = _lookup_hit(scheduler, connector, req, st)
                if ext is None:
                    continue
                loc = st.local_hits.get(rid, 0)
                tot = loc + ext
            best = max(loc, tot)
            c_rem = max(1, req.num_tokens - max(0, best - replay)) if best > 0 else req.num_tokens
            raw_a = max(0, tot - loc)
            is_async = bool(raw_a > 0 and tot > replay)
            a_load = raw_a if is_async else 0
            if c_rem <= short_limit and _allowed_by_b(
                req, c_rem, a_load, cur_ultra, cur_dma
            ):
                return (_k, c_rem, a_load, tier, q_idx, queue, is_async, req)
        return None

    short = {
        r.request_id: _remaining(r)
        for r in scheduler.running
        if 0 < _remaining(r) <= short_limit
    }
    long_exists = any(_remaining(r) > short_limit for r in scheduler.running)

    # Regime 1: running already holds > short_limit prefill work -> share chunk budget
    # exclusively with <= short_limit requests, with ZERO compute-budget bubble on
    # async KV load initiation!
    if long_exists:
        lane_candidates = []
        for item in scanned:
            (
                pkey,
                compute_rem,
                async_load_toks,
                s_idx,
                q_idx,
                queue,
                _res,
                is_async,
                req,
            ) = item
            if compute_rem <= 0 or compute_rem > short_limit:
                continue
            if not _allowed_by_b(
                req,
                compute_rem,
                async_load_toks,
                len(active_ultra_ids),
                len(active_large_dma_ids),
            ):
                continue
            sort_key = pkey if short_first else (compute_rem, s_idx)
            lane_candidates.append(
                (
                    sort_key,
                    compute_rem,
                    async_load_toks,
                    pkey[0],
                    q_idx,
                    queue,
                    is_async,
                    req,
                )
            )

        promoted_ids: list[str] = []
        async_init_ids: set[str] = set()
        sim_dma = len(active_large_dma_ids)
        chosen_by_q: tuple[list, list] = ([], [])

        selected = _resolve_one_lane_candidate(
            lane_candidates, len(active_ultra_ids), sim_dma
        )
        if selected is not None:
            _k, c_rem, a_load, tier, q_idx, _queue, is_async, req = selected
            promoted_ids.append(req.request_id)
            chosen_by_q[q_idx].append(req)
            if short_first:
                st.a_promotions += 1
                if tier == 0:
                    st.a_starve_promotions += 1

            if short_first and is_async:
                # Zero-bubble async initiation: initiating WAITING_FOR_REMOTE_KVS
                # consumes 0 compute tokens this step! Keep full 4096 budget for
                # the running long prefill, OR co-promote one ready synchronous
                # <= short_limit candidate if one is waiting right now.
                async_init_ids.add(req.request_id)
                if async_cap_tokens > 0 and a_load >= async_cap_tokens:
                    sim_dma += 1
                sync_cands = [
                    c
                    for c in lane_candidates
                    if c[-1] is not req
                    and not c[6]  # not is_async
                    and (c[4] >= q_idx)  # preserve skipped_waiting -> waiting visit order
                ]
                sync_sel = _resolve_one_lane_candidate(
                    sync_cands, len(active_ultra_ids), sim_dma
                )
                if sync_sel is not None and not sync_sel[6]:
                    _k2, c_rem2, _a2, tier2, q_idx2, _q2, _isa2, req2 = sync_sel
                    if q_idx2 >= q_idx:
                        promoted_ids.append(req2.request_id)
                        chosen_by_q[q_idx2].append(req2)
                        short[req2.request_id] = c_rem2
                        st.a_promotions += 1
                        if tier2 == 0:
                            st.a_starve_promotions += 1
            else:
                # Synchronous ready short request (or legacy non-short_first mode):
                # reserve bounded short_budget for immediate compute.
                short[req.request_id] = c_rem

        for q_idx, queue in enumerate((scheduler.skipped_waiting, scheduler.waiting)):
            for req in reversed(chosen_by_q[q_idx]):
                if queue.peek_request() is not req:
                    queue.remove_request(req)
                    queue.prepend_request(req)

        if short or async_init_ids:
            return LMCacheBudgetPlan(
                budget,
                short,
                sum(short.values()),
                budget_split=True,
                isolate_waiting=True,
                promoted_waiting_ids=promoted_ids,
                async_init_ids=async_init_ids,
                st=st,
                cap_tokens=cap_tokens,
                cap_max=cap_max,
                cap_idle_max=cap_idle_max,
                release_tokens=release_tokens,
                active_ultra_ids=active_ultra_ids,
                has_short_pressure=has_short_pressure,
                async_cap_tokens=async_cap_tokens,
                async_cap_max=async_cap_max,
                async_cap_idle_max=async_cap_idle_max,
                active_large_dma_ids=active_large_dma_ids,
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
        sim_dma = len(active_large_dma_ids)
        sim_long_admits = 0
        chosen_by_queue: tuple[list, list] = ([], [])
        seen_waiting_queue = False
        max_pack = max(
            1, getattr(scheduler, "max_num_active_reqs", 32) - len(scheduler.running)
        )
        replay = getattr(scheduler, "prefix_replay_tokens", 0)

        for item in sorted_items:
            if len(ordered_ids) >= max_pack or cum_compute >= avail_budget:
                break
            (
                pkey,
                compute_rem,
                async_load_toks,
                _s_idx,
                q_idx,
                queue,
                res_hint,
                async_hint,
                req,
            ) = item
            if not _allowed_by_b(
                req, compute_rem, async_load_toks, sim_ultra, sim_dma
            ):
                continue
            if compute_rem > short_limit and sim_long_admits >= 2:
                continue

            # Check/drive lookup resolution.
            is_resolved = res_hint
            is_async = async_hint
            if not is_resolved:
                if connector is None or scheduler.kv_cache_manager.enable_kv_cache_events:
                    is_resolved = True
                elif ahead_used < ahead_budget:
                    rid = req.request_id
                    if rid in probed and probed[rid] is not None:
                        ext = probed[rid]
                    else:
                        if track_submitted:
                            st.submitted.add(rid)
                        ext = _lookup_hit(scheduler, connector, req, st)
                    ahead_used += 1
                    if ext is not None:
                        is_resolved = True
                        loc = st.local_hits.get(rid, 0)
                        tot = loc + ext
                        best = max(loc, tot)
                        compute_rem = (
                            max(1, req.num_tokens - max(0, best - replay))
                            if best > 0
                            else req.num_tokens
                        )
                        raw_a = max(0, tot - loc)
                        is_async = bool(raw_a > 0 and tot > replay)
                        async_load_toks = raw_a if is_async else 0
                        if not _allowed_by_b(
                            req, compute_rem, async_load_toks, sim_ultra, sim_dma
                        ):
                            continue
                        if compute_rem > short_limit and sim_long_admits >= 2:
                            continue

            if not is_resolved:
                if cum_compute > 0:
                    strict_ordered = True
                    break
                continue

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
            if compute_rem > short_limit:
                sim_long_admits += 1
            if (
                cap_tokens > 0
                and req not in inflight
                and compute_rem >= cap_tokens
                and compute_rem > release_tokens
            ):
                sim_ultra += 1
            if (
                async_cap_tokens > 0
                and req not in inflight
                and is_async
                and async_load_toks >= async_cap_tokens
            ):
                sim_dma += 1
            if not is_async:
                cum_compute += compute_rem

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
        async_cap_tokens=async_cap_tokens,
        async_cap_max=async_cap_max,
        async_cap_idle_max=async_cap_idle_max,
        active_large_dma_ids=active_large_dma_ids,
    )
