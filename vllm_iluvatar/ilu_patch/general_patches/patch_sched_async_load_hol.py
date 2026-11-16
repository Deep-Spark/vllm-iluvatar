# SPDX-License-Identifier: Apache-2.0
"""Let in-flight async KV loads pass an async load their reservation refused.

Corresponds to: vllm/v1/core/sched/scheduler.py
Target   : Scheduler.schedule (waiting-queue loop)
Baseline : v0.30.1rc0 (async-load reservation gate from community #44560)
Why      : #44560 admits an async KV load only if it fits in (free blocks -
           blocks still reserved by in-flight prefills), then ``break``s the
           waiting loop when it does not. In-flight async loads wait in
           ``skipped_waiting`` as WAITING_FOR_REMOTE_KVS and are promoted only
           when that loop reaches them. If they sit behind the refused load,
           their finished receives are never promoted, they never release the
           reservation, and the refused load is refused again every step: the
           engine spins on empty steps with every waiting request frozen; no
           LOOKUP is polled, so no deadline fires either. Hit on DSv4.1 P/D
           prefill with LMCache when a large request's lookup resolved right
           after small loads were admitted behind it. The same holds for loads
           already promoted back to WAITING: they keep their reservation until
           their remaining prefill is scheduled (seen with early LMCache
           lookups: a 400K-token load refused by an 800K load queued behind).
How      : when the reservation refuses an async load while an in-flight
           load is queued behind it (finished but not promoted, or promoted
           and waiting to compute), skip the refused request
           instead of breaking, and for the rest of that step only let
           requests that already hold a reservation (``_inflight_prefills``)
           through. New work keeps its FCFS place behind the refused request.
           Applied on top of the rewritten source that ``patch_pp_spec_schedule``
           and ``patch_prefill_budget`` keep on the function.
Scope    : async-load KV connectors (LMCache MP, NIXL consumer).
           ``VLLM_ILU_SCHED_ASYNC_LOAD_HOL=0`` disables it.
Remove   : when community stops breaking the waiting loop on a reservation
           refusal that in-flight loads behind it could resolve.
"""

from __future__ import annotations

import inspect
import os
import textwrap
import time
from typing import Any

from vllm.logger import init_logger

from vllm_iluvatar.ilu_patch import general_patch as p

logger = init_logger("vllm." + __name__)

_MARK = "_ilu_async_load_hol_patched"
_LOG_INTERVAL_S = 10.0

# Dedented ``Scheduler.schedule`` source fragments.
_OLD_LOOP_HEAD = """\
    if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
        step_skipped_waiting = create_request_queue(self.policy)
"""
_NEW_LOOP_HEAD = """\
    if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
        step_skipped_waiting = create_request_queue(self.policy)
        ilu_refused_async_load = False
"""

_OLD_AFTER_PROMOTE = """\
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue

            if (
                request.num_stale_output_tokens > 0
"""
_NEW_AFTER_PROMOTE = """\
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue

            if ilu_refused_async_load and request not in self._inflight_prefills:
                # New work stays behind the refused async load.
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue

            if (
                request.num_stale_output_tokens > 0
"""

_OLD_REFUSED = """\
                if request.has_encoder_inputs:
                    self.encoder_cache_manager.free(request)
                break

            # KVTransfer: the connector uses this info to determine
"""
_NEW_REFUSED = """\
                if request.has_encoder_inputs:
                    self.encoder_cache_manager.free(request)
                if (
                    load_kv_async
                    and reserved_blocks > 0
                    and _ilu_has_finished_inflight_load(self)
                ):
                    # Loads holding the reservation may be queued behind this
                    # one; they must be promoted to ever release it.
                    _ilu_note_refused_async_load(self, request, reserved_blocks)
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    ilu_refused_async_load = True
                    continue
                break

            # KVTransfer: the connector uses this info to determine
"""

_REPLACEMENTS = (
    (_OLD_LOOP_HEAD, _NEW_LOOP_HEAD),
    (_OLD_AFTER_PROMOTE, _NEW_AFTER_PROMOTE),
    (_OLD_REFUSED, _NEW_REFUSED),
)


def register() -> None:
    if os.environ.get("VLLM_ILU_SCHED_ASYNC_LOAD_HOL", "1") == "0":
        return
    p.register_once(
        __name__,
        _apply,
        detail="in-flight async loads pass a reservation-refused async load",
    )


def has_finished_inflight_load(scheduler: Any) -> bool:
    """Whether a reservation holder is queued and could release it if let past.

    That is an in-flight async load that finished but is not promoted yet, or
    one already promoted back to WAITING to compute the rest of its prompt.
    Both keep their reservation until their prefill runs.
    """
    from vllm.v1.request import RequestStatus

    done = scheduler.finished_recving_kv_req_ids
    return any(
        req.status == RequestStatus.WAITING
        or (
            req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            and req.request_id in done
        )
        for req in scheduler._inflight_prefills
    )


def note_refused_async_load(scheduler: Any, request: Any, reserved: int) -> None:
    """Log (rate-limited) that in-flight loads were let past a refused load."""
    now = time.monotonic()
    if now - scheduler.__dict__.get("_ilu_hol_logged", 0.0) < _LOG_INTERVAL_S:
        return
    scheduler.__dict__["_ilu_hol_logged"] = now
    logger.info(
        "Async KV load for %s refused by %d reserved blocks; letting finished "
        "in-flight loads pass it",
        request.request_id,
        reserved,
    )


def patch_schedule_source(source: str) -> str:
    """Rewrite dedented ``Scheduler.schedule`` source; fail on drift."""
    for old, new in _REPLACEMENTS:
        if new in source:
            continue
        if source.count(old) != 1:
            raise RuntimeError(
                "Scheduler.schedule no longer matches patch_sched_async_load_hol.py; "
                "re-sync the patch"
            )
        source = source.replace(old, new, 1)
    return source


def _apply() -> None:
    from vllm.v1.core.sched import scheduler as sched_mod
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_iluvatar.ilu_patch.general_patches import patch_pp_spec_schedule

    # Compose with the PP / remote replay rewrite regardless of general-plugin
    # discovery order; it must read the community source first.
    patch_pp_spec_schedule.register()
    if not getattr(Scheduler.schedule, "_ilu_pp_spec_schedule_patched", False):
        patch_pp_spec_schedule._apply()
    current = Scheduler.schedule
    if getattr(current, _MARK, False):
        return
    # Schedule rewriters keep their rewritten source on the function; an
    # exec'd function has no source file of its own.
    source = getattr(current, "_ilu_pp_spec_schedule_source", None)
    if source is None:
        source = textwrap.dedent(inspect.getsource(current))
    patched = patch_schedule_source(source)

    sched_mod._ilu_has_finished_inflight_load = has_finished_inflight_load
    sched_mod._ilu_note_refused_async_load = note_refused_async_load
    namespace: dict[str, object] = {}
    exec(patched, sched_mod.__dict__, namespace)  # noqa: S102
    schedule = namespace["schedule"]
    schedule.__dict__.update(current.__dict__)
    schedule._ilu_pp_spec_schedule_source = patched  # type: ignore[attr-defined]
    setattr(schedule, _MARK, True)
    Scheduler.schedule = schedule  # type: ignore[method-assign]
