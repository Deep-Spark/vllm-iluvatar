# SPDX-License-Identifier: Apache-2.0
"""Keep PP+spec schedule rows wide enough for ``drafts + bonus`` logits.

Corresponds to: vllm/v1/core/sched/scheduler.py
Target   : Scheduler.schedule, Scheduler.update_draft_token_ids
Baseline : v0.30.0rc0
Why      : two PP+spec shapes break
           ``assert (num_scheduled_tokens_np >= num_logits).all()`` in
           ``prepare_inputs`` (num_logits = drafts + bonus):

           1. PP batch-queue advances ``num_computed_tokens`` before the
              sampled token is appended, so ``num_new_tokens`` can go
              *negative*. Community only skips ``== 0``.
           2. Healthy decode keeps ``computed == tokens - 1`` so algebra
              schedules ``drafts + 1``. When ``computed == tokens`` and
              drafts attach, algebra schedules exactly ``len(drafts)`` —
              too short for the bonus row.

              Widening ``num_new`` (bonus bump) is wrong: with
              ``computed == tokens`` it writes ``last_sampled`` into a
              *new* sequence position and duplicates text (``419419``).
              Trimming drafts traps ``computed == tokens`` forever.
              Rolling ``computed`` back by 1 restores the natural
              ``tokens - 1`` shape. On V2 the PP ``new_token_ids`` payload
              is unused (sampled tokens ride the GPU broadcast), so this
              does not re-append the last token to the client.

How      : rebind ``schedule`` with idle checks widened to ``<= 0``; in
           ``update_draft_token_ids``, roll ``computed`` back one when
           equal to ``tokens``, and drop drafts when PP is still ahead
           (``computed > tokens``).
Scope    : V2 scheduler under pipeline parallelism + speculative decode.
Remove   : when community schedule skips ``num_new_tokens <= 0`` and always
           keeps ``computed == tokens - 1`` when drafts are attached.
"""

from __future__ import annotations

import inspect
import textwrap
from collections.abc import Callable
from functools import wraps
from typing import Any

from vllm_iluvatar.ilu_patch import general_patch as p

_IDLE_CHECK = "if num_new_tokens == 0:"
_IDLE_CHECK_FIXED = "if num_new_tokens <= 0:"


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="PP+spec schedule bonus-slot and negative-budget guards",
    )


def _patched_schedule_source(source: str) -> str:
    """Widen idle ``num_new_tokens == 0`` checks to ``<= 0``."""
    if _IDLE_CHECK not in source:
        raise RuntimeError(
            "Scheduler.schedule no longer contains "
            f"{_IDLE_CHECK!r}; re-sync patch_pp_spec_schedule.py"
        )
    return source.replace(_IDLE_CHECK, _IDLE_CHECK_FIXED)


def _align_computed_for_drafts(scheduler: Any, draft_token_ids: Any) -> None:
    """Restore drafts+bonus schedule shape; drop drafts if PP is still ahead.

    When ``computed == tokens`` and nothing is in flight, roll ``computed``
    back one so the next schedule is ``num_sched = drafts + 1``.

    When ``computed == tokens`` but ``num_in_flight_tokens > 0``, a sample
    for that position is still in the PP queue. Rewinding would schedule
    another verify from the same token and emit it twice. Drop the drafts
    instead of rewinding.

    When ``computed > tokens``, PP is still ahead — drop drafts.
    """
    for req_id in draft_token_ids.req_ids:
        request = scheduler.requests.get(req_id)
        if request is None or request.is_finished() or not request.spec_token_ids:
            continue
        pending = request.num_tokens + request.num_output_placeholders
        in_flight = int(getattr(request, "num_in_flight_tokens", 0) or 0)
        if request.num_computed_tokens > pending or (
            request.num_computed_tokens == pending and in_flight > 0
        ):
            request.spec_token_ids = []
        elif request.num_computed_tokens == pending:
            request.num_computed_tokens -= 1


def _wrap_update_draft_token_ids(original: Callable[..., None]) -> Callable[..., None]:
    @wraps(original)
    def update_draft_token_ids(self, draft_token_ids) -> None:
        original(self, draft_token_ids)
        _align_computed_for_drafts(self, draft_token_ids)

    return update_draft_token_ids


def _patch_update_draft_token_ids(Scheduler: Any) -> None:
    if getattr(Scheduler.update_draft_token_ids, "_ilu_pp_spec_draft_patched", False):
        return
    wrapped = _wrap_update_draft_token_ids(Scheduler.update_draft_token_ids)
    wrapped._ilu_pp_spec_draft_patched = True  # type: ignore[attr-defined]
    Scheduler.update_draft_token_ids = wrapped  # type: ignore[method-assign]


def _apply() -> None:
    from vllm.v1.core.sched import scheduler as sched_mod
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler.schedule, "_ilu_pp_spec_schedule_patched", False):
        _patch_update_draft_token_ids(Scheduler)
        return

    # Read the community body while it still has a source file. An exec'd
    # replacement is not inspect-able, so never getsource the patched method.
    source = textwrap.dedent(inspect.getsource(Scheduler.schedule))
    patched = _patched_schedule_source(source)
    namespace: dict[str, object] = {}
    exec(patched, sched_mod.__dict__, namespace)  # noqa: S102
    schedule = namespace["schedule"]
    schedule._ilu_pp_spec_schedule_patched = True  # type: ignore[attr-defined]
    schedule._ilu_pp_spec_schedule_source = patched  # type: ignore[attr-defined]
    Scheduler.schedule = schedule  # type: ignore[method-assign]
    _patch_update_draft_token_ids(Scheduler)
