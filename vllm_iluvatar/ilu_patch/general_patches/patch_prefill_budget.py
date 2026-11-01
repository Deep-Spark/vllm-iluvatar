# SPDX-License-Identifier: Apache-2.0
"""Install producer chunk sharing without changing the vLLM source tree."""

import os
from functools import wraps

from vllm_iluvatar.ilu_patch import general_patch as p
from vllm_iluvatar.v1.core.sched import prefill_supply


def register() -> None:
    # D normally has no PP partition. Do not even wrap its scheduler methods.
    if os.environ.get("VLLM_PP_LAYER_PARTITION"):
        p.register_once(__name__, _apply, detail="PP producer prefill chunk sharing")


def patch_schedule_source(source: str) -> str:
    replacements = (
        (
            "    # First, schedule the RUNNING requests.\n",
            (
                "    supply_plan = _ilu_prefill_supply.plan(self, token_budget)\n\n"
                "    # First, schedule the RUNNING requests.\n"
            ),
        ),
        (
            "        # Make sure the input position does not exceed the max model len.\n",
            (
                "        if supply_plan is not None:\n"
                "            num_new_tokens = supply_plan.limit(request.request_id, num_new_tokens)\n\n"
                "        # Make sure the input position does not exceed the max model len.\n"
            ),
        ),
        (
            "        num_scheduled_tokens[request_id] = num_new_tokens\n",
            (
                "        num_scheduled_tokens[request_id] = num_new_tokens\n"
                "        if supply_plan is not None:\n"
                "            supply_plan.consume(request_id, num_new_tokens)\n"
            ),
        ),
    )
    for old, new in replacements:
        if source.count("\n" + old) != 1:
            raise RuntimeError(
                f"Scheduler.schedule changed; re-sync prefill budget hook: {old!r}"
            )
        source = source.replace("\n" + old, "\n" + new, 1)
    return source


def _apply() -> None:
    from vllm.v1.core.sched import scheduler as sched_mod

    from vllm_iluvatar.ilu_patch.general_patches import patch_pp_spec_schedule

    # Compose with the existing PP / remote replay rewrite, regardless of
    # general-plugin discovery order. Exec'd methods have no inspectable source.
    patch_pp_spec_schedule.register()
    scheduler = sched_mod.Scheduler
    if getattr(scheduler.schedule, "_ilu_prefill_budget_patched", False):
        return
    original = scheduler.schedule
    source = patch_schedule_source(original._ilu_pp_spec_schedule_source)
    sched_mod._ilu_prefill_supply = prefill_supply
    namespace = {}
    exec(source, sched_mod.__dict__, namespace)  # noqa: S102
    schedule = namespace["schedule"]
    schedule.__dict__.update(original.__dict__)
    schedule._ilu_pp_spec_schedule_source = source
    schedule._ilu_prefill_budget_patched = True
    scheduler.schedule = schedule

    free_request = scheduler._free_request

    @wraps(free_request)
    def finish(self, request, *args, **kwargs):
        if request.is_finished():
            prefill_supply.finish(self, request)
        return free_request(self, request, *args, **kwargs)

    scheduler._free_request = finish
