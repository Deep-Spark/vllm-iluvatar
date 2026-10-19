# SPDX-License-Identifier: Apache-2.0
"""Recover failed KV loads from scheduler-side connector output.

Corresponds to: vllm/v1/core/sched/scheduler.py
Target   : Scheduler.update_from_output, Scheduler._update_from_kv_xfer_finished
Baseline : v0.30.0rc0 (7ce3d3b074)
Why      : community consumes ``failed_recving`` before
           ``_update_from_kv_xfer_finished`` calls
           ``connector.update_connector_output``. OptNixl records peer
           disconnects on the scheduler, so those failures never reach
           ``_handle_failed_recving`` and one receive error can strand
           unrelated scheduled work.
How      : run that call before ``update_from_output`` handles failed
           receives, and drop the later call so it runs once.
Scope    : scheduler-side KV connectors that add ``failed_recving``.
Remove   : when community vLLM updates the connector output before handling
           failed receives.
"""

from __future__ import annotations

import inspect
import textwrap
from collections.abc import Callable
from functools import wraps
from typing import Any

from vllm_iluvatar.ilu_patch import general_patch as p

# Dedented ``_update_from_kv_xfer_finished`` body.
_OLD_XFER = """\
    if self.connector is not None:
        self.connector.update_connector_output(kv_connector_output)

    # KV Connector:: update recv and send status from last step.
"""
_NEW_XFER = """\
    # update_connector_output already ran before failed receives were consumed.

    # KV Connector:: update recv and send status from last step.
"""


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="request-scoped recovery for scheduler-side KV load failures",
    )


def make_scheduler_update_from_output(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    """Publish scheduler-side receive failures before they are consumed."""

    @wraps(original)
    def _ilu_update_from_output(self, scheduler_output, model_runner_output):
        kv_connector_output = getattr(model_runner_output, "kv_connector_output", None)
        connector = getattr(self, "connector", None)
        if kv_connector_output and connector is not None:
            connector.update_connector_output(kv_connector_output)
        return original(self, scheduler_output, model_runner_output)

    return _ilu_update_from_output


def patch_kv_xfer_finished_source(source: str) -> str:
    """Drop the late connector update from dedented xfer-finished source."""
    if _NEW_XFER in source:
        return source
    if _OLD_XFER not in source:
        raise RuntimeError(
            "Scheduler._update_from_kv_xfer_finished no longer matches "
            "patch_kv_load_failure_recovery.py; re-sync the patch"
        )
    return source.replace(_OLD_XFER, _NEW_XFER, 1)


def _apply() -> None:
    from vllm.v1.core.sched import scheduler as sched_mod
    from vllm.v1.core.sched.scheduler import Scheduler

    current = Scheduler.update_from_output
    if not getattr(current, "_ilu_kv_load_recovery_patched", False):
        wrapped = make_scheduler_update_from_output(current)
        wrapped._ilu_kv_load_recovery_patched = True  # type: ignore[attr-defined]
        Scheduler.update_from_output = wrapped  # type: ignore[method-assign]

    xfer = Scheduler._update_from_kv_xfer_finished
    if getattr(xfer, "_ilu_kv_load_recovery_patched", False):
        return
    # An exec'd replacement has no source file. Read the community body first.
    source = textwrap.dedent(inspect.getsource(xfer))
    patched = patch_kv_xfer_finished_source(source)
    namespace: dict[str, object] = {}
    exec(patched, sched_mod.__dict__, namespace)  # noqa: S102
    finished = namespace["_update_from_kv_xfer_finished"]
    finished._ilu_kv_load_recovery_patched = True  # type: ignore[attr-defined]
    Scheduler._update_from_kv_xfer_finished = finished  # type: ignore[method-assign]
