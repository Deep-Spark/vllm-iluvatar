# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scheduler-side KV load failure recovery."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_iluvatar.ilu_patch.general_patches import (
    patch_kv_load_failure_recovery as patch,
)
from vllm_iluvatar.ilu_patch.general_patches.patch_kv_load_failure_recovery import (
    _NEW_XFER,
    _OLD_XFER,
)


def test_connector_update_runs_before_failed_receive_handling():
    seen: list[str] = []

    class Connector:
        def update_connector_output(self, connector_output):
            seen.append("connector")
            connector_output.failed_recving.add("req-1")

    class Scheduler:
        connector = Connector()

        def update_from_output(self, scheduler_output, model_runner_output):
            seen.append("handle")
            failed = model_runner_output.kv_connector_output.failed_recving
            assert "req-1" in failed
            return "ok"

    wrapped = patch.make_scheduler_update_from_output(Scheduler.update_from_output)
    output = SimpleNamespace(
        kv_connector_output=SimpleNamespace(failed_recving=set())
    )
    assert wrapped(Scheduler(), None, output) == "ok"
    assert seen == ["connector", "handle"]


def test_connector_update_skipped_without_output():
    seen: list[str] = []

    class Scheduler:
        connector = SimpleNamespace(
            update_connector_output=lambda _output: seen.append("connector")
        )

        def update_from_output(self, scheduler_output, model_runner_output):
            seen.append("handle")

    wrapped = patch.make_scheduler_update_from_output(Scheduler.update_from_output)
    wrapped(Scheduler(), None, SimpleNamespace(kv_connector_output=None))
    assert seen == ["handle"]


def test_patch_kv_xfer_finished_source_drops_late_call():
    source = (
        "def _update_from_kv_xfer_finished(self, kv_connector_output):\n"
        + _OLD_XFER
        + "    for req_id in kv_connector_output.finished_recving or ():\n"
        + "        return req_id\n"
    )
    patched = patch.patch_kv_xfer_finished_source(source)
    assert "update_connector_output" not in patched
    assert _NEW_XFER in patched
    assert "finished_recving" in patched
    assert patch.patch_kv_xfer_finished_source(patched) == patched


def test_patch_kv_xfer_finished_source_rejects_drift():
    with pytest.raises(RuntimeError, match="re-sync"):
        patch.patch_kv_xfer_finished_source("def _update_from_kv_xfer_finished(self):\n    return\n")
