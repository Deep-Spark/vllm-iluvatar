# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the sync-PP prepare_inputs_event arming patch."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_iluvatar.ilu_patch.general_patches.patch_input_prep_sync import (
    _maybe_arm_prepare_inputs_event,
    register,
)


def _runner(*, pp_size: int, event):
    return SimpleNamespace(
        prepare_inputs_event=event,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size)
        ),
    )


def test_arms_event_when_pp_and_missing():
    created = object()
    runner = _runner(pp_size=8, event=None)
    _maybe_arm_prepare_inputs_event(runner, lambda: created)
    assert runner.prepare_inputs_event is created


def test_skips_when_pp_is_one():
    runner = _runner(pp_size=1, event=None)
    _maybe_arm_prepare_inputs_event(runner, lambda: object())
    assert runner.prepare_inputs_event is None


def test_does_not_replace_existing_event():
    existing = object()
    runner = _runner(pp_size=8, event=existing)
    _maybe_arm_prepare_inputs_event(runner, lambda: object())
    assert runner.prepare_inputs_event is existing


def test_input_prep_patch_register_is_idempotent_and_wraps():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    register()
    first = GPUModelRunner.__init__
    register()
    register()
    assert GPUModelRunner.__init__ is first
    assert hasattr(GPUModelRunner.__init__, "__wrapped__")
