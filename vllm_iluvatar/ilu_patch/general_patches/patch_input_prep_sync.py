# SPDX-License-Identifier: Apache-2.0
"""Arm prepare_inputs_event for PP under sync scheduling.

Corresponds to: vllm/v1/worker/gpu_model_runner.py

Target   : GPUModelRunner.__init__
Baseline : main_6e448d0
Why      : Community only creates the event under async scheduling. PP
           keeps pp_size in-flight batches, so non-blocking H2D of the
           shared pinned input buffers can race the next step (worse
           with CUDA graphs).
How      : After community ``__init__``, if PP and the event is still
           None, create ``torch.cuda.Event(blocking=True)``.
Scope    : V1 GPUModelRunner. TP-only (PP=1) is unchanged.
Remove   : when upstream keys the event off max_concurrent_batches.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from vllm_iluvatar.ilu_patch import general_patch as p


def _maybe_arm_prepare_inputs_event(
    runner: Any, event_factory: Callable[[], Any]
) -> None:
    if (
        runner.prepare_inputs_event is None
        and runner.vllm_config.parallel_config.pipeline_parallel_size > 1
    ):
        runner.prepare_inputs_event = event_factory()


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="arm prepare_inputs_event for sync-scheduled PP",
    )


def _apply() -> None:
    import torch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    orig_init = GPUModelRunner.__init__

    # [ILU override] GPUModelRunner.__init__ — delta: arm prepare_inputs_event
    # for pipeline-parallel runs that stay on sync scheduling.
    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        _maybe_arm_prepare_inputs_event(
            self, lambda: torch.cuda.Event(blocking=True)
        )

    p.check_signatures(
        GPUModelRunner,
        SimpleNamespace(__init__=__init__),
        ["__init__"],
        strict=True,
    )
    GPUModelRunner.__init__ = __init__  # type: ignore[method-assign]
