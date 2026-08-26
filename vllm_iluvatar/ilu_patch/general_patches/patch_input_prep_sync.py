# SPDX-License-Identifier: Apache-2.0
"""Arm prepare_inputs_event for PP under sync scheduling.

Corresponds to: vllm/v1/worker/gpu_model_runner.py

Target   : GPUModelRunner.__init__
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

from vllm_iluvatar.ilu_patch import general_patch as p


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

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        if (
            self.prepare_inputs_event is None
            and self.vllm_config.parallel_config.pipeline_parallel_size > 1
        ):
            self.prepare_inputs_event = torch.cuda.Event(blocking=True)

    GPUModelRunner.__init__ = __init__  # type: ignore[method-assign]
