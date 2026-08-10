# SPDX-License-Identifier: Apache-2.0
"""Iluvatar PP + speculative-decode wrappers for ``vllm.v1.engine.core``.

Corresponds to: vllm/v1/engine/core.py
Baseline     : main_61ac3680210
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any


def make_init(original_init: Callable[..., None]) -> Callable[..., None]:
    """Set the inline-draft transport mode after community initialization."""

    @wraps(original_init)
    def _ilu_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # [ILU override] EngineCore.__init__ - delta: PP batch-queue outputs
        # carry drafts inline rather than through the delayed worker RPC.
        self.draft_in_model_output = (
            self.batch_queue is not None and self.use_spec_decode
        )

    return _ilu_init


def make_post_step(original_post_step: Callable[..., None]) -> Callable[..., None]:
    """Skip the stale draft-token RPC when outputs already carry drafts."""

    @wraps(original_post_step)
    def _ilu_post_step(self: Any, model_executed: bool) -> None:
        # [ILU override] EngineCore.post_step - delta: Scheduler receives
        # draft tokens from ModelRunnerOutput in update_from_output.
        if getattr(
            self,
            "draft_in_model_output",
            self.batch_queue is not None and self.use_spec_decode,
        ):
            return
        original_post_step(self, model_executed)

    return _ilu_post_step
