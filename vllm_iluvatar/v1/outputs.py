# SPDX-License-Identifier: Apache-2.0
"""Iluvatar extensions for :mod:`vllm.v1.outputs`.

Corresponds to: vllm/v1/outputs.py
Baseline     : main_6e448d0

The upstream output object is a cross-process transport contract.  PP with a
batch queue must carry fresh speculative draft tokens in that contract instead
of fetching them in ``EngineCore.post_step``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any


def install_draft_token_ids_field(model_runner_output_cls: type) -> None:
    """Allow ``ModelRunnerOutput(..., draft_token_ids=...)`` without copying it.

    The community dataclass is deliberately patched in place: all modules which
    imported its class object retain the same identity, and pickle continues to
    resolve the community-qualified type in spawned worker processes.
    """
    if getattr(model_runner_output_cls, "_ilu_pp_mtp_output_patched", False):
        return

    original_init: Callable[..., None] = model_runner_output_cls.__init__

    @wraps(original_init)
    def _ilu_init(
        self: Any,
        *args: Any,
        draft_token_ids: Any = None,
        **kwargs: Any,
    ) -> None:
        original_init(self, *args, **kwargs)
        self.draft_token_ids = draft_token_ids

    # [ILU override] ModelRunnerOutput.__init__ - delta: carry draft tokens
    # through the PP model-output transport contract.
    model_runner_output_cls.__init__ = _ilu_init
    model_runner_output_cls.draft_token_ids = None
    model_runner_output_cls._ilu_pp_mtp_output_patched = True
