# SPDX-License-Identifier: Apache-2.0
"""Allow the V2 PP non-last-rank Mamba state update to use int32 mappings."""

from __future__ import annotations

import types
from collections.abc import Callable
from functools import wraps
from typing import Any

import torch

from vllm_iluvatar.ilu_patch import general_patch as p


def make_postprocess_state(original: Callable[..., None]) -> Callable[..., None]:
    @wraps(original)
    def postprocess_state(
        self: Any,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        if isinstance(num_sampled, int) and idx_mapping.dtype != torch.int64:
            idx_mapping = idx_mapping.to(dtype=torch.int64)
        return original(self, idx_mapping, num_sampled, num_computed_tokens)

    return postprocess_state


def register() -> None:
    p.register_once(__name__, _apply, detail="Mamba PP index_fill index dtype")


def _apply() -> None:
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    replacement = make_postprocess_state(MambaHybridModelState.postprocess_state)
    p.check_signatures(
        MambaHybridModelState,
        types.SimpleNamespace(postprocess_state=replacement),
        ["postprocess_state"],
        strict=True,
    )
    MambaHybridModelState.postprocess_state = replacement
