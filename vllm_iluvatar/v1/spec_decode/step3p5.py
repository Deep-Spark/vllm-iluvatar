# SPDX-License-Identifier: Apache-2.0
"""Iluvatar ``Step3p5MTPProposer.propose`` override.

Corresponds to: vllm/v1/spec_decode/step3p5.py ::
``Step3p5MTPProposer.propose``

The community sequential draft uses ``spec_step_idx = token_index + 1`` and
``set_forward_context(..., self.vllm_config, ...)``. Step-3.5 / Step-3.7 MTP-3
selects the draft layer via ``spec_step_idx % num_mtp_layers``, so rotating
the index draws tokens 2/3 from MTP layers 1/2. The wrapper keeps community
``propose`` intact and only:

1. forces ``spec_step_idx=0`` / ``draft_index=0`` on the draft module, logits
   sample, and drafting attention metadata (fixed MTP layer 0);
2. clones ``draft_vllm_config`` under draft PP=1 and runs that loop under
   ``self.draft_vllm_config``.

Other proposers (EAGLE / DeepSeek MTP / DSpark) are not patched here.

Baseline : community propose on v0.31.0 (341bf7d4d1).
Remove   : when upstream keeps layer 0 and the draft vllm config in this loop.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch


def _ensure_draft_vllm_config(self) -> None:
    """Materialise ``self.draft_vllm_config`` if the community tree omitted it."""
    if getattr(self, "draft_vllm_config", None) is None:
        self.draft_vllm_config = self._create_draft_vllm_config()


def make_create_draft_vllm_config(orig: Callable[..., object]) -> Callable[..., object]:
    """Clone the community draft config at PP=1 when the target uses PP>1.

    Step-3 MTP draft is the in-checkpoint next-token head, not a separately
    supplied draft model. There is no external ``draft_parallel_config``.
    Community ``_create_draft_vllm_config`` copies ``self.vllm_config``, so the
    target pipeline size is replaced for the clone and restored afterwards.
    """

    @functools.wraps(orig)
    def _create_draft_vllm_config(self):
        from copy import copy

        from vllm.config import replace

        vcfg = self.vllm_config
        target_pc = vcfg.parallel_config
        if target_pc.pipeline_parallel_size > 1:
            draft_pc = copy(target_pc)
            draft_pc.pipeline_parallel_size = 1
            self.vllm_config = replace(vcfg, parallel_config=draft_pc)
        try:
            return orig(self)
        finally:
            self.vllm_config = vcfg

    return _create_draft_vllm_config


class _ForceMtpLayer0:
    """Proxy that pins draft ``spec_step_idx`` to MTP layer 0."""

    __slots__ = ("_model",)

    def __init__(self, model: Any) -> None:
        object.__setattr__(self, "_model", model)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["spec_step_idx"] = 0
        return object.__getattribute__(self, "_model")(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_model"), name)


def make_sample_draft_tokens_for_step(
    orig: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
    """Keep sequential draft logits on MTP layer 0."""

    @functools.wraps(orig)
    def _sample_draft_tokens_for_step(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any,
        spec_step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return orig(self, hidden_states, sampling_metadata, 0)

    return _sample_draft_tokens_for_step


def make_build_per_group_and_layer_attn_metadata(
    orig: Callable[..., Any],
) -> Callable[..., Any]:
    """Keep sequential draft attention metadata on MTP layer 0."""

    @functools.wraps(orig)
    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: Any,
        draft_index: int = 0,
    ) -> Any:
        return orig(self, common_attn_metadata, 0)

    return build_per_group_and_layer_attn_metadata


def make_propose(
    orig: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """Wrap community ``propose`` with Step-3 MTP-3 draft semantics."""

    @functools.wraps(orig)
    def propose(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        _ensure_draft_vllm_config(self)
        saved_model = self.model
        saved_cfg = self.vllm_config
        self.model = _ForceMtpLayer0(saved_model)
        self.vllm_config = self.draft_vllm_config
        try:
            return orig(self, *args, **kwargs)
        finally:
            self.model = saved_model
            self.vllm_config = saved_cfg

    return propose