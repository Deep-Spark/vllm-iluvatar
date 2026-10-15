# SPDX-License-Identifier: Apache-2.0
"""Iluvatar model state for DeepSeek-V4.1: raw token ids on every PP rank.

Corresponds to: vllm/models/deepseek_v41/nvidia/model_state.py

Target   : vllm.models.deepseek_v41.nvidia.model_state.DeepseekV41ModelState
Baseline : v0.30.1rc0
Why      : ``GPUModelRunner.execute_model`` blanks ``input_ids`` on every
           non-first PP rank (``model_inputs["input_ids"] = None``), which is
           fine for a model that only embeds on rank 0 but not for V4.1: each
           of the 40 MoE layers routes image tokens by ``gate.bias_vl``, and
           the engram layers hash the ids. With 40 layers over PP=8 those
           consumers sit on every stage -- engram layer 14 lands on stage 2 --
           so stages past the first either raise ("image-token MoE routing
           requires input_ids") or silently skip the engram contribution.
How      : pass the ids under a second name. The runner merges whatever
           ``prepare_inputs`` returns into the model kwargs *before* it blanks
           ``input_ids``, so a key of our own survives; the model falls back to
           it whenever ``input_ids`` is None. The ids are already resident on
           every rank (the input batch is built per rank), so nothing new goes
           on the wire.
Scope    : DeepSeek-V4.1 (text and vision) on any PP size; a no-op at PP=1,
           where ``input_ids`` is never blanked.
Remove   : when the runner keeps ``input_ids`` for models that declare
           ``requires_raw_input_tokens``.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState

# The kwarg the model reads when ``input_ids`` has been blanked. Not named
# ``input_ids``: the runner overwrites that key right after the merge.
RAW_INPUT_IDS_KWARG = "raw_input_ids"


class IluDeepseekV41ModelState(DeepseekV41ModelState):
    """Community V4.1 state plus the raw token ids as a second kwarg."""

    _dummy_input_ids: torch.Tensor | None = None

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        model_inputs[RAW_INPUT_IDS_KWARG] = input_batch.input_ids
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        # The dummy run reaches the same consumers, and an uninitialised id
        # would index the engram token map out of range. Zero is a real token.
        if self._dummy_input_ids is None:
            self._dummy_input_ids = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device
            )
        model_inputs[RAW_INPUT_IDS_KWARG] = self._dummy_input_ids[:num_tokens]
        return model_inputs
