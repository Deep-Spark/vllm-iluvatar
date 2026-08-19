# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT Step3.5 MTP model overrides.

Corresponds to: vllm/model_executor/models/step3p5_mtp.py
Target   : arch ``Step3p5MTP``
Baseline : main_6e448d0
Why      : Iluvatar Step3.5 decoder layers retain a separate residual stream
           and return ``(hidden_states, residual)``.
How      : Promote only the MTP decoder block and consume its first output.
Scope    : Step3.5 speculative decoding on Iluvatar.
Remove   : when upstream exposes the residual-stream decoder contract.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.step3p5_mtp import (
    Step3p5AMultiTokenPredictor as _Step3p5AMultiTokenPredictor,
)
from vllm.model_executor.models.step3p5_mtp import (
    Step3p5AMultiTokenPredictorLayer as _Step3p5AMultiTokenPredictorLayer,
)
from vllm.model_executor.models.step3p5_mtp import Step3p5MTP as _Step3p5MTP

from vllm_iluvatar.model_executor.models.step3p5 import IluStep3p5DecoderLayer


class IluStep3p5AMultiTokenPredictorLayer(_Step3p5AMultiTokenPredictorLayer):
    """MTP layer that consumes the Iluvatar decoder's residual pair."""

    @staticmethod
    def promote(layer: _Step3p5AMultiTokenPredictorLayer) -> None:
        layer.__class__ = IluStep3p5AMultiTokenPredictorLayer
        IluStep3p5DecoderLayer.promote(layer.mtp_block)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        hidden_states, _ = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        return hidden_states


class IluStep3p5AMultiTokenPredictor(_Step3p5AMultiTokenPredictor):
    """Community MTP predictor with promoted Iluvatar blocks."""

    @staticmethod
    def promote(model: _Step3p5AMultiTokenPredictor) -> None:
        model.__class__ = IluStep3p5AMultiTokenPredictor
        for layer in model.layers.values():
            IluStep3p5AMultiTokenPredictorLayer.promote(layer)


class IluStep3p5MTP(_Step3p5MTP):
    """OOT Step3.5 MTP model compatible with Iluvatar decoder output."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        IluStep3p5AMultiTokenPredictor.promote(self.model)


def register_model() -> None:
    """Register the Iluvatar Step3.5 MTP implementation."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Step3p5MTP",
        "vllm_iluvatar.model_executor.models.step3p5_mtp:IluStep3p5MTP",
    )
