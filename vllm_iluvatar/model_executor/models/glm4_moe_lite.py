# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT GLM4-MoE-Lite model override.

Corresponds to: vllm/model_executor/models/glm4_moe_lite.py

Target   : vllm.model_executor.models.glm4_moe_lite.Glm4MoeLiteForCausalLM
           (arch "Glm4MoeLiteForCausalLM")
Baseline : main_6e448d0
Why      : GLM-4.7-Flash on Iluvatar needs bf16 MoE router logits for the
           ixformer grouped-topk kernel.
How      : Use the official ``ModelRegistry.register_model`` hook. The subclass
           inherits community construction, converts the shared GLM4 MoE router
           gate to bf16, and otherwise uses the community GLM4-Lite loading and
           attention paths unchanged.
Scope    : GLM-4.7-Flash models on Iluvatar.
Remove   : when community GLM4-MoE-Lite handles these CoreX paths natively.

Registration: ``register_model()`` below is invoked by
``vllm_iluvatar.ilu_register.apply_oot_registrations`` during register_general.
"""

from __future__ import annotations

from vllm.config import VllmConfig
from vllm.model_executor.models.glm4_moe_lite import (
    Glm4MoeLiteForCausalLM as _Glm4MoeLiteForCausalLM,
)

from vllm_iluvatar.model_executor.models.glm4_moe import (
    glm4_moe_bf16_router_overrides,
)


class Glm4MoeLiteForCausalLM(_Glm4MoeLiteForCausalLM):
    """GLM4-MoE-Lite causal LM wired for Iluvatar loading and router deltas."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        glm4_moe_bf16_router_overrides(self.model)


def register_model() -> None:
    """Model registration hook: point GLM4-MoE-Lite arch at the Iluvatar class."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Glm4MoeLiteForCausalLM",
        "vllm_iluvatar.model_executor.models.glm4_moe_lite:Glm4MoeLiteForCausalLM",
    )
