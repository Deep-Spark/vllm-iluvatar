# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT Qwen3-VL-MoE model override.

Corresponds to:
    vllm/model_executor/models/qwen3_vl_moe.py

Target:
    vllm.model_executor.models.qwen3_vl_moe.Qwen3VLMoeForConditionalGeneration
    (arch "Qwen3VLMoeForConditionalGeneration").
Baseline:
    main_6e448d0.
Why:
    Some Qwen3-VL-MoE text configs do not carry ``architectures``. The normal
    Iluvatar vLLM baseline constructs the embedded language model with
    ``architectures=["Qwen3MoeForCausalLM"]``. The embedded Qwen3-MoE language
    model needs the Iluvatar fused q/k norm and unquantized router-gate
    adaptations, while expert loading stays on the AutoWeightsLoader /
    RoutedExperts path.
How:
    Use ``ModelRegistry.register_model`` for the VL-MoE architecture. Before
    community construction, fill the nested text config architecture if missing;
    after construction, install the same Qwen3-MoE language-model adaptations
    used by the standalone Qwen3-MoE plugin.
Scope:
    Qwen3-VL-MoE models on Iluvatar.
Remove:
    When community preserves nested text config architectures and Qwen3-MoE
    attention/router behavior is device-safe upstream.
"""

from __future__ import annotations

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration as _Qwen3VLMoeForConditionalGeneration,
)

from vllm_iluvatar.model_executor.models.qwen3_moe import (
    qwen3_moe_attention_fusion,
    qwen3_moe_router_gate_overrides,
)


def ensure_qwen3_moe_text_architecture(vllm_config: VllmConfig) -> None:
    """Fill missing nested text architecture for Qwen3-VL-MoE configs."""
    config = vllm_config.model_config.hf_config
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return
    if not getattr(text_config, "architectures", None):
        text_config.architectures = ["Qwen3MoeForCausalLM"]


class Qwen3VLMoeForConditionalGeneration(_Qwen3VLMoeForConditionalGeneration):
    """Qwen3-VL-MoE model wired for Iluvatar Qwen3-MoE text adaptations."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        ensure_qwen3_moe_text_architecture(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        qwen3_moe_attention_fusion(self.language_model.model)
        qwen3_moe_router_gate_overrides(self.language_model.model)


def register_model() -> None:
    """Point Qwen3-VL-MoE architecture resolution at the Iluvatar subclass."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Qwen3VLMoeForConditionalGeneration",
        "vllm_iluvatar.model_executor.models.qwen3_vl_moe:"
        "Qwen3VLMoeForConditionalGeneration",
    )
