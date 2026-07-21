# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT GLM4-MoE-Lite model override.

Corresponds to: vllm/model_executor/models/glm4_moe_lite.py

Target   : vllm.model_executor.models.glm4_moe_lite.Glm4MoeLiteForCausalLM
           (arch "Glm4MoeLiteForCausalLM")
Baseline : main_dcfebf93f4e
Why      : GLM-4.7-Flash on Iluvatar needs bf16 MoE router logits for the
           ixformer grouped-topk kernel, and performs a post-load q/kv projection
           fusion for supported quantization methods.
How      : Use the official ``ModelRegistry.register_model`` hook. The subclass
           inherits community construction, converts the shared GLM4 MoE router
           gate to bf16, overrides ``load_weights`` to preserve the fork's
           GLM4-Lite checkpoint mapping, then installs the optimized fused q/kv
           projection forward path when the loaded quantization method supports it.
Scope    : GLM-4.7-Flash models on Iluvatar.
Remove   : when community GLM4-MoE-Lite handles these CoreX paths natively.

Registration: ``register_model()`` below is invoked by
``vllm_iluvatar.ilu_register.apply_oot_registrations`` during register_general.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteMLAAttention,
)
from vllm.model_executor.models.glm4_moe_lite import (
    Glm4MoeLiteForCausalLM as _Glm4MoeLiteForCausalLM,
)

from vllm_iluvatar.model_executor.models.glm4_moe import (
    glm4_moe_bf16_router_overrides,
)

_OPT_SUPPORT_QUANT_METHODS = {"UnquantizedLinearMethod", "IluCompressedTensorsW8A8Int8"}


def _get_quant_method_name(layer) -> str:
    kv_proj = layer.kv_a_proj_with_mqa
    if hasattr(kv_proj, "scheme"):
        return kv_proj.scheme.__class__.__name__
    return kv_proj.quant_method.__class__.__name__


def _install_optimized_forward(layer, is_mla: bool) -> None:
    if is_mla:
        layer.mla_attn.forward = layer.mla_attn.forward_opt
    else:
        layer.forward = layer.forward_opt


def _inject_optimized_qkv_layer(layer, is_mla: bool) -> None:
    q_lora_rank = getattr(layer, "q_lora_rank", None)
    if q_lora_rank is not None:
        layer.q_a_proj.weight.data = torch.cat(
            [layer.q_a_proj.weight, layer.kv_a_proj_with_mqa.weight],
            dim=0,
        )
        if hasattr(layer.q_a_proj, "weight_scale"):
            layer.q_a_proj.weight_scale.data = torch.cat(
                [
                    layer.q_a_proj.weight_scale,
                    layer.kv_a_proj_with_mqa.weight_scale,
                ],
                dim=0,
            )
            del layer.kv_a_proj_with_mqa.weight_scale
    elif not is_mla:
        layer.q_proj.weight.data = torch.cat(
            [layer.q_proj.weight, layer.kv_a_proj_with_mqa.weight],
            dim=0,
        )
        if hasattr(layer.q_proj, "weight_scale"):
            layer.q_proj.weight_scale.data = torch.cat(
                [
                    layer.q_proj.weight_scale,
                    layer.kv_a_proj_with_mqa.weight_scale,
                ],
                dim=0,
            )
            del layer.kv_a_proj_with_mqa.weight_scale
    else:
        return

    del layer.kv_a_proj_with_mqa.weight
    del layer.kv_a_proj_with_mqa
    _install_optimized_forward(layer, is_mla)


def _inject_optimized_attention_layer(layer, is_mla: bool) -> None:
    quant_method = _get_quant_method_name(layer)
    if quant_method not in _OPT_SUPPORT_QUANT_METHODS:
        return
    _inject_optimized_qkv_layer(layer, is_mla)


def glm4_moe_lite_attention_overrides(model) -> None:
    """Install post-load q/kv projection fusion for supported Lite attention layers."""
    for _, layer in model.named_modules():
        if isinstance(layer, Glm4MoeLiteMLAAttention):
            _inject_optimized_attention_layer(layer, is_mla=True)
        elif isinstance(layer, Glm4MoeLiteAttention):
            _inject_optimized_attention_layer(layer, is_mla=False)


class Glm4MoeLiteForCausalLM(_Glm4MoeLiteForCausalLM):
    """GLM4-MoE-Lite causal LM wired for Iluvatar loading and router deltas."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        glm4_moe_bf16_router_overrides(self.model)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params = super().load_weights(weights)
        glm4_moe_lite_attention_overrides(self.model)
        return loaded_params


def register_model() -> None:
    """Model registration hook: point GLM4-MoE-Lite arch at the Iluvatar class."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Glm4MoeLiteForCausalLM",
        "vllm_iluvatar.model_executor.models.glm4_moe_lite:Glm4MoeLiteForCausalLM",
    )
