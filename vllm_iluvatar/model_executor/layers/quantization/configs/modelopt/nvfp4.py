# SPDX-License-Identifier: Apache-2.0
"""Iluvatar ModelOpt NVFP4 quantization config and method selection."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.nvfp4_moe import (
    IluModelOptNvFp4MoEMethod,
)


@register_quantization_config("modelopt_fp4")
class IluModelOptNvFp4Config(ModelOptNvFp4Config):
    """ModelOpt NVFP4 config whose routed experts land on Iluvatar kernels.

    Inherits ``__init__`` / ``get_name`` / ``override_quantization_method`` /
    ``_from_config`` unchanged, so checkpoint detection and every linear /
    attention branch stay community behaviour.
    """

    # [ILU override] ModelOptQuantConfigBase.get_quant_method — delta: routed
    # experts (not excluded) return IluModelOptNvFp4MoEMethod instead of
    # community ModelOptNvFp4FusedMoE. Everything else (linear, attention, KV,
    # excluded layers) is left to super().
    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, RoutedExperts) and not self.is_layer_excluded(prefix):
            return IluModelOptNvFp4MoEMethod(
                quant_config=self, moe_config=layer.moe_config
            )
        return super().get_quant_method(layer, prefix)
