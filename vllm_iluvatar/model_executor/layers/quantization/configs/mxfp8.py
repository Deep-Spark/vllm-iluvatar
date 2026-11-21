# SPDX-License-Identifier: Apache-2.0
"""Iluvatar configuration for MiniMax-style MXFP8 checkpoints."""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMxFp8Config,
    ModelOptNvFp4Config,
)

from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.mxfp8_moe import (
    IluMxFp8FusedMoE,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.nvfp4_moe import (
    IluModelOptNvFp4MoEMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.mxfp8.linear import (
    mxfp8_int8_linear_method,
)


@register_quantization_config("mxfp8")
class IluMxFp8Config(ModelOptMxFp8Config):
    """MXFP8 config whose routed experts use the Iluvatar ix30 compute path.

    Mixed checkpoints may declare NVFP4 experts alongside MXFP8 linears.
    Dispatch is based on the checkpoint format, independently of model
    architecture.

    For ordinary MXFP8 checkpoints on ivcore30 only ``FusedMoEMethodCls``
    differs: the inherited ``ModelOptQuantConfigBase.get_quant_method``
    dispatches Linear layers through the community ``build_linear_method``
    (which picks up ``IluMxfp8LinearKernel`` from the patched kernel list) and
    ``RoutedExperts`` through ``FusedMoEMethodCls``. On ivcore11 that kernel
    patch is inactive and the dense default is INT8, so Linear is claimed by
    ``mxfp8_int8_linear_method``.

    MiniMax checkpoints tag ``quant_method: "mxfp8"``. NVIDIA ModelOpt
    checkpoints resolve to ``modelopt_mxfp8``, which ``configs/modelopt/mxfp8``
    claims with its own sibling config; the two differ only in the registry
    name and the expert method, and share this Linear dispatch.
    """

    def get_name(self) -> Any:
        return "mxfp8"

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        result = super().from_config(config)
        result.moe_quant_algo = config.get("moe_quant_algo", "MXFP8")
        if result.moe_quant_algo not in ("MXFP8", "NVFP4"):
            raise ValueError(
                f"Unsupported moe_quant_algo {result.moe_quant_algo!r}; "
                "expected MXFP8 or NVFP4"
            )
        if result.moe_quant_algo == "NVFP4" and (
            config.get("moe_quant_format") != "nvfp4-pack-quantized"
            or config.get("group_size") != 16
        ):
            raise ValueError("Unsupported NVFP4 expert checkpoint format")
        return result

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        linear_method = mxfp8_int8_linear_method(self, layer, prefix)
        if linear_method is not None:
            return linear_method
        if (
            getattr(self, "moe_quant_algo", "MXFP8") == "NVFP4"
            and isinstance(layer, RoutedExperts)
            and not self.is_layer_excluded(prefix)
        ):
            return IluModelOptNvFp4MoEMethod(
                ModelOptNvFp4Config(
                    is_checkpoint_nvfp4_serialized=True,
                    group_size=16,
                ),
                layer.moe_config,
                stage_on_cpu=True,
                validate_checkpoint=True,
                expert_chunk_size=2,
            )
        return super().get_quant_method(layer, prefix)


IluMxFp8Config.FusedMoEMethodCls = IluMxFp8FusedMoE
