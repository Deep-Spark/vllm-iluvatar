# SPDX-License-Identifier: Apache-2.0
"""Configuration and backend selection for Iluvatar GPTQ."""

from __future__ import annotations

from copy import deepcopy

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
    get_linear_quant_method,
    override_config,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_layer,
    get_marlin_input_dtype,
)

logger = init_logger(__name__)


def get_moe_quant_method(
    config: IluAutoGPTQConfig,
    layer: RoutedExperts,
    prefix: str,
    moe_method_cls: type,
):
    cloned_config = deepcopy(config)

    assert isinstance(layer, RoutedExperts)
    # False = skip module, None = no override, else = Positive match
    if (
        get_dynamic_override(  # noqa: E712
            cloned_config,  # noqa: E712
            layer_name=prefix,
        )
        == False
    ):  # noqa: E712
        return UnquantizedFusedMoEMethod(layer.moe_config)

    if prefix:
        # Dynamic per module/layer rules may override base config
        override_config(cloned_config, prefix=prefix)

    return moe_method_cls(cloned_config, layer.moe_config)


@register_quantization_config("auto_gptq")
@register_quantization_config("gptq")
@register_quantization_config("gptq_marlin")
class IluAutoGPTQConfig(AutoGPTQConfig):
    """Config class for AutoGPTQ quantization using Marlin kernels."""

    def __repr__(self) -> str:
        return (
            f"IluAutoGPTQConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"dynamic={self.dynamic}, "
            f"modules_in_block_to_quantize={self.modules_in_block_to_quantize})"
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

            if not check_moe_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by GPTQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                return MoeWNA16Config.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            from ..backends.gptq.moe import IluAutoGPTQMoEMethod

            moe_quant_method = get_moe_quant_method(
                self, layer, prefix, IluAutoGPTQMoEMethod
            )
            if moe_quant_method is None:
                return None
            moe_quant_method.input_dtype = get_marlin_input_dtype(prefix)
            return moe_quant_method

        from ..backends.gptq.linear import IluAutoGPTQLinearMethod

        quant_method = get_linear_quant_method(
            self, layer, prefix, IluAutoGPTQLinearMethod
        )
        if quant_method is None:
            return None
        quant_method.input_dtype = get_marlin_input_dtype(prefix)
        return quant_method


__all__ = [
    "IluAutoGPTQConfig",
]
