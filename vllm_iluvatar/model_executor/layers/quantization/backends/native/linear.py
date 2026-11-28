# SPDX-License-Identifier: Apache-2.0
"""Native BF16/FP16 Linear execution on ivcore11."""

import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_iluvatar.model_executor.layers.quantization.common.w8a8_linear import (
    IluW8A8Int8LinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.conversions.native_to_int8 import (
    quantize_native_to_w8a8_int8,
)


class IluNativeW8A8Int8LinearMethod(IluW8A8Int8LinearMethod):
    def __init__(self) -> None:
        super().__init__(UnquantizedLinearMethod())

    def _quantize(self, layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        return quantize_native_to_w8a8_int8(layer.weight.data)
