# SPDX-License-Identifier: Apache-2.0
"""Load-time conversion of a native Linear to the static GLM W8A8 path."""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.linear._w8a8 import (
    _IluRuntimeW8A8Int8LinearMethod,
    _quantize_per_channel_int8,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.registry import (
    IluvatarQuantRegistry,
)


def quantize_native_to_w8a8_int8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one unquantized ``[N, K]`` tensor to INT8 + ``[N, 1]`` scale."""
    if weight.ndim != 2 or weight.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("W8A8 conversion requires a 2-D BF16/FP16 weight")
    return _quantize_per_channel_int8(weight.float())


@IluvatarQuantRegistry.register_method
class IluNativeToW8A8Int8OnlineLinearMethod(_IluRuntimeW8A8Int8LinearMethod):
    """Quantize an unquantized Linear onto the static GLM W8A8 path.

    Used for the MTP ``eh_proj``, which the FP8 checkpoint leaves BF16 while
    the offline W4A8 NN checkpoint quantizes it with the other dense Linears.
    """

    source_format = "native"
    source_dtypes = ("bfloat16",)
    original_disposition = "ignored"
    original_method_cls = UnquantizedLinearMethod

    @classmethod
    def from_ilu_runtime_config(cls, context: Any):
        if set(context.group.weights.source.extra):
            raise ValueError(
                f"{context.group.group_id}: a native source takes no extra "
                "weights.source fields"
            )
        return super().from_ilu_runtime_config(context)

    def _quantize(self, layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        return quantize_native_to_w8a8_int8(layer.weight.data)
