# SPDX-License-Identifier: Apache-2.0
"""Per-channel INT8 conversion for native BF16/FP16 weights."""

import torch

from vllm_iluvatar.model_executor.layers.quantization.conversions.fp8_to_int8 import (
    _quantize_per_channel_int8,
)


def quantize_native_to_w8a8_int8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one unquantized ``[N, K]`` tensor to INT8 + ``[N, 1]`` scale."""
    if weight.ndim != 2 or weight.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("W8A8 conversion requires a 2-D BF16/FP16 weight")
    return _quantize_per_channel_int8(weight.float())
