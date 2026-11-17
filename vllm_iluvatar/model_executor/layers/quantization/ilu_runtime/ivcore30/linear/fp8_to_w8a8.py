# SPDX-License-Identifier: Apache-2.0
"""Load-time conversion of a block-FP8 Linear to the static GLM W8A8 path.

Per-channel scales are computed from this rank's shard. For the column-parallel
and replicated layers that is the whole checkpoint row; for the row-parallel
``o_proj`` / ``down_proj`` it is a K slice, so the scale is finer than the
offline checkpoint's full-K scale. That stays numerically valid -- each rank
dequantizes its own partial product before the reduction -- and unlike W4A8 NN
there is no second scale stage that depends on the INT8 amax being exactly 127.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_iluvatar.model_executor.layers.quantization.backends.ivcore30.fp8.linear import (
    IluFp8LinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.ivcore30.linear._w8a8 import (
    _IluRuntimeW8A8Int8LinearMethod,
    _quantize_per_channel_int8,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.registry import (
    IluvatarQuantRegistry,
)


def quantize_fp8_block_to_w8a8_int8(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    block_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one block-FP8 ``[N, K]`` tensor to INT8 + ``[N, 1]`` scale.

    Dequantize and requantize one source row-block at a time so a large
    projection never needs a full FP32 copy of itself.
    """
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError("W8A8 conversion requires a 2-D float8_e4m3fn weight")
    n, k = weight.shape
    bn, bk = block_shape
    expected = ((n + bn - 1) // bn, (k + bk - 1) // bk)
    if tuple(block_scale.shape) != expected or block_scale.dtype != torch.float32:
        raise ValueError(
            f"block scale shape/dtype {tuple(block_scale.shape)}/{block_scale.dtype} "
            f"does not match expected {expected}/torch.float32"
        )

    qweight = torch.empty_like(weight, dtype=torch.int8)
    channel_scale = torch.empty((n, 1), dtype=torch.float32, device=weight.device)
    for n0 in range(0, n, bn):
        n1 = min(n0 + bn, n)
        row_scale = block_scale[n0 // bn].repeat_interleave(bk)[:k]
        value = weight[n0:n1].float() * row_scale.unsqueeze(0)
        qweight[n0:n1], channel_scale[n0:n1] = _quantize_per_channel_int8(value)
    return qweight, channel_scale


@IluvatarQuantRegistry.register_method
class IluFp8ToW8A8Int8OnlineLinearMethod(_IluRuntimeW8A8Int8LinearMethod):
    """Replace a serialized block-FP8 Linear with the static GLM W8A8 path."""

    source_format = "fp8-block"
    source_dtypes = ("float8_e4m3fn",)
    original_disposition = "quantized"
    original_method_cls = IluFp8LinearMethod

    def __init__(self, context: Any) -> None:
        super().__init__(context)
        self.block_shape = tuple(context.group.weights.source.extra["block_shape"])

    @classmethod
    def from_ilu_runtime_config(cls, context: Any):
        source = context.group.weights.source
        if set(source.extra) != {"block_shape"} or tuple(
            source.extra["block_shape"]
        ) != (128, 128):
            raise ValueError(
                f"{context.group.group_id}: block-FP8 source requires "
                "block_shape [128, 128]"
            )
        if tuple(context.base_quant_config.weight_block_size or ()) != (128, 128):
            raise ValueError(
                f"{context.group.group_id}: block_shape must match the "
                "checkpoint's weight_block_size"
            )
        return super().from_ilu_runtime_config(context)

    def _quantize(self, layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        scale = getattr(layer, "weight_scale_inv", None)
        if scale is None:
            raise RuntimeError("block-FP8 source scale is missing")
        qweight, channel_scale = quantize_fp8_block_to_w8a8_int8(
            layer.weight.data, scale.data, self.block_shape
        )
        del layer._parameters["weight_scale_inv"]
        return qweight, channel_scale
