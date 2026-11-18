# SPDX-License-Identifier: Apache-2.0
"""Load-time conversion of a block-FP8 Linear to channel W8A8.

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

from vllm_iluvatar.model_executor.layers.quantization.backends.fp8.linear import (
    IluFp8LinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.conversions.fp8_to_int8 import (
    quantize_fp8_block_to_w8a8_int8,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.linear._w8a8 import (
    _IluRuntimeW8A8Int8LinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.registry import (
    IluvatarQuantRegistry,
)


@IluvatarQuantRegistry.register_method
class IluFp8ToW8A8Int8OnlineLinearMethod(_IluRuntimeW8A8Int8LinearMethod):
    """Convert a serialized block-FP8 Linear to channel W8A8."""

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
        block_shape = tuple(source.extra.get("block_shape", ()))
        if (
            set(source.extra) - {"block_shape", "scale_format"}
            or block_shape not in ((128, 128), (32, 32))
            or source.extra.get("scale_format", "ue8m0") != "ue8m0"
        ):
            raise ValueError(
                f"{context.group.group_id}: block-FP8 source requires "
                "block_shape [128, 128] or [32, 32], with optional scale_format ue8m0"
            )
        if tuple(context.base_quant_config.weight_block_size or ()) != block_shape:
            raise ValueError(
                f"{context.group.group_id}: block_shape must match the "
                "checkpoint's weight_block_size"
            )
        from vllm_iluvatar.config.ilu_compute import compute_options

        if (
            isinstance(context.original_method, IluFp8LinearMethod)
            and compute_options("linear").get("fp8_backend") == "fp8"
        ):
            raise ValueError(
                "Explicit FP8 execution conflicts with FP8 -> INT8 online quant"
            )
        return super().from_ilu_runtime_config(context)

    def _get_block_scale(self, layer: torch.nn.Module) -> torch.Tensor:
        scale = getattr(layer, "weight_scale_inv", None)
        if scale is None:
            raise RuntimeError("block-FP8 source scale is missing")
        if (
            self.context.group.weights.source.extra.get("scale_format") == "ue8m0"
            and scale.dtype != torch.float8_e8m0fnu
        ):
            raise ValueError("UE8M0 block-FP8 source requires E8M0 scales")
        return scale.data

    def _quantize(self, layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        qweight, channel_scale = quantize_fp8_block_to_w8a8_int8(
            layer.weight.data,
            self._get_block_scale(layer),
            self.block_shape,
        )
        del layer._parameters["weight_scale_inv"]
        return qweight, channel_scale
