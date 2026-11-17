# SPDX-License-Identifier: Apache-2.0
"""Load-time conversion of a block-FP8 GLM-5.2 ``kv_b_proj`` to grouped W8A16."""

from __future__ import annotations

from typing import Any

import torch

from vllm_iluvatar.model_executor.layers.quantization.backends.grouped_bmm.glm_kv_b_proj import (
    IluGlmKVBProjW8A16Method,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.registry import (
    IluvatarQuantRegistry,
)


@IluvatarQuantRegistry.register_method
class IluFp8ToGlmKVBProjW8A16Method(IluGlmKVBProjW8A16Method):
    """Load a block-FP8 ``kv_b_proj`` into the W8A16 grouped-marlin layout.

    The official FP8 checkpoint has no ``mla_bmm_w8a16`` field to route
    ``kv_b_proj`` here, so without this method it stays on
    ``IluFp8LinearMethod`` and ``fp8_attention`` rebuilds the absorbed up-proj
    as grouped FP8 einsums instead. Selecting this group restores the offline
    checkpoint's kernel: the FP8 loader still fills ``weight`` /
    ``weight_scale_inv``, then the block scale is folded into a per-channel INT8
    weight and handed to the inherited split + Marlin repack.

    Activations stay BF16 (W8A16), so the group carries no
    ``input_activations``.
    """

    def __init__(self, context: Any) -> None:
        self.context = context
        self.parent_config = context.base_quant_config
        self.original_method = context.original_method
        self.block_shape = tuple(context.group.weights.source.extra["block_shape"])

    @classmethod
    def from_ilu_runtime_config(cls, context: Any) -> IluFp8ToGlmKVBProjW8A16Method:
        from vllm.model_executor.layers.linear import LinearBase

        from vllm_iluvatar.model_executor.layers.quantization.backends.ivcore30.fp8.linear import (
            IluFp8LinearMethod,
        )

        group = context.group
        weights = group.weights
        source, target = weights.source, weights.target
        where = group.group_id
        if not isinstance(context.layer, LinearBase) or not isinstance(
            context.original_method, IluFp8LinearMethod
        ):
            raise TypeError(f"{where}: W8A16 kv_b_proj requires a block-FP8 Linear")
        if group.require.original_disposition != "quantized":
            raise ValueError(f"{where}: W8A16 kv_b_proj requires quantized sources")
        if (
            weights.loading.mode != "transform"
            or weights.loading.stage != "process_weights_after_loading"
        ):
            raise ValueError(
                f"{where}: W8A16 kv_b_proj requires "
                "loading=transform@process_weights_after_loading"
            )
        if (
            source.format != "fp8-block"
            or tuple(source.dtypes) != ("float8_e4m3fn",)
            or set(source.extra) != {"block_shape"}
            or tuple(source.extra["block_shape"]) != (128, 128)
        ):
            raise ValueError(f"{where}: expected a 128x128 block-FP8 source")
        if tuple(context.base_quant_config.weight_block_size or ()) != (128, 128):
            raise ValueError(
                f"{where}: block_shape must match the checkpoint's weight_block_size"
            )
        if (
            target.format != "int8-channel-marlin"
            or target.dtype != "int8"
            or target.strategy != "channel"
            or set(target.extra)
        ):
            raise ValueError(f"{where}: unsupported W8A16 kv_b_proj weight target")
        if group.input_activations is not None or group.output_activations is not None:
            raise ValueError(
                f"{where}: W8A16 keeps BF16 activations, so both activation "
                "fields must be null"
            )
        if set(group.options):
            raise ValueError(f"{where}: unsupported options: {sorted(group.options)}")
        return cls(context)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self.original_method.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        layer.logical_widths = output_partition_sizes
        self.context.record_shape_result(transformed=True)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.quantization.utils import replace_parameter

        from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.ivcore30.linear.fp8_to_w8a8 import (  # noqa: E501
            quantize_fp8_block_to_w8a8_int8,
        )

        scale = getattr(layer, "weight_scale_inv", None)
        if scale is None:
            raise RuntimeError("block-FP8 kv_b_proj scale is missing")
        qweight, channel_scale = quantize_fp8_block_to_w8a8_int8(
            layer.weight.data, scale.data, self.block_shape
        )
        replace_parameter(layer, "weight", qweight)
        del layer._parameters["weight_scale_inv"]
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(channel_scale, requires_grad=False),
        )
        super().process_weights_after_loading(layer)


__all__ = ["IluFp8ToGlmKVBProjW8A16Method"]
