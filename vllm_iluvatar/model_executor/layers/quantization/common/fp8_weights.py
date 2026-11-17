# SPDX-License-Identifier: Apache-2.0
"""FP8 source parameter assembly without selecting an FP8 compute kernel."""

import torch
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    validate_fp8_block_shape,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
)
from vllm.model_executor.utils import is_weights_pre_processed, replace_parameter


def create_fp8_source_weights(
    method,
    layer,
    input_size_per_partition,
    output_partition_sizes,
    input_size,
    output_size,
    params_dtype,
    **attrs,
):
    if is_weights_pre_processed():
        raise ValueError("FP8 to INT8 conversion requires original checkpoint weights")
    block = method.quant_config.weight_block_size
    loader = attrs.get("weight_loader")
    layer.logical_widths = output_partition_sizes
    layer.input_size_per_partition = input_size_per_partition
    layer.output_size_per_partition = sum(output_partition_sizes)
    layer.orig_dtype = params_dtype
    layer.weight_block_size = block
    if block is not None:
        validate_fp8_block_shape(
            layer,
            input_size,
            output_size,
            input_size_per_partition,
            output_partition_sizes,
            block,
        )
    layer.register_parameter(
        "weight",
        create_fp8_weight_parameter(
            layer.output_size_per_partition, input_size_per_partition, loader
        ),
    )
    layer.register_parameter(
        "weight_scale_inv" if block else "weight_scale",
        create_fp8_scale_parameter(
            BlockQuantScaleParameter if block else ChannelQuantScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            block,
            loader,
            scale_dtype=(
                torch.float8_e8m0fnu
                if getattr(method.quant_config, "is_scale_e8m0", False)
                else torch.float32
            ),
        ),
    )


def finalize_fp8_to_int8(layer):
    from vllm_iluvatar.model_executor.layers.quantization.conversions.fp8_to_int8 import (
        _quantize_per_channel_int8,
        quantize_fp8_block_to_w8a8_int8,
    )

    if layer.weight_block_size is not None:
        weight, scale = quantize_fp8_block_to_w8a8_int8(
            layer.weight.data,
            layer.weight_scale_inv.data,
            tuple(layer.weight_block_size),
        )
        del layer.weight_scale_inv
    else:
        scale = layer.weight_scale.float().reshape(-1)
        if scale.numel() == len(layer.logical_widths):
            scale = torch.repeat_interleave(
                scale, torch.tensor(layer.logical_widths, device=scale.device)
            )
        if scale.numel() not in (1, layer.weight.shape[0]):
            raise ValueError("FP8 channel scale does not match the loaded weight")
        value = layer.weight.float() * scale[:, None]
        weight, scale = _quantize_per_channel_int8(value)
        del layer.weight_scale
    replace_parameter(layer, "weight", weight)
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(scale, requires_grad=False)
    )
    layer.input_scale = None
    layer.input_zero_point = None
    layer.azp_adj = None
