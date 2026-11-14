# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 block32/E8M0 checkpoint conversion to existing W8A8 kernels."""

import torch
from vllm.model_executor.layers.quantization.utils import replace_parameter

from vllm_iluvatar.model_executor.layers.quantization.backends.fp8.deepseek_v41_linear import (
    IluDeepseekV41Bf16DequantLinearMethod,
    IluDeepseekV41FP8LinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.linear._w8a8 import (
    _IluRuntimeW8A8Int8LinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.linear.fp8_to_w8a8 import (
    quantize_fp8_block_to_w8a8_int8,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.registry import (
    IluvatarQuantRegistry,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.grouped_bmm.deepseek_v4_wo_a import IluV4W8A8BMMLinearMethod


@IluvatarQuantRegistry.register_method
class IluDsv41Fp8ToW8A8LinearMethod(_IluRuntimeW8A8Int8LinearMethod):
    source_format = "fp8-block"
    source_dtypes = ("float8_e4m3fn",)
    original_disposition = "quantized"
    original_method_cls = IluDeepseekV41FP8LinearMethod

    @classmethod
    def from_ilu_runtime_config(cls, context):
        source = context.group.weights.source
        if (
            set(source.extra) != {"block_shape", "scale_format"}
            or tuple(source.extra["block_shape"]) != (32, 32)
            or source.extra["scale_format"] != "ue8m0"
        ):
            raise ValueError(
                "DSV4.1 W8A8 requires block_shape [32,32], scale_format ue8m0"
            )
        if tuple(context.base_quant_config.weight_block_size or ()) != (32, 32):
            raise ValueError(
                "DSV4.1 W8A8 requires the original block32 checkpoint config"
            )
        return super().from_ilu_runtime_config(context)

    def _quantize(self, layer):
        scale = getattr(layer, "weight_scale_inv", None)
        if scale is None or scale.dtype != torch.float8_e8m0fnu:
            raise ValueError("DSV4.1 W8A8 expects original E8M0 block scales")
        scales = scale.data.float()
        if (
            not torch.isfinite(scales).all()
            or not torch.isfinite(layer.weight.float()).all()
        ):
            raise ValueError("DSV4.1 FP8 source contains non-finite weights or scales")
        result = quantize_fp8_block_to_w8a8_int8(layer.weight.data, scales, (32, 32))
        del layer._parameters["weight_scale_inv"]
        return result


@IluvatarQuantRegistry.register_method
class IluDsv41Fp8ToW8A8GroupedLinearMethod(
    IluDsv41Fp8ToW8A8LinearMethod, IluV4W8A8BMMLinearMethod
):
    """Keep WO-A's grouped contraction and inverse-RoPE quantization contract."""

    original_method_cls = IluDeepseekV41Bf16DequantLinearMethod

    def __init__(self, context):
        IluDsv41Fp8ToW8A8LinearMethod.__init__(self, context)
        IluV4W8A8BMMLinearMethod.__init__(self, context.base_quant_config, {})

    def process_weights_after_loading(self, layer):
        weight, scales = self._quantize(layer)
        replace_parameter(layer, "weight", weight)
        layer.register_parameter(
            "weight_scale", torch.nn.Parameter(scales, requires_grad=False)
        )
        IluV4W8A8BMMLinearMethod.process_weights_after_loading(self, layer)

    def apply(self, layer, x, bias=None):
        raise RuntimeError("Grouped WO-A must use apply_with_rope or apply_quantized")

    def apply_quantized(self, layer, activations, scales, *, output_dtype):
        self.context.assert_validation_complete()
        return IluV4W8A8BMMLinearMethod.apply_quantized(
            self, layer, activations, scales, output_dtype=output_dtype
        )
