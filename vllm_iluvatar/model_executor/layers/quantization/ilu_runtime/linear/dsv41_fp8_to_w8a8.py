# SPDX-License-Identifier: Apache-2.0
"""Runtime registration for DeepSeek V4.1 grouped output projection."""

from vllm_iluvatar.model_executor.layers.quantization.backends.fp8.deepseek_v41_linear import (
    IluDeepseekV41Bf16DequantLinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.grouped_bmm.deepseek_v4_wo_a import (
    IluDeepseekV41Fp8GroupedInt8Method,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.linear.fp8_to_w8a8 import (
    IluFp8ToW8A8Int8OnlineLinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.registry import (
    IluvatarQuantRegistry,
)

# Existing runtime JSON files may use the model-specific registration name.
IluDsv41Fp8ToW8A8LinearMethod = IluFp8ToW8A8Int8OnlineLinearMethod
IluvatarQuantRegistry.register_method(
    "IluDsv41Fp8ToW8A8LinearMethod", IluFp8ToW8A8Int8OnlineLinearMethod
)


@IluvatarQuantRegistry.register_method
class IluDsv41Fp8ToW8A8GroupedLinearMethod(
    IluDeepseekV41Fp8GroupedInt8Method, IluFp8ToW8A8Int8OnlineLinearMethod
):
    """Validate the runtime override and reuse the grouped FP8-source method."""

    original_method_cls = IluDeepseekV41Bf16DequantLinearMethod

    def __init__(self, context):
        super().__init__(context.base_quant_config)
        self.context = context

    def create_weights(self, *args, **kwargs):
        super().create_weights(*args, **kwargs)
        self.context.record_shape_result(transformed=True)

    def process_weights_after_loading(self, layer):
        self._get_block_scale(layer)
        super().process_weights_after_loading(layer)

    def apply_quantized(self, layer, activations, scales, *, output_dtype):
        self.context.assert_validation_complete()
        return super().apply_quantized(
            layer, activations, scales, output_dtype=output_dtype
        )
