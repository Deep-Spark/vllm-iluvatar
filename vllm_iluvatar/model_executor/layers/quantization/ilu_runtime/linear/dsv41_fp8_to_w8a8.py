# SPDX-License-Identifier: Apache-2.0
"""Compatibility registration for the shared FP8-to-INT8 linear method."""

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
