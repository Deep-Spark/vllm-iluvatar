# SPDX-License-Identifier: Apache-2.0
"""Iluvatar compressed-tensors linear method-selection wrapper.

Corresponds to:
    vllm/model_executor/layers/quantization/compressed_tensors/
    compressed_tensors.py

Target   : CompressedTensorsConfig._get_scheme_from_parts
Baseline : main_4875b4456b5
Why      : preserve the fork's Iluvatar W4A8/W8A8 linear scheme dispatch without
           editing community vLLM source.
How      : ``patch_compressed_tensors`` installs this method wrapper; community
           scheme detection still runs first, then only W4A8/W8A8 INT schemes are
           replaced by Iluvatar implementations.
Scope    : late general-plugin patch during ``register_general()``.
Safety   : no copied community method body; unsupported schemes return the
           original community scheme unchanged.
Remove   : when community compressed-tensors exposes an official backend
           selection hook for vendor linear schemes.
"""

from __future__ import annotations

from collections.abc import Callable

from compressed_tensors.quantization import QuantizationArgs

from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int import (  # noqa: E501
    CompressedTensorsW4A8Int,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (  # noqa: E501
    CompressedTensorsW8A8Int8,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (  # noqa: E501
    IluCompressedTensorsW4A8Int,
    IluCompressedTensorsW8A8Int8,
)

_ORIGINAL_GET_SCHEME_FROM_PARTS: Callable | None = None


def set_original_get_scheme_from_parts(fn: Callable) -> None:
    global _ORIGINAL_GET_SCHEME_FROM_PARTS
    _ORIGINAL_GET_SCHEME_FROM_PARTS = fn


class IluCompressedTensorsConfig:
    """Wrapper surface installed onto community CompressedTensorsConfig."""

    def _get_scheme_from_parts(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        output_quant: QuantizationArgs | None = None,
        format: str | None = None,
        layer_name: str | None = None,
    ):
        if _ORIGINAL_GET_SCHEME_FROM_PARTS is None:
            raise RuntimeError("Original CompressedTensorsConfig method not set")

        scheme = _ORIGINAL_GET_SCHEME_FROM_PARTS(
            self,
            weight_quant=weight_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            format=format,
            layer_name=layer_name,
        )

        # [ILU override] W8A8 INT8 linear scheme - delta: use Iluvatar kernel
        # layout, padding, and ixformer execution while preserving upstream
        # scheme detection.
        if isinstance(scheme, CompressedTensorsW8A8Int8):
            return IluCompressedTensorsW8A8Int8(
                strategy=weight_quant.strategy,
                is_static_input_scheme=not input_quant.dynamic,
                input_symmetric=input_quant.symmetric,
            )

        # [ILU override] W4A8 INT linear scheme - delta: use Iluvatar packed
        # W4A8 layout and ixformer execution.
        if isinstance(scheme, CompressedTensorsW4A8Int):
            return IluCompressedTensorsW4A8Int(
                num_bits=weight_quant.num_bits,
                strategy=weight_quant.strategy,
                group_size=weight_quant.group_size,
                is_static_input_scheme=bool(input_quant and not input_quant.dynamic),
                input_symmetric=input_quant.symmetric,
            )

        return scheme
