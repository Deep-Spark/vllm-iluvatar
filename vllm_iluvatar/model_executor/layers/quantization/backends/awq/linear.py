# SPDX-License-Identifier: Apache-2.0
"""Iluvatar AWQ linear backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.auto_awq import BaseAWQLinearMethod
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    verify_marlin_supported,
)
from vllm.scalar_type import scalar_types

if TYPE_CHECKING:
    from ...configs.awq import IluAutoAWQConfig


class IluAWQMarlinLinearMethod(BaseAWQLinearMethod):
    """Linear AWQ method backed by Iluvatar W4A16 kernels."""

    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quant_config: IluAutoAWQConfig) -> None:
        super().__init__(quant_config)
        self.quant_config = quant_config
        self.quant_type = scalar_types.uint4
        self.input_dtype = None
        verify_marlin_supported(
            quant_type=self.quant_config.quant_type,
            group_size=self.quant_config.group_size,
            has_zp=self.quant_config.zero_point,
        )

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
        super().create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )
        output_size_per_partition = sum(output_partition_sizes)
        group_size = (
            self.quant_config.group_size
            if self.quant_config.group_size != -1
            else input_size
        )
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.num_groups = input_size_per_partition // group_size

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.output_size_per_partition = (
            layer.qweight.data.shape[1] * self.quant_config.pack_factor
        )
        align_size = 64 * 8 // self.quant_config.weight_bits
        if layer.output_size_per_partition % align_size != 0:
            padded_output_size = (
                (layer.output_size_per_partition + align_size - 1)
                // align_size
                * align_size
            )
            layer.output_padding_size = (
                padded_output_size - layer.output_size_per_partition
            )
            device = layer.qweight.device
            packed_output = (
                layer.output_size_per_partition // self.quant_config.pack_factor
            )
            padded_packed_output = padded_output_size // self.quant_config.pack_factor

            pad_qweight = torch.zeros(
                layer.input_size_per_partition,
                padded_packed_output,
                dtype=torch.int32,
                device=device,
            )
            pad_qzeros = torch.zeros(
                layer.num_groups,
                padded_packed_output,
                dtype=torch.int32,
                device=device,
            )
            pad_scales = torch.zeros(
                layer.num_groups,
                padded_output_size,
                dtype=layer.scales.data.dtype,
                device=device,
            )
            pad_qweight[..., :packed_output] = layer.qweight.data
            pad_qzeros[..., :packed_output] = layer.qzeros.data
            pad_scales[..., : layer.output_size_per_partition] = layer.scales.data
            replace_parameter(layer, "qweight", pad_qweight)
            replace_parameter(layer, "qzeros", pad_qzeros)
            replace_parameter(layer, "scales", pad_scales)

        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        out = ops.awq_gemm(
            reshaped_x,
            qweight,
            scales,
            qzeros,
            pack_factor,
            group_size=self.quant_config.group_size,
        )
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
