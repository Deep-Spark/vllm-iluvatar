# SPDX-License-Identifier: Apache-2.0
"""Iluvatar GPTQ linear backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_repeat_scales_on_all_ranks,
    verify_marlin_supported,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
)

from vllm_iluvatar.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)

if TYPE_CHECKING:
    from ...configs.gptq import IluAutoGPTQConfig

logger = init_logger(__name__)


class IluAutoGPTQLinearMethod(LinearMethodBase):
    """Linear method for AutoGPTQ using Marlin kernels.

    Args:
        quant_config: The AutoGPTQ quantization config.
    """

    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quant_config: IluAutoGPTQConfig) -> None:
        self.quant_config = quant_config
        self.input_dtype = None
        self.quant_type = self.quant_config.quant_type

        # Verify supported on platform.
        verify_marlin_supported(
            quant_type=self.quant_config.quant_type,
            group_size=self.quant_config.group_size,
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
        output_size_per_partition = sum(output_partition_sizes)
        is_row_parallel = input_size != input_size_per_partition
        weight_loader = extra_weight_attrs.get("weight_loader")
        input_dtype = self.input_dtype

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_config.quant_type,
            act_type=params_dtype if input_dtype is None else input_dtype,
            group_size=self.quant_config.group_size,
            zero_points=False,
        )

        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for IluAutoGPTQLinearMethod", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        # Determine sharding
        if marlin_repeat_scales_on_all_ranks(
            self.quant_config.group_size, is_row_parallel
        ):
            # By setting scale_dim == None, weight_loader will
            # repeat the scales on each GPU in TP>1 case.
            scales_and_zp_input_dim = None
            scales_and_zp_size = input_size // group_size
        else:
            # By setting scale_dim == 0, weight_loader will
            # shard the scales in TP>1 case.
            scales_and_zp_input_dim = 0
            scales_and_zp_size = input_size_per_partition // group_size

        # Quantized weights
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        qzeros_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader": weight_loader,
        }
        weight_scale_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader": weight_loader,
        }

        if scales_and_zp_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1, **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        else:
            scales = GroupQuantScaleParameter(
                output_dim=1, input_dim=0, **weight_scale_args
            )
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)

        self.kernel = kernel_type(
            mp_linear_kernel_config,
            w_q_param_name="qweight",
            w_s_param_name="scales",
            w_zp_param_name="qzeros",
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)
