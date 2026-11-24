# SPDX-License-Identifier: Apache-2.0
"""Shared weight preparation and execution for dynamic INT8 Linear."""

import torch
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.utils import replace_parameter

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (
    IluCompressedTensorsW8A8Int8,
)


class IluW8A8Int8LinearMethod(LinearMethodBase):
    """Preserve the source loader and convert the assembled rank-local weight."""

    def __init__(
        self,
        original_method: LinearMethodBase,
        *,
        gemm_format: str = "TN",
        mixed_strategy: str = "CHANNEL",
        gemm_mode: str = "PERF",
    ) -> None:
        self.original_method = original_method
        self.scheme = IluCompressedTensorsW8A8Int8(
            strategy="channel",
            is_static_input_scheme=False,
            input_symmetric=True,
            ilu_params={
                "gemm_format": gemm_format,
                "mixed_strategy": mixed_strategy,
                "gemm_mode": gemm_mode,
            },
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
        self.original_method.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        # UnquantizedLinearMethod does not set this, and the scheme reads it to
        # tell a fused module from a plain one.
        layer.logical_widths = output_partition_sizes
        layer.scheme = self.scheme

    def _quantize(self, layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        qweight, channel_scale = self._quantize(layer)
        replace_parameter(layer, "weight", qweight)
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(channel_scale, requires_grad=False),
        )
        # Dynamic symmetric activations: the scheme reads these by name and
        # takes the no-zero-point path when they are absent.
        layer.input_scale = None
        layer.input_zero_point = None
        layer.azp_adj = None
        self.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.scheme.apply_weights(layer, x, bias)
