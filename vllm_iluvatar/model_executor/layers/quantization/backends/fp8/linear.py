# SPDX-License-Identifier: Apache-2.0
"""Serialized FP8 sources with architecture-selected FP8 or INT8 execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ixformer.inference.functions as ixfops
import torch
from ixformer.inference.functions import fp8_per_token_dynamic_scaled
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
)
from vllm.model_executor.parameter import ChannelQuantScaleParameter
from vllm.model_executor.utils import replace_parameter

from vllm_iluvatar.model_executor.layers.quantization import IluvatarQuantRegistry
from vllm_iluvatar.model_executor.layers.quantization.capabilities import is_ivcore30

if TYPE_CHECKING:
    from vllm_iluvatar.model_executor.layers.quantization.configs.fp8 import (
        IluFp8Config,
    )

logger = init_logger(__name__)

_IXMM_FP8_GEMV_MAX_M = 16


def _can_fuse_bf16_fp8_gemv(x, layer, gemm_format, bias):
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and 1 <= x.shape[0] <= 6
        and 64 <= x.shape[1] <= 8192
        and x.shape[1] % 64 == 0
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and gemm_format == "TN"
        and getattr(layer, "_can_use_gemm", False)
        and layer.weight.shape[0] % 32 == 0
        and layer.weight.stride(1) == 1
        and layer.weight.stride(0) % 64 == 0
        and layer.weight_scale.is_contiguous()
        and (bias is None or bias.is_contiguous())
    ):
        return False
    # Runtime pointer checks stay in the native op for compiled graphs.
    if not torch.compiler.is_compiling() and (
        x.data_ptr() % 16
        or layer.weight.data_ptr() % 16
        or (bias is not None and bias.data_ptr() % 4)
    ):
        return False
    return is_ivcore30()


@IluvatarQuantRegistry.register_method
class IluFp8LinearMethod(Fp8LinearMethod):
    """Serialized FP8 source with a load-time FP8 or INT8 compute choice.

    ivcore11 defaults to INT8 W8A8; ivcore30 preserves FP8 unless explicitly
    overridden. Both keep checkpoint parameters and TP loaders until post-load.
    Only the selected compute implementation performs weight finalization.
    """

    def __init__(self, quant_config: IluFp8Config) -> None:
        super().__init__(quant_config)
        from vllm_iluvatar.config.ilu_compute import fp8_linear_backend

        self.gemm_format = getattr(quant_config, "gemm_format", "TN").upper()
        self.compute_backend = fp8_linear_backend()
        self.int8_scheme = None
        if self.compute_backend == "w8a8_int8":
            from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (
                IluCompressedTensorsW8A8Int8,
            )

            if self.act_q_static:
                raise ValueError("FP8 -> INT8 currently requires dynamic activations")
            self.int8_scheme = IluCompressedTensorsW8A8Int8(
                strategy="channel",
                is_static_input_scheme=False,
                input_symmetric=True,
                ilu_params={
                    "gemm_format": "TN",
                    "mixed_strategy": "CHANNEL",
                    "gemm_mode": "PERF",
                },
            )
        self.supports_pre_processed_weights = self.int8_scheme is None
        logger.info_once("FP8 Linear execution backend: %s", self.compute_backend)

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
        if self.int8_scheme is not None:
            from vllm_iluvatar.model_executor.layers.quantization.common.fp8_weights import (
                create_fp8_source_weights,
            )

            create_fp8_source_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                **extra_weight_attrs,
            )
            layer.scheme = self.int8_scheme
            return
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        if self.block_quant:
            # Parent registered weight_scale_inv, not weight_scale. Overlaying
            # a channel weight_scale via replace_parameter(None) plants a
            # regular attribute and KeyErrors on register_parameter.
            return
        weight_loader = extra_weight_attrs.get("weight_loader")
        scale = create_fp8_scale_parameter(
            ChannelQuantScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
        )
        replace_parameter(layer, "weight_scale", None)
        layer.register_parameter("weight_scale", scale)
        weight = layer.weight
        layer._can_use_gemm = (
            weight.shape[0] % 32 == 0
            if self.gemm_format == "TN"
            else weight.shape[1] % 64 == 0
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Finalize the selected precision and layout once, before graph capture."""
        if self.int8_scheme is not None:
            from vllm_iluvatar.model_executor.layers.quantization.common.fp8_weights import (
                finalize_fp8_to_int8,
            )

            finalize_fp8_to_int8(layer)
            self.int8_scheme.process_weights_after_loading(layer)
            return
        if self.block_quant:
            return super().process_weights_after_loading(layer)

        scale = layer.weight_scale.to(torch.float32).reshape(-1)
        n = layer.output_size_per_partition
        if scale.numel() == n:
            channel_scale = scale.contiguous()
        elif scale.numel() == 1:
            channel_scale = scale.expand(n).contiguous()
        elif scale.numel() == len(layer.logical_widths):
            channel_scale = torch.repeat_interleave(
                scale,
                torch.tensor(
                    layer.logical_widths,
                    dtype=torch.int64,
                    device=scale.device,
                ),
            ).contiguous()
        else:
            raise ValueError(
                f"Cannot map weight_scale shape {tuple(layer.weight_scale.shape)} "
                f"to {n} output channels"
            )

        replace_parameter(layer, "weight_scale", channel_scale)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.int8_scheme is not None:
            return self.int8_scheme.apply_weights(layer, x, bias)
        if self.block_quant:
            return super().apply(layer, x, bias)
        x_2d = x.reshape(-1, x.shape[-1])

        output = torch.empty(
            (x_2d.shape[0], layer.output_size_per_partition),
            dtype=torch.bfloat16,
            device=x.device,
        )
        bias = None if bias is None else bias.to(torch.bfloat16)

        if _can_fuse_bf16_fp8_gemv(x_2d, layer, self.gemm_format, bias):
            ixfops.per_channel_gemv_bf16_fp8_bf16(
                x_2d, layer.weight, layer.weight_scale, output, bias
            )
            return output.view(*x.shape[:-1], layer.output_size_per_partition)

        qx, scale_x = fp8_per_token_dynamic_scaled(x_2d)
        scale_w = layer.weight_scale
        m = qx.shape[0]
        can_use_gemm = getattr(layer, "_can_use_gemm", False)

        if can_use_gemm and m <= _IXMM_FP8_GEMV_MAX_M:
            ixfops.fp8_per_channel_gemv(
                qx,
                layer.weight,
                scale_x,
                scale_w,
                output,
                bias,
                format=self.gemm_format,
                backend="ixmm",
            )
        else:
            ixfops.fp8_per_channel_mm(
                qx, layer.weight, scale_x, scale_w, output, bias, backend="ixmm"
            )

        result = output.to(x.dtype) if x.dtype != torch.bfloat16 else output
        return result.view(*x.shape[:-1], layer.output_size_per_partition)
