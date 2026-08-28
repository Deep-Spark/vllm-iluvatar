# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar W4A8 INT compressed-tensors linear scheme."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import ixformer.inference.functions as ixf_ops
import torch
from compressed_tensors.quantization import QuantizationStrategy
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (  # noqa: E501
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.scalar_type import scalar_types

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (  # noqa: E501
    W4A8GemmInfo,
    pad_tensor,
    replace_with_padded,
    unpad_tensor,
)


__all__ = ["IluCompressedTensorsW4A8Int"]

W4A8_SUPPORTED_TYPES_MAP = {
    4: scalar_types.int4,
}
W4A8_SUPPORTED_BITS = list(W4A8_SUPPORTED_TYPES_MAP.keys())


class IluCompressedTensorsW4A8Int(CompressedTensorsScheme):
    def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: int | None = None,
        is_static_input_scheme: bool = False,
        input_symmetric: bool = True,
        ilu_params: dict[str, Any] | None = None,
    ):
        ilu_params = ilu_params or {}
        if strategy != QuantizationStrategy.CHANNEL:
            raise ValueError(
                "IluCompressedTensorsW4A8Int only supports channel-wise quantization"
            )
        if group_size not in (None, -1):
            raise ValueError("IluCompressedTensorsW4A8Int only supports group_size=-1")
        self.strategy = strategy
        self.group_size = -1 if group_size is None else group_size
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric

        if num_bits not in W4A8_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {W4A8_SUPPORTED_TYPES_MAP.keys()}"
            )
        self.quant_type = W4A8_SUPPORTED_TYPES_MAP[num_bits]
        self.layer_param_names = [
            "weight",
            "weight_scale",
        ]
        # ``format`` is reserved by Compressed-Tensors for the serialization
        # format (for example ``pack-quantized``).  Use a distinct extension
        # key for the Iluvatar GEMM layout.
        self.format = ilu_params.get("gemm_format", "")
        assert self.format == "NN", f"Invalid format:{self.format}"

    def _get_layer_params(
        self, layer: torch.nn.Module
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        w_q, w_s = self.layer_param_names
        return (
            getattr(layer, w_q),
            getattr(layer, w_s),
        )

    @classmethod
    def get_min_capability(cls) -> int:
        return 1

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        self.gemm_info = W4A8GemmInfo(
            format_str=self.format,
            k=input_size_per_partition,
            n=sum(output_partition_sizes),
        )
        weight = ModelWeightParameter(
            data=torch.empty(self.gemm_info.get_weight_shape(), dtype=torch.int8),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((self.gemm_info.get_n(), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            assert self.strategy == QuantizationStrategy.TENSOR
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q_name, w_s_name = self.layer_param_names

        # WEIGHT
        replace_with_padded(
            layer,
            w_q_name,
            self.gemm_info.get_pad_weight_shape(),
            pad_value=0,
            log_prefix="[IluCompressedTensorsW4A8Int process_weights_after_loading weight]",
        )

        # BIAS (optional)
        # Keep bias aligned with padded N so epilogue can read contiguous
        # tensors without shape mismatch on the padded channels.
        if getattr(layer, "bias", None) is not None:
            replace_with_padded(
                layer,
                "bias",
                (self.gemm_info.get_pad_n(),),
                pad_value=0,
                log_prefix="[IluCompressedTensorsW4A8Int process_weights_after_loading bias]",
            )

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = getattr(layer, w_s_name)
            if weight_scale.shape[0] != self.gemm_info.get_pad_n():
                target_shape = (self.gemm_info.get_pad_n(), weight_scale.shape[1])
                replace_with_padded(
                    layer,
                    w_s_name,
                    target_shape,
                    pad_value=1,
                    log_prefix="[IluCompressedTensorsW4A8Int process_weights_after_loading weight_scale]",
                )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        w_q, w_s = self._get_layer_params(layer)
        i_s, i_zp, azp_adj = None, None, None
        symmetric = azp_adj is None

        if isinstance(x, tuple):
            # unpack tensor which has been quantified
            x_q, x_s, out_dtype = x
            x_zp = None
        else:
            out_dtype = x.dtype
            x_q, x_s, x_zp = ops.scaled_int8_quant(
                x.contiguous(), i_s, i_zp, symmetric=symmetric
            )
        assert x_zp is None, (
            "IluCompressedTensorsW4A8Int does not support asymmetric quantization for input"
        )

        # pad tensor
        if x_q.shape[-1] != self.gemm_info.pad_k:
            x_q = pad_tensor(x_q, self.gemm_info.pad_k)

        # w4a8 call
        out = torch.empty(
            (x_q.shape[0], self.gemm_info.get_pad_n()),
            dtype=out_dtype,
            device=x_q.device,
        )
        ixf_ops.w4a8(
            input=x_q,
            weight=w_q,
            i_scales=x_s,
            w_scales=w_s,
            w_i8scales=None,
            w_i8zeros=None,
            bias=bias,
            output=out,
            format=self.gemm_info.format_str,
            group_size=self.group_size,
            output_dtype=out_dtype,
        )

        # unpad tensor
        out = unpad_tensor(out, self.gemm_info.get_n(), contiguous=True)
        return out
