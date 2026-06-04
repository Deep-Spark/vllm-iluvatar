# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from types import SimpleNamespace

import ixformer.inference.functions as ixf_ops
import torch
from compressed_tensors.quantization import QuantizationStrategy
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.ilu_utils import (
    W8A8GemmInfo,
    pad_tensor,
    replace_with_padded,
    unpad_tensor,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (  # noqa: E501
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

logger = init_logger(__name__)


class IluCompressedTensorsW8A8Int8(CompressedTensorsScheme):
    def __init__(
        self,
        strategy: str,
        is_static_input_scheme: bool,
        input_symmetric: bool,
    ):
        import vllm.envs as env

        if env.VLLM_MIX_QUANTIZATION_TYPE == "TENSOR":
            self.strategy = QuantizationStrategy.TENSOR
        elif env.VLLM_MIX_QUANTIZATION_TYPE == "CHANNEL":
            self.strategy = QuantizationStrategy.CHANNEL
        else:
            self.strategy = strategy
        assert (
            self.strategy == QuantizationStrategy.CHANNEL
        ), "IluCompressedTensorsW8A8Int8 only supports channel-wise quantization"
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric
        self.layer_param_names = [
            "weight",
            "weight_scale",
            "input_scale",
            "input_zero_point",
            "azp_adj",
        ]
        self.config = SimpleNamespace(
            is_channelwise=(self.strategy == QuantizationStrategy.CHANNEL),
            is_static_input_scheme=self.is_static_input_scheme,
            input_symmetric=self.input_symmetric,
        )
        self.format = "TN"

    def _get_layer_params(self, layer: torch.nn.Module) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        w_q, w_s, i_s, i_zp, azp_adj = self.layer_param_names
        return (
            getattr(layer, w_q),
            getattr(layer, w_s),
            getattr(layer, i_s, None),
            getattr(layer, i_zp, None),
            getattr(layer, azp_adj, None),
        )

    @classmethod
    def get_min_capability(cls) -> int:
        # turing and up
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        # TN format
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
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

        # INPUT SCALE
        input_zero_point = None
        input_scale = None
        if self.is_static_input_scheme:
            input_scale = BasevLLMParameter(
                data=torch.empty(1, dtype=torch.float32), weight_loader=weight_loader
            )
            if not self.input_symmetric:
                # Note: compressed-tensors stores the zp using the same dtype
                # as the weights
                # AZP loaded as int8 but used as int32
                input_zero_point = BasevLLMParameter(
                    data=torch.empty(1, dtype=torch.int8), weight_loader=weight_loader
                )

        layer.register_parameter("input_zero_point", input_zero_point)
        layer.register_parameter("input_scale", input_scale)
        if not hasattr(layer, "azp_adj"):
            layer.register_parameter("azp_adj", None)

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names
        config = self.config
        # WEIGHT
        weight = getattr(layer, w_q_name)
        self.format = "TN"
        n, k = weight.shape
        """
            auto choose format now
        """
        if n % 64 == 0 and k % 64 == 0:
            self.format = "NN"
            self.gemm_info = W8A8GemmInfo(
                format_str="NN",
                k=k,
                n=n,
            )
            replace_parameter(
                layer,
                w_q_name,
                torch.nn.Parameter(weight.data.t().contiguous(), requires_grad=False),
            )
            logger.info_once(
                f"[IluCompressedTensorsW8A8Int8 process_weights_after_loading weight] use NN format out_size:{n} in_size:{k} weight:{weight.shape} -> {getattr(layer, w_q_name).shape}"
            )
        else:
            self.gemm_info = W8A8GemmInfo(
                format_str="TN",
                k=k,
                n=n,
            )
            replace_with_padded(
                layer,
                w_q_name,
                self.gemm_info.get_pad_weight_shape(),
                pad_value=0,
                log_prefix="[IluCompressedTensorsW8A8Int8 process_weights_after_loading weight]",
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
                log_prefix="[IluCompressedTensorsW8A8Int8 process_weights_after_loading bias]",
            )

        # WEIGHT SCALE
        # Cutlass kernels support only per-tensor and per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = getattr(layer, w_s_name)
        if is_fused_module and not config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        if self.strategy == QuantizationStrategy.CHANNEL:
            if weight_scale.shape[0] != self.gemm_info.get_pad_n():
                target_shape = (self.gemm_info.get_pad_n(), weight_scale.shape[1])
                replace_with_padded(
                    layer,
                    w_s_name,
                    target_shape,
                    pad_value=1,
                    log_prefix="[IluCompressedTensorsW8A8Int8 process_weights_after_loading weight_scale]",
                )

        # INPUT SCALE
        if config.is_static_input_scheme:
            input_scale = getattr(layer, i_s_name)

            if config.input_symmetric:
                replace_parameter(
                    layer,
                    i_s_name,
                    torch.nn.Parameter(input_scale.max(), requires_grad=False),
                )
                setattr(layer, i_zp_name, None)
            else:
                input_zero_point = getattr(layer, i_zp_name)

                # reconstruct the ranges
                int8_traits = torch.iinfo(torch.int8)
                azps = input_zero_point.to(dtype=torch.int32)
                range_max = (input_scale * (int8_traits.max - azps)).max()
                range_min = (input_scale * (int8_traits.min - azps)).min()

                scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
                replace_parameter(
                    layer, i_s_name, torch.nn.Parameter(scale, requires_grad=False)
                )

                # AZP loaded as int8 but used as int32
                azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)
                replace_parameter(
                    layer, i_zp_name, torch.nn.Parameter(azp, requires_grad=False)
                )

        # azp_adj is the AZP adjustment term, used to account for weights.
        # It does not depend on scales or azp, so it is the same for
        # static and dynamic quantization.
        # For more details, see csrc/quantization/w8a8/cutlass/Epilogues.md
        # https://github.com/vllm-project/vllm/blob/main/csrc/quantization/w8a8/cutlass/Epilogues.md
        if not config.input_symmetric:
            weight = getattr(layer, w_q_name)
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
            if config.is_static_input_scheme:
                # cutlass_w8a8 requires azp to be folded into azp_adj
                # in the per-tensor case
                azp_adj = getattr(layer, i_zp_name) * azp_adj
            setattr(
                layer,
                azp_adj_name,
                torch.nn.Parameter(azp_adj, requires_grad=False),
            )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
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

        if x_zp is not None:
            # Currently, static is always per-tensor and dynamic is per-token
            static = i_zp is not None
            azp = None if static else x_zp
            return ops.cutlass_scaled_mm_azp(
                x_q,
                w_q,
                scale_a=x_s,
                scale_b=w_s,
                out_dtype=out_dtype,
                azp_adj=azp_adj,
                azp=azp,
                bias=bias,
            )

        # pad tensor
        if x_q.shape[-1] != self.gemm_info.pad_k:
            x_q = pad_tensor(x_q, self.gemm_info.pad_k)

        # w8a8 call
        out = torch.empty(
            (x_q.shape[0], self.gemm_info.get_pad_n()),
            dtype=out_dtype,
            device=x_q.device,
        )
        ixf_ops.w8a8(
            x_q,
            w_q,
            x_s,
            w_s,
            bias,
            format=self.gemm_info.format_str,
            output=out,
            out_dtype=out_dtype,
        )

        # unpad tensor
        out = unpad_tensor(out, self.gemm_info.get_n(), contiguous=True)
        return out
