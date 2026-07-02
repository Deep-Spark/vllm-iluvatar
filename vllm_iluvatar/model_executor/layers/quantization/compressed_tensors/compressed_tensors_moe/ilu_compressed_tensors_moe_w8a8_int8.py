# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)
import ixformer.inference.functions as ixfops  # type: ignore
import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    SharedExperts,
    FusedMoeWeightScaleSupported,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    make_int8_moe_kernel,
    make_int8_moe_quant_config,
    select_int8_moe_backend,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata

logger = init_logger(__name__)


def _get_ilu_output_scale(layer: RoutedExperts) -> float:
    return getattr(layer, "_ilu_output_scale", 1.0)


def _is_supported_decode_metadata(value) -> bool:
    """Accept community metadata plus Iluvatar fork metadata by class name."""
    return isinstance(value, MLACommonMetadata) or (
        value.__class__.__name__ == "IluFlashAttentionMetadata"
    )


class IluCompressedTensorsW8A8Int8MoEMethod(CompressedTensorsMoEMethod):
    """W8A8 Int8 MoE quantization using compressed tensors."""

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(moe)
        self.has_bias = self.moe.has_bias
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        
        self.group_size = -1 if self.weight_quant.group_size is None else self.weight_quant.group_size
        self.gemm_format = envs.VLLM_W8A8_FORMAT
        assert self.gemm_format in ["TN", "NN"], f"W8A8 INT8 MoE only supports TN or NN format, but got {self.gemm_format}"
        self.padding_dim = -1 if self.gemm_format == "TN" else -2

        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}"
            )

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales."
            )

        # Select Int8 MoE backend.
        self.int8_backend, self.experts_cls = select_int8_moe_backend(
            config=self.moe,
            weight_key=kInt8StaticChannelSym,
            activation_key=kInt8DynamicTokenSym,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        params_dtype = torch.int8
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        w13_remainder = hidden_size % 64
        w2_remainder = intermediate_size_per_partition % 64
        if w13_remainder != 0:
            hidden_size_padded = hidden_size + (64 - w13_remainder)
        else:
            hidden_size_padded = hidden_size
        if w2_remainder != 0:
            intermediate_size_per_partition_padded = intermediate_size_per_partition + (64 - w2_remainder)
        else:
            intermediate_size_per_partition_padded = intermediate_size_per_partition

        if self.gemm_format == "NN":
            w13_shape = (num_experts, hidden_size_padded, w13_num_shards * intermediate_size_per_partition_padded)
            w2_shape = (num_experts, intermediate_size_per_partition_padded, hidden_size_padded)
        else:
            w13_shape = (num_experts, w13_num_shards * intermediate_size_per_partition, hidden_size_padded)
            w2_shape = (num_experts, hidden_size, intermediate_size_per_partition_padded)
             
        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.zeros(
            w13_shape, 
            dtype=params_dtype), requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.zeros(
            w2_shape, 
            dtype=params_dtype), requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.gemm_format == "TN":
            w13_bias_shape = (num_experts, 2 * intermediate_size_per_partition)
            w2_bias_shape = (num_experts, hidden_size)
        elif self.gemm_format == "NN":
            w13_bias_shape = (num_experts, 2 * intermediate_size_per_partition_padded)
            w2_bias_shape = (num_experts, hidden_size_padded)

        if self.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    w13_bias_shape, 
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        else:
            setattr(layer, "w13_bias", None)

        if self.has_bias:
            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    w2_bias_shape,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            setattr(layer, "w2_bias", None)

        # WEIGHT_SCALES
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        if self.gemm_format == "NN":
            w13_weight_scale_shape = (num_experts, w13_num_shards * intermediate_size_per_partition_padded, 1)
            w2_weight_scale_shape = (num_experts, hidden_size_padded, 1)
        else:
            w13_weight_scale_shape = (num_experts, w13_num_shards * intermediate_size_per_partition, 1)
            w2_weight_scale_shape = (num_experts, hidden_size, 1)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(w13_weight_scale_shape, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(w2_weight_scale_shape, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
            w13_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, hidden_size, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, intermediate_size_per_partition, dtype=torch.float32),
                                                requires_grad=False)
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        pass


    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        return make_int8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        # Assign the value of shared_experts_output to variable shared_experts_input for fusion
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        attn_metadata = get_forward_context().attn_metadata
        use_ep = layer.expert_map is not None
        # unsupported ep now
        if attn_metadata:
            deepseek_instance = None
            for value in attn_metadata.values():
                if _is_supported_decode_metadata(value) and hasattr(value, 'num_prefills') and hasattr(value, 'num_decodes'):
                    deepseek_instance = value
                    break
            value_types = {type(value).__name__ for value in attn_metadata.values()}
            is_same_class = len(value_types) == 1
            if is_same_class:
                assert deepseek_instance
                only_decode = (use_ep == False and all(t.num_decodes > 0 and t.num_prefills ==0 for t in list(attn_metadata.values())))
            else:
                if deepseek_instance:
                    only_decode = (use_ep == False and deepseek_instance.num_decodes > 0 and deepseek_instance.num_prefills ==0)
                else:
                    only_decode = False
        else:
            only_decode = False

        if use_ep:
            ep_rank = layer.moe_config.ep_rank
            start_eid = ep_rank * layer.local_num_experts
            end_eid = min((ep_rank + 1) * layer.local_num_experts, layer.global_num_experts)
        
        dtype = x.dtype
        num_tokens = topk_ids.shape[0]
        num_experts = layer.global_num_experts
        
        if use_ep:
            hidden_size = x.shape[1]
            (
                src_to_dst,
                sorted_token_ids,
                expert_sizes_gpu,
                expert_sizes_cpu,
                expand_tokens,
            ) = ixfops.moe_compute_token_index_ep(
                topk_ids=topk_ids,
                num_experts=num_experts,
                start_expert_id=start_eid,
                end_expert_id=end_eid,
            )
            if expert_sizes_cpu.sum() == 0:
                return torch.zeros(
                    (num_tokens, hidden_size),
                    device=x.device,
                    dtype=x.dtype,
                )
        else:
            expand_tokens = num_tokens * layer.top_k
            (
                src_to_dst,
                sorted_token_ids,
                expert_sizes_gpu,
                expert_sizes_cpu,
            ) = ixfops.moe_compute_token_index(
                topk_ids=topk_ids,
                num_experts=num_experts,
            )

        if only_decode and self.gemm_format == "NN":

            # expand + reorder + quant
            i8_hidden_states, a_scale = ixfops.moe_expand_input_dynamic_scaled_int8(
                hidden_states=x,
                dst_to_src=sorted_token_ids,
                dst_tokens=expand_tokens,
                topk=layer.top_k,
                src_to_dst=src_to_dst,
                topk_ids=None,
                smooth_scales=layer.w13_input_scale,
            )

            if i8_hidden_states.shape[-1] != layer.w13_weight.shape[-2]:
                padding = layer.w13_weight.shape[-2] - i8_hidden_states.shape[-1]
                i8_hidden_states_align = torch.nn.functional.pad(i8_hidden_states, (0, padding), mode='constant', value=0)
            else:
                i8_hidden_states_align = i8_hidden_states

            # w8a8 group gemm 1
            pt_output_1 = ixfops.moe_w8a8_group_gemv(
                input=i8_hidden_states_align,
                weight=layer.w13_weight,
                i_scales=a_scale,
                w_scales=layer.w13_weight_scale,
                output_dtype=dtype,
                tokens_per_experts=expert_sizes_gpu,
                dst_to_src=None,
                bias=layer.w13_bias,
                format=0,
                group_size=self.group_size,
            )
                
            # act + quant
            if layer.activation.value == "swigluoai":
                pt_output_2, a2_scale = ixfops.activation_swigluoai_dynamic_scaled_int8(
                    input=pt_output_1,
                    bias=None,
                    smooth_scales=layer.w2_input_scale,
                    dst_to_src=sorted_token_ids,
                    topk_ids=None,
                )
            else:
                pt_output_2, a2_scale = ixfops.activation_dynamic_scaled_int8(
                    input=pt_output_1,
                    bias=None,
                    smooth_scales=layer.w2_input_scale,
                    dst_to_src=sorted_token_ids,
                    topk_ids=None,
                    act_type="swiglu",
                )


            if pt_output_2.shape[-1] != layer.w2_weight.shape[-2]:
                padding = layer.w2_weight.shape[-2] - pt_output_2.shape[-1]
                pt_output_2_align = torch.nn.functional.pad(pt_output_2, (0, padding), mode='constant', value=0)
            else:
                pt_output_2_align = pt_output_2

            # w8a8 group gemm 2 + reorder
            pt_output_3 = ixfops.moe_w8a8_group_gemv(
                input=pt_output_2_align,
                weight=layer.w2_weight,
                i_scales=a2_scale,
                w_scales=layer.w2_weight_scale,
                output_dtype=dtype,
                tokens_per_experts=expert_sizes_gpu,
                dst_to_src=sorted_token_ids,
                bias=layer.w2_bias,
                format=0,
                group_size=self.group_size,
            )
            
            final_hidden_states = ixfops.moe_output_reduce_sum(
                input=pt_output_3.view(num_tokens, layer.top_k, -1),
                topk_weight=topk_weights,
                scaling_factor=_get_ilu_output_scale(layer),
                extra_residual=shared_experts_input,
            )
        else:
            expert_sizes_cpu = expert_sizes_gpu.cpu()
            # expand + reorder + quant
            i8_hidden_states, a_scale = ixfops.moe_expand_input_dynamic_scaled_int8(
                hidden_states=x,
                dst_to_src=sorted_token_ids,
                dst_tokens=expand_tokens,
                topk=layer.top_k,
                src_to_dst=src_to_dst,
                topk_ids=None,
                smooth_scales=layer.w13_input_scale,
            )

            if i8_hidden_states.shape[-1] != layer.w13_weight.shape[self.padding_dim]:
                padding = layer.w13_weight.shape[self.padding_dim] - i8_hidden_states.shape[-1]
                i8_hidden_states_align = torch.nn.functional.pad(i8_hidden_states, (0, padding), mode='constant', value=0)
            else:
                i8_hidden_states_align = i8_hidden_states

            # w8a8 group gemm 1
            pt_output_1 = ixfops.moe_w8a8_group_gemm(
                input=i8_hidden_states_align,
                weight=layer.w13_weight,
                i_scales=a_scale,
                w_scales=layer.w13_weight_scale,
                output_dtype=dtype,
                tokens_per_experts=expert_sizes_cpu,
                dst_to_src=None,
                bias=layer.w13_bias,
                format=self.gemm_format,
            )
            
            # act + quant
            if layer.activation.value == "swigluoai":
                pt_output_2, a2_scale = ixfops.activation_swigluoai_dynamic_scaled_int8(
                    input=pt_output_1,
                    bias=None,
                    smooth_scales=layer.w2_input_scale,
                    dst_to_src=sorted_token_ids,
                    topk_ids=None,
                )
            else:
                pt_output_2, a2_scale = ixfops.activation_dynamic_scaled_int8(
                    input=pt_output_1,
                    bias=None,
                    smooth_scales=layer.w2_input_scale,
                    dst_to_src=sorted_token_ids,
                    topk_ids=None,
                    act_type="swiglu",
                )

            if pt_output_2.shape[-1] != layer.w2_weight.shape[self.padding_dim]:
                padding = layer.w2_weight.shape[self.padding_dim] - pt_output_2.shape[-1]
                pt_output_2_align = torch.nn.functional.pad(pt_output_2, (0, padding), mode='constant', value=0)
            else:
                pt_output_2_align = pt_output_2
            
                
            # w8a8 group gemm 2 + reorder
            if use_ep:
                pt_output_3 = torch.empty(
                    (num_tokens * layer.top_k, hidden_size),
                    device=x.device,
                    dtype=x.dtype,
                )
                ixfops.moe_w8a8_group_gemm(
                    input=pt_output_2_align,
                    weight=layer.w2_weight,
                    i_scales=a2_scale,
                    w_scales=layer.w2_weight_scale,
                    output_dtype=dtype,
                    tokens_per_experts=expert_sizes_cpu,
                    dst_to_src=sorted_token_ids,
                    bias=layer.w2_bias,
                    format=self.gemm_format,
                    output=pt_output_3,
                )

                reduce_mask = src_to_dst == -1
                final_hidden_states = ixfops.moe_output_reduce_sum(
                    input=pt_output_3.view(num_tokens, layer.top_k, -1),
                    topk_weight=topk_weights,
                    scaling_factor=_get_ilu_output_scale(layer),
                    mask=reduce_mask,
                    extra_residual=shared_experts_input,
                )
            else:
                pt_output_3 = ixfops.moe_w8a8_group_gemm(
                    input=pt_output_2_align,
                    weight=layer.w2_weight,
                    i_scales=a2_scale,
                    w_scales=layer.w2_weight_scale,
                    output_dtype=dtype,
                    tokens_per_experts=expert_sizes_cpu,
                    dst_to_src=sorted_token_ids,
                    bias=layer.w2_bias,
                    format=self.gemm_format,
                )

                # mul + reduce_sum
                final_hidden_states = ixfops.moe_output_reduce_sum(
                    input=pt_output_3.view(num_tokens, layer.top_k, -1),
                    topk_weight=topk_weights,
                    scaling_factor=_get_ilu_output_scale(layer),
                    extra_residual=shared_experts_input,
                )

        return final_hidden_states
