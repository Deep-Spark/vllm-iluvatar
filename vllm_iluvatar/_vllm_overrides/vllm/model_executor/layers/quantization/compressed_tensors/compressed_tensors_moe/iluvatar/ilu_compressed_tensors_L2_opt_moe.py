import ixformer.inference.functions as ixfops  # type: ignore
import torch
import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
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
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backends.ilu_flash_attn import IluFlashAttentionMetadata

logger = init_logger(__name__)


class IluCompressedTensorsL2OptMoEMethod(CompressedTensorsMoEMethod):

    def __init__(
        self,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.has_bias = self.moe.has_bias
        self.pack_factor = 2
        self.group_size = -1
        self.version = 2
        self.gemm_format = envs.VLLM_W4A8_FORMAT
        self.format_mapping = {"NN": 0, "NT": 1, "TN": 2}
        assert self.gemm_format in ["TN", "NN"]

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
        w13_remainder = (hidden_size // self.pack_factor) % 64
        w2_remainder = (intermediate_size_per_partition // self.pack_factor) % 64
        if self.gemm_format == "TN":
            if w13_remainder != 0:
                w13_shape = (
                    num_experts,
                    2 * intermediate_size_per_partition,
                    (hidden_size // self.pack_factor) + 64 - w13_remainder,
                )
            else:
                w13_shape = (
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // self.pack_factor,
                )

            if w2_remainder != 0:
                w2_shape = (
                    num_experts,
                    hidden_size,
                    (intermediate_size_per_partition // self.pack_factor)
                    + 64
                    - w2_remainder,
                )
            else:
                w2_shape = (
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // self.pack_factor,
                )
        else:
            w13_shape = (
                num_experts,
                hidden_size,
                2 * intermediate_size_per_partition // self.pack_factor,
            )
            w2_shape = (
                num_experts,
                intermediate_size_per_partition,
                hidden_size // self.pack_factor,
            )

        # WEIGHTS
        # use process_weights_after_loading to get get right layout if gemm_format is NN
        w13_weight = torch.nn.Parameter(
            torch.empty(w13_shape, dtype=params_dtype), requires_grad=False
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        if self.gemm_format == "NN":
            setattr(w13_weight, "shard_dim", 1)

        if self.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        else:
            setattr(layer, "w13_bias", None)

        if w2_remainder != 0:
            w2_weight = torch.nn.Parameter(
                torch.zeros(w2_shape, dtype=params_dtype), requires_grad=False
            )
        else:
            w2_weight = torch.nn.Parameter(
                torch.empty(w2_shape, dtype=params_dtype), requires_grad=False
            )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        if self.gemm_format == "NN":
            setattr(w2_weight, "shard_dim", 0)

        if self.has_bias:
            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            setattr(layer, "w2_bias", None)

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        # The following scale or zero will use permute(0,2,1) to get right layout, init here to avoid rewrite data_loader
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts, 1, 2 * intermediate_size_per_partition, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        setattr(w13_weight_scale, "shard_dim", 1)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, 1, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        setattr(w2_weight_scale, "shard_dim", 0)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        setattr(layer, "w13_i8_weight_scale", None)
        setattr(layer, "w2_i8_weight_scale", None)
        setattr(layer, "w13_i8_weight_zero", None)
        setattr(layer, "w2_i8_weight_zero", None)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

        self.gemm_format = self.format_mapping[self.gemm_format]

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

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
                if (
                    isinstance(value, (MLACommonMetadata, IluFlashAttentionMetadata))
                    and hasattr(value, "num_prefills")
                    and hasattr(value, "num_decodes")
                ):
                    deepseek_instance = value
                    break
            value_types = {type(value).__name__ for value in attn_metadata.values()}
            is_same_class = len(value_types) == 1
            if is_same_class:
                assert deepseek_instance
                only_decode = use_ep == False and all(
                    t.num_decodes > 0 and t.num_prefills == 0
                    for t in list(attn_metadata.values())
                )
            else:
                if deepseek_instance:
                    only_decode = (
                        use_ep == False
                        and deepseek_instance.num_decodes > 0
                        and deepseek_instance.num_prefills == 0
                    )
                else:
                    only_decode = False
        else:
            only_decode = False
        if use_ep:
            start_eid = layer.ep_rank * layer.local_num_experts
            end_eid = min(
                (layer.ep_rank + 1) * layer.local_num_experts, layer.global_num_experts
            )

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

        if only_decode and self.gemm_format == 2:
            i8_hidden_states, a_scale = ixfops.moe_expand_input_dynamic_scaled_int8(
                hidden_states=x,
                dst_to_src=sorted_token_ids,
                dst_tokens=expand_tokens,
                topk=layer.top_k,
                src_to_dst=src_to_dst,
                topk_ids=None,
                smooth_scales=layer.w13_input_scale,
                output_format=1,
            )

            if i8_hidden_states.shape[-1] != layer.w13_weight.shape[-1] * 2:
                padding = layer.w13_weight.shape[-1] * 2 - i8_hidden_states.shape[-1]
                i8_hidden_states_align = torch.nn.functional.pad(
                    i8_hidden_states, (0, padding), mode="constant", value=0
                )
            else:
                i8_hidden_states_align = i8_hidden_states

            pt_output_1 = ixfops.moe_w4a8_group_gemv(
                input=i8_hidden_states_align,
                weight=layer.w13_weight,
                i_scales=a_scale,
                w_scales=layer.w13_weight_scale,
                output_dtype=dtype,
                tokens_per_experts=expert_sizes_gpu,
                w_i8scales=layer.w13_i8_weight_scale,
                w_i8zeros=layer.w13_i8_weight_zero,
                dst_to_src=None,
                bias=layer.w13_bias,
                format=self.gemm_format,
                group_size=self.group_size,
            )

            if layer.activation.value == "swigluoai":
                pt_output_2, a2_scale = ixfops.activation_swigluoai_dynamic_scaled_int8(
                    input=pt_output_1,
                    bias=None,
                    smooth_scales=layer.w2_input_scale,
                    dst_to_src=sorted_token_ids,
                    topk_ids=None,
                    output_format=1,
                )
            else:
                pt_output_2, a2_scale = ixfops.activation_dynamic_scaled_int8(
                    input=pt_output_1,
                    bias=None,
                    smooth_scales=layer.w2_input_scale,
                    dst_to_src=sorted_token_ids,
                    topk_ids=None,
                    act_type="swiglu",
                    output_format=1,
                )

            if pt_output_2.shape[-1] != layer.w2_weight.shape[-1] * 2:
                padding = layer.w2_weight.shape[-1] * 2 - pt_output_2.shape[-1]
                pt_output_2_align = torch.nn.functional.pad(
                    pt_output_2, (0, padding), mode="constant", value=0
                )
            else:
                pt_output_2_align = pt_output_2

            pt_output_3 = ixfops.moe_w4a8_group_gemv(
                input=pt_output_2_align,
                weight=layer.w2_weight,
                i_scales=a2_scale,
                w_scales=layer.w2_weight_scale,
                output_dtype=dtype,
                tokens_per_experts=expert_sizes_gpu,
                w_i8scales=layer.w2_i8_weight_scale,
                w_i8zeros=layer.w2_i8_weight_zero,
                dst_to_src=sorted_token_ids,
                bias=layer.w2_bias,
                format=self.gemm_format,
                group_size=self.group_size,
            )
            # mul + reduce_sum
            final_hidden_states = ixfops.moe_output_reduce_sum(
                input=pt_output_3.view(num_tokens, layer.top_k, -1),
                topk_weight=topk_weights,
                scaling_factor=layer.runner_scaling_factor,
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

            if i8_hidden_states.shape[-1] != layer.w13_weight.shape[-1] * 2:
                padding = layer.w13_weight.shape[-1] * 2 - i8_hidden_states.shape[-1]
                i8_hidden_states_align = torch.nn.functional.pad(
                    i8_hidden_states, (0, padding), mode="constant", value=0
                )
            else:
                i8_hidden_states_align = i8_hidden_states

            # w4a8 group gemm 1
            pt_output_1 = ixfops.moe_w4a8_group_gemm(
                input=i8_hidden_states_align,
                weight=layer.w13_weight,
                i_scales=a_scale,
                w_scales=layer.w13_weight_scale,
                output_dtype=dtype,
                tokens_per_experts=expert_sizes_cpu,
                w_i8scales=layer.w13_i8_weight_scale,
                w_i8zeros=layer.w13_i8_weight_zero,
                dst_to_src=None,
                bias=layer.w13_bias,
                format=self.gemm_format,
                group_size=self.group_size,
                version=self.version,
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

            if (
                pt_output_2.shape[-1] != layer.w2_weight.shape[-1] * 2
                and self.gemm_format == 2
            ):
                padding = layer.w2_weight.shape[-1] * 2 - pt_output_2.shape[-1]
                pt_output_2_align = torch.nn.functional.pad(
                    pt_output_2, (0, padding), mode="constant", value=0
                )
            else:
                pt_output_2_align = pt_output_2

            # w4a8 group gemm 2 + reorder
            if use_ep:
                pt_output_3 = torch.empty(
                    (num_tokens * layer.top_k, hidden_size),
                    device=x.device,
                    dtype=x.dtype,
                )

                ixfops.moe_w4a8_group_gemm(
                    input=pt_output_2_align,
                    weight=layer.w2_weight,
                    i_scales=a2_scale,
                    w_scales=layer.w2_weight_scale,
                    output_dtype=dtype,
                    tokens_per_experts=expert_sizes_cpu,
                    w_i8scales=layer.w2_i8_weight_scale,
                    w_i8zeros=layer.w2_i8_weight_zero,
                    dst_to_src=sorted_token_ids,
                    bias=layer.w2_bias,
                    format=self.gemm_format,
                    group_size=self.group_size,
                    version=self.version,
                    output=pt_output_3,
                )

                reduce_mask = src_to_dst == -1
                final_hidden_states = ixfops.moe_output_reduce_sum(
                    input=pt_output_3.view(num_tokens, layer.top_k, -1),
                    topk_weight=topk_weights,
                    scaling_factor=layer.runner_scaling_factor,
                    mask=reduce_mask,
                )
            else:
                pt_output_3 = ixfops.moe_w4a8_group_gemm(
                    input=pt_output_2_align,
                    weight=layer.w2_weight,
                    i_scales=a2_scale,
                    w_scales=layer.w2_weight_scale,
                    output_dtype=dtype,
                    tokens_per_experts=expert_sizes_cpu,
                    w_i8scales=layer.w2_i8_weight_scale,
                    w_i8zeros=layer.w2_i8_weight_zero,
                    dst_to_src=sorted_token_ids,
                    bias=layer.w2_bias,
                    format=self.gemm_format,
                    group_size=self.group_size,
                    version=self.version,
                )

                # mul + reduce_sum
                final_hidden_states = ixfops.moe_output_reduce_sum(
                    input=pt_output_3.view(num_tokens, layer.top_k, -1),
                    topk_weight=topk_weights,
                    scaling_factor=layer.runner_scaling_factor,
                    extra_residual=shared_experts_input,
                )
        return final_hidden_states
