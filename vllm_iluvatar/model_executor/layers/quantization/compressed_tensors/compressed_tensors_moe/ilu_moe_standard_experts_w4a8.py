# SPDX-License-Identifier: Apache-2.0
"""Iluvatar standard-layout W4A8 MoE expert implementation.

Corresponds to:
    vllm/model_executor/layers/quantization/compressed_tensors/
    compressed_tensors_moe/*
Target   : W4A8 standard-layout expert kernel for Iluvatar MoE.
Baseline : plugin/vllm 0.24.0 runtime source
Why      : Iluvatar W4A8 MoE uses ixformer kernels with fork-specific format
           and metadata handling.
How      : subclass ``BaseStandardExperts`` and route W4A8 compute through
           ixformer inference functions.
Scope    : Iluvatar compressed-tensors MoE W4A8 standard-layout paths.
Remove   : when community compressed-tensors MoE supports this vendor kernel.
"""

from typing import Literal

import ixformer.inference.functions as ixff
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_moe_standard_experts import (
    BaseStandardExperts,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (
    hidden_padding,
    unpad_tensor,
)


class IluMoEStandardExpertsW4A8(BaseStandardExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        quant_mode: Literal["none", "w8a8", "w4a8"],
        gemm1_info,
        gemm2_info,
        w13_input_scale: torch.Tensor | None = None,
        w2_input_scale: torch.Tensor | None = None,
        runner_scaling_factor: float = 1.0,
        format: Literal["TN", "NT", "NN"] | int = "TN",
        version: int = 2,
        group_size: int = None,
        out_dtype: torch.dtype = torch.bfloat16,
        swiglu_limit: float | None = None,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
        )

        assert quant_mode == "w4a8", f"not support {quant_mode} now."

        self.quant_mode = quant_mode
        self.format = format
        self.version = version
        self.group_size = group_size
        self.out_dtype = out_dtype
        self.swiglu_limit = (
            swiglu_limit if swiglu_limit is None else float(swiglu_limit)
        )

        self.local_num_experts: int = None
        self.w13_input_scale: torch.Tensor | None = w13_input_scale
        self.w2_input_scale: torch.Tensor | None = w2_input_scale
        self.runner_scaling_factor = runner_scaling_factor
        self.gemm1_info = gemm1_info
        self.gemm2_info = gemm2_info

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None = None,
        apply_router_weight_on_input: bool = False,
    ):
        assert self.local_num_experts is not None

        ep_rank = get_ep_group().rank
        local_num_experts = self.local_num_experts
        topk = topk_ids.shape[-1]

        start_eid = ep_rank * local_num_experts
        end_eid = min(
            (ep_rank + 1) * local_num_experts,
            global_num_experts,
        )

        dtype = hidden_states.dtype
        num_tokens = topk_ids.shape[0]
        num_experts = global_num_experts
        hidden_size = hidden_states.shape[1]

        (
            src_to_dst,
            sorted_token_ids,
            expert_sizes_gpu,
            expert_sizes_cpu,
            expand_tokens,
        ) = ixff.moe_compute_token_index_ep(
            topk_ids=topk_ids,
            num_experts=num_experts,
            start_expert_id=start_eid,
            end_expert_id=end_eid,
        )
        if expert_sizes_cpu.sum() == 0:
            return torch.zeros(
                (num_tokens, hidden_size),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        i8_hidden_states, a_scale = self._expand_and_quant(
            input=hidden_states,
            expand_tokens=expand_tokens,
            sorted_token_ids=sorted_token_ids,
            src_to_dst=src_to_dst,
            topk=topk_weights.shape[-1],
        )

        pt_output_1 = ixff.moe_w4a8_group_gemm(
            input=i8_hidden_states,
            weight=w1,
            i_scales=a_scale,
            w_scales=self.w1_scale,
            output_dtype=dtype,
            tokens_per_experts=expert_sizes_cpu,
            w_i8scales=self.g1_alphas,
            w_i8zeros=self.w1_zp,
            dst_to_src=None,
            bias=self.w1_bias,
            format=self.gemm1_info.format,
            group_size=self.group_size,
            version=self.version,
        )

        pt_output_2, a2_scale = self._activation_and_quant(
            input=pt_output_1,
            sorted_token_ids=sorted_token_ids,
            activation=activation,
        )
        pt_output_3 = torch.empty(
            (num_tokens * topk, self.gemm2_info.get_pad_n()),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        ixff.moe_w4a8_group_gemm(
            input=pt_output_2,
            weight=w2,
            i_scales=a2_scale,
            w_scales=self.w2_scale,
            output_dtype=dtype,
            tokens_per_experts=expert_sizes_cpu,
            w_i8scales=self.g2_alphas,
            w_i8zeros=self.w2_zp,
            dst_to_src=sorted_token_ids,
            bias=self.w2_bias,
            format=self.gemm2_info.format,
            group_size=self.group_size,
            version=self.version,
            output=pt_output_3,
        )
        pt_output_3 = unpad_tensor(
            pt_output_3, self.gemm2_info.get_n(), contiguous=False
        )

        reduce_mask = src_to_dst == -1
        out = ixff.moe_output_reduce_sum(
            input=pt_output_3.view(num_tokens, topk, -1),
            topk_weight=topk_weights,
            scaling_factor=self.runner_scaling_factor,
            mask=reduce_mask,
            output=output,
            # extra_residual=shared_experts_input,
        )

        return out

    # expand + quant + pad
    def _expand_and_quant(
        self,
        input,
        expand_tokens,
        sorted_token_ids,
        src_to_dst,
        output_format=0,
        topk: int = None,
    ):
        output, scale = ixff.moe_expand_input_dynamic_scaled_int8(
            hidden_states=input,
            dst_to_src=sorted_token_ids,
            dst_tokens=expand_tokens,
            topk=topk,
            src_to_dst=src_to_dst,
            topk_ids=None,
            smooth_scales=self.w13_input_scale,
            output_format=output_format,
            output_padding=hidden_padding(input.shape[-1], self.gemm1_info.pad_k),
        )

        return output, scale

    # act + quant + pad
    def _activation_and_quant(
        self,
        input,
        sorted_token_ids,
        output_format=None,
        activation: MoEActivation = None,
    ):
        # unpad
        input = unpad_tensor(input, self.gemm1_info.get_n(), contiguous=True)

        swiglu_limit = self.swiglu_limit
        act_value = activation.value
        use_fused_swiglu_limit = swiglu_limit is not None and act_value in (
            "silu",
            "swiglu",
        )

        kwargs = dict(
            input=input,
            smooth_scales=self.w2_input_scale,
            dst_to_src=sorted_token_ids,
            topk_ids=None,
        )
        if output_format is not None:
            kwargs["output_format"] = output_format
        kwargs["output_padding"] = hidden_padding(
            self.gemm1_info.get_n() // 2, self.gemm2_info.pad_k
        )

        if use_fused_swiglu_limit:
            output, scale = ixff.swiglu_limit_dynamic_scaled_int8(
                limit=float(swiglu_limit), **kwargs
            )
        elif act_value == "swigluoai":
            kwargs["bias"] = None
            output, scale = ixff.activation_swigluoai_dynamic_scaled_int8(**kwargs)
        elif act_value == "swiglustep":
            kwargs["bias"] = None
            kwargs["act_type"] = "swiglustep"
            output, scale = ixff.activation_dynamic_scaled_int8(**kwargs)
        elif act_value in ("silu", "swiglu"):
            kwargs["bias"] = None
            kwargs["act_type"] = "swiglu"
            output, scale = ixff.activation_dynamic_scaled_int8(**kwargs)
        else:
            raise ValueError(
                f"IluCompressedTensorsW4A8Int8MoEMethod: unsupported MoE "
                f"activation {act_value!r}. Supported activations: "
                f"'silu', 'swiglu', 'swiglustep', 'swigluoai'."
            )

        return output, scale
