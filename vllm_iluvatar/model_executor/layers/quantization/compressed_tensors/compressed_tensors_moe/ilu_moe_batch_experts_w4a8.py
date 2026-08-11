# SPDX-License-Identifier: Apache-2.0
"""Iluvatar batched W4A8 MoE expert implementation.

Corresponds to:
    vllm/model_executor/layers/quantization/compressed_tensors/
    compressed_tensors_moe/*
Target   : W4A8 batched expert kernel for Iluvatar MoE.
Baseline : plugin/vllm 0.24.0 runtime source
Why      : Iluvatar DP+EP MoE uses ixformer grouped kernels and fork metadata
           for W4A8 expert dispatch.
How      : subclass ``BaseBatchExperts`` and route W4A8 compute through ixformer
           inference functions.
Scope    : Iluvatar compressed-tensors MoE W4A8 paths.
Remove   : when community compressed-tensors MoE supports this vendor kernel.
"""

import ixformer.inference.functions as ixff
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe import MoEActivation

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_moe_batch_experts import (
    MOE_ALGO,
    BaseBatchExperts,
    _group_gemv_format,
    grouped_act_int8,
)


class IluMoEBatchedExpertsW4A8(BaseBatchExperts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.quant_mode != "w4a8":
            raise ValueError(
                f"IluMoEBatchedExpertsW4A8 requires quant_mode='w4a8', got {self.quant_mode!r}"
            )

    def apply(self, *args, **kwargs):
        if MOE_ALGO == "FUSED":
            return self.apply_fused(*args, **kwargs)
        if MOE_ALGO == "GEMV":
            return self.apply_group_gemv(*args, **kwargs)
        if MOE_ALGO == "GEMM":
            return self.apply_group_gemm(*args, **kwargs)
        raise RuntimeError(
            f"not support MoE Algo: {MOE_ALGO}, expect: GEMM, GEMV, FUSED"
        )

    def apply_fused(
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
        del (
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            a2_scale,
            workspace13,
            workspace2,
            apply_router_weight_on_input,
        )

        if activation != MoEActivation.SILU:
            raise NotImplementedError(
                "ixformer fused W4A8 MoE supports SILU activation only"
            )
        if expert_tokens_meta is None:
            raise ValueError("expert_tokens_meta is required for fused decode MoE")
        if any(
            value is not None
            for value in (
                self.g1_alphas,
                self.g2_alphas,
                self.w1_zp,
                self.w2_zp,
                self.w1_bias,
                self.w2_bias,
            )
        ):
            raise NotImplementedError(
                "ixformer fused W4A8 MoE does not support bias, zero-point, "
                "or auxiliary weight scales"
            )
        assert self.quant_mode == "w4a8", "only support w4a8 for fused moe."
        if self.version != 2 or self.group_size != -1:
            raise NotImplementedError(
                "ixformer fused W4A8 MoE requires version=2 and group_size=-1"
            )
        if self.out_dtype != torch.bfloat16:
            raise NotImplementedError(
                "ixformer fused W4A8 MoE requires bfloat16 output"
            )

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        output = output.reshape(-1, output.shape[-1])

        num_experts = w1.size(0)
        num_tokens, _ = hidden_states.shape
        assert num_tokens % num_experts == 0
        max_num_tokens_per_expert = num_tokens // num_experts

        w1_scale = self.w1_scale
        w2_scale = self.w2_scale
        if w1_scale.ndim == 3 and w1_scale.shape[1] == 1:
            w1_scale = w1_scale.reshape(w1_scale.shape[0], w1_scale.shape[-1])
            w2_scale = w2_scale.reshape(w2_scale.shape[0], w2_scale.shape[-1])

        kwargs = dict(
            hidden_states=hidden_states,
            a1q_scale=a1q_scale,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            expert_num_tokens=expert_tokens_meta.expert_num_tokens,
            max_num_tokens_per_expert=max_num_tokens_per_expert,
            format=self.format,
            output=output,
            limit=self.swiglu_limit,
        )

        return ixff.fused_moe_w4a8(**kwargs)

    def apply_group_gemv(
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
        del global_num_experts, expert_map, a2_scale, workspace13, workspace2

        assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert output.is_contiguous(), "Output must be contiguous"
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"

        if self.quant_mode != "w4a8":
            raise NotImplementedError(
                f"Unsupported quant_mode for GEMV path: {self.quant_mode}"
            )
        if self.version != 2 or self.group_size != -1:
            raise NotImplementedError(
                "GEMV path currently supports version=2 and group_size=-1 only"
            )

        gemv_format = _group_gemv_format(self.format)
        if gemv_format not in (0, 2):
            raise NotImplementedError(
                f"moe_w4a8_group_gemv only supports NN/TN format, got {self.format!r}"
            )

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        output = output.reshape(-1, output.shape[-1])
        num_tokens, _ = hidden_states.shape
        if num_tokens == 0:
            return output

        num_experts = w1.size(0)
        assert num_tokens % num_experts == 0
        max_num_tokens_per_expert = num_tokens // num_experts

        if expert_tokens_meta is None:
            expert_num_tokens_gpu = torch.full(
                [num_experts],
                max_num_tokens_per_expert,
                dtype=torch.int32,
                device=hidden_states.device,
            )
        else:
            expert_num_tokens_gpu = expert_tokens_meta.expert_num_tokens.to(
                device=hidden_states.device,
                dtype=torch.int32,
                non_blocking=True,
            ).contiguous()

        if expert_num_tokens_gpu.numel() != num_experts:
            raise ValueError(
                f"expert_num_tokens must have {num_experts} values, "
                f"got {expert_num_tokens_gpu.numel()}"
            )

        if hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            hidden_states, a1q_scale = ixff.scaled_int8_quant(hidden_states)
            a1q_scale = a1q_scale.flatten()
        elif a1q_scale is None:
            raise ValueError("a1q_scale is required when hidden_states is already int8")

        assert hidden_states.dtype == torch.int8
        a1q_scale = a1q_scale.reshape(-1).contiguous()
        assert hidden_states.shape[0] == a1q_scale.shape[0], (
            f"got hidden_states: {hidden_states.shape}, scales: {a1q_scale.shape}"
        )

        (
            input_1,
            scale_1,
            dense_tokens_per_experts,
        ) = ixff.moe_group_gemv_reformat_input_fused_dense(
            input=hidden_states,
            input_scales=a1q_scale,
            expert_num_tokens=expert_num_tokens_gpu,
            max_num_tokens_per_expert=max_num_tokens_per_expert,
        )

        w1_bias = self.w1_bias
        w2_bias = self.w2_bias

        output_1 = ixff.moe_w4a8_group_gemv(
            input=input_1,
            weight=w1,
            i_scales=scale_1,
            w_scales=self.w1_scale,
            w_i8scales=self.g1_alphas,
            w_i8zeros=self.w1_zp,
            output_dtype=self.out_dtype,
            tokens_per_experts=expert_num_tokens_gpu,
            dst_to_src=None,
            bias=w1_bias,
            format=gemv_format,
            group_size=self.group_size,
            dense_tokens_per_experts=dense_tokens_per_experts,
        )

        output_2, a2_scale = ixff.moe_group_gemv_swiglu_quant(
            input=output_1,
            expert_num_tokens=expert_num_tokens_gpu,
            max_num_tokens_per_expert=max_num_tokens_per_expert,
        )

        ixff.moe_w4a8_group_gemv(
            input=output_2,
            weight=w2,
            i_scales=a2_scale,
            w_scales=self.w2_scale,
            w_i8scales=self.g2_alphas,
            w_i8zeros=self.w2_zp,
            output_dtype=self.out_dtype,
            tokens_per_experts=expert_num_tokens_gpu,
            dst_to_src=None,
            bias=w2_bias,
            format=gemv_format,
            group_size=self.group_size,
            output=output,
            dense_tokens_per_experts=dense_tokens_per_experts,
        )

        return output

    def apply_group_gemm(
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
        assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
        ]

        origin_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        num_tokens, hidden_size = hidden_states.size()
        if num_tokens == 0:
            return hidden_states

        num_experts, N, _ = w1.size()
        top_k = topk_weights.size(1)

        w1_bias = self.w1_bias
        w2_bias = self.w2_bias

        max_num_tokens_per_expert = hidden_states.shape[0] // num_experts
        num_tokens_per_local_experts = expert_tokens_meta.expert_num_tokens.cpu()

        if self.quant_mode != "w4a8":
            raise NotImplementedError(
                f"Unsupported quant_mode for W4A8 batch experts: {self.quant_mode}"
            )
        inter_size = self.w1_scale.shape[-1]

        self._prepare_group_gemm_offset(
            num_experts=num_experts,
            max_num_tokens=max_num_tokens_per_expert,
            hidden_size=hidden_size,
            inter_size=inter_size,
        )

        if hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            hidden_states, a1q_scale = ixff.scaled_int8_quant(hidden_states)
            a1q_scale = a1q_scale.flatten()

        assert hidden_states.shape[0] == a1q_scale.shape[0], (
            f"got hidden_states: {hidden_states.shape}, scales: {a1q_scale.shape}"
        )
        assert hidden_states.dtype == torch.int8

        output_1 = ixff.moe_w4a8_group_gemm(
            input=hidden_states,
            weight=w1,
            i_scales=a1q_scale,
            w_scales=self.w1_scale,
            w_i8scales=self.g1_alphas,
            w_i8zeros=self.w1_zp,
            output_dtype=self.out_dtype,
            tokens_per_experts=num_tokens_per_local_experts,
            dst_to_src=None,
            format=self.format,
            group_size=self.group_size,
            input_expert_token_offset=self.input_expert_token_offset_l1,
            output_expert_token_offset=self.output_expert_token_offset_l1,
        )

        output_2, a2_scale = grouped_act_int8(
            input=output_1,
            num_tokens_per_expert=num_tokens_per_local_experts,
            max_num_tokens_per_expert=max_num_tokens_per_expert,
            act=activation,
            swiglu_limit=self.swiglu_limit,
        )

        output = output.reshape(-1, output.shape[-1])
        ixff.moe_w4a8_group_gemm(
            input=output_2,
            weight=w2,
            i_scales=a2_scale,
            w_scales=self.w2_scale,
            w_i8scales=self.g2_alphas,
            w_i8zeros=self.w2_zp,
            output_dtype=self.out_dtype,
            tokens_per_experts=num_tokens_per_local_experts,
            dst_to_src=None,
            format=self.format,
            group_size=self.group_size,
            output=output,
            input_expert_token_offset=self.input_expert_token_offset_l2,
            output_expert_token_offset=self.output_expert_token_offset_l2,
        )
