# SPDX-License-Identifier: Apache-2.0
"""Iluvatar batched W16A16 MoE expert implementation.

Corresponds to:
    vllm/model_executor/layers/quantization/compressed_tensors/
    compressed_tensors_moe/*
Target   : W16A16 batched expert kernel for Iluvatar MoE.
Baseline : plugin/vllm 0.24.0 runtime source
Why      : Iluvatar DP+EP MoE uses ixformer grouped GEMM/GEMV kernels for
           batched W16A16 experts.
How      : subclass ``BaseBatchExperts`` and delegate compute to ixformer
           inference functions.
Scope    : Iluvatar compressed-tensors MoE W16A16 paths.
Remove   : when community compressed-tensors MoE supports this vendor kernel.
"""

import ixformer.inference.functions as ixff
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe import MoEActivation

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_moe_batch_experts import (
    BaseBatchExperts,
)


class IluMoEBatchedExpertsW16A16(BaseBatchExperts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.quant_mode != "none":
            raise ValueError(
                f"IluMoEBatchedExpertsW16A16 requires quant_mode='none', got {self.quant_mode!r}"
            )

    def apply(self, *args, **kwargs):
        return self.apply_group_gemm(*args, **kwargs)

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

        if self.quant_mode != "none":
            raise NotImplementedError(
                f"Unsupported quant_mode for W16A16 batch experts: {self.quant_mode}"
            )
        inter_size = w1.shape[1] if self.format[0] == "T" else w1.shape[2]

        self._prepare_group_gemm_offset(
            num_experts=num_experts,
            max_num_tokens=max_num_tokens_per_expert,
            hidden_size=hidden_size,
            inter_size=inter_size,
        )

        output_1 = ixff.moe_w16a16_group_gemm(
            input=hidden_states,
            weight=w1,
            output_dtype=hidden_states.dtype,
            tokens_per_experts=num_tokens_per_local_experts,
            dst_to_src=None,
            bias=w1_bias,
            format=self.format,
            input_expert_token_offset=self.input_expert_token_offset_l1,
            output_expert_token_offset=self.output_expert_token_offset_l1,
        )

        if activation == "silu":
            output_2 = ixff.silu_and_mul(output_1)
        elif activation == "gelu":
            output_2 = ixff.gelu_and_mul(output_1)
        elif activation == "swigluoai":
            output_2 = ixff.swigluoai_and_mul(output_1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        output = output.reshape(-1, output.shape[-1])
        ixff.moe_w16a16_group_gemm(
            input=output_2,
            weight=w2,
            output_dtype=hidden_states.dtype,
            tokens_per_experts=num_tokens_per_local_experts,
            format="TN",
            bias=w2_bias,
            output=output,
            input_expert_token_offset=self.input_expert_token_offset_l2,
            output_expert_token_offset=self.output_expert_token_offset_l2,
        )
