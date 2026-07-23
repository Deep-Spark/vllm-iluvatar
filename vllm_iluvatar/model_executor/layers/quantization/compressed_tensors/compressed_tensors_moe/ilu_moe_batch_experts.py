# SPDX-License-Identifier: Apache-2.0
"""Iluvatar batched MoE expert base implementations.

Corresponds to:
    vllm/model_executor/layers/fused_moe/modular_kernel.py
    vllm/model_executor/layers/quantization/compressed_tensors/
    compressed_tensors_moe/*
Target   : batched expert kernels selected by Iluvatar compressed-tensors MoE.
Baseline : plugin/vllm 0.24.0 runtime source
Why      : Iluvatar W4A8/W16A16 DP+EP kernels consume batched expert layouts and
           may own activation quantization internally.
How      : plugin-only modular-kernel expert classes selected by
           ``ilu_compressed_tensors_moe`` method dispatch.
Scope    : Iluvatar compressed-tensors MoE quantization paths.
Remove   : when community modular MoE exposes equivalent vendor expert kernels.
"""

from typing import Literal

import ixformer.inference.functions as ixff
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.model_executor.layers.fused_moe import (
    FusedMoEActivationFormat,
    FusedMoEParallelConfig,
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

MOE_ALGO = envs.VLLM_ILU_MOE_EP_LL_MOE_ALGO


def _group_gemv_format(format: Literal["TN", "NT", "NN"] | int) -> int:
    if isinstance(format, str):
        format_mapping = {"NN": 0, "NT": 1, "TN": 2}
        return format_mapping[format]
    return int(format)


def grouped_act_int8(
    act: MoEActivation,
    input: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    max_num_tokens_per_expert: int,
    swiglu_limit: float = None,
):
    assert act == MoEActivation.SILU

    num_tokens = input.shape[0]
    in_hidden = input.shape[1]
    output = torch.empty(
        [num_tokens, in_hidden // 2], dtype=torch.int8, device=input.device
    )
    scales = torch.empty([num_tokens], dtype=torch.float, device=input.device)

    offset = 0
    for n in num_tokens_per_expert:
        if n > 0:
            if swiglu_limit is not None:
                ixff.swiglu_limit_dynamic_scaled_int8(
                    input=input[offset : offset + n],
                    limit=swiglu_limit,
                    i8_output=output[offset : offset + n],
                    output_scales=scales[offset : offset + n],
                )
            else:
                ixff.activation_dynamic_scaled_int8(
                    input=input[offset : offset + n],
                    bias=None,
                    smooth_scales=None,
                    topk_ids=None,
                    act_type="swiglu",
                    i8_output=output[offset : offset + n],
                    output_scales=scales[offset : offset + n],
                )

        offset += max_num_tokens_per_expert

    return output, scales


class BaseBatchExperts(mk.FusedMoEExpertsModular):
    @staticmethod
    def _supports_current_device() -> bool:
        raise NotImplementedError(
            "BaseBatchExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        raise NotImplementedError(
            "BaseBatchExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        raise NotImplementedError(
            "BaseBatchExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        raise NotImplementedError(
            "BaseBatchExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        raise NotImplementedError(
            "BaseBatchExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def activation_format() -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.BatchedExperts

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    input_expert_token_offset_l1: torch.Tensor = None
    output_expert_token_offset_l1: torch.Tensor = None
    input_expert_token_offset_l2: torch.Tensor = None
    output_expert_token_offset_l2: torch.Tensor = None

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
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
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

        if self.quant_mode == "none":
            global MOE_ALGO
            MOE_ALGO = "GEMM"

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        a1 = a1.reshape(-1, a1.shape[-1])
        assert w1.dim() == 3 and w2.dim() == 3
        E, N, _ = w1.size()
        K = a1.size(-1)
        M = a1.size(0)
        topk = topk_ids.size(1)

        return E, M, N, K, topk

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (0, N // 2)
        workspace2 = (0, K)

        max_num_tokens = M // local_num_experts
        output = (local_num_experts, max_num_tokens, K)

        return workspace1, workspace2, output

    def _prepare_group_gemm_offset(
        self, num_experts: int, max_num_tokens: int, hidden_size: int, inter_size: int
    ):
        if self.input_expert_token_offset_l1 is not None:
            return

        self.input_expert_token_offset_l1 = (
            ixff.moe_group_gemm_compute_expert_token_offset(
                num_experts=num_experts,
                num_tokens_per_expert=max_num_tokens,
                channels=hidden_size,
            )
        )
        self.output_expert_token_offset_l1 = (
            ixff.moe_group_gemm_compute_expert_token_offset(
                num_experts=num_experts,
                num_tokens_per_expert=max_num_tokens,
                channels=inter_size,
            )
        )

        self.input_expert_token_offset_l2 = (
            ixff.moe_group_gemm_compute_expert_token_offset(
                num_experts=num_experts,
                num_tokens_per_expert=max_num_tokens,
                channels=inter_size // 2,
            )
        )
        self.output_expert_token_offset_l2 = (
            ixff.moe_group_gemm_compute_expert_token_offset(
                num_experts=num_experts,
                num_tokens_per_expert=max_num_tokens,
                channels=hidden_size,
            )
        )
