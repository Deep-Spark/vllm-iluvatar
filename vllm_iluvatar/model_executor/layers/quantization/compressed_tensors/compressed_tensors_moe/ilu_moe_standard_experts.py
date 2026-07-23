# SPDX-License-Identifier: Apache-2.0
"""Iluvatar standard-layout MoE expert base implementation.

Corresponds to:
    vllm/model_executor/layers/fused_moe/modular_kernel.py
    vllm/model_executor/layers/quantization/compressed_tensors/
    compressed_tensors_moe/*
Target   : standard activation-format expert kernels selected by Iluvatar MoE.
Baseline : plugin/vllm 0.24.0 runtime source
Why      : Iluvatar W4A8 kernels need a standard expert layout that performs
           activation quantization internally.
How      : plugin-only modular-kernel expert base selected by
           ``ilu_compressed_tensors_moe`` method dispatch.
Scope    : Iluvatar compressed-tensors MoE standard-layout paths.
Remove   : when community modular MoE exposes equivalent vendor expert kernels.
"""

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe import (
    FusedMoEActivationFormat,
    FusedMoEParallelConfig,
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


class BaseStandardExperts(mk.FusedMoEExpertsModular):
    @staticmethod
    def _supports_current_device() -> bool:
        raise NotImplementedError(
            "BaseStandardExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        raise NotImplementedError(
            "BaseStandardExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        raise NotImplementedError(
            "BaseStandardExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        raise NotImplementedError(
            "BaseStandardExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        raise NotImplementedError(
            "BaseStandardExperts is not yet used by an Oracle. "
            "This method should not be called."
        )

    @staticmethod
    def activation_format() -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @property
    def expects_unquantized_inputs(self) -> bool:
        # This expert implementation performs expand + activation quantization
        # internally, so DeepEP prepare/finalize should not pre-quantize input.
        return True

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
        self.local_num_experts = local_num_experts

        workspace1 = (0, N // 2)
        workspace2 = (0, K)

        output = (M, K)

        return workspace1, workspace2, output
