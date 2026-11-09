# SPDX-License-Identifier: Apache-2.0
# Source: ilu_unquantized_fused_moe_method.py at 47fb804; independent expert pipeline.
"""Unquantized routed experts using the existing ixaite grouped pipeline."""

import torch
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op


# Keep community weight creation while selecting a complete executable lifecycle.
class IluUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    # Fixed activation and dispatch contracts are known before weights exist.
    def __init__(self, moe):
        if (
            moe.has_bias
            or moe.activation.value not in ("silu", "swiglu")
            or moe.in_dtype not in (torch.float16, torch.bfloat16)
            or moe.moe_parallel_config.use_ep
            or moe.is_lora_enabled
        ):
            raise RuntimeError(
                "ixaite unquantized MoE requires FP16/BF16, SiLU, no bias, no EP or LoRA"
            )
        implementation, reason = load_ixaite_python_op(
            "moe_bf16_grouped_gemm", "moe_bf16_grouped_gemm"
        )
        if implementation is None:
            raise RuntimeError(f"ixaite unquantized MoE is unavailable: {reason}")
        # IluPlatform reports CUDA; bypass the community NVIDIA kernel oracle.
        FusedMoEMethodBase.__init__(self, moe)
        self.unquantized_backend = UnquantizedMoeBackend.OOT
        self.experts_cls = None
        self._experts = implementation

    # RoutedExperts applies routing separately from the expert pipeline.
    @property
    def is_monolithic(self):
        return False

    # Clamp and input-weight routing cannot silently change the library formula.
    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if layer.apply_router_weight_on_input or getattr(layer, "swiglu_limit", None):
            raise RuntimeError(
                "ixaite unquantized MoE activation/routing is unsupported"
            )
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    # Keep the runner's shared residual separate from its routed output scaling.
    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        output = self._experts(
            x, layer.w13_weight, layer.w2_weight, topk_weights, topk_ids
        )
        if shared_experts_input is not None:
            output = output + shared_experts_input
        return output
