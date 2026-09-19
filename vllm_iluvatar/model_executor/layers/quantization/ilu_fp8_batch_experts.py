# SPDX-License-Identifier: Apache-2.0
"""Iluvatar batched FP4-weight/FP8-activation MoE experts for DP+EP."""

import ixformer.inference.functions as ixff
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_moe_batch_experts import (
    BaseBatchExperts,
)


class IluMoEBatchedExpertsFp8(BaseBatchExperts):
    """Run ix30 FP4 weights with FP8 activations in batched expert layout."""

    # Shared by every MoE layer: expert chains run one layer at a time, so a
    # single set of intermediates serves the whole model. Row counts come from
    # the batched dispatch layout, which is fixed for the life of the engine,
    # so the buffers are allocated once during warmup and never resized under
    # a captured graph.
    _scratch_cache: dict[str, torch.Tensor] = {}

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
        swiglu_limit: float | None = None,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            quant_mode="w4a8",
            gemm1_info=None,
            gemm2_info=None,
            out_dtype=out_dtype,
            swiglu_limit=swiglu_limit,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )

    def _scratch(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        buf = self._scratch_cache.get(name)
        if (
            buf is None
            or buf.dtype != dtype
            or buf.device != device
            or buf.ndim != len(shape)
            or buf.shape[1:] != shape[1:]
            or buf.shape[0] < shape[0]
        ):
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._scratch_cache[name] = buf
        return buf[: shape[0]]

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
    ) -> torch.Tensor:
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
        if expert_tokens_meta is None:
            raise ValueError("expert_tokens_meta is required for batched FP8 MoE")
        if hidden_states.ndim not in (2, 3):
            raise ValueError("IluMoEBatchedExpertsFp8 expects 2D or batched 3D input")
        num_experts = w1.shape[0]
        if hidden_states.ndim == 2:
            if hidden_states.shape[0] % num_experts != 0:
                raise ValueError("flattened token rows must be divisible by experts")
            hidden_states = hidden_states.reshape(
                num_experts,
                hidden_states.shape[0] // num_experts,
                hidden_states.shape[1],
            )
        if hidden_states.shape[0] != num_experts:
            raise ValueError("activation and weight expert dimensions must match")
        if w1.dtype != torch.uint8 or w2.dtype != torch.uint8:
            raise TypeError("batched FP8 MoE expects ix30-packed uint8 FP4 weights")
        if self.out_dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise NotImplementedError("batched FP4/FP8 MoE requires BF16 output")

        max_num_tokens_per_expert = hidden_states.shape[1]
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        rows, hidden_size = hidden_states.shape
        device = hidden_states.device
        if hidden_states.dtype in (torch.float16, torch.bfloat16):
            hidden_states, a1q_scale = ixff.dynamic_scaled_fp8_quant(
                self._scratch("a1q", (rows, hidden_size), torch.float8_e4m3fn, device),
                hidden_states,
                self._scratch("a1q_scale", (rows, 1), torch.float32, device),
            )
        elif hidden_states.dtype == torch.float8_e4m3fn:
            if a1q_scale is None:
                raise ValueError("a1q_scale is required for FP8 dispatched inputs")
        else:
            raise TypeError(
                f"unsupported batched FP8 MoE input dtype {hidden_states.dtype}"
            )
        assert a1q_scale is not None
        a1q_scale = a1q_scale.reshape(-1)
        output_2d = output.reshape(-1, output.shape[-1])

        expert_num_tokens = (
            expert_tokens_meta.expert_num_tokens.to(
                device=hidden_states.device,
                dtype=torch.int32,
                non_blocking=True,
            )
            .reshape(-1)
            .contiguous()
        )
        if expert_num_tokens.numel() != num_experts:
            raise ValueError(
                f"expert_num_tokens must have {num_experts} values, "
                f"got {expert_num_tokens.numel()}"
            )
        group_layout = ixff.moe_make_padded_dense_tokens_per_experts_fp8(
            expert_num_tokens=expert_num_tokens,
            max_num_tokens_per_expert=max_num_tokens_per_expert,
            group_layout=self._scratch(
                "group_layout", (num_experts, 2), torch.int32, device
            ),
        )

        # ix30-packed w1 is [expert, k/2, 2 * intermediate].
        inter_x2 = w1.shape[2]
        gemm1_output = ixff.group_gemm_fp4(
            a=hidden_states,
            scale_a=a1q_scale,
            weight=w1,
            scale_weight=self.w1_scale,
            group_layout=group_layout,
            output=self._scratch("gemm1", (rows, inter_x2), torch.bfloat16, device),
            backend="ixmm",
        )

        act_type = activation.value
        if act_type in ("silu", "swiglu"):
            act_type = "swiglu"
        elif act_type != "swiglustep":
            raise ValueError(
                f"IluMoEBatchedExpertsFp8 does not support activation {act_type!r}"
            )
        activated, a2_scale = ixff.moe_group_gemv_swiglu_quant_fp8(
            input=gemm1_output,
            expert_num_tokens=expert_num_tokens,
            max_num_tokens_per_expert=max_num_tokens_per_expert,
            act_type=act_type,
            limit=self.swiglu_limit,
            output=self._scratch(
                "act", (rows, inter_x2 // 2), torch.float8_e4m3fn, device
            ),
            output_scales=self._scratch("act_scale", (rows,), torch.float32, device),
        )

        ixff.group_gemm_fp4(
            a=activated,
            scale_a=a2_scale.reshape(-1),
            weight=w2,
            scale_weight=self.w2_scale,
            group_layout=group_layout,
            output=output_2d,
            backend="ixmm",
        )
        return output
