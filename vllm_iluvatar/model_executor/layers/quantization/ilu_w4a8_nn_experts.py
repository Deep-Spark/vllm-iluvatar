# SPDX-License-Identifier: Apache-2.0
"""ivcore30 NN standard and batched experts using ixformer ixmm."""

import ixformer.inference.functions as ixff
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe import MoEActivation

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_moe_batch_experts import (
    BaseBatchExperts,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_moe_standard_experts_w4a8 import (
    IluMoEStandardExpertsW4A8,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (
    unpad_tensor,
)

from .ilu_w4a8_nn import require_ivcore30_nn


class _NNExpertsInit:
    """Initialize the fixed NN contract without legacy packing-version state."""

    def __init__(
        self,
        moe_config,
        quant_config,
        gemm1_info,
        gemm2_info,
        runner_scaling_factor=1.0,
        swiglu_limit=None,
        max_num_tokens=None,
        num_dispatchers=None,
        out_dtype=torch.bfloat16,
    ):
        require_ivcore30_nn()
        batched = self.activation_format() == mk.FusedMoEActivationFormat.BatchedExperts
        mk.FusedMoEExpertsModular.__init__(
            self,
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens if batched else None,
            num_dispatchers=num_dispatchers if batched else None,
        )
        self.quant_mode = "w4a8"
        self.format = "NN"
        self.group_size = -1
        self.out_dtype = out_dtype
        self.swiglu_limit = None if swiglu_limit is None else float(swiglu_limit)
        self.local_num_experts = None
        self.w13_input_scale = self.w2_input_scale = None
        self.runner_scaling_factor = runner_scaling_factor
        self.gemm1_info, self.gemm2_info = gemm1_info, gemm2_info


class IluMoEBatchedExpertsW4A8NN(_NNExpertsInit, BaseBatchExperts):
    """NN GEMMs preserve the fixed-capacity LL dispatch layout."""

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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None = None,
        apply_router_weight_on_input: bool = False,
    ):
        """NN LL buffers: [expert, capacity, hidden], valid rows first."""
        if self.group_size != -1 or activation != MoEActivation.SILU:
            raise NotImplementedError("NN LL experts require per-channel W4A8 and SILU")
        if any(
            value is not None
            for value in (
                self.g1_alphas,
                self.g2_alphas,
                self.w1_zp,
                self.w2_zp,
                self.w1_bias,
                self.w2_bias,
                self.w13_input_scale,
                self.w2_input_scale,
            )
        ):
            raise NotImplementedError(
                "NN LL experts require symmetric weights and dynamic INT8"
            )
        if expert_tokens_meta is None:
            raise ValueError("NN LL experts require GPU expert token counts")
        expert_num_tokens = expert_tokens_meta.expert_num_tokens
        experts = w1.shape[0]
        if not expert_num_tokens.is_cuda or expert_num_tokens.numel() != experts:
            raise ValueError("NN LL experts require one CUDA token count per expert")
        if output.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("NN LL experts require FP16/BF16 output")
        if hidden_states.ndim not in (2, 3):
            raise ValueError("NN LL experts require 2D or batched 3D input")
        if hidden_states.ndim == 2:
            if hidden_states.shape[0] % experts != 0:
                raise ValueError("flattened token rows must be divisible by experts")
            hidden_states = hidden_states.reshape(
                experts,
                hidden_states.shape[0] // experts,
                hidden_states.shape[1],
            )
        if hidden_states.shape[0] != experts:
            raise ValueError("activation and weight expert dimensions must match")
        max_num_tokens_per_expert = hidden_states.shape[1]
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        out = output.view(-1, output.shape[-1])
        rows = x.shape[0]
        if out.shape[0] != rows:
            raise ValueError("NN LL buffers must have equal capacity for every expert")
        if rows == 0:
            return output

        if x.dtype != torch.int8:
            x, a1q_scale = ixff.scaled_int8_quant(x)
        if a1q_scale is None or a1q_scale.numel() != rows:
            raise ValueError("NN LL INT8 input requires one activation scale per row")
        if x.shape[-1] != w1.shape[1]:
            raise ValueError("NN LL dispatch hidden width must match W13 K")

        if w1.shape[-1] * 2 != self.gemm1_info.get_n():
            raise NotImplementedError("NN LL W13 requires an unpadded output width")
        if w2.shape[-1] * 2 != out.shape[-1]:
            raise NotImplementedError("NN LL W2 must write the final output directly")
        if not a1q_scale.is_contiguous():
            raise ValueError("NN LL requires contiguous activation scales")

        # Fixed-capacity slots must NOT be compacted: combine consumes the
        # dispatch layout. All offsets/counts remain on device during replay.
        expert_num_tokens = expert_num_tokens.reshape(-1).to(torch.int32).contiguous()
        layout = ixff.moe_make_padded_dense_tokens_per_experts_fp8(
            expert_num_tokens=expert_num_tokens,
            max_num_tokens_per_expert=max_num_tokens_per_expert,
            backend="ixformer_perf",
        )
        act = torch.empty((rows, w1.shape[-1] * 2), dtype=output.dtype, device=x.device)
        ixff.moe_w4a8_group_gemm_ixmm(
            a=x,
            weight=w1,
            scale_a=a1q_scale.view(-1),
            weight_scale=self.w1_scale.reshape(experts, -1),
            group_layout=layout,
            output=act,
            backend="ixmm",
        )
        # Grouped activation visits valid rows directly, without GEMV reordering.
        # Quantize before W2 K padding so padding does not affect scales.
        padding = w2.shape[1] - act.shape[-1] // 2
        act, scale = ixff.moe_group_gemm_swiglu_quant(
            input=act,
            expert_num_tokens=expert_num_tokens,
            max_num_tokens_per_expert=max_num_tokens_per_expert,
            limit=self.swiglu_limit,
        )
        if padding:
            act = torch.nn.functional.pad(act, (0, padding))
        # GEMM writes active rows; LL combine reads only those same rows.
        # Inactive fixed-capacity slots need not be initialized.
        ixff.moe_w4a8_group_gemm_ixmm(
            a=act,
            weight=w2,
            scale_a=scale.view(-1),
            weight_scale=self.w2_scale.reshape(experts, -1),
            group_layout=layout,
            output=out,
            backend="ixmm",
        )
        # Router weights and routed scaling are owned by finalize/combine.
        return output


class IluMoEStandardExpertsW4A8NN(_NNExpertsInit, IluMoEStandardExpertsW4A8):
    """Share the activation helper; own all NN standard-layout execution."""

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
        return self._apply_nn(
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
        )

    def _apply_nn(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
    ) -> torch.Tensor:
        """NN prefill with device-side expert ranges and ixmm-packed weights."""
        if self.group_size != -1 or self.w13_input_scale is not None:
            raise NotImplementedError("NN prefill requires dynamic per-channel W4A8")
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("NN prefill requires BF16 or FP16 activations")
        start = get_ep_group().rank * self.local_num_experts
        end = min(start + self.local_num_experts, global_num_experts)
        rows = topk_ids.numel()
        group_layout = torch.empty(
            (end - start, 2), device=topk_ids.device, dtype=torch.int32
        )
        packed_to_expert_row = torch.empty(
            (rows,), device=topk_ids.device, dtype=torch.int32
        )
        src_to_dst, _, weights_packed, row_offsets, _, valid_rows = (
            ixff.moe_compute_token_index_ep_packed(
                topk_ids=topk_ids.to(torch.int32),
                num_experts=global_num_experts,
                start_expert_id=start,
                end_expert_id=end,
                topk_weights=topk_weights,
                graph_safe=True,
                group_layout=group_layout,
                packed_to_expert_row=packed_to_expert_row,
            )
        )
        quantized, input_scale = ixff.scaled_int8_quant(hidden_states)
        act = torch.empty(
            (rows, self.gemm1_info.pad_k), device=hidden_states.device, dtype=torch.int8
        )
        act_scale = torch.empty(
            (rows,), device=hidden_states.device, dtype=torch.float32
        )
        ixff.moe_expand_input_ep(
            hidden_states=quantized,
            src_to_expert_row=src_to_dst,
            expand_tokens=rows,
            topk=topk_ids.shape[-1],
            output=act,
            output_scale=act_scale,
            input_scale=input_scale,
        )
        hidden = ixff.moe_w4a8_group_gemm_ixmm(
            a=act,
            weight=w1,
            scale_a=act_scale,
            weight_scale=self.w1_scale.reshape(w1.shape[0], -1),
            group_layout=group_layout,
            out_dtype=output.dtype,
            backend="ixmm",
        )
        act, act_scale = self._activation_and_quant(
            input=hidden,
            sorted_token_ids=None,
            output_format=0,
            activation=activation,
            check_valid=True,
            expand_tokens_gpu=valid_rows,
        )
        routed = ixff.moe_w4a8_group_gemm_ixmm(
            a=act,
            weight=w2,
            scale_a=act_scale.reshape(-1).contiguous(),
            weight_scale=self.w2_scale.reshape(w2.shape[0], -1),
            group_layout=group_layout,
            out_dtype=output.dtype,
            backend="ixmm",
        )
        routed = unpad_tensor(routed, self.gemm2_info.get_n(), contiguous=False)
        return ixff.moe_output_reduce_sum_ep(
            input=routed,
            token_row_offsets=row_offsets,
            weights=weights_packed,
            output=output,
            scaling_factor=self.runner_scaling_factor,
            packed_to_expert_row=packed_to_expert_row,
        )
