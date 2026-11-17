# SPDX-License-Identifier: Apache-2.0
"""Iluvatar AWQ mixture-of-experts backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ixformer.inference.functions as ixfops
import torch
from torch.nn import Parameter
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.linear import set_weight_attrs
from vllm.scalar_type import scalar_types

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_decode_metadata import (
    _ensure_cuda_graph_safe_dispatch,
    _is_only_decode,
)

if TYPE_CHECKING:
    from ...configs.awq import IluAutoAWQConfig


def _get_ilu_output_scale(layer: RoutedExperts) -> float:
    return getattr(layer, "_ilu_output_scale", 1.0)


def _apply_decode_w4a16_awq(
    *,
    x: torch.Tensor,
    layer: RoutedExperts,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    group_size: int,
    num_tokens: int,
    num_experts: int,
) -> torch.Tensor:
    """Decode-only AWQ MoE path using GEMV and avoiding D2H sync."""
    expand_tokens = num_tokens * layer.top_k
    (
        src_to_dst,
        sorted_token_ids,
        expert_sizes_gpu,
    ) = ixfops.moe_compute_token_index_for_decode(
        topk_ids=topk_ids,
        num_experts=num_experts,
    )
    expand_hidden_states = ixfops.moe_expand_input(
        hidden_states=x,
        dst_to_src=sorted_token_ids,
        dst_tokens=expand_tokens,
        topk=layer.top_k,
        src_to_dst=src_to_dst,
    )
    pt_output_1 = ixfops.moe_w4a16_group_gemv(
        input=expand_hidden_states,
        weight=layer.w13_qweight,
        w_scales=layer.w13_scales,
        quant_type="awq",
        w_zeros=layer.w13_qzeros,
        group_size=group_size,
        dst_to_src=None,
        format="NN",
        tokens_per_experts_gpu=expert_sizes_gpu,
    )
    pt_output_2 = ixfops.silu_and_mul(pt_output_1)
    pt_output_3 = ixfops.moe_w4a16_group_gemv(
        input=pt_output_2,
        weight=layer.w2_qweight,
        w_scales=layer.w2_scales,
        quant_type="awq",
        w_zeros=layer.w2_qzeros,
        group_size=group_size,
        dst_to_src=sorted_token_ids,
        format="NN",
        tokens_per_experts_gpu=expert_sizes_gpu,
    )
    return ixfops.moe_output_reduce_sum(
        input=pt_output_3.view(num_tokens, layer.top_k, -1),
        topk_weight=topk_weights,
        scaling_factor=_get_ilu_output_scale(layer),
        extra_residual=shared_experts_input,
    )


class IluAutoAWQMoEMethod(FusedMoEMethodBase):
    """Iluvatar AWQ MoE method using ixformer routed expert kernels."""

    def __init__(
        self,
        quant_config: IluAutoAWQConfig,
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        if self.quant_config.weight_bits != 4:
            raise ValueError("IluAutoAWQMoEMethod only supports 4-bit weights.")
        self.quant_type = scalar_types.uint4
        self.input_dtype = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        layer.input_dtype = self.input_dtype
        extra_weight_attrs.update(
            {
                "is_transposed": True,
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            }
        )

        intermediate_size_full = extra_weight_attrs.pop(
            "intermediate_size_full",
            intermediate_size_per_partition,
        )
        self.is_k_full = intermediate_size_per_partition == intermediate_size_full

        w13_qweight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.quant_config.group_size
        num_groups_w2 = intermediate_size_per_partition // self.quant_config.group_size
        layer.num_groups_w13 = num_groups_w13
        layer.num_groups_w2 = num_groups_w2

        w13_scales = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                intermediate_size_per_partition * 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = Parameter(
            torch.empty(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        w13_qzeros = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # ixformer kernels consume the loaded AWQ layout directly.
        return

    def get_fused_moe_quant_config(
        self,
        layer: torch.nn.Module,
    ) -> FusedMoEQuantConfig | None:
        return

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert layer.activation.value == "silu", "Only SiLU activation is supported."
        use_ep = layer.expert_map is not None
        attn_metadata = None
        try:
            from vllm.forward_context import get_forward_context

            attn_metadata = get_forward_context().attn_metadata
        except AssertionError:
            attn_metadata = None
        # unsupported ep now
        only_decode = not use_ep and _is_only_decode(attn_metadata)
        _ensure_cuda_graph_safe_dispatch(only_decode, attn_metadata)

        if use_ep:
            start_eid = layer.ep_rank * layer.local_num_experts
            end_eid = min(
                (layer.ep_rank + 1) * layer.local_num_experts,
                layer.global_num_experts,
            )

        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for fused "
                "Iluvatar AWQ MoE method."
            )

        num_tokens = topk_ids.shape[0]
        num_experts = layer.global_num_experts

        # Decode-optimized path (CG-safe).
        if only_decode:
            return _apply_decode_w4a16_awq(
                x=x,
                layer=layer,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
                group_size=self.quant_config.group_size,
                num_tokens=num_tokens,
                num_experts=num_experts,
            )
        # ---- General / EP path ----
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

        expand_hidden_states = ixfops.moe_expand_input(
            hidden_states=x,
            dst_to_src=sorted_token_ids,
            dst_tokens=expand_tokens,
            topk=layer.top_k,
            src_to_dst=src_to_dst,
        )

        expert_sizes_cpu = expert_sizes_gpu.cpu()
        pt_output_1 = ixfops.moe_w4a16_group_gemm(
            input=expand_hidden_states,
            weight=layer.w13_qweight,
            w_scales=layer.w13_scales,
            quant_type="awq",
            tokens_per_experts=expert_sizes_cpu,
            w_zeros=layer.w13_qzeros,
            group_size=self.quant_config.group_size,
            dst_to_src=None,
            format="NN",
            tokens_per_experts_gpu=expert_sizes_gpu,
        )
        pt_output_2 = ixfops.silu_and_mul(pt_output_1)

        if use_ep:
            pt_output_3 = torch.empty(
                (num_tokens * layer.top_k, hidden_size),
                device=x.device,
                dtype=x.dtype,
            )
            ixfops.moe_w4a16_group_gemm(
                input=pt_output_2,
                weight=layer.w2_qweight,
                w_scales=layer.w2_scales,
                quant_type="awq",
                tokens_per_experts=expert_sizes_cpu,
                w_zeros=layer.w2_qzeros,
                group_size=self.quant_config.group_size,
                dst_to_src=sorted_token_ids,
                format="NN",
                output=pt_output_3,
            )
            reduce_mask = src_to_dst == -1
            return ixfops.moe_output_reduce_sum(
                input=pt_output_3.view(num_tokens, layer.top_k, -1),
                topk_weight=topk_weights,
                scaling_factor=_get_ilu_output_scale(layer),
                mask=reduce_mask,
                extra_residual=shared_experts_input,
            )

        pt_output_3 = ixfops.moe_w4a16_group_gemm(
            input=pt_output_2,
            weight=layer.w2_qweight,
            w_scales=layer.w2_scales,
            quant_type="awq",
            tokens_per_experts=expert_sizes_cpu,
            w_zeros=layer.w2_qzeros,
            group_size=self.quant_config.group_size,
            dst_to_src=sorted_token_ids,
            format="NN",
            tokens_per_experts_gpu=expert_sizes_gpu,
        )
        return ixfops.moe_output_reduce_sum(
            input=pt_output_3.view(num_tokens, layer.top_k, -1),
            topk_weight=topk_weights,
            scaling_factor=_get_ilu_output_scale(layer),
            extra_residual=shared_experts_input,
        )
