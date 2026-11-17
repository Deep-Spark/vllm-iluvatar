# SPDX-License-Identifier: Apache-2.0
"""Iluvatar GPTQ mixture-of-experts backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ixformer.inference.functions as ixfops
import torch
from vllm import _custom_ops as ops
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.linear import set_weight_attrs
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.scalar_type import scalar_types

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_decode_metadata import (
    _ensure_cuda_graph_safe_dispatch,
    _is_only_decode,
)

if TYPE_CHECKING:
    from ...configs.gptq import IluAutoGPTQConfig


def _get_ilu_output_scale(layer: RoutedExperts) -> float:
    return getattr(layer, "_ilu_output_scale", 1.0)


def _apply_decode_w4a16_gptq(
    *,
    expand_hidden_states: torch.Tensor,
    layer: RoutedExperts,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_sizes_gpu: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    group_size: int,
    num_tokens: int,
) -> torch.Tensor:
    """Decode-only GPTQ MoE path using GEMV and avoiding D2H sync."""
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


def unpack_k_batch(packed_w: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """
    Efficient vectorized unpacking for 3D tensor (batch version).
    Converts [B, K // pack_factor, N] int32 tensor → [B, K, N] int8 tensor.

    Args:
        packed_w: torch.int32 tensor of shape [B, K // pack_factor, N].
        num_bits: Number of bits per packed element (e.g., 4).

    Returns:
        unpacked: torch.int8 tensor of shape [B, K, N].
    """
    B, k_packed, n = packed_w.shape
    pack_factor = 32 // num_bits
    k = k_packed * pack_factor

    mask = (1 << num_bits) - 1

    # [pack_factor, 1, 1, 1]
    shifts = (num_bits * torch.arange(pack_factor, device=packed_w.device)).view(
        -1, 1, 1, 1
    )

    # [1, B, k_packed, N]
    packed_expanded = packed_w.unsqueeze(0)

    # Extract each group of num_bits using bitwise ops
    unpacked_groups = ((packed_expanded >> shifts) & mask).to(torch.int8)

    # [pack_factor, B, k_packed, N] → [B, K, N]
    unpacked = unpacked_groups.permute(1, 2, 0, 3).reshape(B, k, n)

    return unpacked


def pack_n_batch(x: torch.Tensor, pack_num: int = 8, order_map=None) -> torch.Tensor:
    """
    Efficient vectorized batch packing: [B, K, N] int4 -> [B, K, N//pack_num] int32.

    Args:
        x: torch.int32 tensor of shape [B, K, N], each element 0-15 (int4).
        pack_num: Number of 4-bit elements per packed int32 (default=8).
        order_map: Optional order of elements within each packed int32.

    Returns:
        torch.int32 tensor of shape [B, K, N//pack_num].
    """

    B, K, N = x.shape
    assert N % pack_num == 0, "N must be divisible by pack_num"
    cols = N // pack_num

    if order_map is None:
        order_map = list(range(pack_num))
    order_map = torch.tensor(order_map, device=x.device)

    unit = 32 // pack_num  # number of bits per element

    # reshape to [B, K, cols, pack_num]
    pack_num_int = int(pack_num)

    x_reshape = x.view(B, K, cols, pack_num_int)

    # reorder according to order_map
    x_reorder = torch.gather(
        x_reshape, 3, order_map.view(1, 1, 1, -1).expand(B, K, cols, -1)
    )

    # mask low 4 bits
    x_reorder = x_reorder & 0xF

    # bit shifts [pack_num] -> [1,1,1,pack_num] broadcastable
    shifts = (unit * torch.arange(pack_num_int, device=x.device)).view(1, 1, 1, -1)

    # shift and sum along last dimension to combine bits
    packed = (x_reorder << shifts).sum(dim=-1).to(torch.int32)

    return packed


class IluAutoGPTQMoEMethod(FusedMoEMethodBase):
    """MoE Marlin method with quantization."""

    def __init__(
        self,
        quant_config: IluAutoGPTQConfig,
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        if self.quant_config.quant_type.size_bits != 4:
            raise ValueError("IluAutoGPTQMoEMethod only supports int4 now.")
        self.quant_type = scalar_types.uint4
        self.input_dtype = None
        self.use_marlin = True

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.input_dtype = self.input_dtype
        is_a_8bit = self.input_dtype is not None and self.input_dtype.itemsize == 1

        if is_a_8bit:
            assert self.quant_config.quant_type.size_bits == 8, (
                "W8A8-INT8 is not supported by marlin kernel."
            )

        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            w2_scales_size = intermediate_size_per_partition
            scales_size2 = w2_scales_size // self.quant_config.group_size
            strategy = FusedMoeWeightScaleSupported.GROUP.value
        else:
            scales_size13 = 1
            scales_size2 = 1
            strategy = FusedMoeWeightScaleSupported.CHANNEL.value

        layer.num_groups_w13 = scales_size13
        layer.num_groups_w2 = scales_size2

        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})
        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.quant_config.pack_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.quant_config.pack_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        # up_proj scales
        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)
        # down_proj scales
        w2_scales = torch.nn.Parameter(
            torch.empty(num_experts, scales_size2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        # up_proj scales
        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)
        # down_proj scales
        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        is_a_8bit = self.input_dtype is not None and self.input_dtype.itemsize == 1

        if is_a_8bit:
            assert self.quant_config.quant_type.size_bits == 8, (
                "W8A8-INT8 is not supported by marlin kernel."
            )
        if self.input_dtype == torch.float8_e4m3fn:
            ops.marlin_int4_fp8_preprocess(layer.w13_qweight, inplace=True)
            ops.marlin_int4_fp8_preprocess(layer.w2_qweight, inplace=True)
            layer.w13_scales.data = layer.w13_scales.data * 512
            layer.w2_scales.data = layer.w2_scales.data * 512

        if self.quant_config.desc_act:
            raise NotImplementedError(
                "GPTQMarlinMoEMethod now not support  desc_act. please fix it"
            )
        w13_qweight_unpacked = unpack_k_batch(layer.w13_qweight)
        w13_qweight_repacked = pack_n_batch(
            w13_qweight_unpacked,
            self.quant_config.pack_factor,
            order_map=[0, 2, 4, 6, 1, 3, 5, 7],
        )
        replace_parameter(layer, "w13_qweight", w13_qweight_repacked)

        w13_zp = torch.full_like(
            layer.w13_scales, self.quant_config.quant_type.bias, dtype=torch.int32
        )
        w13_zp_pack = pack_n_batch(
            w13_zp, self.quant_config.pack_factor, order_map=[0, 2, 4, 6, 1, 3, 5, 7]
        ).contiguous()
        replace_parameter(layer, "w13_qzeros", w13_zp_pack)

        w2_qweight_unpacked = unpack_k_batch(layer.w2_qweight)
        w2_qweight_repacked = pack_n_batch(
            w2_qweight_unpacked,
            self.quant_config.pack_factor,
            order_map=[0, 2, 4, 6, 1, 3, 5, 7],
        )
        replace_parameter(layer, "w2_qweight", w2_qweight_repacked)

        w2_zp = torch.full_like(
            layer.w2_scales, self.quant_config.quant_type.bias, dtype=torch.int32
        )
        w2_zp_pack = pack_n_batch(
            w2_zp, self.quant_config.pack_factor, order_map=[0, 2, 4, 6, 1, 3, 5, 7]
        ).contiguous()
        replace_parameter(layer, "w2_qzeros", w2_zp_pack)

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        from vllm.model_executor.layers.fused_moe.config import (
            gptq_marlin_moe_quant_config,
        )

        return gptq_marlin_moe_quant_config(
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            weight_bits=self.quant_config.weight_bits,
            group_size=self.quant_config.group_size,
            w1_zp=getattr(layer, "w13_qzeros", None)
            if not self.quant_config.is_sym
            else None,
            w2_zp=getattr(layer, "w2_qzeros", None)
            if not self.quant_config.is_sym
            else None,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )

    def select_gemm_impl(
        self,
        prepare_finalize,
        layer: RoutedExperts,
    ):
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel "
            "initialization logic. This function should not be called."
        )

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
        attn_metadata = get_forward_context().attn_metadata
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
                "Apply router weight on input is not supported for "
                "fused Marlin MoE method."
            )

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

        # expand + reorder
        # TODO use kernel
        expand_hidden_states = ixfops.moe_expand_input(
            hidden_states=x,
            dst_to_src=sorted_token_ids,
            dst_tokens=expand_tokens,
            topk=layer.top_k,
            src_to_dst=src_to_dst,
        )
        if only_decode:
            final_hidden_states = _apply_decode_w4a16_gptq(
                expand_hidden_states=expand_hidden_states,
                layer=layer,
                topk_weights=topk_weights,
                sorted_token_ids=sorted_token_ids,
                expert_sizes_gpu=expert_sizes_gpu,
                shared_experts_input=shared_experts_input,
                group_size=self.quant_config.group_size,
                num_tokens=num_tokens,
            )
        else:
            expert_sizes_cpu = expert_sizes_gpu.cpu()
            # w4a16 group gemm 1
            # pt_output_1: (expand_tokens, 2n) dtype
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

            # act
            pt_output_2 = ixfops.silu_and_mul(pt_output_1)

            # w4a16 group gemm 2 + reorder
            # pt_output_3: (expand_tokens, k) dtype
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
                final_hidden_states = ixfops.moe_output_reduce_sum(
                    input=pt_output_3.view(num_tokens, layer.top_k, -1),
                    topk_weight=topk_weights,
                    scaling_factor=_get_ilu_output_scale(layer),
                    mask=reduce_mask,
                    extra_residual=shared_experts_input,
                )
            else:
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

                # mul + reduce_sum
                # final_hidden_states: (num_tokens, k)
                final_hidden_states = ixfops.moe_output_reduce_sum(
                    input=pt_output_3.view(num_tokens, layer.top_k, -1),
                    topk_weight=topk_weights,
                    scaling_factor=_get_ilu_output_scale(layer),
                    extra_residual=shared_experts_input,
                )
        return final_hidden_states
