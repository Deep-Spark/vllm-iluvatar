# SPDX-License-Identifier: Apache-2.0
"""Opt-in DSV4.1 TP2/DP4 decode dispatch with one packed FP8 AllGather."""

import ixformer.distributed as ixfd
import ixformer.inference.functions as ops
import torch
import triton
import triton.language as tl
from vllm import envs
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger

logger = init_logger(__name__)
_IDENTITY_ROUTES = {}


@triton.jit
def _pack_quantized_routes(
    quantized_input,
    input_scales,
    expert_indices,
    expert_weights,
    output,
    hidden_size: tl.constexpr,
    packet_stride: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * block_size + tl.arange(0, block_size)
    quantized_values = tl.load(
        quantized_input + row * hidden_size + columns, columns < hidden_size, other=0
    )
    scale_values = tl.load(
        input_scales + row * 4 + columns - hidden_size,
        (columns >= hidden_size) & (columns < hidden_size + 4),
        other=0,
    )
    index_values = tl.load(
        expert_indices + row * 24 + columns - hidden_size - 4,
        (columns >= hidden_size + 4) & (columns < hidden_size + 28),
        other=0,
    )
    weight_values = tl.load(
        expert_weights + row * 24 + columns - hidden_size - 28,
        (columns >= hidden_size + 28) & (columns < packet_stride),
        other=0,
    )
    tl.store(
        output + row * packet_stride + columns,
        tl.where(
            columns < hidden_size,
            quantized_values,
            tl.where(
                columns < hidden_size + 4,
                scale_values,
                tl.where(columns < hidden_size + 28, index_values, weight_values),
            ),
        ),
        columns < packet_stride,
    )


@triton.jit
def _unpack_route_metadata(
    packet,
    input_scales,
    expert_indices,
    expert_weights,
    hidden_size: tl.constexpr,
    packet_stride: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, 64)
    value = tl.load(
        packet + row * packet_stride + hidden_size + columns, columns < 52, other=0
    )
    tl.store(input_scales + row * 4 + columns, value, columns < 4)
    tl.store(
        expert_indices + row * 24 + columns - 4, value, (columns >= 4) & (columns < 28)
    )
    tl.store(
        expert_weights + row * 24 + columns - 28,
        value,
        (columns >= 28) & (columns < 52),
    )


@triton.jit
def _expand_quantized_rows(
    quantized_input,
    input_scales,
    routing_map,
    output,
    output_scales,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    block_size: tl.constexpr,
    input_stride: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * block_size + tl.arange(0, block_size)
    route = tl.load(routing_map + row)
    token = tl.maximum(route, 0) // top_k
    value = tl.load(
        quantized_input + token * input_stride + columns,
        (route >= 0) & (columns < hidden_size),
        other=0,
    )
    tl.store(output + row * hidden_size + columns, value, columns < hidden_size)
    if tl.program_id(1) == 0:
        tl.store(
            output_scales + row, tl.load(input_scales + token, route >= 0, other=1.0)
        )


@triton.jit
def _quantize_and_pack_routes(
    hidden_states,
    expert_indices,
    expert_weights,
    output,
    hidden_size: tl.constexpr,
    packet_stride: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_size)
    values = tl.load(
        hidden_states + row * hidden_size + columns, columns < hidden_size, other=0
    ).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), 0) * (1.0 / 448.0), 1.0e-12)
    quantized_input = tl.minimum(
        tl.maximum(tl.div_rn(values, scale), -448.0), 448.0
    ).to(tl.float8e4nv)
    tl.store(
        (output + row * packet_stride).to(tl.pointer_type(tl.float8e4nv)) + columns,
        quantized_input,
        columns < hidden_size,
    )
    tl.store(
        (output + row * packet_stride + hidden_size).to(tl.pointer_type(tl.float32)),
        scale,
    )
    route_columns = tl.arange(0, 8)
    expert_indices = tl.load(
        expert_indices + row * 6 + route_columns, route_columns < 6, other=0
    )
    expert_weights = tl.load(
        expert_weights + row * 6 + route_columns, route_columns < 6, other=0
    )
    tl.store(
        (output + row * packet_stride + hidden_size + 4).to(tl.pointer_type(tl.int32))
        + route_columns,
        expert_indices,
        route_columns < 6,
    )
    tl.store(
        (output + row * packet_stride + hidden_size + 28).to(
            tl.pointer_type(tl.float32)
        )
        + route_columns,
        expert_weights,
        route_columns < 6,
    )


def expand_prequantized_input(quantized_input, input_scales, mapping, topk):
    rows = mapping.numel()
    hidden = quantized_input.shape[1]
    output = torch.empty(
        (rows, hidden), device=quantized_input.device, dtype=torch.float8_e4m3fn
    )
    output_scales = torch.empty(
        rows, device=quantized_input.device, dtype=torch.float32
    )
    _expand_quantized_rows[(rows, triton.cdiv(hidden, 1024))](
        quantized_input.view(torch.uint8),
        input_scales,
        mapping,
        output.view(torch.uint8),
        output_scales,
        hidden,
        topk,
        1024,
        quantized_input.stride(0),
    )
    return output, output_scales


def maybe_packed_experts(owner, hidden_states, input_ids, ctx, group, sizes):
    runner = owner.experts
    layer = runner.routed_experts
    communicator = group.device_communicator
    # Every collective decision uses replicated configuration/global DP state.
    if not (
        type(owner).__module__
        in (
            "vllm_iluvatar.models.deepseek_v4.model",
            "vllm_iluvatar.models.deepseek_v41.model",
        )
        and type(runner._quant_method).__name__ == "IluMxfp4MoEMethod"
        and ctx.dp_metadata is not None
        and ctx.dp_metadata.all_ranks_decode_only
        and not runner.moe_config.is_sequence_parallel
        and get_tensor_model_parallel_world_size() == 2
        and len(sizes) == 4
        and all(n == sizes[0] for n in sizes)
        and 24 <= sizes[0] <= 96
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.shape[1] == 5120
        and getattr(runner.moe_config, "experts_per_token", 0) == 6
        and layer.w13_weight.dtype == torch.uint8
        and getattr(layer, "w13_bias", None) is None
        and getattr(layer, "w2_bias", None) is None
        and communicator is not None
        and communicator.ixfd_group is not None
        and not getattr(ctx, "_ilu_is_padding_gathered", False)
    ):
        return None
    # A rank-local stride must not choose a different collective from peers.
    # Normalize locally instead of taking a rank-local BF16 fallback.
    hidden_states = hidden_states.contiguous()
    from .ilu_mxfp4_fused_moe import _ilu_mxfp4_fused_experts_ixformer

    local_tokens = hidden_states.shape[0]
    total_tokens = sum(sizes)
    gate = runner.gate if runner.gate is not None else owner.gate
    logits, _ = gate(hidden_states)
    # Keep the local padding mask and local hash/vision token IDs here.
    local_weights, local_expert_indices = runner.router.select_experts(
        hidden_states=hidden_states,
        router_logits=logits,
        topk_indices_dtype=torch.int32,
        input_ids=owner._router_input_ids(runner, input_ids),
    )
    local_expert_indices = local_expert_indices.to(torch.int32).contiguous()
    local_weights = local_weights.float().contiguous()
    packet = torch.empty(
        (local_tokens, 5172), device=hidden_states.device, dtype=torch.uint8
    )
    gathered = torch.empty(
        (total_tokens, 5172), device=hidden_states.device, dtype=torch.uint8
    )
    if envs.VLLM_ILU_V41_PACKED_FP8_FUSED_QUANT:
        _quantize_and_pack_routes[(local_tokens,)](
            hidden_states,
            local_expert_indices,
            local_weights,
            packet,
            5120,
            5172,
            8192,
            num_warps=8,
        )
    else:
        key = (hidden_states.device, local_tokens)
        if key not in _IDENTITY_ROUTES:
            _IDENTITY_ROUTES[key] = torch.arange(
                local_tokens, device=hidden_states.device, dtype=torch.int32
            )
        quantized_input, input_scales = ops.moe_expand_input_dynamic_scaled_fp8(
            hidden_states, _IDENTITY_ROUTES[key], 1
        )
        _pack_quantized_routes[(local_tokens, 6)](
            quantized_input.view(torch.uint8),
            input_scales.view(torch.uint8),
            local_expert_indices.view(torch.uint8),
            local_weights.view(torch.uint8),
            packet,
            5120,
            5172,
            1024,
        )
    ixfd.all_gather_into_tensor(
        gathered, packet, group=communicator.ixfd_group, async_op=True
    )
    full_scales = torch.empty(
        total_tokens, device=hidden_states.device, dtype=torch.float32
    )
    expert_indices = torch.empty(
        (total_tokens, 6), device=hidden_states.device, dtype=torch.int32
    )
    expert_weights = torch.empty(
        (total_tokens, 6), device=hidden_states.device, dtype=torch.float32
    )
    _unpack_route_metadata[(total_tokens,)](
        gathered,
        full_scales.view(torch.uint8),
        expert_indices.view(torch.uint8),
        expert_weights.view(torch.uint8),
        5120,
        5172,
    )
    logger.info_once("V4.1 packed FP8 dispatch active")
    # Only shape/device/dtype are consumed when prequantized input is provided.
    shape_view = hidden_states[:1].expand(total_tokens, -1)
    mode = 1 if envs.VLLM_ILU_V41_DECODE_MOE_MODE1 and total_tokens <= 768 else None
    return _ilu_mxfp4_fused_experts_ixformer(
        shape_view,
        layer.w13_weight,
        layer.w13_weight_scale,
        layer.w2_weight,
        layer.w2_weight_scale,
        expert_weights,
        expert_indices,
        swiglu_limit=layer.swiglu_limit,
        expert_map=layer.expert_map,
        output_scale=getattr(layer, "_ilu_output_scale", 1.0),
        extra_residual=None,
        w13_bias=None,
        w2_bias=None,
        gemm_backend="ixmm",
        gemm_mode=mode,
        swiglu_variant="dsv4_clamp",
        quantized_input=(gathered[:, :5120].view(torch.float8_e4m3fn), full_scales),
    )
