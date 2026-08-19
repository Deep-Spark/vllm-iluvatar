# SPDX-License-Identifier: Apache-2.0
"""Iluvatar W16A16 fused-MoE expert execution.

Corresponds to:
    vllm/model_executor/layers/fused_moe/fused_moe.py

Target   : fused_experts / fused_experts_impl W16A16 unquantized path
Baseline : main_6e448d0 + fork fused_moe.py ixformer delta
Why      : keep the fork's ixformer W16A16 MoE execution out of community vLLM
           while preserving the Iluvatar unquantized MoE path.
How      : callers in ``IluUnquantizedFusedMoEMethod`` and elastic-EP wrappers
           call ``ilu_w16a16_fused_experts`` directly. Community
           ``fused_moe.py`` remains identical to upstream main.
Scope    : ILUVATAR_W16A16 unquantized MoE only.
Remove   : when upstream exposes a vendor fused-experts backend hook.
"""

from __future__ import annotations

import ixformer.inference.functions as ixfops  # type: ignore
import torch
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_decode_metadata import (
    _ensure_cuda_graph_safe_dispatch,
    _is_only_decode,
)


def _activation_name(activation) -> str:
    return getattr(activation, "value", activation)


def _check_unquantized(quant_config) -> None:
    if quant_config is None:
        return
    if (
        getattr(quant_config, "use_fp8_w8a8", False)
        or getattr(quant_config, "use_int8_w8a8", False)
        or getattr(quant_config, "use_int8_w8a16", False)
        or getattr(quant_config, "use_int4_w4a16", False)
        or getattr(quant_config, "w1_scale", None) is not None
        or getattr(quant_config, "w2_scale", None) is not None
        or getattr(quant_config, "w1_zp", None) is not None
        or getattr(quant_config, "w2_zp", None) is not None
        or getattr(quant_config, "a1_scale", None) is not None
        or getattr(quant_config, "a2_scale", None) is not None
    ):
        raise ValueError("Iluvatar W16A16 fused experts only supports unquantized MoE")


def _apply_activation(x: torch.Tensor, activation: str, top_k: int) -> torch.Tensor:
    if activation == "silu":
        return ixfops.silu_and_mul(x)
    if activation == "gelu":
        return ixfops.gelu_and_mul(x)
    if activation == "swigluoai":
        return ixfops.swigluoai_and_mul(x)
    if activation == "swiglustep":
        from vllm.model_executor.layers.activation import swiglustep_and_mul_triton

        output_dim = x.shape[1]
        out = torch.empty(
            (x.shape[0], output_dim // 2),
            device=x.device,
            dtype=x.dtype,
        )
        swiglustep_and_mul_triton(out, x)
        return out
    raise ValueError(f"Unsupported activation: {activation}")


def _apply_decode_w16a16(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    w1_bias,
    w2_bias,
) -> torch.Tensor:
    """Decode-only path: GPU token-index + GEMV, no D2H sync (CG-safe)."""
    num_tokens = topk_ids.shape[0]
    num_experts = w1.size(0)
    top_k = topk_weights.size(1)
    expand_tokens = num_tokens * top_k

    (
        src_to_dst,
        sorted_token_ids,
        expert_sizes_gpu,
    ) = ixfops.moe_compute_token_index_for_decode(
        topk_ids=topk_ids,
        num_experts=num_experts,
    )

    expanded = ixfops.moe_expand_input(
        hidden_states=hidden_states,
        dst_to_src=sorted_token_ids,
        dst_tokens=expand_tokens,
        topk=top_k,
        src_to_dst=src_to_dst,
    )
    pt_output_1 = ixfops.moe_w16a16_group_gemv(
        input=expanded,
        weight=w1,
        output_dtype=hidden_states.dtype,
        tokens_per_experts_gpu=expert_sizes_gpu,
        dst_to_src=None,
        bias=w1_bias,
        format="TN",
    )
    pt_output_2 = _apply_activation(pt_output_1, activation, top_k)
    pt_output_3 = ixfops.moe_w16a16_group_gemv(
        input=pt_output_2,
        weight=w2,
        output_dtype=hidden_states.dtype,
        tokens_per_experts_gpu=expert_sizes_gpu,
        dst_to_src=sorted_token_ids,
        bias=w2_bias,
        format="TN",
    )
    return ixfops.moe_output_reduce_sum(
        input=pt_output_3.view(num_tokens, top_k, -1),
        topk_weight=topk_weights,
    )


def ilu_w16a16_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    quant_config=None,
) -> torch.Tensor:
    """Run ixformer W16A16 MoE experts for Iluvatar.

    This is the fork's ``fused_experts_impl_opt`` scoped to the only path the
    plugin needs: unquantized W16A16. EP uses ixformer's EP token-index helper;
    decode-only non-EP batches use GEMV to avoid the general GEMM path.
    """
    _check_unquantized(quant_config)
    activation = _activation_name(activation)
    w1_bias = getattr(quant_config, "w1_bias", None) if quant_config else None
    w2_bias = getattr(quant_config, "w2_bias", None) if quant_config else None

    attn_metadata = get_forward_context().attn_metadata
    use_ep = expert_map is not None
    only_decode = not use_ep and _is_only_decode(attn_metadata)
    _ensure_cuda_graph_safe_dispatch(only_decode, attn_metadata)

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    if only_decode:
        return _apply_decode_w16a16(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )

    num_tokens = hidden_states.size(0)
    num_experts = w1.size(0)
    top_k = topk_weights.size(1)

    if use_ep:
        local_num_experts = w1.size(0)
        ep_rank = get_ep_group().device_group.rank()
        start_eid = ep_rank * local_num_experts
        end_eid = min((ep_rank + 1) * local_num_experts, global_num_experts)
        hidden_size = hidden_states.shape[1]
        (
            src_to_dst,
            sorted_token_ids,
            expert_sizes_gpu,
            expert_sizes_cpu,
            expand_tokens,
        ) = ixfops.moe_compute_token_index_ep(
            topk_ids=topk_ids,
            num_experts=global_num_experts,
            start_expert_id=start_eid,
            end_expert_id=end_eid,
        )
        if expert_sizes_cpu.sum() == 0:
            return torch.zeros(
                (num_tokens, hidden_size),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
    else:
        expand_tokens = num_tokens * top_k
        (
            src_to_dst,
            sorted_token_ids,
            expert_sizes_gpu,
            expert_sizes_cpu,
        ) = ixfops.moe_compute_token_index(
            topk_ids=topk_ids,
            num_experts=num_experts,
        )

    hidden_states = ixfops.moe_expand_input(
        hidden_states=hidden_states,
        dst_to_src=sorted_token_ids,
        dst_tokens=expand_tokens,
        topk=top_k,
        src_to_dst=src_to_dst,
    )
    expert_sizes_cpu = expert_sizes_gpu.cpu()
    pt_output_1 = ixfops.moe_w16a16_group_gemm(
        input=hidden_states,
        weight=w1,
        output_dtype=hidden_states.dtype,
        tokens_per_experts=expert_sizes_cpu,
        dst_to_src=None,
        bias=w1_bias,
        format="TN",
    )
    pt_output_2 = _apply_activation(pt_output_1, activation, top_k)

    if use_ep:
        pt_output_3 = torch.empty(
            (num_tokens * top_k, hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        pt_output_3 = ixfops.moe_w16a16_group_gemm(
            input=pt_output_2,
            weight=w2,
            output_dtype=hidden_states.dtype,
            tokens_per_experts=expert_sizes_cpu,
            dst_to_src=sorted_token_ids,
            format="TN",
            bias=w2_bias,
            output=pt_output_3,
        )
        return ixfops.moe_output_reduce_sum(
            input=pt_output_3.view(num_tokens, top_k, -1),
            topk_weight=topk_weights,
            mask=src_to_dst == -1,
        )

    pt_output_3 = ixfops.moe_w16a16_group_gemm(
        input=pt_output_2,
        weight=w2,
        output_dtype=hidden_states.dtype,
        tokens_per_experts=expert_sizes_cpu,
        dst_to_src=sorted_token_ids,
        bias=w2_bias,
        format="TN",
    )
    return ixfops.moe_output_reduce_sum(
        input=pt_output_3.view(num_tokens, top_k, -1),
        topk_weight=topk_weights,
    )
