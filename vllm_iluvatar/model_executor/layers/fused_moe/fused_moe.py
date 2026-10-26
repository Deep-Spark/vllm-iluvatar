# SPDX-License-Identifier: Apache-2.0
"""Iluvatar W16A16 fused-MoE expert execution.

Corresponds to: vllm/model_executor/layers/fused_moe/fused_moe.py

The method selects TN or NN while loading weights. This module exposes the
shared execution entry point and keeps both implementations private: the
original TN GEMM/GEMV path and the CoreX30 NN device-routing path.
"""

from __future__ import annotations

import ixformer.inference.functions as ixfops
import torch
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_decode_metadata import (
    _ensure_cuda_graph_safe_dispatch,
    _is_only_decode,
)


def validate_nn_weights(w13: torch.Tensor, w2: torch.Tensor) -> None:
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("NN MoE weights must be rank-3 [E,K,N]")
    if (
        w13.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
        or w13.device != w2.device
    ):
        raise ValueError("NN MoE weights must be BF16 on the same device")
    if (
        w13.shape[0] != w2.shape[0]
        or w13.shape[1] != w2.shape[2]
        or w13.shape[2] != 2 * w2.shape[1]
    ):
        raise ValueError("NN MoE requires w13=[E,H,2I], w2=[E,I,H]")
    for weight in (w13, w2):
        if (
            weight.stride(-1) != 1
            or weight.shape[1] < 32
            or any(v % 32 for v in (*weight.shape[1:], weight.stride(-2)))
        ):
            raise ValueError("NN K/N/leading dimensions must align to 32")


def normalize_nn_weights(layer) -> None:
    """Normalize explicit loader layouts once; preserve original NN storage."""
    if layer.__dict__.get("_ilu_w16a16_weight_format") == "NN":
        validate_nn_weights(layer.w13_weight, layer.w2_weight)
        return
    from vllm.model_executor.utils import replace_parameter

    formats = layer.__dict__.get(
        "_ilu_w16a16_input_formats", {"w13_weight": "TN", "w2_weight": "TN"}
    )
    converted = {}
    for name in ("w13_weight", "w2_weight"):
        weight = getattr(layer, name)
        fmt = formats.get(name)
        if fmt not in ("TN", "NN"):
            raise ValueError(f"Explicit input layout required for {name}: {fmt}")
        if fmt == "TN":
            tensor = weight.detach().transpose(-1, -2).contiguous()
            tensor.__dict__.update(weight.__dict__)
            converted[name] = tensor
        else:
            converted[name] = weight
    validate_nn_weights(converted["w13_weight"], converted["w2_weight"])
    for name, tensor in converted.items():
        if tensor is not getattr(layer, name):
            replace_parameter(layer, name, tensor)
    layer._ilu_w16a16_weight_format = "NN"


def ilu_w16a16_fused_experts(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    activation,
    expert_map=None,
    step5_native_activation=False,
    swiglu_limit=7.0,
    *,
    apply_router_weight_on_input=False,
    global_num_experts=-1,
    quant_config=None,
    weight_format="NN",
) -> torch.Tensor:
    from vllm_iluvatar.ilu_util.arch import is_ivcore11, is_ivcore30

    if not (is_ivcore11() or is_ivcore30()):
        raise ValueError("W16A16 MoE requires CoreX11 or CoreX30")
    if weight_format == "TN":
        return _fused_experts_tn(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            quant_config=quant_config,
            swiglu_limit=swiglu_limit,
        )
    if weight_format != "NN" or not is_ivcore30():
        raise ValueError("NN W16A16 MoE requires CoreX30 and NN weights")
    return _fused_experts_nn(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        expert_map,
        step5_native_activation,
        swiglu_limit,
    )


def _fused_experts_nn(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    activation,
    expert_map=None,
    step5_native_activation=False,
    swiglu_limit=7.0,
) -> torch.Tensor:
    import ixformer.inference.functions as ops

    from vllm_iluvatar.model_executor.layers.fused_moe.activation import (
        swiglustep_limit,
    )

    activation = getattr(activation, "value", activation)
    if activation == "swiglustep":
        swiglu_limit = swiglustep_limit(swiglu_limit)

    validate_nn_weights(w1, w2)
    if (
        hidden_states.dtype != torch.bfloat16
        or not hidden_states.is_cuda
        or hidden_states.device != w1.device
        or (not hidden_states.is_contiguous())
        or (hidden_states.ndim != 2)
        or (hidden_states.shape[1] != w1.shape[1])
    ):
        raise ValueError("W16A16 MoE input must be contiguous CUDA BF16 [T,H]")
    if (
        topk_ids.ndim != 2
        or topk_ids.shape != topk_weights.shape
        or topk_ids.shape[0] != hidden_states.shape[0]
        or (topk_ids.shape[1] < 1)
        or (topk_ids.device != hidden_states.device)
        or (topk_weights.device != hidden_states.device)
    ):
        raise ValueError("MoE routing requires matching device [T,top_k] tensors")
    if activation not in ("silu", "swiglustep"):
        raise ValueError(f"W16A16 MoE activation unsupported: {activation}")
    if hidden_states.shape[0] == 0:
        return torch.empty_like(hidden_states)
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        topk_ids = topk_ids.to(
            dtype=torch.int32, memory_format=torch.contiguous_format, copy=True
        )
    if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
        topk_weights = topk_weights.to(
            dtype=torch.float32, memory_format=torch.contiguous_format, copy=True
        )
    top_k = topk_ids.shape[1]
    capacity = topk_ids.numel()
    src_to_dst, dst_to_src, layout = ops.moe_compute_token_index_ep_group_layout(
        topk_ids, 0, w1.shape[0], expert_map=expert_map, backend="ixformer_native"
    )
    expanded = ops.moe_expand_input(
        hidden_states,
        dst_to_src,
        capacity,
        top_k,
        src_to_dst=None,
        check_valid=True,
        backend="ixformer_native",
    )
    masked_step5 = step5_native_activation and activation == "swiglustep"
    gate_up = torch.empty((capacity, w1.shape[2]), device=w1.device, dtype=w1.dtype)
    ops.moe_w16a16_group_gemm_ixmm(
        expanded, w1, layout, output=gate_up, format="NN", backend="ixmm"
    )
    if masked_step5:
        from ixformer.inference.functions.models.step5.moe.float.moe_swiglu import (
            moe_swiglu,
        )

        activated = moe_swiglu(gate_up, dst_to_src, limit=swiglu_limit)
    elif activation == "swiglustep":
        activated = ops.swiglustep_and_mul(
            gate_up, limit=swiglu_limit, backend="ixformer_native"
        )
    else:
        activated = ops.silu_and_mul(gate_up)
    expert_output = ops.moe_w16a16_group_gemm_ixmm(
        activated, w2, layout, format="NN", backend="ixmm"
    )
    return ops.moe_gather_output_reduce_sum(
        expert_output,
        topk_weights,
        src_to_dst,
        scaling_factor=1.0,
        backend="ixformer_native",
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


def _apply_tn_activation(
    x: torch.Tensor, activation: str, swiglu_limit: float
) -> torch.Tensor:
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
        swiglustep_and_mul_triton(out, x, limit=swiglu_limit)
        return out
    raise ValueError(f"Unsupported activation: {activation}")


def _fused_experts_tn_decode(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    w1_bias,
    w2_bias,
    swiglu_limit: float,
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
    pt_output_2 = _apply_tn_activation(pt_output_1, activation, swiglu_limit)
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


def _fused_experts_tn(
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
    swiglu_limit: float | None = 7.0,
) -> torch.Tensor:
    """Run ixformer W16A16 MoE experts for Iluvatar.

    This is the fork's ``fused_experts_impl_opt`` scoped to the only path the
    plugin needs: unquantized W16A16. EP uses ixformer's EP token-index helper;
    decode-only non-EP batches use GEMV to avoid the general GEMM path.
    """
    _check_unquantized(quant_config)
    activation = _activation_name(activation)
    if activation == "swiglustep":
        from vllm_iluvatar.model_executor.layers.fused_moe.activation import (
            swiglustep_limit,
        )

        swiglu_limit = swiglustep_limit(swiglu_limit)
    if hidden_states.shape[0] == 0:
        return torch.empty_like(hidden_states)
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
        return _fused_experts_tn_decode(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            swiglu_limit=swiglu_limit,
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
    pt_output_2 = _apply_tn_activation(pt_output_1, activation, swiglu_limit)

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
