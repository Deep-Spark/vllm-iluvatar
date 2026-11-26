# SPDX-License-Identifier: Apache-2.0
"""Local-expert FP4-E2M1 / FP8-E4M3 computation for serialized F4F8."""

from __future__ import annotations

import torch
from vllm.v1.worker import workspace as _workspace
from vllm.v1.worker.workspace import current_workspace_manager

F4F8_RUNNER_ALL2ALL_BACKENDS = frozenset({"naive", "allgather_reducescatter"})


def ilu_f4f8_moe_backends_available() -> tuple[bool, str | None]:
    from ixformer.inference import functions as _functions  # noqa: F401
    from ixformer.inference.backend_registry import registry

    required = (
        ("moe_compute_token_index_ep_group_layout", "ixformer_native"),
        ("moe_expand_input_dynamic_scaled_fp8", "ixformer_native"),
        ("group_gemm_fp4", "ixmm"),
        ("activation_dynamic_scaled_fp8", "ixformer_perf"),
        ("moe_gather_output_reduce_sum", "ixformer_native"),
    )
    for op_name, backend in required:
        if backend not in registry.available_backends(op_name):
            return False, f"ilu_f4f8 requires {op_name!r} backend {backend!r}"
    return True, None


def _prepare_group_gemm(
    group_layout: torch.Tensor, tokens: int, routes: int
) -> tuple[int, torch.Tensor]:
    """Select ixMM's mode and the task layout shared by gate/up and down."""
    if tokens <= 64:
        return 1, group_layout  # GEMV consumes the original routing ranges.

    experts = group_layout.shape[0]
    split = 128 <= experts <= 512 and routes >= 16 * experts
    if split or (tokens <= 512 and 16 <= experts <= 128):
        from .f4f8_layout import sort_group_layout

        group_layout = sort_group_layout(group_layout, split_large=split)
    return 0, group_layout


class IluF4F8Experts:
    """Compute local experts; the MoE runner owns DP/EP dispatch and combine."""

    def __init__(self, swiglu_limit: float | None) -> None:
        self.swiglu_limit = swiglu_limit

    def apply(
        self,
        *,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
        shared_expert_output: torch.Tensor | None = None,
        scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        from ixformer.inference.functions import (
            activation_dynamic_scaled_fp8,
            group_gemm_fp4,
            moe_compute_token_index_ep_group_layout,
            moe_expand_input_dynamic_scaled_fp8,
            moe_gather_output_reduce_sum,
        )

        tokens, hidden = hidden_states.shape
        output = torch.empty_like(hidden_states)
        if tokens == 0:
            return output
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError("ilu_f4f8 expects BF16 input")
        if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
            raise TypeError("ilu_f4f8 expects int32 top-k IDs and FP32 top-k weights")

        experts, _, width = w13.shape
        intermediate = width // 2
        topk = topk_ids.shape[-1]
        routes = tokens * topk
        # Routing and quantization outputs require distinct storage in ixformer.
        group_layout = torch.empty(
            (experts, 2), dtype=torch.int32, device=hidden_states.device
        )
        src_to_dst, dst_to_src, group_layout = moe_compute_token_index_ep_group_layout(
            topk_ids=topk_ids,
            start_expert_id=0,
            end_expert_id=experts,
            group_layout=group_layout,
            expert_map=expert_map,
            backend="ixformer_native",
        )
        a1, a1_scale = moe_expand_input_dynamic_scaled_fp8(
            hidden_states=hidden_states,
            dst_to_src=dst_to_src,
            top_k=topk,
            zero_invalid=False,  # Downstream ops only consume compact local routes.
            backend="ixformer_native",
        )
        a2 = torch.empty(
            (routes, intermediate),
            dtype=torch.float8_e4m3fn,
            device=hidden_states.device,
        )
        a2_scale = torch.empty(
            (routes,), dtype=torch.float32, device=hidden_states.device
        )
        common_shape = (routes * max(width, hidden),)
        if _workspace._manager is not None:
            (common,) = current_workspace_manager().get_simultaneous(
                (common_shape, torch.bfloat16),
            )
        else:
            common = torch.empty(
                common_shape, dtype=torch.bfloat16, device=hidden_states.device
            )

        # The two GEMMs reuse storage after the intervening activation consumes it.
        gate_up = common[: routes * width].view(routes, width)
        routed_output = common[: routes * hidden].view(routes, hidden)
        mode, gemm_layout = _prepare_group_gemm(group_layout, tokens, routes)
        group_gemm_fp4(
            a=a1,
            scale_a=a1_scale,
            weight=w13,
            scale_weight=w13_scale,
            group_layout=gemm_layout,
            output=gate_up,
            mode=mode,
            backend="ixmm",
        )
        activation_dynamic_scaled_fp8(
            input=gate_up,
            act_type="swiglu",
            limit=self.swiglu_limit,
            output=a2,
            output_scales=a2_scale,
            expand_tokens_gpu=group_layout.view(-1)[-1:],
            backend="ixformer_perf",
        )
        group_gemm_fp4(
            a=a2,
            scale_a=a2_scale,
            weight=w2,
            scale_weight=w2_scale,
            group_layout=gemm_layout,
            output=routed_output,
            mode=mode,
            backend="ixmm",
        )
        return moe_gather_output_reduce_sum(
            input=routed_output,
            topk_weight=topk_weights,
            src_to_dst=src_to_dst,
            output=output,
            extra_residual=shared_expert_output,
            scaling_factor=scaling_factor,
            backend="ixformer_native",
        )
