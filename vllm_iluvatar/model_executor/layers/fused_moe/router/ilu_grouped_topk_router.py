# SPDX-License-Identifier: Apache-2.0
"""Iluvatar grouped-topk router implementation.

Corresponds to: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py

Target   : vllm.model_executor.layers.fused_moe.router.grouped_topk_router.grouped_topk
Baseline : main_6e448d0
Why      : Iluvatar uses ixformer's grouped-topk kernel instead of community
           CUDA/Torch grouped-topk paths. ixformer requires the noaux_tc
           correction bias dtype to match the router logits dtype.
How      : general patch rebinds the module-level ``grouped_topk`` function to
           this implementation. No community method body is copied.
Scope    : grouped-topk MoE routing on Iluvatar.
Remove   : when upstream exposes a grouped-topk backend hook or supports the
           Iluvatar ixformer kernel natively.
"""

from __future__ import annotations

import torch


def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run grouped-topk through ixformer and preserve upstream scale semantics."""
    from ixformer.inference.functions import (  # type: ignore[import]
        moe_grouped_topk as _ixf_grouped_topk,
    )

    # [ILU override] grouped_topk - delta: route selection through ixformer.
    if (
        e_score_correction_bias is not None
        and e_score_correction_bias.dtype != gating_output.dtype
    ):
        e_score_correction_bias = e_score_correction_bias.to(dtype=gating_output.dtype)

    topk_weights, topk_ids = _ixf_grouped_topk(
        gating_output,
        topk,
        num_expert_group,
        topk_group,
        scoring_func,
        e_score_correction_bias,
        renormalize=renormalize,
    )
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids
