# SPDX-License-Identifier: Apache-2.0
"""Iluvatar grouped-topk router implementation.

Corresponds to: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py

Target   : vllm.model_executor.layers.fused_moe.router.grouped_topk_router.grouped_topk
Baseline : v0.31.0 (db9527a468)
Why      : Iluvatar delegates grouped-topk routing (including architecture-specific
           single-group sigmoid fast paths and general fallbacks) to ixformer.
How      : general patch rebinds the module-level ``grouped_topk`` function to
           this implementation. No community method body is copied.
Scope    : grouped-topk MoE routing on Iluvatar.
Remove   : when upstream exposes a grouped-topk backend hook or supports the
           Iluvatar ixformer kernel natively.
"""

from __future__ import annotations

import torch

from vllm_iluvatar.ilu_util.arch import is_ivcore30


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

    group_scoring = "max"
    if e_score_correction_bias is not None:
        group_scoring = "top2sum"
        bias_dtype = e_score_correction_bias.dtype
        if bias_dtype != gating_output.dtype and (
            bias_dtype != torch.float32 or not is_ivcore30()
        ):
            e_score_correction_bias = e_score_correction_bias.to(gating_output.dtype)

    return _ixf_grouped_topk(
        gating_output,
        topk,
        num_expert_group,
        topk_group,
        scoring_func,
        e_score_correction_bias,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        group_scoring_func=group_scoring,
    )
