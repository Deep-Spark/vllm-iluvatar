# SPDX-License-Identifier: Apache-2.0
"""Iluvatar grouped-topk router implementation.

Corresponds to: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py

Target   : vllm.model_executor.layers.fused_moe.router.grouped_topk_router.grouped_topk
Baseline : v0.31.0 (db9527a468)
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
    from vllm import envs

    from vllm_iluvatar.ilu_util.arch import is_ivcore30

    if (
        envs.VLLM_ILU_GLM_MOE_PREPARE_FUSION
        and is_ivcore30()
        and gating_output.is_cuda
        and gating_output.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and gating_output.ndim == 2
        and gating_output.is_contiguous()
        and gating_output.shape[-1] == 256
        and topk == 8
        and num_expert_group == 1
        and topk_group == 1
        and scoring_func == "sigmoid"
        and renormalize
        and e_score_correction_bias is not None
        and e_score_correction_bias.is_contiguous()
        and e_score_correction_bias.dtype in (gating_output.dtype, torch.float32)
    ):
        from ixformer.inference.functions import moe_sigmoid_topk

        return moe_sigmoid_topk(
            gating_output,
            e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
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
        routed_scaling_factor=routed_scaling_factor,
    )
    return topk_weights, topk_ids
