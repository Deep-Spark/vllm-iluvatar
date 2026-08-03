# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar grouped-topk router patch."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

import torch
import vllm.model_executor.layers.fused_moe.router.grouped_topk_router as community

from vllm_iluvatar.ilu_patch.general_patches import patch_grouped_topk_router
from vllm_iluvatar.model_executor.layers.fused_moe.router import (
    ilu_grouped_topk_router,
)


def _install_fake_ixformer(monkeypatch, fake_grouped_topk):
    ixformer = ModuleType("ixformer")
    inference = ModuleType("ixformer.inference")
    functions = ModuleType("ixformer.inference.functions")
    functions.moe_grouped_topk = fake_grouped_topk
    inference.functions = functions
    ixformer.inference = inference

    monkeypatch.setitem(sys.modules, "ixformer", ixformer)
    monkeypatch.setitem(sys.modules, "ixformer.inference", inference)
    monkeypatch.setitem(sys.modules, "ixformer.inference.functions", functions)


def test_grouped_topk_calls_ixformer_and_applies_routed_scale(monkeypatch):
    calls = []
    raw_weights = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    raw_ids = torch.tensor([[3, 4]], dtype=torch.int32)

    def fake_grouped_topk(*args, **kwargs):
        calls.append((args, kwargs))
        return raw_weights, raw_ids

    _install_fake_ixformer(monkeypatch, fake_grouped_topk)
    gating_output = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.bfloat16)
    bias = torch.ones(4, dtype=torch.float32)

    topk_weights, topk_ids = ilu_grouped_topk_router.grouped_topk(
        hidden_states=torch.empty((1, 4)),
        gating_output=gating_output,
        topk=2,
        renormalize=True,
        num_expert_group=2,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=3.0,
        e_score_correction_bias=bias,
    )

    torch.testing.assert_close(topk_weights, raw_weights * 3.0)
    assert topk_ids is raw_ids
    args, kwargs = calls[0]
    assert args[:5] == (gating_output, 2, 2, 1, "sigmoid")
    assert args[5].dtype is torch.bfloat16
    assert kwargs == {"renormalize": True}


def test_grouped_topk_patch_rebinds_module_function_and_new_custom_op():
    patch_grouped_topk_router._apply()

    assert community.grouped_topk is ilu_grouped_topk_router.grouped_topk

    with patch("vllm.model_executor.custom_op.CustomOp.__init__", return_value=None):
        grouped_topk_op = community.GroupedTopk(
            topk=2,
            renormalize=True,
            num_expert_group=2,
            topk_group=1,
        )
    assert grouped_topk_op.native_impl is ilu_grouped_topk_router.grouped_topk
