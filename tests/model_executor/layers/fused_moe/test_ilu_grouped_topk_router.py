# SPDX-License-Identifier: Apache-2.0
"""Tests for the Iluvatar grouped-topk router patch."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

import pytest
import torch
import vllm.model_executor.layers.fused_moe.router.grouped_topk_router as community

from vllm_iluvatar.ilu_patch.general_patches import patch_grouped_topk_router
from vllm_iluvatar.model_executor.layers.fused_moe.router import (
    ilu_grouped_topk_router,
)

requires_cq = pytest.mark.skipif(
    not torch.cuda.is_available() or getattr(torch, "corex_arch", None) != "ivcore30",
    reason="CQ grouped top-k kernel",
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


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("cq", [False, True])
def test_grouped_topk_delegates_cleanly_to_ixformer(monkeypatch, has_bias, cq):
    monkeypatch.setattr(ilu_grouped_topk_router, "is_ivcore30", lambda: cq)
    calls = []
    raw_weights = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    raw_ids = torch.tensor([[3, 4]], dtype=torch.int32)

    def fake_grouped_topk(*args, **kwargs):
        calls.append((args, kwargs))
        scale = kwargs.get("routed_scaling_factor", 1.0)
        return raw_weights * scale, raw_ids

    _install_fake_ixformer(monkeypatch, fake_grouped_topk)
    gating_output = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.bfloat16)
    bias = torch.ones(4, dtype=torch.float32) if has_bias else None

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
    if bias is not None and not cq:
        torch.testing.assert_close(args[5], bias.to(gating_output.dtype))
    else:
        assert args[5] is bias
    assert kwargs == {
        "renormalize": True,
        "routed_scaling_factor": 3.0,
        "group_scoring_func": "top2sum" if has_bias and cq else "max",
    }


@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.float64])
def test_grouped_topk_cq_casts_non_fp32_mismatch_bias(monkeypatch, bias_dtype):
    monkeypatch.setattr(ilu_grouped_topk_router, "is_ivcore30", lambda: True)
    captured = []
    _install_fake_ixformer(
        monkeypatch,
        lambda *args, **kwargs: (
            captured.append(args[5])
            or (
                torch.ones(1, 2, dtype=torch.float32),
                torch.zeros(1, 2, dtype=torch.int32),
            )
        ),
    )
    gating_output = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.bfloat16)
    bias = torch.ones(4, dtype=bias_dtype)
    ilu_grouped_topk_router.grouped_topk(
        hidden_states=torch.empty((1, 4)),
        gating_output=gating_output,
        topk=2,
        renormalize=True,
        num_expert_group=2,
        topk_group=1,
        e_score_correction_bias=bias,
    )
    assert captured[0].dtype == torch.bfloat16


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


def _routing_reference(
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = logits.cpu().float()
    scores = scores.softmax(dim=-1) if scoring_func == "softmax" else scores.sigmoid()
    original = scores
    if bias is not None:
        scores = scores + bias.cpu().to(logits.dtype).float().unsqueeze(0)
    num_tokens = scores.shape[0]
    grouped = scores.view(num_tokens, num_expert_group, -1)
    group_scores = (
        grouped.topk(2, dim=-1).values.sum(dim=-1)
        if bias is not None
        else grouped.max(dim=-1).values
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_tokens, -1)
    )
    masked = scores.masked_fill(~score_mask.bool(), float("-inf"))
    expected_ids = masked.argsort(dim=-1, descending=True, stable=True)[:, :topk]
    selected = original.gather(1, expected_ids)
    if renormalize:
        selected = selected / selected.sum(-1, keepdim=True)
    return selected * routed_scaling_factor, expected_ids


@requires_cq
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 129])
@pytest.mark.parametrize("experts", [256, 288, 320])
def test_single_group_sigmoid_routing_native(dtype, tokens, experts):
    torch.manual_seed(1701 + experts)
    logits = torch.randn(tokens, experts, device="cuda", dtype=dtype)
    bias = torch.randn(experts, device="cuda", dtype=torch.float32) * 0.01
    weights, ids = ilu_grouped_topk_router.grouped_topk(
        torch.empty(tokens, 1, device="cuda"),
        logits,
        8,
        True,
        1,
        1,
        "sigmoid",
        2.5,
        bias,
    )
    expected_weights, expected_ids = _routing_reference(
        logits, bias, 8, 1, 1, "sigmoid", True, 2.5
    )
    torch.testing.assert_close(ids.cpu().long(), expected_ids)
    torch.testing.assert_close(weights.cpu(), expected_weights, rtol=2e-5, atol=2e-6)


@requires_cq
@pytest.mark.parametrize(
    "dtype,experts,groups,topk_group,scoring_func",
    [
        (torch.bfloat16, 256, 8, 4, "sigmoid"),
        (torch.float32, 256, 8, 4, "sigmoid"),
        (torch.float32, 288, 8, 4, "sigmoid"),
        (torch.bfloat16, 320, 8, 4, "softmax"),
    ],
)
def test_multi_group_and_fallback_routing(
    dtype, experts, groups, topk_group, scoring_func
):
    torch.manual_seed(2301 + experts)
    logits = torch.randn(32, experts, device="cuda", dtype=dtype)
    bias = torch.randn(experts, device="cuda", dtype=torch.float32) * 0.05
    weights, ids = ilu_grouped_topk_router.grouped_topk(
        torch.empty(32, 1, device="cuda"),
        logits,
        8,
        True,
        groups,
        topk_group,
        scoring_func,
        2.5,
        bias,
    )
    expected_weights, expected_ids = _routing_reference(
        logits, bias, 8, groups, topk_group, scoring_func, True, 2.5
    )
    # Compare sorted selected scores so ties across multi-group kernels remain valid.
    scores = logits.cpu().float()
    scores = scores.softmax(dim=-1) if scoring_func == "softmax" else scores.sigmoid()
    ranked = scores + bias.cpu().to(dtype).float().unsqueeze(0)
    torch.testing.assert_close(
        ranked.gather(1, ids.cpu().long()).sort(dim=-1).values,
        ranked.gather(1, expected_ids).sort(dim=-1).values,
        rtol=2e-5,
        atol=2e-6,
    )
    ref_weights_from_ids = scores.gather(1, ids.cpu().long())
    ref_weights_from_ids = (
        ref_weights_from_ids / ref_weights_from_ids.sum(-1, keepdim=True) * 2.5
    )
    torch.testing.assert_close(
        weights.cpu(), ref_weights_from_ids, rtol=2e-5, atol=2e-6
    )


@requires_cq
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_bias", [False, True])
def test_correction_bias_selects_groups_by_top2_sum(dtype, has_bias):
    probabilities = torch.tensor(
        [
            [
                0.9,
                0.1,
                0.08,
                0.07,
                0.06,
                0.05,
                0.04,
                0.03,
                0.65,
                0.64,
                0.2,
                0.19,
                0.18,
                0.17,
                0.16,
                0.15,
            ]
        ],
        device="cuda",
    )
    logits = probabilities.logit().to(dtype)
    bias = torch.zeros(16, device="cuda") if has_bias else None
    weights, ids = ilu_grouped_topk_router.grouped_topk(
        torch.empty(1, 1, device="cuda"),
        logits,
        8,
        True,
        2,
        1,
        "sigmoid",
        2.5,
        bias,
    )
    start = 8 if has_bias else 0
    assert torch.equal(
        ids.long().sort().values[0], torch.arange(start, start + 8, device="cuda")
    )
    expected = logits.float().sigmoid().gather(1, ids.long())
    expected = expected / expected.sum(-1, keepdim=True) * 2.5
    torch.testing.assert_close(weights, expected, rtol=2e-5, atol=2e-6)
