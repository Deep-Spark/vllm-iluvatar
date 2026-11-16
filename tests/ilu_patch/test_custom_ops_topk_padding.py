# SPDX-License-Identifier: Apache-2.0
"""Tests for Iluvatar custom top-k wrappers."""

from __future__ import annotations

import inspect

import torch

import vllm_iluvatar._custom_ops as ilu_ops


def _fill_topk_outputs(topk_weight, topk_ids):
    topk_weight.copy_(
        torch.arange(topk_weight.numel(), dtype=topk_weight.dtype).reshape_as(
            topk_weight
        )
    )
    topk_ids.copy_(
        torch.arange(topk_ids.numel(), dtype=topk_ids.dtype).reshape_as(topk_ids)
    )


def test_topk_wrappers_accept_is_padding():
    for name in ("topk_softmax", "topk_sigmoid", "topk_hash_softplus_sqrt"):
        assert "is_padding" in inspect.signature(getattr(ilu_ops, name)).parameters


def test_topk_softmax_masks_padding_rows(monkeypatch):
    def fake_moe_fused_topk_bias(**kwargs):
        _fill_topk_outputs(kwargs["topk_weight"], kwargs["topk_ids"])

    monkeypatch.setattr(
        ilu_ops.ixf_ops,
        "moe_fused_topk_bias",
        fake_moe_fused_topk_bias,
    )
    topk_weights = torch.empty(3, 2)
    topk_ids = torch.empty(3, 2, dtype=torch.int32)
    token_expert_indices = torch.empty_like(topk_ids)

    ilu_ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        torch.randn(3, 4),
        is_padding=torch.tensor([False, True, False]),
    )

    assert torch.equal(topk_weights[1], torch.zeros_like(topk_weights[1]))
    assert torch.equal(topk_ids[1], torch.full_like(topk_ids[1], -1))
    assert torch.count_nonzero(topk_weights[[0, 2]]) > 0


def test_topk_sigmoid_masks_padding_rows(monkeypatch):
    def fake_moe_fused_topk_bias(**kwargs):
        _fill_topk_outputs(kwargs["topk_weight"], kwargs["topk_ids"])

    monkeypatch.setattr(
        ilu_ops.ixf_ops,
        "moe_fused_topk_bias",
        fake_moe_fused_topk_bias,
    )
    topk_weights = torch.empty(3, 2)
    topk_ids = torch.empty(3, 2, dtype=torch.int32)
    token_expert_indices = torch.empty_like(topk_ids)

    ilu_ops.topk_sigmoid(
        topk_weights,
        topk_ids,
        token_expert_indices,
        torch.randn(3, 4),
        is_padding=torch.tensor([True, False, True]),
    )

    assert torch.equal(topk_weights[0], torch.zeros_like(topk_weights[0]))
    assert torch.equal(topk_ids[0], torch.full_like(topk_ids[0], -1))
    assert torch.equal(topk_weights[2], torch.zeros_like(topk_weights[2]))
    assert torch.equal(topk_ids[2], torch.full_like(topk_ids[2], -1))


def test_topk_hash_softplus_sqrt_masks_padding_rows(monkeypatch):
    def fake_topk_hash_softplus_sqrt(**kwargs):
        _fill_topk_outputs(kwargs["topk_weights"], kwargs["topk_indices"])

    monkeypatch.setattr(
        ilu_ops.ixf_ops,
        "topk_hash_softplus_sqrt",
        fake_topk_hash_softplus_sqrt,
    )
    topk_weights = torch.empty(3, 2)
    topk_indices = torch.empty(3, 2, dtype=torch.int64)
    token_expert_indices = torch.empty_like(topk_indices)

    ilu_ops.topk_hash_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        torch.randn(3, 4),
        is_padding=torch.tensor([False, True, False]),
    )

    assert torch.equal(topk_weights[1], torch.zeros_like(topk_weights[1]))
    assert torch.equal(topk_indices[1], torch.full_like(topk_indices[1], -1))


def test_mask_topk_padding_skips_row_mismatch():
    topk_weights = torch.ones(8, 6)
    topk_ids = torch.arange(48, dtype=torch.int32).reshape(8, 6)
    # Local dummy rows vs gathered Hybrid EP naive tokens.
    is_padding = torch.ones(2, 1, dtype=torch.bool)

    ilu_ops._mask_topk_padding(topk_weights, topk_ids, is_padding)

    assert torch.equal(topk_weights, torch.ones(8, 6))
    assert torch.equal(topk_ids, torch.arange(48, dtype=torch.int32).reshape(8, 6))


def test_mask_topk_padding_preserves_dp1_and_skips_ungathered_dp(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    ctx = SimpleNamespace(dp_metadata=None)
    fc = ModuleType("vllm.forward_context")
    fc.is_forward_context_available = lambda: True
    fc.get_forward_context = lambda: ctx
    monkeypatch.setitem(sys.modules, "vllm.forward_context", fc)

    # DP=1 (dp_metadata is None): local padding must be masked.
    w = torch.ones(2, 2)
    ids = torch.ones(2, 2, dtype=torch.int32)
    ilu_ops._mask_topk_padding(w, ids, torch.tensor([False, True]))
    assert w[:, 0].tolist() == [1.0, 0.0]
    assert ids[:, 0].tolist() == [1, -1]

    # DP>1 non-decode without gathered mask: skip even if row count matches.
    ctx.dp_metadata = SimpleNamespace(all_ranks_decode_only=False)
    w = torch.ones(2, 2)
    ids = torch.ones(2, 2, dtype=torch.int32)
    ilu_ops._mask_topk_padding(w, ids, torch.tensor([False, True]))
    assert w[:, 0].tolist() == [1.0, 1.0]
    assert ids[:, 0].tolist() == [1, 1]

    # DP>1 with gathered mask flag set: mask is applied.
    ctx._ilu_is_padding_gathered = True
    ilu_ops._mask_topk_padding(w, ids, torch.tensor([False, True]))
    assert w[:, 0].tolist() == [1.0, 0.0]
    assert ids[:, 0].tolist() == [1, -1]
