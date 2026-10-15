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


def _run_vl_topk(input_tokens, *, image_sentinel_lo=100, is_padding=None, top_k=2):
    """Route four fixed rows; returns (weights, indices, bias_vl).

    Which bias a row selects is decided inside the kernel, so these tests cover
    only what the wrapper still owns: what it forwards and the padding mask.
    The routing semantics are tested in ixformer
    (``tests/python/inference/functions/test_topk_hash_softplus_sqrt.py``).
    """
    gating_output = torch.arange(1.0, 5.0).outer(torch.arange(1.0, 5.0))
    topk_weights = torch.empty(4, top_k)
    topk_indices = torch.empty(4, top_k, dtype=torch.int64)
    token_expert_indices = torch.empty_like(topk_indices)
    bias_vl = torch.tensor([100.0, 99.0, 0.0, 0.0])

    ilu_ops.topk_hash_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        False,
        2.0,
        torch.zeros(4),
        input_tokens,
        None,
        is_padding=is_padding,
        bias_vl=bias_vl,
        image_sentinel_lo=image_sentinel_lo,
    )
    return topk_weights, topk_indices, bias_vl


def _capture_kernel(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        _fill_topk_outputs(kwargs["topk_weights"], kwargs["topk_indices"])

    monkeypatch.setattr(ilu_ops.ixf_ops, "topk_hash_softplus_sqrt", fake_kernel)
    return calls


def test_topk_hash_softplus_sqrt_forwards_bias_vl_to_the_kernel(monkeypatch):
    calls = _capture_kernel(monkeypatch)

    tokens = torch.tensor([99, 100, 104, 105], dtype=torch.int32)
    _, _, bias_vl = _run_vl_topk(tokens)

    assert len(calls) == 1
    assert calls[0]["bias_vl"] is bias_vl
    assert calls[0]["image_sentinel_lo"] == 100
    # The kernel reads the ids to tell image rows apart, so it needs int32.
    assert calls[0]["input_tokens"].dtype == torch.int32


def test_topk_hash_softplus_sqrt_bias_vl_disabled_by_zero_sentinel(monkeypatch):
    calls = _capture_kernel(monkeypatch)

    tokens = torch.tensor([100, 100, 100, 100], dtype=torch.int32)
    _run_vl_topk(tokens, image_sentinel_lo=0)

    # The kernel rejects a bias_vl that arrives with a disabled sentinel, so
    # the wrapper has to drop it.
    assert len(calls) == 1
    assert calls[0]["bias_vl"] is None
    assert calls[0]["image_sentinel_lo"] == 0


def test_topk_hash_softplus_sqrt_bias_vl_masks_padding_rows(monkeypatch):
    monkeypatch.setattr(
        ilu_ops.ixf_ops,
        "topk_hash_softplus_sqrt",
        lambda **kwargs: None,
    )
    tokens = torch.tensor([100, 100, 100, 100], dtype=torch.int32)
    weights, indices, _ = _run_vl_topk(
        tokens, is_padding=torch.tensor([False, True, False, False])
    )

    assert torch.equal(weights[1], torch.zeros_like(weights[1]))
    assert torch.equal(indices[1], torch.full_like(indices[1], -1))
