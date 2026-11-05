# SPDX-License-Identifier: Apache-2.0
"""Tests for ixaite activation provider routing."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import vllm._custom_ops as ops

import vllm_iluvatar._custom_ops as ilu_ops

# Reset every routed target so tests do not leak provider choices.


# Check the stable wrapper starts on the original ixformer implementation.
def test_silu_and_mul_custom_op_dispatches_to_ixformer(monkeypatch):
    ixformer = Mock()
    monkeypatch.setattr(ilu_ops.ixf_ops, "silu_and_mul", ixformer)

    x = object()
    out = object()

    assert ilu_ops.silu_and_mul(out, x) is None
    ixformer.assert_called_once_with(x, out)


# Check both exported module paths observe the one startup-selected target.


# Check auto chooses the original callable before forward when ixaite is unsupported.


# Check the exported vLLM wrapper reaches the real ixaite op on an Iluvatar GPU.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_silu_and_mul_ixaite_wrapper_matches_ixformer(monkeypatch):
    from vllm_iluvatar.backends.ixaite import ops as ixaite_ops

    monkeypatch.setattr(ops, "silu_and_mul", ixaite_ops.silu_and_mul, raising=False)
    torch.manual_seed(7)
    x = torch.randn((8, 24576), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty((8, 12288), device="cuda", dtype=torch.bfloat16)
    expected = torch.empty_like(actual)

    ops.silu_and_mul(actual, x)
    ilu_ops.silu_and_mul(expected, x)

    assert ops.silu_and_mul is torch.ops.ixaite.silu_and_mul.default
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
