# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for ``IluMxfp4MoEMethod.process_weights_after_loading``.

Covers dtype normalisation: the four routed uint8
parameters are re-labelled in place to the dtypes the ixformer pipeline / torch
oracle expect, without moving a single byte. The two silent-failure modes this
guards against are:

* §2.1: ``uint8`` scale passed to ``group_gemm_fp4``'s facade ``.float()`` would
  decode to ``0..255`` integers, not ``2 ** (v - 127)`` -- weights scaled by
  ~1e38.
* §2.2: original-layout ``uint8`` weight would be misread as the ixmm
  preprocessed ``uint8 [E, k/2, n]`` format (dtype-keyed dispatch).

The torch oracle's invariance under this normalisation is exercised in
``test_ilu_mxfp4_fused_moe.py``; this file pins the method-side transform.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import vllm_iluvatar.model_executor.layers.quantization.ilu_mxfp4 as ilu_mxfp4
from vllm_iluvatar.model_executor.layers.quantization.ilu_mxfp4 import (
    IluMxfp4MoEMethod,
)

_H = 64
_I = 32
_E = 4


def _rand_mxfp4(num_experts, out_dim, in_dim, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(
        0, 256, (num_experts, out_dim, in_dim // 2), dtype=torch.uint8, generator=g
    )
    s = torch.randint(
        125, 130, (num_experts, out_dim, in_dim // 32), dtype=torch.uint8, generator=g
    )
    return w, s


class _FakeRoutedExperts(nn.Module):
    """Minimal ``nn.Module`` carrying the four routed MXFP4 parameters."""

    def __init__(self, w13, s13, w2, s2):
        super().__init__()
        self.w13_weight = nn.Parameter(w13, requires_grad=False)
        self.w13_weight_scale = nn.Parameter(s13, requires_grad=False)
        self.w2_weight = nn.Parameter(w2, requires_grad=False)
        self.w2_weight_scale = nn.Parameter(s2, requires_grad=False)


def _make_method():
    """Build an ``IluMxfp4MoEMethod`` without running its (heavy) ``__init__``."""
    return IluMxfp4MoEMethod.__new__(IluMxfp4MoEMethod)


def _make_layer():
    w13, s13 = _rand_mxfp4(_E, 2 * _I, _H, seed=1)
    w2, s2 = _rand_mxfp4(_E, _H, _I, seed=2)
    return _FakeRoutedExperts(w13, s13, w2, s2), (w13, s13, w2, s2)


@pytest.fixture(autouse=True)
def _force_original_layout(monkeypatch):
    """Most tests below pin the zero-copy fallback."""
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_ILU_MXFP4_MOE_BACKEND", "torch", raising=False)


def test_process_weights_relables_dtypes():
    method = _make_method()
    layer, _ = _make_layer()

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.dtype is torch.int8
    assert layer.w2_weight.dtype is torch.int8
    assert layer.w13_weight_scale.dtype is torch.float8_e8m0fnu
    assert layer.w2_weight_scale.dtype is torch.float8_e8m0fnu


def test_process_weights_is_zero_copy():
    """``view(dtype)`` must not move storage: ``data_ptr()`` is unchanged and
    shapes are preserved (zero-copy claim)."""
    method = _make_method()
    layer, (w13, s13, w2, s2) = _make_layer()

    w13_ptr = layer.w13_weight.data_ptr()
    s13_ptr = layer.w13_weight_scale.data_ptr()
    w2_ptr = layer.w2_weight.data_ptr()
    s2_ptr = layer.w2_weight_scale.data_ptr()
    w13_shape = layer.w13_weight.shape
    s13_shape = layer.w13_weight_scale.shape
    w2_shape = layer.w2_weight.shape
    s2_shape = layer.w2_weight_scale.shape

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.data_ptr() == w13_ptr
    assert layer.w13_weight_scale.data_ptr() == s13_ptr
    assert layer.w2_weight.data_ptr() == w2_ptr
    assert layer.w2_weight_scale.data_ptr() == s2_ptr
    assert layer.w13_weight.shape == w13_shape
    assert layer.w13_weight_scale.shape == s13_shape
    assert layer.w2_weight.shape == w2_shape
    assert layer.w2_weight_scale.shape == s2_shape


def test_process_weights_rejects_non_uint8_input():
    method = _make_method()
    layer, _ = _make_layer()
    # Simulate a double-run: already int8.
    method.process_weights_after_loading(layer)
    with pytest.raises(AssertionError):
        method.process_weights_after_loading(layer)


def test_process_weights_rejects_non_contiguous_input():
    method = _make_method()
    layer, _ = _make_layer()
    # Break contiguity of w13_weight.
    layer.w13_weight = nn.Parameter(
        layer.w13_weight.data.transpose(1, 2).contiguous().transpose(1, 2),
        requires_grad=False,
    )
    assert not layer.w13_weight.is_contiguous()
    with pytest.raises(AssertionError):
        method.process_weights_after_loading(layer)


def test_process_weights_preprocesses_ixmm_layout(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_ILU_MXFP4_MOE_BACKEND", "ixmm", raising=False)
    monkeypatch.setattr(ilu_mxfp4, "_ixmm_backend_available", lambda: True)

    fake_ixfops = MagicMock()

    def _preprocess(weight, scale):
        assert weight.dtype is torch.int8
        assert scale.dtype is torch.float8_e8m0fnu
        expert, out_dim, packed_k = weight.shape
        return (
            torch.zeros(expert, packed_k, out_dim, dtype=torch.uint8),
            torch.ones(
                expert, (packed_k * 2) // 256, out_dim, dtype=torch.float32
            ),
        )

    fake_ixfops.preprocess_weight_for_ixmm.side_effect = _preprocess
    fake_ixformer = MagicMock()
    fake_inference = MagicMock()
    fake_inference.functions = fake_ixfops
    fake_ixformer.inference = fake_inference
    monkeypatch.setitem(sys.modules, "ixformer", fake_ixformer)
    monkeypatch.setitem(sys.modules, "ixformer.inference", fake_inference)
    monkeypatch.setitem(sys.modules, "ixformer.inference.functions", fake_ixfops)

    method = _make_method()
    layer, (w13, _, w2, _) = _make_layer()
    method.process_weights_after_loading(layer)

    assert layer._ilu_mxfp4_ixmm is True
    assert layer.w13_weight.dtype is torch.uint8
    assert layer.w13_weight.shape == (w13.shape[0], w13.shape[2], w13.shape[1])
    assert layer.w13_weight_scale.dtype is torch.float32
    assert layer.w2_weight.dtype is torch.uint8
    assert layer.w2_weight.shape == (w2.shape[0], w2.shape[2], w2.shape[1])
    assert layer.w2_weight_scale.dtype is torch.float32
    assert fake_ixfops.preprocess_weight_for_ixmm.call_count == 2
