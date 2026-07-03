# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar rotary and Conv3D OOT replacements."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import torch
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

import vllm_iluvatar._custom_ops as ilu_ops
from vllm_iluvatar.model_executor.layers.conv import IluConv3dLayer
from vllm_iluvatar.model_executor.layers.rotary_embedding.base import (
    IluDynamicNTKAlphaRotaryEmbedding,
    IluDynamicNTKScalingRotaryEmbedding,
    IluFourierRotaryEmbedding,
    IluGemma4RotaryEmbedding,
    IluLinearScalingRotaryEmbedding,
    IluLlama3RotaryEmbedding,
    IluNTKScalingRotaryEmbedding,
    IluRotaryEmbedding,
    IluTeleChat3RoPEScaledRotaryEmbedding,
    IluYaRNScalingRotaryEmbedding,
    forward_cuda as ilu_rotary_forward_cuda,
)
from vllm_iluvatar.model_executor.layers.rotary_embedding.common import (
    IluApplyRotaryEmb,
)


def test_rotary_embedding_preserves_explicit_rotary_dim(monkeypatch):
    calls = []

    def fake_rotary_embedding(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(ilu_ops.ixf_ops, "vllm_rotary_embedding", fake_rotary_embedding)

    ilu_ops.rotary_embedding(
        torch.tensor([0], dtype=torch.int64),
        torch.empty((1, 1, 8)),
        None,
        head_size=8,
        cos_sin_cache=torch.empty((4, 6)),
        is_neox=False,
        rotary_dim=4,
    )

    assert calls[0][1]["rotary_dim"] == 4


def test_rotary_embedding_oot_passes_rotary_dim(monkeypatch):
    assert op_registry_oot["RotaryEmbedding"] is IluRotaryEmbedding

    calls = []

    def fake_rotary_embedding(*args):
        calls.append(args)

    monkeypatch.setattr("vllm._custom_ops.rotary_embedding", fake_rotary_embedding)
    op = object.__new__(IluRotaryEmbedding)
    op.use_flashinfer = False
    op.head_size = 8
    op.rotary_dim = 6
    op.is_neox_style = True
    cos_sin_cache = torch.empty((4, 6))
    op._match_cos_sin_cache_dtype = MagicMock(return_value=cos_sin_cache)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    query = torch.empty((2, 1, 8))
    key = torch.empty((2, 1, 8))

    assert IluRotaryEmbedding.forward_cuda(op, positions, query, key) == (query, key)

    op._match_cos_sin_cache_dtype.assert_called_once_with(query)
    assert calls == [
        (positions, query, key, 8, cos_sin_cache, True, 6),
    ]


def test_rotary_embedding_subclass_oot_passes_rotary_dim(monkeypatch):
    assert op_registry_oot["Llama3RotaryEmbedding"] is IluLlama3RotaryEmbedding

    calls = []

    def fake_rotary_embedding(*args):
        calls.append(args)

    monkeypatch.setattr("vllm._custom_ops.rotary_embedding", fake_rotary_embedding)
    op = object.__new__(IluLlama3RotaryEmbedding)
    op.use_flashinfer = False
    op.head_size = 8
    op.rotary_dim = 6
    op.is_neox_style = True
    cos_sin_cache = torch.empty((4, 6))
    op._match_cos_sin_cache_dtype = MagicMock(return_value=cos_sin_cache)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    query = torch.empty((2, 1, 8))
    key = torch.empty((2, 1, 8))

    assert IluLlama3RotaryEmbedding.forward_cuda(op, positions, query, key) == (
        query,
        key,
    )
    assert calls == [
        (positions, query, key, 8, cos_sin_cache, True, 6),
    ]


def test_rotary_embedding_subclass_oot_registrations():
    assert op_registry_oot["LinearScalingRotaryEmbedding"] is (
        IluLinearScalingRotaryEmbedding
    )
    assert op_registry_oot["NTKScalingRotaryEmbedding"] is IluNTKScalingRotaryEmbedding
    assert op_registry_oot["DynamicNTKAlphaRotaryEmbedding"] is (
        IluDynamicNTKAlphaRotaryEmbedding
    )
    assert op_registry_oot["DynamicNTKScalingRotaryEmbedding"] is (
        IluDynamicNTKScalingRotaryEmbedding
    )
    assert op_registry_oot["FourierRotaryEmbedding"] is IluFourierRotaryEmbedding
    assert op_registry_oot["YaRNScalingRotaryEmbedding"] is (
        IluYaRNScalingRotaryEmbedding
    )
    assert op_registry_oot["TeleChat3RoPEScaledRotaryEmbedding"] is (
        IluTeleChat3RoPEScaledRotaryEmbedding
    )
    assert op_registry_oot["Gemma4RotaryEmbedding"] is IluGemma4RotaryEmbedding


def test_rotary_embedding_forward_cuda_ilu_delta():
    assert "self.is_neox_style,\n        self.rotary_dim," in (
        inspect.getsource(ilu_rotary_forward_cuda)
    )


def test_apply_rotary_emb_oot_uses_native_cuda_path():
    assert op_registry_oot["ApplyRotaryEmb"] is IluApplyRotaryEmb

    op = object.__new__(IluApplyRotaryEmb)
    expected = torch.empty((1, 2, 1, 4))
    op.forward_native = MagicMock(return_value=expected)
    x = torch.empty((1, 2, 1, 4))
    cos = torch.empty((2, 2))
    sin = torch.empty((2, 2))

    assert IluApplyRotaryEmb.forward_cuda(op, x, cos, sin) is expected
    op.forward_native.assert_called_once_with(x, cos, sin)


def test_conv3d_oot_uses_matmul_cuda_path():
    assert op_registry_oot["Conv3dLayer"] is IluConv3dLayer

    conv = object.__new__(IluConv3dLayer)
    expected = torch.empty((1, 1, 1, 1, 1))
    conv._forward_mulmat = MagicMock(return_value=expected)
    x = torch.empty((1, 1, 1, 1, 1))

    assert IluConv3dLayer.forward_cuda(conv, x) is expected
    conv._forward_mulmat.assert_called_once_with(x)


def test_rotary_package_is_auto_discovered():
    from vllm_iluvatar.ilu_register import _OOT_OP_PACKAGES

    assert "vllm_iluvatar.model_executor.layers.rotary_embedding" in _OOT_OP_PACKAGES
