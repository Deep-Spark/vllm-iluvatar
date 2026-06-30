# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar compressed-tensors MoE method selection."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    ilu_compressed_tensors_moe as ilu_moe,
)


class _FakeMoEMethod:
    def __init__(self, weight_quant, input_quant, moe_config):
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.moe_config = moe_config


class _FakeUnquantizedFusedMoEMethod:
    def __init__(self, moe_config):
        self.moe_config = moe_config


def _quant_config_with_schemes(*schemes):
    quant_config = MagicMock()
    quant_config.get_scheme_dict.side_effect = list(schemes)
    quant_config._is_wNa16_group_channel.return_value = False
    quant_config._is_dynamic_token_w8a8.return_value = False
    return quant_config


def _install_fake_method_module(monkeypatch, module_name: str, class_name: str):
    module = types.ModuleType(module_name)
    setattr(module, class_name, _FakeMoEMethod)
    monkeypatch.setitem(sys.modules, module_name, module)


def test_scheme_none_returns_unquantized_fused_moe_method(monkeypatch):
    moe_config = object()
    quant_config = _quant_config_with_schemes(None, None, None)
    layer = SimpleNamespace(moe_config=moe_config)
    monkeypatch.setattr(
        ilu_moe, "UnquantizedFusedMoEMethod", _FakeUnquantizedFusedMoEMethod
    )

    method = ilu_moe.IluCompressedTensorsMoEMethod.get_moe_method(
        quant_config, layer, "model.layers.0.mlp.experts"
    )

    assert isinstance(method, _FakeUnquantizedFusedMoEMethod)
    assert method.moe_config is moe_config


def test_mismatched_moe_projection_schemes_raise():
    quant_config = _quant_config_with_schemes(
        {"weights": object()},
        {"weights": object()},
        {"weights": object()},
    )
    layer = SimpleNamespace(moe_config=object())

    with pytest.raises(ValueError, match="same quantization scheme"):
        ilu_moe.IluCompressedTensorsMoEMethod.get_moe_method(
            quant_config, layer, "model.layers.0.mlp.experts"
        )


def test_w4a8_canonical_env_selects_ilu_w4a8_method(monkeypatch):
    module_name = (
        "vllm_iluvatar.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.ilu_compressed_tensors_moe_w4a8_int8"
    )
    _install_fake_method_module(
        monkeypatch, module_name, "IluCompressedTensorsW4A8Int8MoEMethod"
    )
    monkeypatch.setattr(ilu_moe.envs, "VLLM_ILU_W8A8_MOE_USE_W4A8", True, raising=False)
    monkeypatch.setattr(ilu_moe.envs, "VLLM_W8A8_MOE_USE_W4A8", False, raising=False)
    weight_quant = SimpleNamespace(num_bits=8)
    input_quant = SimpleNamespace(num_bits=8)
    quant_config = _quant_config_with_schemes(
        {"weights": weight_quant, "input_activations": input_quant},
        {"weights": weight_quant, "input_activations": input_quant},
        {"weights": weight_quant, "input_activations": input_quant},
    )
    layer = SimpleNamespace(moe_config=object())

    method = ilu_moe.IluCompressedTensorsMoEMethod.get_moe_method(
        quant_config, layer, "model.layers.0.mlp.experts"
    )

    assert isinstance(method, _FakeMoEMethod)
    assert method.weight_quant is weight_quant
    assert method.input_quant is input_quant
    assert method.moe_config is layer.moe_config


def test_w4a8_legacy_env_selects_ilu_w4a8_method(monkeypatch):
    module_name = (
        "vllm_iluvatar.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.ilu_compressed_tensors_moe_w4a8_int8"
    )
    _install_fake_method_module(
        monkeypatch, module_name, "IluCompressedTensorsW4A8Int8MoEMethod"
    )
    monkeypatch.setattr(ilu_moe.envs, "VLLM_ILU_W8A8_MOE_USE_W4A8", False, raising=False)
    monkeypatch.setattr(ilu_moe.envs, "VLLM_W8A8_MOE_USE_W4A8", True, raising=False)
    weight_quant = SimpleNamespace(num_bits=8)
    input_quant = SimpleNamespace(num_bits=8)
    quant_config = _quant_config_with_schemes(
        {"weights": weight_quant, "input_activations": input_quant},
        {"weights": weight_quant, "input_activations": input_quant},
        {"weights": weight_quant, "input_activations": input_quant},
    )
    layer = SimpleNamespace(moe_config=object())

    method = ilu_moe.IluCompressedTensorsMoEMethod.get_moe_method(
        quant_config, layer, "model.layers.0.mlp.experts"
    )

    assert isinstance(method, _FakeMoEMethod)
    assert method.weight_quant is weight_quant
    assert method.input_quant is input_quant


def test_wna16_env_selects_ilu_w4a8_method(monkeypatch):
    module_name = (
        "vllm_iluvatar.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.ilu_compressed_tensors_moe_w4a8_int8"
    )
    _install_fake_method_module(
        monkeypatch, module_name, "IluCompressedTensorsW4A8Int8MoEMethod"
    )
    monkeypatch.setattr(ilu_moe.envs, "VLLM_ILU_W8A8_MOE_USE_W4A8", False, raising=False)
    monkeypatch.setattr(ilu_moe.envs, "VLLM_W8A8_MOE_USE_W4A8", False, raising=False)
    monkeypatch.setattr(
        ilu_moe.envs, "VLLM_ILU_WNA16_MOE_USE_W4A8", True, raising=False
    )
    monkeypatch.setattr(ilu_moe.envs, "VLLM_WNA16_MOE_USE_W4A8", False, raising=False)
    weight_quant = SimpleNamespace(num_bits=4, group_size=None)
    input_quant = SimpleNamespace(num_bits=16)
    scheme = {
        "weights": weight_quant,
        "input_activations": input_quant,
        "format": ilu_moe.CompressionFormat.pack_quantized.value,
    }
    quant_config = _quant_config_with_schemes(scheme, scheme, scheme)
    quant_config._is_wNa16_group_channel.return_value = True
    layer = SimpleNamespace(moe_config=object())

    method = ilu_moe.IluCompressedTensorsMoEMethod.get_moe_method(
        quant_config, layer, "model.layers.0.mlp.experts"
    )

    assert isinstance(method, _FakeMoEMethod)
    assert method.weight_quant is weight_quant
    assert method.input_quant is input_quant


def test_wna16_env_rejects_unsupported_format(monkeypatch):
    monkeypatch.setattr(ilu_moe.envs, "VLLM_ILU_W8A8_MOE_USE_W4A8", False, raising=False)
    monkeypatch.setattr(ilu_moe.envs, "VLLM_W8A8_MOE_USE_W4A8", False, raising=False)
    monkeypatch.setattr(
        ilu_moe.envs, "VLLM_ILU_WNA16_MOE_USE_W4A8", True, raising=False
    )
    monkeypatch.setattr(ilu_moe.envs, "VLLM_WNA16_MOE_USE_W4A8", False, raising=False)
    weight_quant = SimpleNamespace(num_bits=4, group_size=None)
    input_quant = SimpleNamespace(num_bits=16)
    scheme = {
        "weights": weight_quant,
        "input_activations": input_quant,
        "format": "unsupported",
    }
    quant_config = _quant_config_with_schemes(scheme, scheme, scheme)
    quant_config._is_wNa16_group_channel.return_value = True
    layer = SimpleNamespace(moe_config=object())

    with pytest.raises(ValueError, match="unsupported"):
        ilu_moe.IluCompressedTensorsMoEMethod.get_moe_method(
            quant_config, layer, "model.layers.0.mlp.experts"
        )


def test_dynamic_token_w8a8_selects_ilu_w8a8_method(monkeypatch):
    module_name = (
        "vllm_iluvatar.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.ilu_compressed_tensors_moe_w8a8_int8"
    )
    _install_fake_method_module(
        monkeypatch, module_name, "IluCompressedTensorsW8A8Int8MoEMethod"
    )
    monkeypatch.setattr(ilu_moe.envs, "VLLM_ILU_W8A8_MOE_USE_W4A8", False, raising=False)
    monkeypatch.setattr(ilu_moe.envs, "VLLM_W8A8_MOE_USE_W4A8", False, raising=False)
    weight_quant = SimpleNamespace(num_bits=8)
    input_quant = SimpleNamespace(num_bits=8)
    quant_config = _quant_config_with_schemes(
        {"weights": weight_quant, "input_activations": input_quant},
        {"weights": weight_quant, "input_activations": input_quant},
        {"weights": weight_quant, "input_activations": input_quant},
    )
    quant_config._is_dynamic_token_w8a8.return_value = True
    layer = SimpleNamespace(moe_config=object())

    method = ilu_moe.IluCompressedTensorsMoEMethod.get_moe_method(
        quant_config, layer, "model.layers.0.mlp.experts"
    )

    assert isinstance(method, _FakeMoEMethod)
    assert method.weight_quant is weight_quant
    assert method.input_quant is input_quant
