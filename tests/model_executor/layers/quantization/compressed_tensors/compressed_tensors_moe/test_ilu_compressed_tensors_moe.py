# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar compressed-tensors MoE method selection."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_iluvatar.ilu_patch import general_patch as p
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    ilu_compressed_tensors_moe as ilu_moe,
)

_MAIN_6E448D0_SCHEME_DISCOVERY_PREFIX_SHA = (
    "5d77ca86af26fe1381f7d00825ca648176edf35d6834544323638fe96494fa06"
)


class _FakeMoEMethod:
    def __init__(self, weight_quant, input_quant, moe_config, ilu_params=None):
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.moe_config = moe_config
        self.ilu_params = ilu_params


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


def test_scheme_discovery_fingerprint_baseline_matches_declared_main_6e448d0_source():
    assert p.source_sha(ilu_moe._SCHEME_DISCOVERY_PREFIX_BASELINE) == (
        _MAIN_6E448D0_SCHEME_DISCOVERY_PREFIX_SHA
    )


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


def test_empty_ilu_params_falls_back_to_official(monkeypatch):
    fallback = object()
    monkeypatch.setattr(
        ilu_moe,
        "_ORIGINAL_GET_MOE_METHOD",
        lambda quant_config, layer, layer_name: fallback,
    )
    weight_quant = SimpleNamespace(num_bits=4)
    input_quant = SimpleNamespace(num_bits=8)
    scheme = {
        "weights": weight_quant,
        "input_activations": input_quant,
        "ilu_params": {},
    }
    quant_config = _quant_config_with_schemes(scheme, scheme, scheme)
    layer = SimpleNamespace(moe_config=object())

    method = ilu_moe.IluCompressedTensorsMoEMethod.get_moe_method(
        quant_config, layer, "model.layers.0.mlp.experts"
    )

    assert method is fallback


def test_w4a8_config_selects_ilu_w4a8_method(monkeypatch):
    module_name = (
        "vllm_iluvatar.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.ilu_compressed_tensors_moe_w4a8_int8"
    )
    _install_fake_method_module(
        monkeypatch, module_name, "IluCompressedTensorsW4A8Int8MoEMethod"
    )
    weight_quant = SimpleNamespace(num_bits=4)
    input_quant = SimpleNamespace(num_bits=8)
    ilu_params = {"gemm_format": "TN", "pack_version": 2}
    scheme = {
        "weights": weight_quant,
        "input_activations": input_quant,
        "ilu_params": ilu_params,
    }
    quant_config = _quant_config_with_schemes(scheme, scheme, scheme)
    layer = SimpleNamespace(moe_config=object())

    method = ilu_moe.IluCompressedTensorsMoEMethod.get_moe_method(
        quant_config, layer, "model.layers.0.mlp.experts"
    )

    assert isinstance(method, _FakeMoEMethod)
    assert method.weight_quant is weight_quant
    assert method.input_quant is input_quant
    assert method.moe_config is layer.moe_config
    assert method.ilu_params is ilu_params


def test_ilu_params_with_unsupported_scheme_raises_runtime_error():
    weight_quant = SimpleNamespace(num_bits=4)
    input_quant = SimpleNamespace(num_bits=16)
    scheme = {
        "weights": weight_quant,
        "input_activations": input_quant,
        "ilu_params": {"gemm_format": "TN"},
    }
    quant_config = _quant_config_with_schemes(scheme, scheme, scheme)
    layer = SimpleNamespace(moe_config=object())

    with pytest.raises(RuntimeError, match="Unsupported FusedMoe scheme"):
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
    weight_quant = SimpleNamespace(num_bits=8)
    input_quant = SimpleNamespace(num_bits=8)
    ilu_params = {"gemm_format": "TN"}
    scheme = {
        "weights": weight_quant,
        "input_activations": input_quant,
        "ilu_params": ilu_params,
    }
    quant_config = _quant_config_with_schemes(scheme, scheme, scheme)
    quant_config._is_dynamic_token_w8a8.return_value = True
    layer = SimpleNamespace(moe_config=object())

    method = ilu_moe.IluCompressedTensorsMoEMethod.get_moe_method(
        quant_config, layer, "model.layers.0.mlp.experts"
    )

    assert isinstance(method, _FakeMoEMethod)
    assert method.weight_quant is weight_quant
    assert method.input_quant is input_quant
    assert method.ilu_params is ilu_params
