# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar GPTQ quantization registration."""

from __future__ import annotations

import importlib
import sys
import types


def _install_fake_ixformer(monkeypatch) -> None:
    ixformer = types.ModuleType("ixformer")
    core = types.ModuleType("ixformer.core")
    config = types.SimpleNamespace(IXFORMER_USE_TORCH_OPS=False)
    inference = types.ModuleType("ixformer.inference")
    functions = types.ModuleType("ixformer.inference.functions")
    core.config = config
    ixformer.core = core
    ixformer.inference = inference
    inference.functions = functions
    monkeypatch.setitem(sys.modules, "ixformer", ixformer)
    monkeypatch.setitem(sys.modules, "ixformer.core", core)
    monkeypatch.setitem(sys.modules, "ixformer.core.config", config)
    monkeypatch.setitem(sys.modules, "ixformer.inference", inference)
    monkeypatch.setitem(sys.modules, "ixformer.inference.functions", functions)


def _load_ilu_auto_gptq(monkeypatch):
    _install_fake_ixformer(monkeypatch)
    return importlib.import_module(
        "vllm_iluvatar.model_executor.layers.quantization.ilu_auto_gptq"
    )


def test_ilu_auto_gptq_registers_gptq_entry_names(monkeypatch):
    mod = _load_ilu_auto_gptq(monkeypatch)

    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
    from vllm.model_executor.layers.quantization import (
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG,
    )

    assert issubclass(mod.IluAutoGPTQConfig, AutoGPTQConfig)
    assert _CUSTOMIZED_METHOD_TO_QUANT_CONFIG["gptq"] is mod.IluAutoGPTQConfig
    assert _CUSTOMIZED_METHOD_TO_QUANT_CONFIG["auto_gptq"] is mod.IluAutoGPTQConfig
    assert _CUSTOMIZED_METHOD_TO_QUANT_CONFIG["gptq_marlin"] is mod.IluAutoGPTQConfig


def test_ilu_auto_gptq_inherits_from_config(monkeypatch):
    mod = _load_ilu_auto_gptq(monkeypatch)

    config = mod.IluAutoGPTQConfig.from_config(
        {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
        }
    )

    assert isinstance(config, mod.IluAutoGPTQConfig)
    assert config.weight_bits == 4
    assert config.group_size == 128


def test_ilu_auto_gptq_override_selects_auto_gptq(monkeypatch):
    mod = _load_ilu_auto_gptq(monkeypatch)

    selected = mod.IluAutoGPTQConfig.override_quantization_method(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
        },
        user_quant=None,
    )

    assert selected == "auto_gptq"


def test_ilu_auto_gptq_imports_plugin_linear_selector(monkeypatch):
    mod = _load_ilu_auto_gptq(monkeypatch)
    selector = importlib.import_module("vllm_iluvatar.model_executor.kernels.linear")

    assert mod.choose_mp_linear_kernel is selector.choose_mp_linear_kernel
    assert mod.MPLinearLayerConfig is selector.MPLinearLayerConfig


def test_ilu_auto_gptq_get_quant_method_uses_ilu_linear_method(monkeypatch):
    mod = _load_ilu_auto_gptq(monkeypatch)

    class DummyRoutedExperts:
        pass

    selected = types.SimpleNamespace(input_dtype=None)

    def fake_get_linear_quant_method(config, layer, prefix, method_cls):
        assert isinstance(config, mod.IluAutoGPTQConfig)
        assert method_cls is mod.IluAutoGPTQLinearMethod
        return selected

    monkeypatch.setattr(mod, "RoutedExperts", DummyRoutedExperts)
    monkeypatch.setattr(mod, "get_linear_quant_method", fake_get_linear_quant_method)
    monkeypatch.setattr(mod, "get_marlin_input_dtype", lambda prefix: "bfloat16")

    config = mod.IluAutoGPTQConfig.from_config(
        {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
        }
    )

    quant_method = config.get_quant_method(object(), "model.layers.0.mlp.down_proj")

    assert quant_method is selected
    assert quant_method.input_dtype == "bfloat16"
