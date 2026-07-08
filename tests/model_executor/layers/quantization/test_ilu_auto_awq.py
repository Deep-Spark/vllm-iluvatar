# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar AWQ quantization registration."""

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


def _load_ilu_auto_awq(monkeypatch):
    _install_fake_ixformer(monkeypatch)
    return importlib.import_module(
        "vllm_iluvatar.model_executor.layers.quantization.ilu_auto_awq"
    )


def test_ilu_auto_awq_registers_awq_entry_names(monkeypatch):
    mod = _load_ilu_auto_awq(monkeypatch)

    from vllm.model_executor.layers.quantization import (
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG,
    )

    assert _CUSTOMIZED_METHOD_TO_QUANT_CONFIG["awq"] is mod.IluAutoAWQConfig
    assert _CUSTOMIZED_METHOD_TO_QUANT_CONFIG["auto_awq"] is mod.IluAutoAWQConfig
    assert _CUSTOMIZED_METHOD_TO_QUANT_CONFIG["awq_marlin"] is mod.IluAutoAWQConfig


def test_ilu_auto_awq_override_selects_awq_marlin(monkeypatch):
    mod = _load_ilu_auto_awq(monkeypatch)

    selected = mod.IluAutoAWQConfig.override_quantization_method(
        {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
        },
        user_quant="awq_marlin",
    )

    assert selected == "auto_awq"


def test_ilu_auto_awq_overrides_plain_awq_by_default(monkeypatch):
    mod = _load_ilu_auto_awq(monkeypatch)

    selected = mod.IluAutoAWQConfig.override_quantization_method(
        {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
        },
        user_quant=None,
    )

    assert selected == "auto_awq"
