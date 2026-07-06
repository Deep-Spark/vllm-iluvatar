# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar layer utility overrides."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import torch
import vllm.model_executor.layers.utils as community

from vllm_iluvatar.ilu_patch.general_patches import patch_layer_utils


def _install_fake_ixformer(monkeypatch):
    calls: list[tuple[str, object, object, object | None]] = []

    def fake_mixed_type_linear(*, input, weight):
        calls.append(("mixed_type_linear", input, weight, None))
        return "mixed"

    def fake_linear(input, weight, bias=None):
        calls.append(("linear", input, weight, bias))
        return "linear"

    ixformer = ModuleType("ixformer")
    inference = ModuleType("ixformer.inference")
    functions = ModuleType("ixformer.inference.functions")
    functions.mixed_type_linear = fake_mixed_type_linear
    functions.linear = fake_linear
    inference.functions = functions
    ixformer.inference = inference

    monkeypatch.setitem(sys.modules, "ixformer", ixformer)
    monkeypatch.setitem(sys.modules, "ixformer.inference", inference)
    monkeypatch.setitem(sys.modules, "ixformer.inference.functions", functions)
    return calls


def _load_impl(monkeypatch):
    calls = _install_fake_ixformer(monkeypatch)
    module_name = "vllm_iluvatar.model_executor.layers.utils"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name), calls


def test_layer_utils_patch_rebinds_default_unquantized_gemm(monkeypatch):
    original = community.default_unquantized_gemm
    fake_consumer = ModuleType("vllm.fake_layer_utils_consumer")
    fake_consumer.default_unquantized_gemm = original
    monkeypatch.setitem(sys.modules, fake_consumer.__name__, fake_consumer)

    impl, _ = _load_impl(monkeypatch)
    patch_layer_utils._apply()

    assert community.default_unquantized_gemm is impl.default_unquantized_gemm
    assert fake_consumer.default_unquantized_gemm is impl.default_unquantized_gemm


def test_default_unquantized_gemm_dispatches_to_ixformer(monkeypatch):
    impl, calls = _load_impl(monkeypatch)

    x = SimpleNamespace(dtype=torch.bfloat16)
    weight = SimpleNamespace(dtype=torch.float32)
    layer = SimpleNamespace(weight=weight)

    assert impl.default_unquantized_gemm(layer, x, weight, bias=None) == "mixed"
    assert calls == [("mixed_type_linear", x, weight, None)]

    bias = object()
    x = SimpleNamespace(dtype=torch.bfloat16)
    weight = SimpleNamespace(dtype=torch.bfloat16)

    assert impl.default_unquantized_gemm(layer, x, weight, bias=bias) == "linear"
    assert calls[-1] == ("linear", x, weight, bias)
