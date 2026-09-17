# SPDX-License-Identifier: Apache-2.0
"""W4A8 (compressed-tensors) MiniMax-M3 architecture registration."""

from __future__ import annotations

import importlib

import vllm

import vllm_iluvatar.ilu_register as ilu_register

PACKAGE = "vllm_iluvatar.models.minimax_m3.w4a8"
EXPECTED_REGISTRATIONS = [
    (
        "IluMiniMaxM3SparseForCausalLM",
        f"{PACKAGE}.model:IluMiniMaxM3SparseForCausalLM",
    ),
    (
        "IluMiniMaxM3SparseForConditionalGeneration",
        f"{PACKAGE}.model:IluMiniMaxM3SparseForConditionalGeneration",
    ),
]


def test_w4a8_package_is_discovered():
    parent = "vllm_iluvatar.models.minimax_m3"
    assert parent in ilu_register._OOT_MODEL_PACKAGES
    assert PACKAGE not in ilu_register._OOT_MODEL_PACKAGES


def test_w4a8_register_model_points_at_ilu_classes(monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: calls.append((arch, path)),
    )
    model = importlib.import_module("vllm_iluvatar.models.minimax_m3.model")
    model.register_model()
    assert calls[: len(EXPECTED_REGISTRATIONS)] == EXPECTED_REGISTRATIONS
    for _, class_path in EXPECTED_REGISTRATIONS:
        module_name, class_name = class_path.split(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        assert cls.__module__ == f"{PACKAGE}.model"


def test_w4a8_exports_ilu_architecture_names():
    module = importlib.import_module(f"{PACKAGE}.model")
    assert hasattr(module, "IluMiniMaxM3SparseForCausalLM")
    assert hasattr(module, "IluMiniMaxM3SparseForConditionalGeneration")
