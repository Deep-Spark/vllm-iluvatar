# SPDX-License-Identifier: Apache-2.0
"""mxfp8 MiniMax-M3 architecture registration."""

from __future__ import annotations

import importlib

import vllm

import vllm_iluvatar.ilu_register as ilu_register

PACKAGE = "vllm_iluvatar.models.minimax_m3.mxfp8"
EXPECTED_REGISTRATIONS = [
    (
        "MiniMaxM3SparseForCausalLM",
        f"{PACKAGE}.model:MiniMaxM3SparseForCausalLM",
    ),
    (
        "MiniMaxM3SparseForConditionalGeneration",
        f"{PACKAGE}.model:MiniMaxM3SparseForConditionalGeneration",
    ),
]


def test_mxfp8_package_is_discovered():
    parent = "vllm_iluvatar.models.minimax_m3"
    assert parent in ilu_register._OOT_MODEL_PACKAGES
    assert PACKAGE not in ilu_register._OOT_MODEL_PACKAGES


def test_mxfp8_register_model_points_at_minimax_classes(monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: calls.append((arch, path)),
    )
    model = importlib.import_module("vllm_iluvatar.models.minimax_m3.model")
    model.register_model()
    assert all(call in calls for call in EXPECTED_REGISTRATIONS)
    for _, class_path in EXPECTED_REGISTRATIONS:
        module_name, class_name = class_path.split(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        assert cls.__module__ == f"{PACKAGE}.model"


def test_mxfp8_exports_minimax_architecture_names():
    module = importlib.import_module(f"{PACKAGE}.model")
    assert hasattr(module, "MiniMaxM3SparseForCausalLM")
    assert hasattr(module, "MiniMaxM3SparseForConditionalGeneration")
