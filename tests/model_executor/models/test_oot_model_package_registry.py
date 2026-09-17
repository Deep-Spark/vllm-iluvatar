# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for migrated multi-file OOT model package registration."""

from __future__ import annotations

import importlib

import pytest

import vllm_iluvatar.ilu_register as ilu_register

_MIGRATED_MODEL_PACKAGES = {
    "vllm_iluvatar.models.deepseek_v32": [
        (
            "DeepseekV32ForCausalLM",
            "vllm_iluvatar.models.deepseek_v32.model:IluDeepseekV32ForCausalLM",
        ),
        (
            "DeepseekV32MTPModel",
            "vllm_iluvatar.models.deepseek_v32.mtp:IluDeepseekV32MTP",
        ),
    ],
    "vllm_iluvatar.models.deepseek_v4": [
        (
            "IluDeepseekV4ForCausalLM",
            "vllm_iluvatar.models.deepseek_v4.model:IluDeepseekV4ForCausalLM",
        ),
        (
            "IluDeepSeekV4MTPModel",
            "vllm_iluvatar.models.deepseek_v4.mtp:IluDeepSeekV4MTP",
        ),
    ],
    "vllm_iluvatar.models.glm_moe_dsa": [
        (
            "IluGlmMoeDsaForCausalLM",
            "vllm_iluvatar.models.glm_moe_dsa.model:IluGlmMoeDsaForCausalLM",
        ),
        (
            "IluGlmMoeDsaMTPModel",
            "vllm_iluvatar.models.glm_moe_dsa.mtp:IluGlmMoeDsaMTP",
        ),
    ],
    "vllm_iluvatar.models.minimax_m3": [
        (
            "IluMiniMaxM3SparseForCausalLM",
            "vllm_iluvatar.models.minimax_m3.w4a8.model:IluMiniMaxM3SparseForCausalLM",
        ),
        (
            "IluMiniMaxM3SparseForConditionalGeneration",
            "vllm_iluvatar.models.minimax_m3.w4a8.model:"
            "IluMiniMaxM3SparseForConditionalGeneration",
        ),
        (
            "MiniMaxM3SparseForCausalLM",
            "vllm_iluvatar.models.minimax_m3.mxfp8.model:MiniMaxM3SparseForCausalLM",
        ),
        (
            "MiniMaxM3SparseForConditionalGeneration",
            "vllm_iluvatar.models.minimax_m3.mxfp8.model:"
            "MiniMaxM3SparseForConditionalGeneration",
        ),
    ],
}


def _register_hook_module(package_name: str) -> str:
    return f"{package_name}.model"


def test_migrated_model_packages_are_explicitly_discovered():
    for package in _MIGRATED_MODEL_PACKAGES:
        assert package in ilu_register._OOT_MODEL_PACKAGES


@pytest.mark.parametrize(
    "package_name",
    list(_MIGRATED_MODEL_PACKAGES),
    ids=list(_MIGRATED_MODEL_PACKAGES),
)
def test_migrated_model_package_exposes_register_model_hook(package_name: str):
    model_module = importlib.import_module(_register_hook_module(package_name))

    assert callable(model_module.register_model)


@pytest.mark.parametrize(
    "package_name, expected_calls",
    list(_MIGRATED_MODEL_PACKAGES.items()),
    ids=list(_MIGRATED_MODEL_PACKAGES),
)
def test_migrated_model_register_model_points_at_plugin_paths(
    monkeypatch,
    package_name: str,
    expected_calls: list[tuple[str, str]],
):
    calls = []

    import vllm

    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: calls.append((arch, path)),
    )

    model_module = importlib.import_module(_register_hook_module(package_name))
    model_module.register_model()

    for call, expected_call in zip(calls, expected_calls):
        assert call == expected_call
