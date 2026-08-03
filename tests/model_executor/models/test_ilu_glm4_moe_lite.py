# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar GLM4-MoE-Lite OOT model wiring."""

from __future__ import annotations

import torch

from vllm_iluvatar.model_executor.models import glm4_moe_lite as ilu_glm4_moe_lite


def test_glm4_moe_lite_register_model(monkeypatch):
    calls = []

    import vllm

    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: calls.append((arch, path)),
    )

    ilu_glm4_moe_lite.register_model()

    assert calls == [
        (
            "Glm4MoeLiteForCausalLM",
            "vllm_iluvatar.model_executor.models.glm4_moe_lite:Glm4MoeLiteForCausalLM",
        )
    ]


def test_glm4_moe_lite_plugin_class_subclasses_community():
    from vllm.model_executor.models.glm4_moe_lite import (
        Glm4MoeLiteForCausalLM as CommunityGlm4MoeLiteForCausalLM,
    )

    assert issubclass(
        ilu_glm4_moe_lite.Glm4MoeLiteForCausalLM,
        CommunityGlm4MoeLiteForCausalLM,
    )


def test_glm4_moe_lite_load_weights_delegates_to_community_loader(monkeypatch):
    calls = []
    loaded = {"model.layers.0.weight"}
    model = object.__new__(ilu_glm4_moe_lite.Glm4MoeLiteForCausalLM)

    def fake_load_weights(self, weights):
        calls.append(("load_weights", self, weights))
        return loaded

    monkeypatch.setattr(
        ilu_glm4_moe_lite._Glm4MoeLiteForCausalLM,
        "load_weights",
        fake_load_weights,
    )
    weights = [("weight", torch.empty(1))]

    assert model.load_weights(weights) is loaded
    assert calls == [("load_weights", model, weights)]


def test_glm4_moe_lite_init_applies_router_override(monkeypatch):
    calls = []

    def fake_init(self, *, vllm_config, prefix=""):
        self.model = "inner-model"
        calls.append(("base_init", vllm_config, prefix))

    monkeypatch.setattr(
        ilu_glm4_moe_lite._Glm4MoeLiteForCausalLM,
        "__init__",
        fake_init,
    )
    monkeypatch.setattr(
        ilu_glm4_moe_lite,
        "glm4_moe_bf16_router_overrides",
        lambda inner_model: calls.append(("router_overrides", inner_model)),
    )

    config = object()
    ilu_glm4_moe_lite.Glm4MoeLiteForCausalLM(vllm_config=config, prefix="model")

    assert calls == [
        ("base_init", config, "model"),
        ("router_overrides", "inner-model"),
    ]
