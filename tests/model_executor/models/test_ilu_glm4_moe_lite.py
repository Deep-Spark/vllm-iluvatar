# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar GLM4-MoE-Lite OOT model wiring."""

from __future__ import annotations

from types import SimpleNamespace

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
    object.__setattr__(model, "model", "inner-model")

    def fake_load_weights(self, weights):
        calls.append(("load_weights", self, weights))
        return loaded

    monkeypatch.setattr(
        ilu_glm4_moe_lite._Glm4MoeLiteForCausalLM,
        "load_weights",
        fake_load_weights,
    )
    monkeypatch.setattr(
        ilu_glm4_moe_lite,
        "glm4_moe_lite_attention_overrides",
        lambda inner_model: calls.append(("overrides", inner_model)),
    )

    weights = [("weight", torch.empty(1))]

    assert model.load_weights(weights) is loaded
    assert calls == [
        ("load_weights", model, weights),
        ("overrides", "inner-model"),
    ]


def test_glm4_moe_lite_optimized_qkv_injection_merges_q_and_kv_weights():
    class FakeAttention:
        def forward_opt(self):
            return "opt"

    q_a_proj = SimpleNamespace(
        weight=torch.tensor([[1.0], [2.0]]),
        weight_scale=torch.tensor([0.5, 0.6]),
    )
    kv_a_proj = SimpleNamespace(
        weight=torch.tensor([[3.0], [4.0]]),
        weight_scale=torch.tensor([0.7, 0.8]),
    )
    layer = SimpleNamespace(
        q_lora_rank=2,
        q_a_proj=q_a_proj,
        kv_a_proj_with_mqa=kv_a_proj,
        mla_attn=FakeAttention(),
    )

    ilu_glm4_moe_lite._inject_optimized_qkv_layer(layer, is_mla=True)

    torch.testing.assert_close(
        layer.q_a_proj.weight,
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )
    torch.testing.assert_close(
        layer.q_a_proj.weight_scale,
        torch.tensor([0.5, 0.6, 0.7, 0.8]),
    )
    assert not hasattr(layer, "kv_a_proj_with_mqa")
    assert layer.mla_attn.forward == layer.mla_attn.forward_opt


def test_glm4_moe_lite_attention_overrides_skips_unsupported_quant_method():
    layer = SimpleNamespace(
        kv_a_proj_with_mqa=SimpleNamespace(quant_method=SimpleNamespace())
    )

    # Unsupported quant methods are intentionally left untouched.
    ilu_glm4_moe_lite._inject_optimized_attention_layer(layer, is_mla=False)

    assert hasattr(layer, "kv_a_proj_with_mqa")
