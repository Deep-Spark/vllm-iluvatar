# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar Qwen3-VL-MoE OOT model wiring."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_iluvatar.model_executor.models import qwen3_vl_moe as ilu_qwen3_vl_moe


def test_qwen3_vl_moe_register_model(monkeypatch):
    calls = []

    import vllm

    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: calls.append((arch, path)),
    )

    ilu_qwen3_vl_moe.register_model()

    assert calls == [
        (
            "Qwen3VLMoeForConditionalGeneration",
            "vllm_iluvatar.model_executor.models.qwen3_vl_moe:"
            "Qwen3VLMoeForConditionalGeneration",
        )
    ]


def test_ensure_qwen3_moe_text_architecture_fills_missing_architectures():
    text_config = SimpleNamespace()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config),
        )
    )

    ilu_qwen3_vl_moe.ensure_qwen3_moe_text_architecture(vllm_config)

    assert text_config.architectures == ["Qwen3MoeForCausalLM"]


def test_ensure_qwen3_moe_text_architecture_preserves_existing_architectures():
    text_config = SimpleNamespace(architectures=["CustomArch"])
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config),
        )
    )

    ilu_qwen3_vl_moe.ensure_qwen3_moe_text_architecture(vllm_config)

    assert text_config.architectures == ["CustomArch"]


def test_qwen3_vl_moe_init_installs_text_model_adaptations(monkeypatch):
    calls = []
    text_model = SimpleNamespace()
    text_config = SimpleNamespace(architectures=None)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config),
        )
    )

    def fake_base_init(self, *, vllm_config, prefix=""):
        calls.append(("base_init", text_config.architectures, prefix))
        self.language_model = SimpleNamespace(model=text_model)

    monkeypatch.setattr(
        ilu_qwen3_vl_moe._Qwen3VLMoeForConditionalGeneration,
        "__init__",
        fake_base_init,
    )
    monkeypatch.setattr(
        ilu_qwen3_vl_moe,
        "qwen3_moe_attention_fusion",
        lambda model: calls.append(("attention", model)),
    )
    monkeypatch.setattr(
        ilu_qwen3_vl_moe,
        "qwen3_moe_router_gate_overrides",
        lambda model: calls.append(("gate", model)),
    )

    ilu_qwen3_vl_moe.Qwen3VLMoeForConditionalGeneration(
        vllm_config=vllm_config,
        prefix="vl",
    )

    assert calls == [
        ("base_init", ["Qwen3MoeForCausalLM"], "vl"),
        ("attention", text_model),
        ("gate", text_model),
    ]


def test_qwen3_vl_moe_init_preserves_existing_text_architectures(monkeypatch):
    calls = []
    text_model = SimpleNamespace()
    text_config = SimpleNamespace(architectures=["CustomArch"])
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config),
        )
    )

    def fake_base_init(self, *, vllm_config, prefix=""):
        calls.append(("base_init", text_config.architectures, prefix))
        self.language_model = SimpleNamespace(model=text_model)

    monkeypatch.setattr(
        ilu_qwen3_vl_moe._Qwen3VLMoeForConditionalGeneration,
        "__init__",
        fake_base_init,
    )
    monkeypatch.setattr(
        ilu_qwen3_vl_moe,
        "qwen3_moe_attention_fusion",
        lambda model: calls.append(("attention", model)),
    )
    monkeypatch.setattr(
        ilu_qwen3_vl_moe,
        "qwen3_moe_router_gate_overrides",
        lambda model: calls.append(("gate", model)),
    )

    ilu_qwen3_vl_moe.Qwen3VLMoeForConditionalGeneration(
        vllm_config=vllm_config,
        prefix="vl",
    )

    assert calls == [
        ("base_init", ["CustomArch"], "vl"),
        ("attention", text_model),
        ("gate", text_model),
    ]
