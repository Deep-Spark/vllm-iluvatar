# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar Step-3.7 OOT model wiring.

Covers:
  * OOT registration: ``Step3p7ForConditionalGeneration`` resolves to the plugin
    subclass of the community model.
  * Nested ``tie_word_embeddings`` is filled before community construction.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from vllm_iluvatar.model_executor.models import step3p7 as ilu_step3p7


def test_step3p7_oot_registration_points_at_plugin_subclass():
    from vllm import ModelRegistry
    from vllm.model_executor.models.registry import _VLLM_MODELS

    from vllm_iluvatar.ilu_register import apply_oot_registrations

    assert "Step3p7ForConditionalGeneration" in _VLLM_MODELS

    try:
        community = importlib.import_module("vllm.model_executor.models.step3p7")
    except ImportError as exc:
        raise AssertionError(
            "Failed to import Step3p7 community model dependencies"
        ) from exc
    assert hasattr(community, "Step3p7ForConditionalGeneration")

    apply_oot_registrations()
    entry = ModelRegistry.models["Step3p7ForConditionalGeneration"]
    assert entry.module_name == "vllm_iluvatar.model_executor.models.step3p7"
    assert entry.class_name == "Step3p7ForConditionalGeneration"


def test_plugin_class_subclasses_community():
    from vllm.model_executor.models.step3p7 import (
        Step3p7ForConditionalGeneration as _Community,
    )
    from vllm_iluvatar.model_executor.models.step3p7 import (
        Step3p7ForConditionalGeneration,
    )

    assert issubclass(Step3p7ForConditionalGeneration, _Community)


def test_step3p7_init_fills_text_config_before_base(monkeypatch):
    calls = []
    text_config = SimpleNamespace()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config),
        )
    )

    def fake_base_init(self, *, vllm_config, prefix=""):
        calls.append(("base_init", text_config.tie_word_embeddings, prefix))

    monkeypatch.setattr(
        ilu_step3p7._Step3p7ForConditionalGeneration,
        "__init__",
        fake_base_init,
    )

    ilu_step3p7.Step3p7ForConditionalGeneration(
        vllm_config=vllm_config,
        prefix="lm",
    )

    assert calls == [("base_init", False, "lm")]
