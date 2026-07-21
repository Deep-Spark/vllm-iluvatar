# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar Pixtral OOT model wiring."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_iluvatar.ilu_patch import patch_platform_import_hooks
from vllm_iluvatar.model_executor.models import pixtral as ilu_pixtral


def test_pixtral_register_model(monkeypatch):
    calls = []

    import vllm

    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: calls.append((arch, path)),
    )

    ilu_pixtral.register_model()

    assert calls == [
        (
            "PixtralForConditionalGeneration",
            "vllm_iluvatar.model_executor.models.pixtral:"
            "PixtralForConditionalGeneration",
        )
    ]


def test_pixtral_plugin_class_subclasses_community():
    from vllm.model_executor.models.pixtral import (
        PixtralForConditionalGeneration as CommunityPixtralForConditionalGeneration,
    )

    assert issubclass(
        ilu_pixtral.PixtralForConditionalGeneration,
        CommunityPixtralForConditionalGeneration,
    )


def test_pixtral_platform_hook_forces_xformers_on():
    module = SimpleNamespace(USE_XFORMERS_OPS=False)

    patch_platform_import_hooks._patch_pixtral_use_xformers_ops(module)

    assert module.USE_XFORMERS_OPS is True


def test_pixtral_vision_overrides_swap_attention():
    attention = object.__new__(ilu_pixtral._Attention)
    layer = SimpleNamespace(attention=attention)
    transformer = SimpleNamespace(layers=[layer])
    vision_encoder = SimpleNamespace(transformer=transformer)
    model = SimpleNamespace(vision_encoder=vision_encoder)

    ilu_pixtral.pixtral_vision_overrides(model)

    assert attention.__class__ is ilu_pixtral.Attention
