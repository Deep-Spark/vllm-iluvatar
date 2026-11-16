# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar GPT-OSS OOT model wiring."""

from __future__ import annotations

from vllm_iluvatar.model_executor.models import gpt_oss as ilu_gpt_oss


def test_gpt_oss_register_model(monkeypatch):
    calls = []

    import vllm

    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: calls.append((arch, path)),
    )

    ilu_gpt_oss.register_model()

    assert calls == [
        (
            "GptOssForCausalLM",
            "vllm_iluvatar.model_executor.models.gpt_oss:GptOssForCausalLM",
        )
    ]


def test_gpt_oss_plugin_classes_subclass_community():
    from vllm.model_executor.models.gpt_oss import (
        GptOssForCausalLM as CommunityGptOssForCausalLM,
    )
    from vllm.model_executor.models.gpt_oss import GptOssModel as CommunityGptOssModel

    assert issubclass(ilu_gpt_oss.GptOssForCausalLM, CommunityGptOssForCausalLM)
    assert issubclass(ilu_gpt_oss.GptOssModel, CommunityGptOssModel)


def test_gpt_oss_route_expert_name_targets_routed_experts():
    assert (
        ilu_gpt_oss._route_expert_name("model.layers.0.mlp.experts.w13_weight")
        == "model.layers.0.mlp.experts.routed_experts.w13_weight"
    )
