# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""routed_scaling / shared-expert fusion gates for Flash MoE."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_iluvatar.backends.ixaite.models.minimax_m3.flash import (
    _MIXED_FORMAT_FUSION_REASON,
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)
from vllm_iluvatar.backends.ixaite.quantization.minimax_m3_flash import (
    METHOD_NAME,
    IluNvfp4ToW4A8MoEMethod,
    IxaiteMiniMaxM3FlashConfig,
)


def test_flash_quant_disables_shared_expert_fusion():
    quant = IxaiteMiniMaxM3FlashConfig()
    assert quant.get_name() == METHOD_NAME

    class _Text:
        n_shared_experts = 1

    class _Vl:
        text_config = _Text()

    for cls, cfg in (
        (MiniMaxM3SparseForCausalLM, _Text()),
        (MiniMaxM3SparseForConditionalGeneration, _Vl()),
    ):
        assert (
            cls.shared_experts_fusion_disable_reason(cfg, quant)
            == _MIXED_FORMAT_FUSION_REASON
        )
    assert (
        MiniMaxM3SparseForCausalLM.shared_experts_fusion_disable_reason(object(), quant)
        == "No shared experts are defined in the config."
    )
    assert (
        MiniMaxM3SparseForCausalLM.shared_experts_fusion_disable_reason(
            _Text(), SimpleNamespace(get_name=lambda: "compressed-tensors")
        )
        is None
    )


def test_nvfp4_moe_has_no_shared_expert_int4_slot():
    method = IluNvfp4ToW4A8MoEMethod(SimpleNamespace())
    layer = torch.nn.Module()
    layer.num_fused_shared_experts = 1
    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=64,
        intermediate_size_per_partition=32,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    assert not hasattr(layer, "_ilu_shared_mxfp8")
    assert not hasattr(layer, "w13_weight_scale_inv")
    assert not hasattr(layer, "w2_weight_scale_inv")
    assert hasattr(layer, "w13_weight_packed")


def test_moe_construct_logs_routed_scaling(monkeypatch):
    import vllm_iluvatar.backends.ixaite.models.minimax_m3.flash as flash_mod

    monkeypatch.setattr(flash_mod, "get_tensor_model_parallel_world_size", lambda: 1)

    class _Gate:
        out_dtype = torch.float32

        def __init__(self, *a, **k):
            pass

    captured: dict = {}
    logs: list[str] = []

    class _Factory:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(flash_mod, "GateLinear", _Gate)
    monkeypatch.setattr(flash_mod, "FusedMoEFactory", _Factory)
    monkeypatch.setattr(
        flash_mod,
        "MiniMaxM3MLP",
        lambda **k: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        flash_mod.logger,
        "info_once",
        lambda msg, *args, **kwargs: logs.append(msg % args if args else msg),
    )

    config = SimpleNamespace(
        num_local_experts=128,
        num_experts_per_tok=4,
        hidden_size=64,
        intermediate_size=32,
        scoring_func="sigmoid",
        use_routing_bias=True,
        routed_scaling_factor=2.0,
        n_shared_experts=1,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        rms_norm_eps=1e-6,
    )
    moe = flash_mod.MiniMaxM3MoE(config, layer_id=0, quant_config=None, prefix="l0")
    assert moe.routed_scaling_factor == 2.0
    assert moe.shared_experts is not None
    assert captured["routed_scaling_factor"] == 2.0
    assert captured["num_experts"] == 128
    assert captured["top_k"] == 4
    assert captured["shared_experts"] is moe.shared_experts
    assert logs and "backend=ixaite" in logs[0]
    assert "experts=128" in logs[0] and "k=4" in logs[0]
    assert "scaling=2.0" in logs[0]
