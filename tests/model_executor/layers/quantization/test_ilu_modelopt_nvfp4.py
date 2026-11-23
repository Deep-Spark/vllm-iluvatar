# SPDX-License-Identifier: Apache-2.0
"""NVFP4 dispatch, host staging, local-expert coverage and activation contracts."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

import vllm_iluvatar.model_executor.layers.quantization.backends.fp4.nvfp4_moe as impl
import vllm_iluvatar.model_executor.layers.quantization.configs.mxfp8 as mixed
from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
    MoEComputeSelection,
)

# The conversion tests pin the ix30 target, which is selected on ivcore30; the
# method reaches it from any host because selection is an explicit field.
IX30 = MoEComputeSelection("ixmm_fp4fp8", "fp4fp8", "ivcore30")


def method(selection=IX30, **kwargs):
    m = impl.IluModelOptNvFp4MoEMethod(
        ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16),
        MagicMock(),
        **kwargs,
    )
    m.selection = selection
    return m


def weights():
    layer = torch.nn.Module()
    layer.num_experts = 2
    # What ``_compute_config`` reads: hidden 256, intermediate 256, 2 experts.
    layer.moe_config = SimpleNamespace(
        hidden_dim=256, intermediate_size_per_partition=256, ep_rank=0
    )
    layer.local_num_experts = layer.global_num_experts = 2
    layer.activation = "silu"
    layer.apply_router_weight_on_input = False
    for name, shape, dtype in (
        ("w13_weight", (2, 512, 128), torch.uint8),
        ("w2_weight", (2, 256, 128), torch.uint8),
        ("w13_weight_scale", (2, 512, 16), torch.float8_e4m3fn),
        ("w2_weight_scale", (2, 256, 16), torch.float8_e4m3fn),
        ("w13_weight_scale_2", (2, 2), torch.float32),
        ("w2_weight_scale_2", (2,), torch.float32),
        ("w13_input_scale", (2, 2), torch.float32),
        ("w2_input_scale", (2,), torch.float32),
    ):
        layer.register_parameter(
            name,
            torch.nn.Parameter(
                torch.ones(shape).to(dtype),
                requires_grad=False,
            ),
        )
    return layer


@pytest.mark.parametrize("stage_on_cpu", [False, True])
def test_conversion_is_bounded_and_idempotent(monkeypatch, stage_on_cpu):
    calls = []

    def convert(w, s, g):
        calls.append(w.shape[0])
        e, n, k = w.shape
        return torch.full((e, k, n), 7, dtype=torch.uint8), torch.ones(
            e, k * 2 // 256, n
        )

    monkeypatch.setattr(impl, "preprocess_nvfp4_weight_for_ixmm", convert)
    layer = weights()
    m = method(stage_on_cpu=stage_on_cpu, expert_chunk_size=1)
    m.process_weights_after_loading(layer)
    assert calls == [1, 1, 1, 1]
    assert layer.w13_weight.shape == (2, 128, 512)
    assert layer.w2_weight_scale.shape == (2, 1, 256)
    assert torch.all(layer.w13_weight == 7)
    assert not hasattr(layer, "w13_weight_scale_2")
    assert ("w13_weight" in layer._buffers) is stage_on_cpu
    m.process_weights_after_loading(layer)
    assert len(calls) == 4


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_global_scales_fail_before_conversion(bad):
    layer = weights()
    layer.w13_weight_scale_2.data.fill_(bad)
    with pytest.raises(ValueError, match="global scales"):
        method().process_weights_after_loading(layer)


def test_gate_up_scales_must_match():
    layer = weights()
    layer.w13_weight_scale_2.data[:, 1] = 2
    with pytest.raises(ValueError, match="gate/up"):
        method().process_weights_after_loading(layer)


def test_an_activation_the_backend_cannot_run_fails_before_conversion():
    """The conversion target also decides which activations are runnable."""
    layer = weights()
    layer.activation = "gelu"
    with pytest.raises(ValueError, match="activation gelu"):
        method().process_weights_after_loading(layer)


def test_loader_tracks_only_successful_local_experts(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    m = method(stage_on_cpu=True, validate_checkpoint=True)
    m.moe.is_act_and_mul = True
    layer = torch.nn.Module()
    seen = []

    def loader(param, weight, name, *, shard_id, expert_id, return_success):
        seen.append((shard_id, expert_id, return_success))
        return expert_id in (5, 9)

    m.create_weights(layer, 2, 256, 256, torch.bfloat16, weight_loader=loader)
    assert layer.w13_weight.device.type == "cpu"
    checked = layer.w13_weight.weight_loader
    for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
        for shard in ("w1", "w2", "w3"):
            for expert in (5, 9, 10):
                assert checked(
                    layer.w13_weight,
                    None,
                    f"experts.{'w2' if shard == 'w2' else 'w13'}_{kind}",
                    shard,
                    expert,
                    return_success=True,
                ) is (expert != 10)
    m._validate_loaded_experts(layer)
    assert all(row[2] for row in seen)
    # A correct total count is insufficient if the wrong expert was loaded.
    layer._ilu_nvfp4_loaded[("w13_input_scale", "w3")] = {5, 10}
    with pytest.raises(ValueError, match="Incomplete"):
        m._validate_loaded_experts(layer)
    del layer._ilu_nvfp4_loaded[("w13_input_scale", "w3")]
    with pytest.raises(ValueError, match="Incomplete"):
        m._validate_loaded_experts(layer)


def test_swigluoai_needs_an_explicit_clamp():
    """It has no unclamped form, and ``requires_activation_limit`` is off."""
    layer = weights()
    layer.activation = "swigluoai_uninterleave"
    layer.swiglu_alpha, layer.swiglu_beta = 1.702, 1.0
    with pytest.raises(ValueError, match="clamp limit"):
        method().process_weights_after_loading(layer)


def test_mixed_config_dispatch_and_exclusions(monkeypatch):
    class Experts(torch.nn.Module):
        moe_config = MagicMock()

    monkeypatch.setattr(mixed, "RoutedExperts", Experts)
    fallback = object()
    monkeypatch.setattr(
        mixed.ModelOptMxFp8Config, "get_quant_method", lambda *args: fallback
    )
    cfg = mixed.IluMxFp8Config.from_config(
        dict(
            quant_method="mxfp8",
            moe_quant_algo="NVFP4",
            moe_quant_format="nvfp4-pack-quantized",
            group_size=16,
            ignored_layers=["ignored"],
        )
    )
    # One method on both architectures; the conversion target is its own
    # decision, so the config does not branch on the arch any more.
    selected = cfg.get_quant_method(Experts(), "experts")
    assert type(selected) is impl.IluModelOptNvFp4MoEMethod
    assert selected.stage_on_cpu and selected.validate_checkpoint
    assert cfg.get_quant_method(Experts(), "ignored") is fallback
    assert cfg.get_quant_method(torch.nn.Linear(2, 2), "dense") is fallback
    normal = mixed.IluMxFp8Config.from_config({"quant_method": "mxfp8"})
    assert normal.get_quant_method(Experts(), "experts") is fallback


@pytest.mark.parametrize(
    "field,value",
    [
        ("group_size", 32),
        ("moe_quant_format", "unknown"),
        ("moe_quant_algo", "NVFP4_typo"),
        ("moe_quant_algo", ""),
        ("moe_quant_algo", None),
    ],
)
def test_mixed_format_is_validated(field, value):
    cfg = dict(
        quant_method="mxfp8",
        moe_quant_algo="NVFP4",
        moe_quant_format="nvfp4-pack-quantized",
        group_size=16,
    )
    cfg[field] = value
    with pytest.raises(ValueError, match="checkpoint format|moe_quant_algo"):
        mixed.IluMxFp8Config.from_config(cfg)
