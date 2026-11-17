# SPDX-License-Identifier: Apache-2.0
"""NVFP4 dispatch, host staging, local-expert coverage and activation contracts."""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

import vllm_iluvatar.model_executor.layers.quantization.backends.fp4.nvfp4_moe as impl
import vllm_iluvatar.model_executor.layers.quantization.configs.mxfp8 as mixed


def method(**kwargs):
    return impl.IluModelOptNvFp4MoEMethod(
        ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16),
        MagicMock(),
        **kwargs,
    )


def weights():
    layer = torch.nn.Module()
    layer.num_experts = 2
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
    monkeypatch.setattr(impl, "_ixmm_backend_available", lambda: True)

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
def test_invalid_global_scales_fail_before_conversion(monkeypatch, bad):
    monkeypatch.setattr(impl, "_ixmm_backend_available", lambda: True)
    layer = weights()
    layer.w13_weight_scale_2.data.fill_(bad)
    with pytest.raises(ValueError, match="global scales"):
        method().process_weights_after_loading(layer)


def test_gate_up_scales_must_match(monkeypatch):
    monkeypatch.setattr(impl, "_ixmm_backend_available", lambda: True)
    layer = weights()
    layer.w13_weight_scale_2.data[:, 1] = 2
    with pytest.raises(ValueError, match="gate/up"):
        method().process_weights_after_loading(layer)


def test_missing_backend_fails_loudly(monkeypatch):
    monkeypatch.setattr(impl, "_ixmm_backend_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires the ixmm"):
        method().process_weights_after_loading(weights())


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


@pytest.mark.parametrize(
    "activation,limit,variant",
    [
        ("silu", 0.0, "step3_swiglustep"),
        ("swiglustep", 9.0, "step3_swiglustep"),
        ("swigluoai_uninterleave", 9.0, "swigluoai"),
    ],
)
def test_activation_dispatch_preserves_scale_and_shared_output(
    monkeypatch, activation, limit, variant
):
    layer = weights()
    layer.activation = activation
    layer.swiglu_limit, layer.swiglu_alpha, layer.swiglu_beta = 9.0, 1.702, 1.0
    layer.expert_map = object()
    layer._ilu_output_scale = 2.0
    residual = object()
    monkeypatch.setattr(impl, "ilu_mxfp4_fused_experts", lambda **kwargs: kwargs)
    got = method().apply(layer, object(), object(), object(), None, residual)
    assert got["swiglu_variant"] == variant and got["swiglu_limit"] == limit
    assert got["extra_residual"] is residual
    assert got["expert_map"] is layer.expert_map and got["output_scale"] == 2.0
    if variant == "swigluoai":
        assert got["swiglu_alpha"] == 1.702 and got["swiglu_beta"] == 1.0


@pytest.mark.parametrize(
    "activation,missing",
    [
        ("swigluoai_uninterleave", "swiglu_limit"),
        ("swigluoai_uninterleave", "swiglu_alpha"),
        ("swigluoai_uninterleave", "swiglu_beta"),
        ("swiglustep", "swiglu_limit"),
    ],
)
def test_activation_parameters_are_not_defaulted(activation, missing):
    layer = weights()
    layer.activation = activation
    layer.swiglu_limit, layer.swiglu_alpha, layer.swiglu_beta = 7.0, 1.702, 1.0
    setattr(layer, missing, None)
    with pytest.raises(ValueError, match="explicit"):
        method().apply(layer, None, None, None, None, None)


def test_unknown_activation_is_rejected():
    layer = weights()
    layer.activation = "gelu"
    with pytest.raises(NotImplementedError, match="activation"):
        method().apply(layer, None, None, None, None, None)


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
