# SPDX-License-Identifier: Apache-2.0
"""Dispatch policy for native BF16/FP16 checkpoints on ivcore11."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_iluvatar.config import ilu_compute
from vllm_iluvatar.model_executor.layers.quantization.backends.native.linear import (
    IluNativeW8A8Int8LinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.native.moe import (
    IluNativeToW4A8MoEMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.configs import native as mod


def _vllm_config(**overrides):
    config = {
        "model_config": SimpleNamespace(quantization=None, dtype=torch.bfloat16),
        "quant_config": None,
        "additional_config": {},
        "load_config": SimpleNamespace(load_format="auto"),
    }
    config.update(overrides)
    return SimpleNamespace(**config)


@pytest.mark.parametrize(
    "claimed,overrides",
    [
        (True, {}),
        (
            True,
            {"model_config": SimpleNamespace(quantization=None, dtype=torch.float16)},
        ),
        # A checkpoint that declares its own quantization already has a Config.
        (
            False,
            {"model_config": SimpleNamespace(quantization="fp8", dtype=torch.bfloat16)},
        ),
        (False, {"quant_config": object()}),
        (
            False,
            {"model_config": SimpleNamespace(quantization=None, dtype=torch.float32)},
        ),
        # An explicit ilu_quant group owns the dispatch it selected.
        (False, {"additional_config": {"ilu_quant": {"enabled": True}}}),
        # Sharded state is written post-conversion; there is nothing to convert.
        (False, {"load_config": SimpleNamespace(load_format="sharded_state")}),
        (False, {"load_config": SimpleNamespace(load_format="runai_streamer_sharded")}),
        (False, {"model_config": None}),
    ],
)
def test_only_unquantized_half_precision_is_claimed(monkeypatch, claimed, overrides):
    monkeypatch.setattr(
        "vllm_iluvatar.ilu_util.arch.is_ivcore11", lambda: True, raising=False
    )
    config = mod.native_quant_config(_vllm_config(**overrides))
    assert isinstance(config, mod.IluNativeQuantConfig) is claimed


def test_other_architectures_keep_their_own_dispatch(monkeypatch):
    monkeypatch.setattr(
        "vllm_iluvatar.ilu_util.arch.is_ivcore11", lambda: False, raising=False
    )
    assert mod.native_quant_config(_vllm_config()) is None


def _with_compute(monkeypatch, linear=None, moe=None):
    monkeypatch.setattr(
        ilu_compute,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            additional_config={
                "ilu_compute": {"linear": linear or {}, "moe": moe or {}}
            }
        ),
    )


@pytest.mark.parametrize(
    "linear,prefix,dtype,expected",
    [
        (
            {},
            "model.layers.0.mlp.gate_proj",
            torch.bfloat16,
            IluNativeW8A8Int8LinearMethod,
        ),
        (
            {"native_backend": "original"},
            "model.layers.0.mlp.gate_proj",
            torch.bfloat16,
            UnquantizedLinearMethod,
        ),
        # Attention reads these two for grouped BMM instead of calling apply().
        (
            {},
            "model.layers.0.self_attn.kv_b_proj",
            torch.bfloat16,
            UnquantizedLinearMethod,
        ),
        ({}, "model.layers.0.self_attn.wo_a", torch.bfloat16, UnquantizedLinearMethod),
        ({}, "model.layers.0.mlp.gate_proj", torch.float32, UnquantizedLinearMethod),
    ],
)
def test_linear_dispatch(monkeypatch, linear, prefix, dtype, expected):
    _with_compute(monkeypatch, linear=linear)
    method = mod.IluNativeQuantConfig().get_quant_method(
        _layer(mod.LinearBase, dtype), prefix
    )
    assert type(method) is expected


def test_expert_dispatch_converts_by_default_and_opts_out(monkeypatch):
    _with_compute(monkeypatch)
    layer = _layer(mod.RoutedExperts, torch.bfloat16, moe_config=object())
    method = mod.IluNativeQuantConfig().get_quant_method(layer, "experts")
    assert isinstance(method, IluNativeToW4A8MoEMethod)
    assert (method.selection.name, method.selection.layout) == ("ixinfer_w4a8", "NN")

    _with_compute(monkeypatch, moe={"mode": "original"})
    layer = _layer(mod.RoutedExperts, torch.bfloat16, moe_config=object())
    # Returning None hands the layer back to the W16A16 OOT method, which then
    # has to reject a checkpoint that is not actually native.
    assert mod.IluNativeQuantConfig().get_quant_method(layer, "experts") is None
    assert layer._ilu_require_native_expert_weights is True


def test_expert_dispatch_rejects_non_native_weights(monkeypatch):
    _with_compute(monkeypatch)
    layer = _layer(mod.RoutedExperts, torch.float32, moe_config=object())
    with pytest.raises(ValueError, match="requires BF16/FP16"):
        mod.IluNativeQuantConfig().get_quant_method(layer, "experts")


def test_unrelated_layers_stay_unquantized(monkeypatch):
    _with_compute(monkeypatch)
    assert (
        mod.IluNativeQuantConfig().get_quant_method(SimpleNamespace(), "norm") is None
    )


def test_checkpoint_config_is_rejected():
    with pytest.raises(ValueError, match="no checkpoint quantization config"):
        mod.IluNativeQuantConfig.from_config({"quant_method": "fp8"})
    assert isinstance(
        mod.IluNativeQuantConfig.from_config({}), mod.IluNativeQuantConfig
    )


def _layer(cls, params_dtype, **attrs):
    """A bare layer of the right type; dispatch only reads these attributes."""
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.params_dtype = params_dtype
    for name, value in attrs.items():
        setattr(layer, name, value)
    return layer
