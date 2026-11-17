# SPDX-License-Identifier: Apache-2.0
"""Mixed FP8/MXFP4 checkpoint routing; no weights or GPU kernels are needed."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts

from vllm_iluvatar.model_executor.layers.quantization.backends.fp4 import (
    mxfp4_moe as ilu_mxfp4,
)
from vllm_iluvatar.model_executor.layers.quantization.configs import fp8


def _config(**storage):
    return fp8.IluFp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            **storage,
        }
    )


@pytest.mark.parametrize(
    "storage",
    [
        {"expert_dtype": "mxfp4"},
        {"store_dtype": "mxfp4"},
        {"store_dtype": None, "expert_dtype": "mxfp4"},
        {"store_dtype": "mxfp4", "expert_dtype": "mxfp4"},
    ],
)
def test_mixed_checkpoint_recognizes_mxfp4_storage(storage):
    config = _config(**storage)
    assert config.store_dtype == "mxfp4"
    assert config.weight_block_size == [128, 128]
    assert config.is_checkpoint_fp8_serialized


def test_missing_expert_storage_preserves_block_fp8_default():
    assert _config().store_dtype is None


def test_conflicting_storage_fields_fail_before_loading():
    with pytest.raises(ValueError, match="Conflicting store_dtype and expert_dtype"):
        _config(store_dtype="fp8", expert_dtype="mxfp4")


@pytest.mark.parametrize("with_runtime", [False, True])
def test_mxfp4_experts_use_ilu_method_and_keep_runtime_overlay(
    monkeypatch, with_runtime
):
    config = _config(expert_dtype="mxfp4")
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace()
    override = object() if with_runtime else None
    selected = object()
    seen = []

    class Method:
        def __init__(self, moe_config):
            assert moe_config is layer.moe_config

    def select(actual_layer, original, runtime):
        assert actual_layer is layer
        assert isinstance(original, Method)
        assert runtime is override
        return selected

    def resolve(**kwargs):
        seen.append(kwargs)
        return override

    monkeypatch.setattr(ilu_mxfp4, "IluMxfp4MoEMethod", Method)
    monkeypatch.setattr(ilu_mxfp4, "select_mxfp4_method", select)
    monkeypatch.setattr(fp8, "resolve_ilu_runtime_method", resolve)
    prefix = "language_model.model.layers.1.mlp.experts"
    assert config.get_quant_method(layer, prefix) is selected
    assert isinstance(seen[0]["original_method"], Method)
    assert seen[0]["original_disposition"] == "quantized"
    assert seen[0]["prefix"] == prefix
    assert seen[0]["layer"] is layer
