# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar compressed-tensors configuration dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

from vllm_iluvatar.model_executor.layers.quantization import IluvatarQuantRegistry
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    IluQuantConfig,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (  # noqa: E501
    IluCompressedTensorsW4A8Int,
    IluCompressedTensorsW8A8Int8,
)

_TARGET = "model.layers.0.linear"


def _make_config(
    *,
    weight_bits: int = 8,
    iluvatar: dict[str, object],
) -> dict[str, object]:
    return {
        "format": "int-quantized",
        "config_groups": {
            "group_0": {
                "format": "int-quantized",
                "targets": [_TARGET],
                "weights": {
                    "num_bits": weight_bits,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                    "group_size": None,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "token",
                    "group_size": None,
                    "dynamic": True,
                },
                "output_activations": None,
                "iluvatar": iluvatar,
            }
        },
        "ignore": [],
    }


def test_from_config_preserves_iluvatar_params_per_target():
    iluvatar = {
        "gemm_format": "TN",
        "mixed_strategy": "CHANNEL",
        "gemm_mode": "PERF",
    }

    quant_config = IluQuantConfig.from_config(_make_config(iluvatar=iluvatar))

    assert quant_config.target_scheme_map[_TARGET]["ilu_params"] == iluvatar


@pytest.mark.parametrize(
    ("weight_bits", "iluvatar", "expected_type"),
    [
        (
            8,
            {
                "gemm_format": "TN",
                "mixed_strategy": "CHANNEL",
                "gemm_mode": "NORMAL",
            },
            IluCompressedTensorsW8A8Int8,
        ),
        (
            4,
            {"gemm_format": "NN"},
            IluCompressedTensorsW4A8Int,
        ),
    ],
)
def test_get_scheme_selects_iluvatar_linear_implementation(
    weight_bits: int,
    iluvatar: dict[str, object],
    expected_type: type,
):
    quant_config = IluQuantConfig.from_config(
        _make_config(weight_bits=weight_bits, iluvatar=iluvatar)
    )
    layer = MagicMock(spec=LinearBase)

    scheme = quant_config.get_scheme(layer, _TARGET)

    assert isinstance(scheme, expected_type)
    assert scheme.format == iluvatar["gemm_format"]


class _CustomQuantMethod(QuantizeMethodBase):
    def __init__(self, parent_config, ilu_params):
        self.parent_config = parent_config
        self.ilu_params = ilu_params

    def create_weights(self, *args, **kwargs) -> None:
        pass

    def apply(self, *args, **kwargs):
        return None


def test_get_quant_method_dispatches_registered_method(monkeypatch):
    method_name = "test_custom_quant_method"
    monkeypatch.setitem(
        IluvatarQuantRegistry.methods,
        method_name,
        _CustomQuantMethod,
    )
    quant_config = IluQuantConfig.from_config(
        _make_config(
            iluvatar={
                "method": method_name,
                "custom_option": 7,
            }
        )
    )
    layer = MagicMock(spec=LinearBase)

    method = quant_config.get_quant_method(layer, _TARGET)

    assert isinstance(method, _CustomQuantMethod)
    assert method.parent_config is quant_config
    assert method.ilu_params == {"custom_option": 7}


def test_registry_supports_class_and_named_registration():
    registry = type(IluvatarQuantRegistry)()

    assert registry.register_method(_CustomQuantMethod) is _CustomQuantMethod
    assert registry.get_method("_CustomQuantMethod") is _CustomQuantMethod

    assert (
        registry.register_method("custom_name", _CustomQuantMethod)
        is _CustomQuantMethod
    )
    assert registry.get_method("custom_name") is _CustomQuantMethod

    with pytest.raises(TypeError, match="QuantizeMethodBase subclass"):
        registry.register_method("invalid", object)
