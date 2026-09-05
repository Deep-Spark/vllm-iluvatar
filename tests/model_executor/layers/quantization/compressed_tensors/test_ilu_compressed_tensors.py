# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar compressed-tensors configuration dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

from vllm_iluvatar.model_executor.layers.quantization import IluvatarQuantRegistry
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    IluQuantConfig,
    _convert_legacy_ilu_deepseek_v4_w4a8,
    _is_legacy_ilu_deepseek_v4_w4a8,
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


def test_w4a8_weight_layout_packed_nn_or_unset():
    packed = IluCompressedTensorsW4A8Int(
        strategy="channel",
        num_bits=4,
        group_size=-1,
        ilu_params={"gemm_format": "NN", "weight_layout": "packed_nn"},
    )
    assert packed.weight_layout == "packed_nn"

    omitted = IluCompressedTensorsW4A8Int(
        strategy="channel",
        num_bits=4,
        group_size=-1,
        ilu_params={"gemm_format": "NN"},
    )
    assert omitted.weight_layout is None

    other = IluCompressedTensorsW4A8Int(
        strategy="channel",
        num_bits=4,
        group_size=-1,
        ilu_params={"gemm_format": "NN", "weight_layout": "future_layout"},
    )
    assert other.weight_layout is None


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


_DEEPSEEK_LEGACY = {
    "quant_method": "ilu_deepseek_v4_w4a8",
    "format": "ilu-mixed-w4a8",
    "version": 2,
    "routed_experts": {
        "targets": ["re:.*\\.experts\\.\\d+\\.w[123]$"],
        "num_bits": 4,
        "strategy": "channel",
        "group_size": None,
        "symmetric": True,
        "type": "int",
        "pack_factor": 2,
        "gemm_format": "TN",
        "pack_version": 2,
        "input_activations": {
            "num_bits": 8,
            "strategy": "token",
            "dynamic": True,
            "symmetric": True,
            "type": "int",
        },
    },
    "shared_experts": {
        "targets": ["re:.*\\.shared_experts\\.w[123]$"],
        "num_bits": 8,
        "strategy": "channel",
        "group_size": None,
        "symmetric": True,
        "type": "int",
        "input_activations": {
            "num_bits": 8,
            "strategy": "token",
            "dynamic": True,
            "symmetric": True,
            "type": "int",
        },
    },
    "dense_linears": {
        "targets": [
            "re:.*\\.attn\\.(fused_wqa_wkv|wq_b|wo_b)$",
            "re:.*\\.attn\\.indexer\\.wq_b$",
            "re:.*\\.e_proj$",
            "re:.*\\.h_proj$",
        ],
        "num_bits": 8,
        "strategy": "channel",
        "group_size": None,
        "symmetric": True,
        "type": "int",
        "input_activations": {
            "num_bits": 8,
            "strategy": "token",
            "dynamic": True,
            "symmetric": True,
            "type": "int",
        },
    },
    "bmm_w8a16_layers": {
        "targets": ["re:.*\\.attn\\.wo_a$"],
        "num_bits": 8,
        "strategy": "channel",
        "group_size": None,
        "symmetric": True,
        "type": "int",
        "input_activations": None,
        "gemm_format": "k16n16",
        "input_dtype": "bfloat16",
    },
    "bf16_layers": ["re:.*\\.attn\\.compressor\\..*"],
    "ignore": ["embed", "head", "re:.*\\.norm$"],
}

_GLM_LEGACY = {
    "quant_method": "ilu_deepseek_v4_w4a8",
    "format": "ilu-mixed-w4a8",
    "version": 2,
    "routed_experts": {
        "targets": ["re:.*\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$"],
        "num_bits": 4,
        "strategy": "channel",
        "group_size": None,
        "symmetric": True,
        "type": "int",
        "gemm_format": "TN",
        "pack_version": 2,
        "input_activations": {
            "num_bits": 8,
            "strategy": "token",
            "dynamic": True,
            "symmetric": True,
            "type": "int",
        },
    },
    "shared_experts": {
        "targets": ["re:.*\\.shared_experts\\.(gate_proj|up_proj|down_proj)$"],
        "num_bits": 8,
        "strategy": "channel",
        "group_size": None,
        "symmetric": True,
        "type": "int",
        "input_activations": {
            "num_bits": 8,
            "strategy": "token",
            "dynamic": True,
            "symmetric": True,
            "type": "int",
        },
    },
    "dense_linears": {
        "targets": [
            "re:.*\\.self_attn\\.(fused_qkv_a_proj|q_b_proj|o_proj)$",
            "re:.*\\.mlp\\.(gate_up_proj|down_proj)$",
        ],
        "num_bits": 8,
        "strategy": "channel",
        "group_size": None,
        "symmetric": True,
        "type": "int",
        "input_activations": {
            "num_bits": 8,
            "strategy": "token",
            "dynamic": True,
            "symmetric": True,
            "type": "int",
        },
    },
    "mla_bmm_w8a16": {
        "targets": ["re:.*\\.self_attn\\.kv_b_proj$"],
        "num_bits": 8,
        "strategy": "channel",
        "group_size": None,
        "symmetric": True,
        "type": "int",
        "input_activations": None,
    },
    "bf16_layers": ["re:.*\\.self_attn\\.indexer\\.wk$"],
    "ignore": ["embed", "head"],
}


def test_legacy_name_registers_ilu_quant_config():
    assert get_quantization_config("ilu_deepseek_v4_w4a8") is IluQuantConfig
    assert get_quantization_config("compressed-tensors") is IluQuantConfig


def test_convert_legacy_deepseek_schema():
    assert _is_legacy_ilu_deepseek_v4_w4a8(_DEEPSEEK_LEGACY)
    converted = _convert_legacy_ilu_deepseek_v4_w4a8(_DEEPSEEK_LEGACY)

    assert converted["quant_method"] == "compressed-tensors"
    assert converted["format"] == "iluvatar-mixed"
    groups = converted["config_groups"]
    assert groups["routed_experts"]["targets"] == ["re:.*\\.experts"]
    assert groups["routed_experts"]["iluvatar"] == {
        "gemm_format": "TN",
        "pack_version": 2,
    }
    assert groups["shared_experts"]["format"] == "int-quantized"
    assert groups["dense_linears"]["iluvatar"]["mixed_strategy"] == "CHANNEL"
    assert groups["wo_a_w8a16"]["iluvatar"] == {
        "method": "IluV4W8A16BMMLinearMethod",
        "bmm_w8a16_gemm_format": "k16n16",
    }
    assert "re:.*\\.attn\\.compressor\\..*" in converted["ignore"]
    assert "kv_b_w8a16" not in groups


def test_convert_legacy_glm_schema():
    converted = _convert_legacy_ilu_deepseek_v4_w4a8(_GLM_LEGACY)
    groups = converted["config_groups"]
    assert "wo_a_w8a16" not in groups
    assert groups["kv_b_w8a16"]["iluvatar"]["method"] == "IluGlmKVBProjW8A16Method"
    assert "re:.*\\.self_attn\\.indexer\\.wk$" in converted["ignore"]


def test_from_config_converts_legacy_deepseek_and_selects_schemes():
    quant_config = IluQuantConfig.from_config(_DEEPSEEK_LEGACY)
    layer = MagicMock(spec=LinearBase)

    dense_scheme = quant_config.get_scheme(layer, "model.layers.0.attn.wq_b")
    assert isinstance(dense_scheme, IluCompressedTensorsW8A8Int8)
    assert dense_scheme.format == "TN"

    wo_a_dict = quant_config.get_scheme_dict(layer, "model.layers.0.attn.wo_a")
    assert wo_a_dict["ilu_params"]["method"] == "IluV4W8A16BMMLinearMethod"
    assert wo_a_dict["ilu_params"]["bmm_w8a16_gemm_format"] == "k16n16"

    routed_dict = quant_config.get_scheme_dict(layer, "model.layers.0.ffn.experts")
    assert routed_dict["weights"].num_bits == 4
    assert routed_dict["ilu_params"]["pack_version"] == 2


def test_from_config_converts_legacy_glm_kv_b():
    quant_config = IluQuantConfig.from_config(_GLM_LEGACY)
    layer = MagicMock(spec=LinearBase)
    scheme_dict = quant_config.get_scheme_dict(
        layer, "model.layers.3.self_attn.kv_b_proj"
    )
    assert scheme_dict["ilu_params"]["method"] == "IluGlmKVBProjW8A16Method"


def test_native_compressed_tensors_config_is_not_treated_as_legacy():
    native = _make_config(
        iluvatar={
            "gemm_format": "TN",
            "mixed_strategy": "CHANNEL",
            "gemm_mode": "PERF",
        }
    )
    assert not _is_legacy_ilu_deepseek_v4_w4a8(native)
