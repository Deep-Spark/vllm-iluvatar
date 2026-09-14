# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter as parameter_module
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.parameter import ModelWeightParameter

from vllm_iluvatar.model_executor.layers.quantization import IluvatarQuantRegistry
from vllm_iluvatar.model_executor.layers.quantization.ilu_fp8 import (
    IluFp8Config,
    IluFp8PerChannelOnlineLinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization import ilu_fp8
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime_quant_config import (
    parse_ilu_runtime_quant_config,
)

PREFIX = "model.layers.0.mlp.shared_expert.gate_proj"


@pytest.fixture(autouse=True)
def _single_rank_parameter_state(monkeypatch):
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(ilu_fp8, "initialize_online_processing", lambda layer: None)


def _payload(
    *,
    targets: list[str] | None = None,
    selected: int = 1,
    transformed: int = 1,
    fallback: int = 0,
) -> dict:
    return {
        "schema_version": 1,
        "enabled": True,
        "applies_to": {
            "model_types": ["qwen3_5_moe"],
            "quant_methods": ["fp8"],
        },
        "ignore": [],
        "config_groups": {
            "online_fp8": {
                "priority": 100,
                "targets": targets or [PREFIX],
                "require": {
                    "original_disposition": "ignored",
                    "checkpoint_dtypes": ["bfloat16"],
                },
                "weights": {
                    "source": {
                        "format": "native",
                        "dtypes": ["bfloat16"],
                    },
                    "target": {
                        "format": "fp8",
                        "dtype": "float8_e4m3fn",
                        "strategy": "channel",
                    },
                    "loading": {
                        "mode": "transform",
                        "stage": "process_weights_after_loading",
                    },
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "token",
                    "dynamic": True,
                },
                "output_activations": None,
                "iluvatar": {
                    "method": "IluFp8PerChannelOnlineLinearMethod",
                    "gemm_format": "TN",
                    "shape_mismatch_policy": "fallback_original",
                },
                "validation": {
                    "expected_selected": selected,
                    "expected_transformed": transformed,
                    "expected_fallback": fallback,
                },
            }
        },
    }


def _parse(payload: dict):
    return parse_ilu_runtime_quant_config(
        {"source": "inline", **payload},
        model_type="qwen3_5_moe",
        quant_method="fp8",
    )


def _linear() -> LinearBase:
    layer = object.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.prefix = PREFIX
    return layer


def _vllm_config(payload: dict) -> VllmConfig:
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="qwen3_5_moe"),
        dtype=torch.bfloat16,
    )
    config.additional_config = {"ilu_quant": {"source": "inline", **payload}}
    return config


def _fp8_config() -> IluFp8Config:
    return IluFp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=[PREFIX],
        weight_block_size=[128, 128],
        none_moe_use_weight_block_size=True,
        none_moe_is_checkpoint_fp8_serialized=True,
    )


def test_inline_payload_is_strict_and_immutable():
    config = _parse(_payload())

    assert config.source == "inline"
    assert config.config_groups[0].weights.loading.mode == "transform"
    with pytest.raises(TypeError):
        config.config_groups[0].options["gemm_format"] = "NN"

    bad = _payload()
    bad["typo"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        _parse(bad)


def test_file_payload_requires_matching_sha256(tmp_path):
    path = tmp_path / "runtime-quant.json"
    raw = json.dumps(_payload(), sort_keys=True).encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()

    config = parse_ilu_runtime_quant_config(
        {
            "source": "file",
            "file": str(path),
            "sha256": f"sha256:{digest}",
        },
        model_type="qwen3_5_moe",
        quant_method="fp8",
    )
    assert config.source == "file"
    assert config.canonical_digest == _parse(_payload()).canonical_digest

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        parse_ilu_runtime_quant_config(
            {
                "source": "file",
                "file": str(path),
                "sha256": f"sha256:{'0' * 64}",
            },
            model_type="qwen3_5_moe",
            quant_method="fp8",
        )


def test_file_payload_rejects_relative_invalid_and_oversized_files(tmp_path):
    with pytest.raises(ValueError, match="must be absolute"):
        parse_ilu_runtime_quant_config(
            {
                "source": "file",
                "file": "relative.json",
                "sha256": f"sha256:{'0' * 64}",
            },
            model_type="qwen3_5_moe",
            quant_method="fp8",
        )

    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff")
    invalid_sha = hashlib.sha256(invalid.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="invalid UTF-8 JSON"):
        parse_ilu_runtime_quant_config(
            {
                "source": "file",
                "file": str(invalid),
                "sha256": f"sha256:{invalid_sha}",
            },
            model_type="qwen3_5_moe",
            quant_method="fp8",
        )

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="exceeds"):
        parse_ilu_runtime_quant_config(
            {
                "source": "file",
                "file": str(oversized),
                "sha256": f"sha256:{'0' * 64}",
            },
            model_type="qwen3_5_moe",
            quant_method="fp8",
        )


def test_applies_to_and_loading_contract_fail_closed():
    with pytest.raises(ValueError, match="model_type"):
        parse_ilu_runtime_quant_config(
            {"source": "inline", **_payload()},
            model_type="other",
            quant_method="fp8",
        )

    payload = _payload()
    payload["config_groups"]["online_fp8"]["weights"]["loading"]["stage"] = "none"
    with pytest.raises(ValueError, match="transform \\+ none"):
        _parse(payload)

    payload = _payload(targets=["re:("])
    with pytest.raises(ValueError, match="invalid regex"):
        _parse(payload)


def test_priority_tie_and_target_coverage_are_rejected():
    payload = _payload(targets=[PREFIX, "never.matches"])
    config = _parse(payload)
    group = config.select_group(_linear(), PREFIX, {})
    assert group is not None
    with pytest.raises(RuntimeError, match="targets matched no layers"):
        config.record_shape_result(group.group_id, transformed=True)

    payload = _payload()
    payload["config_groups"]["second"] = dict(payload["config_groups"]["online_fp8"])
    config = _parse(payload)
    with pytest.raises(ValueError, match="multiple ilu_quant groups"):
        config.select_group(_linear(), PREFIX, {})


def test_fp8_dispatch_overlays_ignored_linear_and_records_transform():
    config = _vllm_config(_payload())
    fp8_config = _fp8_config()
    layer = _linear()

    with set_current_vllm_config(config):
        method = fp8_config.get_quant_method(layer, PREFIX)

    assert isinstance(method, IluFp8PerChannelOnlineLinearMethod)
    method.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[32],
        input_size=64,
        output_size=32,
        params_dtype=torch.bfloat16,
        weight_loader=lambda param, loaded: None,
    )
    method._get_runtime_context().assert_validation_complete()
    assert layer.weight.dtype == torch.float8_e4m3fn


def test_fp8_dispatch_preserves_original_dtype_on_shape_fallback():
    config = _vllm_config(_payload(transformed=0, fallback=1))
    fp8_config = _fp8_config()
    layer = _linear()

    with set_current_vllm_config(config):
        method = fp8_config.get_quant_method(layer, PREFIX)

    method.create_weights(
        layer,
        input_size_per_partition=65,
        output_partition_sizes=[32],
        input_size=65,
        output_size=32,
        params_dtype=torch.bfloat16,
        weight_loader=lambda param, loaded: None,
    )
    method._get_runtime_context().assert_validation_complete()
    assert layer.weight.dtype == torch.bfloat16


def test_runtime_fp8_rejects_parameter_dtype_outside_source_contract():
    with set_current_vllm_config(_vllm_config(_payload())):
        method = _fp8_config().get_quant_method(_linear(), PREFIX)

    with pytest.raises(ValueError, match="parameter dtype"):
        method.create_weights(
            _linear(),
            input_size_per_partition=64,
            output_partition_sizes=[32],
            input_size=64,
            output_size=32,
            params_dtype=torch.float16,
            weight_loader=lambda param, loaded: None,
        )


def test_runtime_fp8_rejects_loaded_shard_dtype_outside_source_contract():
    layer = _linear()
    with set_current_vllm_config(_vllm_config(_payload())):
        method = _fp8_config().get_quant_method(layer, PREFIX)
    method.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[32],
        input_size=64,
        output_size=32,
        params_dtype=torch.bfloat16,
        weight_loader=lambda param, loaded: None,
    )
    meta_weight = layer.weight
    materialized = ModelWeightParameter(
        data=torch.empty((32, 64), dtype=torch.float8_e4m3fn),
        input_dim=meta_weight.input_dim,
        output_dim=meta_weight.output_dim,
        weight_loader=meta_weight.weight_loader,
    )

    with pytest.raises(ValueError, match="checkpoint dtype"):
        meta_weight.weight_loader(
            materialized, torch.empty((32, 64), dtype=torch.float16)
        )


def test_global_ignore_and_unmatched_target_preserve_original_method():
    for update in (
        {"ignore": [PREFIX]},
        {
            "config_groups": {
                **_payload()["config_groups"],
            }
        },
    ):
        payload = _payload()
        if "ignore" in update:
            payload["ignore"] = update["ignore"]
        else:
            payload["config_groups"]["online_fp8"]["targets"] = ["other.layer"]
        config = _vllm_config(payload)
        with set_current_vllm_config(config):
            method = _fp8_config().get_quant_method(_linear(), PREFIX)
        assert not isinstance(method, IluFp8PerChannelOnlineLinearMethod)


def test_fp8_factory_rejects_unknown_options():
    payload = _payload()
    payload["config_groups"]["online_fp8"]["iluvatar"]["typo"] = True
    with set_current_vllm_config(_vllm_config(payload)):
        with pytest.raises(ValueError, match="unsupported options"):
            _fp8_config().get_quant_method(_linear(), PREFIX)


def test_registry_rejects_methods_without_runtime_factory():
    with pytest.raises(TypeError, match="does not implement"):
        IluvatarQuantRegistry.create_runtime_method("IluFp8LinearMethod", None)
    with pytest.raises(KeyError, match="not registered"):
        IluvatarQuantRegistry.create_runtime_method("NoSuchRuntimeMethod", None)
