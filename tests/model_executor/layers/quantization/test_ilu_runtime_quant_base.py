# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter as parameter_module
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

import vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.ivcore30.linear.native_to_fp8 as ilu_fp8
from vllm_iluvatar.model_executor.layers.quantization import capabilities
from vllm_iluvatar.model_executor.layers.quantization.capabilities import (
    require_ivcore30,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime import (
    config as ilu_runtime_config,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.base_config import (
    IluRuntimeQuantBaseConfig,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.ivcore30.linear.native_to_fp8 import (
    IluFp8PerChannelOnlineLinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.ivcore30.moe.fp16_to_fp4 import (
    IluFp16ToFp4OnlineMoEMethod,
)

MODEL_TYPE = "qwen3_5_text"
TARGET_PREFIX = "language_model.model.layers.0.mlp.gate_up_proj"
OTHER_PREFIX = "visual.blocks.0.mlp.linear_fc1"


@pytest.fixture(autouse=True)
def _single_rank_parameter_state(monkeypatch):
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(ilu_fp8, "initialize_online_processing", lambda layer: None)


def _payload(*, enabled: bool = True) -> dict:
    return {
        "source": "inline",
        "schema_version": 1,
        "enabled": enabled,
        "applies_to": {
            "model_types": [MODEL_TYPE],
            "quant_methods": ["ilu_runtime"],
        },
        "ignore": [],
        "config_groups": {
            "qwen38_27b_bf16_to_fp8": {
                "priority": 100,
                "targets": [r"re:.*\.mlp\.(?:gate_up_proj|down_proj)$"],
                "require": {
                    "original_disposition": "ignored",
                    "checkpoint_dtypes": ["bfloat16"],
                },
                "weights": {
                    "source": {"format": "native", "dtypes": ["bfloat16"]},
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
            }
        },
    }


def _vllm_config(payload: dict | None, *, model_type: str = MODEL_TYPE) -> VllmConfig:
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type=model_type),
        dtype=torch.bfloat16,
    )
    config.additional_config = {} if payload is None else {"ilu_quant": payload}
    return config


def _linear(prefix: str) -> LinearBase:
    layer = object.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.prefix = prefix
    return layer


def test_registered_under_ilu_runtime_and_reads_no_checkpoint_json():
    assert get_quantization_config("ilu_runtime") is IluRuntimeQuantBaseConfig
    assert IluRuntimeQuantBaseConfig.get_config_filenames() == []
    assert IluRuntimeQuantBaseConfig.from_config({}).get_name() == "ilu_runtime"
    with pytest.raises(ValueError, match="takes no checkpoint quantization"):
        IluRuntimeQuantBaseConfig.from_config({"quant_method": "fp8"})


def test_matching_target_gets_the_runtime_fp8_method():
    config = IluRuntimeQuantBaseConfig()
    with set_current_vllm_config(_vllm_config(_payload())):
        method = config.get_quant_method(_linear(TARGET_PREFIX), TARGET_PREFIX)
    assert isinstance(method, IluFp8PerChannelOnlineLinearMethod)
    # The factory copies the base config and must find both attributes on it.
    assert method.gemm_format == "TN"
    assert method.quant_config.weight_block_size is None


def test_unmatched_linear_stays_unquantized_and_non_linear_is_none():
    config = IluRuntimeQuantBaseConfig()
    with set_current_vllm_config(_vllm_config(_payload())):
        method = config.get_quant_method(_linear(OTHER_PREFIX), OTHER_PREFIX)
        embedding = object.__new__(VocabParallelEmbedding)
        torch.nn.Module.__init__(embedding)
        assert config.get_quant_method(embedding, "lm_head") is None
    assert isinstance(method, UnquantizedLinearMethod)


def test_missing_or_disabled_payload_fails_fast():
    config = IluRuntimeQuantBaseConfig()
    with (
        set_current_vllm_config(_vllm_config(None)),
        pytest.raises(ValueError, match="requires additional_config.ilu_quant"),
    ):
        config.get_quant_method(_linear(TARGET_PREFIX), TARGET_PREFIX)

    disabled = IluRuntimeQuantBaseConfig()
    with (
        set_current_vllm_config(_vllm_config(_payload(enabled=False))),
        pytest.raises(ValueError, match="ilu_quant.enabled is false"),
    ):
        disabled.get_quant_method(_linear(TARGET_PREFIX), TARGET_PREFIX)


@pytest.mark.parametrize("detected", ["ivcore11", None])
def test_enabled_payload_requires_ivcore30(monkeypatch, detected):
    monkeypatch.setattr(ilu_runtime_config, "require_ivcore30", require_ivcore30)
    monkeypatch.setattr(capabilities, "cuda_arch", lambda: detected)
    config = IluRuntimeQuantBaseConfig()
    with (
        set_current_vllm_config(_vllm_config(_payload())),
        pytest.raises(
            RuntimeError, match=r"additional_config\.ilu_quant requires ivcore30"
        ),
    ):
        config.get_quant_method(_linear(TARGET_PREFIX), TARGET_PREFIX)


def test_enabled_payload_passes_the_gate_on_ivcore30(monkeypatch):
    monkeypatch.setattr(ilu_runtime_config, "require_ivcore30", require_ivcore30)
    monkeypatch.setattr(capabilities, "cuda_arch", lambda: "ivcore30")
    config = IluRuntimeQuantBaseConfig()
    with set_current_vllm_config(_vllm_config(_payload())):
        method = config.get_quant_method(_linear(TARGET_PREFIX), TARGET_PREFIX)
    assert isinstance(method, IluFp8PerChannelOnlineLinearMethod)


def test_disabled_payload_skips_the_arch_gate(monkeypatch):
    monkeypatch.setattr(ilu_runtime_config, "require_ivcore30", require_ivcore30)
    monkeypatch.setattr(capabilities, "cuda_arch", lambda: "ivcore11")
    config = IluRuntimeQuantBaseConfig()
    with (
        set_current_vllm_config(_vllm_config(_payload(enabled=False))),
        pytest.raises(ValueError, match="ilu_quant.enabled is false"),
    ):
        config.get_quant_method(_linear(TARGET_PREFIX), TARGET_PREFIX)


def test_payload_declaring_another_quant_method_is_rejected():
    payload = _payload()
    payload["applies_to"]["quant_methods"] = ["fp8"]
    config = IluRuntimeQuantBaseConfig()
    with (
        set_current_vllm_config(_vllm_config(payload)),
        pytest.raises(ValueError, match="does not apply to quant_method"),
    ):
        config.get_quant_method(_linear(TARGET_PREFIX), TARGET_PREFIX)


def _minimax_payload() -> dict:
    import json
    from pathlib import Path

    example = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "iluvatar-runtime-quant"
        / "MiniMaxAI"
        / "MiniMax-M3"
        / "fp8-fp4-runtime.json"
    )
    return {"source": "inline", **json.loads(example.read_text())}


def _experts(prefix: str) -> RoutedExperts:
    layer = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.prefix = prefix
    layer.moe_config = SimpleNamespace(
        is_act_and_mul=True,
        w13_num_shards=2,
        tp_shard_with_padding=False,
        hidden_dim=256,
        hidden_dim_unpadded=256,
        intermediate_size=256,
        intermediate_size_per_partition=256,
        intermediate_size_per_partition_unpadded=None,
        tp_size=1,
        moe_backend="auto",
    )
    return layer


def test_minimax_payload_dispatches_linear_fp8_and_moe_fp4():
    payload = _minimax_payload()
    config = IluRuntimeQuantBaseConfig()
    linear_prefix = "language_model.model.layers.0.self_attn.qkv_proj"
    moe_prefix = "language_model.model.layers.3.block_sparse_moe.experts"
    other_linear = "language_model.model.layers.3.block_sparse_moe.gate"
    with set_current_vllm_config(_vllm_config(payload, model_type="minimax_m3_text")):
        linear = config.get_quant_method(_linear(linear_prefix), linear_prefix)
        moe = config.get_quant_method(_experts(moe_prefix), moe_prefix)
        moe_nested = config.get_quant_method(
            _experts(moe_prefix + ".routed_experts"),
            moe_prefix + ".routed_experts",
        )
        unmatched_linear = config.get_quant_method(_linear(other_linear), other_linear)
        skipped_experts = config.get_quant_method(
            _experts("visual.blocks.0.mlp.experts"), "visual.blocks.0.mlp.experts"
        )
    assert isinstance(linear, IluFp8PerChannelOnlineLinearMethod)
    assert isinstance(moe, IluFp16ToFp4OnlineMoEMethod)
    assert isinstance(moe_nested, IluFp16ToFp4OnlineMoEMethod)
    assert isinstance(unmatched_linear, UnquantizedLinearMethod)
    assert skipped_experts is None
