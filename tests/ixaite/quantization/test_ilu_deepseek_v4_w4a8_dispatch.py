# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch tests for legacy DeepSeek configs through IluQuantConfig.

These tests cover prefix -> method mapping. They do not add INT4 W4A8
projection kernels. Schema examples cover the legacy layout (v2
dense_linears / bmm_w8a16_layers). Production checkpoint regex copy is
still outstanding.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)

from vllm_iluvatar.backends.ixaite.quantization import config as ixaite_ct
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    IluQuantConfig,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (
    IluCompressedTensorsW8A8Int8,
)
from vllm_iluvatar.models.deepseek_v4.model import IluV4W8A16BMMLinearMethod
from vllm_iluvatar.models.glm_moe_dsa.model import IluGlmKVBProjW8A16Method

_W8A8_SECTION = {
    "num_bits": 8,
    "strategy": "channel",
    "symmetric": True,
    "type": "int",
    "input_activations": {
        "num_bits": 8,
        "strategy": "token",
        "dynamic": True,
        "symmetric": True,
        "type": "int",
    },
}

_ROUTED_SECTION = {
    "num_bits": 4,
    "strategy": "channel",
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
}

# Representative v2 regexes from ilu_deepseek_v4_w4a8.py docstring, plus
# indexer.wq_b / MTP e_proj|h_proj named in that same file. Not copied from
# a production config.json.
_V2_DENSE_TARGETS = [
    "re:.*\\.attn\\.(fused_wqa_wkv|wq_b|wo_b)$",
    "re:.*\\.attn\\.indexer\\.wq_b$",
    "re:.*\\.(e_proj|h_proj)$",
]

_V1_CONFIG = {
    "quant_method": "ilu_deepseek_v4_w4a8",
    "format": "ilu-mixed-w4a8",
    "version": 1,
    "routed_experts": _ROUTED_SECTION,
    "shared_experts": _W8A8_SECTION,
}

_V2_CONFIG = {
    **_V1_CONFIG,
    "version": 2,
    "dense_linears": {
        **_W8A8_SECTION,
        "targets": _V2_DENSE_TARGETS,
    },
    "bmm_w8a16_layers": {
        "targets": ["re:.*\\.attn\\.wo_a$"],
        "num_bits": 8,
        "strategy": "channel",
        "symmetric": True,
        "type": "int",
        "input_activations": None,
        "gemm_format": "k16n16",
        "input_dtype": "bfloat16",
    },
}

_GLM_CONFIG = {
    **_V2_CONFIG,
    "mla_bmm_w8a16": {
        "targets": ["re:.*\\.self_attn\\.kv_b_proj$"],
        "num_bits": 8,
        "strategy": "channel",
        "symmetric": True,
        "type": "int",
        "input_activations": None,
    },
}


class _DummyLinear(LinearBase):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)


class _DummyRoutedExperts(RoutedExperts):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.moe_config = types.SimpleNamespace()


# Isolate the startup provider binding while testing checkpoint dispatch.
@pytest.fixture(autouse=True)
def _reset_w8a8_provider(monkeypatch):
    import vllm.model_executor.layers.quantization as registry

    monkeypatch.setattr(
        registry,
        "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG",
        dict(registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG),
    )
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "baseline")
    ixaite_ct._select_w8a8_linear_scheme_cls.cache_clear()
    ixaite_ct.register_ixaite_quantization_config()
    yield
    ixaite_ct._select_w8a8_linear_scheme_cls.cache_clear()


@pytest.fixture
def v1_cfg() -> IluQuantConfig:
    return IluQuantConfig.from_config(_V1_CONFIG)


@pytest.fixture
def v2_cfg() -> IluQuantConfig:
    return IluQuantConfig.from_config(_V2_CONFIG)


@pytest.fixture
def w4a8_moe_method_stub(monkeypatch):
    """Replace the lazy MoE import even when another test loaded it first."""
    module_name = (
        "vllm_iluvatar.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.ilu_compressed_tensors_moe_w4a8_int8"
    )
    module = types.ModuleType(module_name)
    method_type = type(
        "IluCompressedTensorsW4A8Int8MoEMethod",
        (),
        {"__init__": lambda self, *args, **kwargs: None},
    )
    module.IluCompressedTensorsW4A8Int8MoEMethod = method_type
    monkeypatch.setitem(sys.modules, module_name, module)
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsMoEMethod,
    )

    from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.ilu_compressed_tensors_moe import (
        IluCompressedTensorsMoEMethod,
    )

    # Model startup installs this wrapper; keep that dispatch in the CPU fixture.
    monkeypatch.setattr(
        CompressedTensorsMoEMethod,
        "get_moe_method",
        staticmethod(IluCompressedTensorsMoEMethod.get_moe_method),
    )
    return method_type


# Legacy names resolve to the registered config and are converted on parsing.
def test_legacy_quantization_method_resolves_registered_config() -> None:
    assert get_quantization_config("ilu_deepseek_v4_w4a8") is IluQuantConfig
    assert get_quantization_config("fp8") is not IluQuantConfig


def test_v1_and_v2_routed_experts_are_w4a8(
    v1_cfg, v2_cfg, w4a8_moe_method_stub
) -> None:
    layer = _DummyRoutedExperts()
    prefix = "model.layers.5.ffn.experts"
    assert isinstance(
        v1_cfg.get_quant_method(layer, prefix),
        w4a8_moe_method_stub,
    )
    assert isinstance(
        v2_cfg.get_quant_method(layer, prefix),
        w4a8_moe_method_stub,
    )


def test_shared_expert_linear_is_w8a8(v2_cfg) -> None:
    layer = _DummyLinear()
    method = v2_cfg.get_quant_method(
        layer, "model.layers.5.ffn.shared_experts.gate_up_proj"
    )
    assert isinstance(method, CompressedTensorsLinearMethod)
    assert isinstance(layer.scheme, IluCompressedTensorsW8A8Int8)


def test_v1_projections_stay_bf16(v1_cfg) -> None:
    layer = _DummyLinear()
    prefixes = (
        "model.layers.0.attn.fused_wqa_wkv",
        "model.layers.0.attn.wq_b",
        "model.layers.0.attn.wo_a",
        "model.layers.0.attn.wo_b",
        "model.layers.0.attn.indexer.wq_b",
        "model.layers.0.attn.indexer.weights_proj",
        "model.layers.0.attn.indexer.compressor.fused_wkv_wgate",
        "model.layers.42.mtp.e_proj",
        "model.layers.42.mtp.h_proj",
    )
    for prefix in prefixes:
        method = v1_cfg.get_quant_method(layer, prefix)
        assert isinstance(method, UnquantizedLinearMethod), prefix


def test_v2_dense_and_bmm_dispatch(v2_cfg) -> None:
    layer = _DummyLinear()
    cases = (
        ("model.layers.0.attn.fused_wqa_wkv", CompressedTensorsLinearMethod),
        ("model.layers.0.attn.wq_b", CompressedTensorsLinearMethod),
        ("model.layers.0.attn.wo_b", CompressedTensorsLinearMethod),
        ("model.layers.0.attn.indexer.wq_b", CompressedTensorsLinearMethod),
        ("model.layers.42.mtp.e_proj", CompressedTensorsLinearMethod),
        ("model.layers.42.mtp.h_proj", CompressedTensorsLinearMethod),
        ("model.layers.0.attn.wo_a", IluV4W8A16BMMLinearMethod),
    )
    for prefix, expected in cases:
        method = v2_cfg.get_quant_method(layer, prefix)
        assert isinstance(method, expected), prefix
        if expected is CompressedTensorsLinearMethod:
            assert isinstance(layer.scheme, IluCompressedTensorsW8A8Int8), prefix


def test_v2_untargeted_projections_stay_bf16(v2_cfg) -> None:
    layer = _DummyLinear()
    prefixes = (
        "model.layers.0.attn.indexer.weights_proj",
        "model.layers.0.attn.indexer.compressor.fused_wkv_wgate",
        "model.layers.0.attn.q_norm",
    )
    for prefix in prefixes:
        method = v2_cfg.get_quant_method(layer, prefix)
        assert isinstance(method, UnquantizedLinearMethod), prefix


# The private parser keeps BMM metadata when a later group repeats a dense target.
def test_wo_a_metadata_wins_over_dense_if_both_could_match() -> None:
    cfg = ixaite_ct.IxaiteCompressedTensorsConfig.from_config(
        {
            **_V2_CONFIG,
            "dense_linears": {
                **_W8A8_SECTION,
                "targets": ["re:.*\\.attn\\.wo_a$"],
            },
        }
    )
    layer = _DummyLinear()
    prefix = "model.layers.0.attn.wo_a"
    parts = cfg._vendor_config().get_scheme_dict(layer, prefix)
    assert parts["input_activations"] is None
    assert parts["ilu_params"]["method"] == "IluV4W8A16BMMLinearMethod"
    # Preserving the checkpoint contract does not permit a native BMM fallback.
    with pytest.raises(RuntimeError, match="does not support this quantization contract"):
        cfg.get_quant_method(layer, prefix)


def test_glm_kv_b_proj_is_w8a16_split() -> None:
    cfg = IluQuantConfig.from_config(_GLM_CONFIG)
    method = cfg.get_quant_method(_DummyLinear(), "model.layers.0.self_attn.kv_b_proj")
    assert isinstance(method, IluGlmKVBProjW8A16Method)


def test_no_int4_linear_method_for_projections(v2_cfg) -> None:
    layer = _DummyLinear()
    prefixes = (
        "model.layers.0.attn.fused_wqa_wkv",
        "model.layers.0.attn.wq_b",
        "model.layers.0.attn.wo_a",
        "model.layers.0.attn.wo_b",
        "model.layers.0.attn.indexer.wq_b",
        "model.layers.0.attn.indexer.compressor.fused_wkv_wgate",
    )
    for prefix in prefixes:
        method = v2_cfg.get_quant_method(layer, prefix)
        assert (
            type(getattr(layer, "scheme", None)).__name__
            != "IluCompressedTensorsW4A8Int"
        )
        assert type(method).__name__ != "IluCompressedTensorsW4A8Int8MoEMethod"
