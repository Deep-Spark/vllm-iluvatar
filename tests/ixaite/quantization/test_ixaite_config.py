# SPDX-License-Identifier: Apache-2.0
"""Route boundaries for standard W8A8 and vendor checkpoint formats."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import WeightsMapper

from tests.ixaite.quantization.config_fixtures import (
    _DEEPSEEK_LEGACY,
    _TARGET,
    _make_config,
)
from vllm_iluvatar.backends.ixaite.quantization import config as route
from vllm_iluvatar.backends.ixaite.quantization.w8a8 import (
    IxaiteCompressedTensorsW8A8Int8,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (
    IluCompressedTensorsW8A8Int8,
)


# Isolate cached choices and restore registry entries after each case.
@pytest.fixture(autouse=True)
def _provider(monkeypatch):
    from vllm.model_executor.layers.quantization import (
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG,
    )

    for name in ("compressed-tensors", "ilu_deepseek_v4_w4a8"):
        monkeypatch.setitem(
            _CUSTOMIZED_METHOD_TO_QUANT_CONFIG,
            name,
            _CUSTOMIZED_METHOD_TO_QUANT_CONFIG.get(name, route.IluQuantConfig),
        )
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(
        route, "load_ixaite_python_op", lambda *_: (object(), "available")
    )
    route._select_w8a8_linear_scheme_cls.cache_clear()
    route.register_ixaite_quantization_config()
    yield
    route._select_w8a8_linear_scheme_cls.cache_clear()


# Both quantization names must select the right config after upstream registration.
@pytest.mark.parametrize("mode", ["baseline", "ixaite"])
def test_registration(monkeypatch, mode):
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", mode)
    route.register_ixaite_quantization_config()
    expected = (
        route.IluQuantConfig
        if mode == "baseline"
        else route.IxaiteCompressedTensorsConfig
    )
    for name in ("compressed-tensors", "ilu_deepseek_v4_w4a8"):
        assert get_quantization_config(name) is expected
    assert route.IxaiteCompressedTensorsConfig.__bases__ == (CompressedTensorsConfig,)


# Unquantized attention retains its community scale lifecycle under W8A8 config.
def test_attention_without_quantized_kv_is_supported():
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsKVCacheMethod,
    )

    config = route.IxaiteCompressedTensorsConfig.from_config(_make_config(iluvatar={}))
    method = config.get_quant_method(object.__new__(Attention), "self_attn.attn")
    assert type(method) is CompressedTensorsKVCacheMethod
    assert method.quant_config.kv_cache_scheme is None


# Standard scale semantics win over conflicting vendor options and legacy aliases.
@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"mixed_strategy": "TENSOR", "gemm_format": "invalid", "gemm_mode": "invalid"},
    ],
)
@pytest.mark.parametrize(
    "dynamic,symmetric", [(True, True), (False, True), (True, False), (False, False)]
)
def test_standard_contract(monkeypatch, metadata, dynamic, symmetric):
    monkeypatch.setenv("VLLM_ILU_MIX_QUANTIZATION_TYPE", "TENSOR")
    monkeypatch.setenv("VLLM_MIX_QUANTIZATION_TYPE", "TENSOR")
    config = _make_config(iluvatar=metadata or {})
    group = config["config_groups"]["group_0"]
    if metadata is None:
        group.pop("iluvatar")
    group["input_activations"].update(
        dynamic=dynamic, symmetric=symmetric, strategy="token" if dynamic else "tensor"
    )
    original = deepcopy(config)
    quant = route.IxaiteCompressedTensorsConfig.from_config(config)
    layer = MagicMock(spec=LinearBase)
    quant.get_quant_method(layer, _TARGET)
    assert type(layer.scheme) is IxaiteCompressedTensorsW8A8Int8
    assert layer.scheme.strategy == "channel"
    assert layer.scheme.is_static_input_scheme is not dynamic
    assert layer.scheme.input_symmetric is symmetric
    assert "_ilu_config" not in quant.__dict__
    assert config == original


# Missing lifecycle functions must fail before any weights are allocated.
@pytest.mark.parametrize("missing", ["apply", "bind", "warmup"])
def test_missing_runtime(monkeypatch, missing):
    monkeypatch.setattr(
        route,
        "load_ixaite_python_op",
        lambda _op, name: (None if name == missing else object(), "test unavailable"),
    )
    quant = route.IxaiteCompressedTensorsConfig.from_config(_make_config(iluvatar={}))
    with pytest.raises(RuntimeError, match="test unavailable"):
        quant.get_scheme(MagicMock(spec=LinearBase), _TARGET)


# Unsupported tensor scales stay tensor scales, even with contradictory vendor hints.
def test_tensor_scales_are_not_reinterpreted(monkeypatch):
    monkeypatch.setattr(
        route, "load_ixaite_python_op", lambda *_: pytest.fail("tensor loaded ixaite")
    )
    config = _make_config(iluvatar={"mixed_strategy": "CHANNEL"})
    config["config_groups"]["group_0"]["weights"]["strategy"] = "tensor"
    quant = route.IxaiteCompressedTensorsConfig.from_config(config)
    with pytest.raises(RuntimeError, match="channel-wise"):
        quant.get_scheme(MagicMock(spec=LinearBase), _TARGET)


# Baseline must preserve upstream's vendor gate and never inspect the optional runtime.
@pytest.mark.parametrize(
    "metadata",
    [{}, {"mixed_strategy": "CHANNEL", "gemm_format": "TN", "gemm_mode": "NORMAL"}],
)
def test_baseline(monkeypatch, metadata):
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "baseline")
    monkeypatch.setattr(
        route, "load_ixaite_python_op", lambda *_: pytest.fail("baseline loaded ixaite")
    )
    route.register_ixaite_quantization_config()
    quant = get_quantization_config("compressed-tensors").from_config(
        _make_config(iluvatar=metadata)
    )
    scheme = quant.get_scheme(MagicMock(spec=LinearBase), _TARGET)
    assert type(scheme) is (
        IluCompressedTensorsW8A8Int8 if metadata else CompressedTensorsW8A8Int8
    )


# W4A8 retains its packing contract after model-name mapping and packed aliases.
def test_vendor_route_mapping_and_ignore():
    config = _make_config(
        weight_bits=4, iluvatar={"gemm_format": "NN", "weight_layout": "packed_nn"}
    )
    config["config_groups"]["group_0"]["targets"] = [
        "model.layers.0.q_proj",
        "model.layers.0.k_proj",
    ]
    config["ignore"] = ["model.layers.1.q_proj", "model.layers.1.k_proj"]
    quant = route.IxaiteCompressedTensorsConfig.from_config(config)
    quant.apply_vllm_mapper(
        WeightsMapper(orig_to_new_prefix={"model.": "language_model."})
    )
    quant.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj"]}
    layer = MagicMock(spec=LinearBase)
    parts = quant._vendor_config().get_scheme_dict(
        layer, "language_model.layers.0.qkv_proj"
    )
    assert parts["ilu_params"]["weight_layout"] == "packed_nn"
    with pytest.raises(RuntimeError, match="quantization contract"):
        quant.get_scheme(layer, "language_model.layers.0.qkv_proj")
    assert quant.get_scheme(layer, "language_model.layers.1.qkv_proj") is None


# Legacy conversion keeps BMM/MoE metadata while routing dense W8A8 through ixaite.
def test_legacy_mixed_routes(monkeypatch):
    quant = route.IxaiteCompressedTensorsConfig.from_config(_DEEPSEEK_LEGACY)
    layer = MagicMock(spec=LinearBase)
    assert (
        type(quant.get_scheme(layer, "model.layers.0.attn.wq_b"))
        is IxaiteCompressedTensorsW8A8Int8
    )
    vendor = quant._vendor_config()
    bmm = vendor.get_scheme_dict(layer, "model.layers.0.attn.wo_a")["ilu_params"]
    assert bmm["method"] == "IluV4W8A16BMMLinearMethod"
    assert bmm["bmm_w8a16_gemm_format"] == "k16n16"
    moe = vendor.get_scheme_dict(layer, "model.layers.0.ffn.experts")
    assert moe["weights"].num_bits == 4
    assert moe["ilu_params"]["pack_version"] == 2
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    from vllm_iluvatar.backends.ixaite.moe import w4a8

    moe_layer = MagicMock(spec=RoutedExperts)
    moe_layer.moe_config = SimpleNamespace(has_bias=False)
    monkeypatch.setattr(w4a8, "_moe_w4a8_group_gemm", object())
    method = quant.get_quant_method(moe_layer, "model.layers.0.ffn.experts")
    assert type(method) is w4a8.IluCompressedTensorsW4A8Int8MoEMethod
    assert method.gemm_format == "TN" and method.version == 2


# A quantized output head is a separate community dispatch case, not LinearBase.
def test_quantized_lm_head_uses_ixaite():
    quant = route.IxaiteCompressedTensorsConfig.from_config(_make_config(iluvatar={}))
    layer = MagicMock(spec=ParallelLMHead)
    assert quant.get_quant_method(layer, _TARGET) is not None
    assert type(layer.scheme) is IxaiteCompressedTensorsW8A8Int8


# MiniMax's baseline predicate stays unchanged; ixaite has no supported M3 model.
def test_minimax_keeps_mainline_quant_config_predicate():
    from vllm_iluvatar.models.minimax_m3.w4a8.model import _is_ilu_quant_config

    assert _is_ilu_quant_config(
        route.IluQuantConfig.from_config(_make_config(iluvatar={}))
    )
    assert not _is_ilu_quant_config(
        route.IxaiteCompressedTensorsConfig.from_config(_make_config(iluvatar={}))
    )
    assert not _is_ilu_quant_config(None)
    assert not _is_ilu_quant_config(
        CompressedTensorsConfig.from_config(_make_config(iluvatar={}))
    )
