# SPDX-License-Identifier: Apache-2.0
"""CPU-only OOT-interface tests for the Iluvatar block-FP8 MoE config.

Pins the registration / routing contract of ``ilu_fp8.py`` without a GPU or the
ixmm backend:

* ``@register_quantization_config("fp8")`` shadows the community name and the
  class stays a ``Fp8Config`` subclass (so checkpoint detection / linear /
  attention branches are inherited).
* ``get_quant_method`` narrows to the *block-wise* routed-expert branch only:
  block-FP8 routed experts get ``IluFp8MoEMethod``; per-tensor FP8, skipped
  layers and everything non-MoE delegate to ``super()`` (plan §7 R1 -- do not
  shadow other fp8 models).
* ``IluFp8MoEMethod.__init__`` skips community ``select_fp8_moe_backend`` (the
  NVIDIA-only selection that lands off-NVIDIA on an unusable backend).

The ix30 numeric path (``preprocess_fp8_block_weight_for_ixmm`` +
``process_weights_after_loading`` + ``apply``) needs ixformer/ixmm and is
exercised on the remote GPU env, not here.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch


def _install_fake_ixformer(monkeypatch) -> None:
    ixformer = types.ModuleType("ixformer")
    core = types.ModuleType("ixformer.core")
    config = types.SimpleNamespace(IXFORMER_USE_TORCH_OPS=False)
    inference = types.ModuleType("ixformer.inference")
    functions = types.ModuleType("ixformer.inference.functions")
    core.config = config
    ixformer.core = core
    ixformer.inference = inference
    inference.functions = functions
    monkeypatch.setitem(sys.modules, "ixformer", ixformer)
    monkeypatch.setitem(sys.modules, "ixformer.core", core)
    monkeypatch.setitem(sys.modules, "ixformer.core.config", config)
    monkeypatch.setitem(sys.modules, "ixformer.inference", inference)
    monkeypatch.setitem(sys.modules, "ixformer.inference.functions", functions)


def _load_ilu_fp8(monkeypatch):
    _install_fake_ixformer(monkeypatch)
    return importlib.import_module(
        "vllm_iluvatar.model_executor.layers.quantization.ilu_fp8"
    )


_BLOCK_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
}
_PER_TENSOR_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
}


class _DummyRoutedExperts:
    pass


def test_ilu_fp8_shadows_community_fp8_name(monkeypatch):
    mod = _load_ilu_fp8(monkeypatch)

    from vllm.model_executor.layers.quantization import (
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    assert issubclass(mod.IluFp8Config, Fp8Config)
    assert issubclass(mod.IluFp8MoEMethod, mod.Fp8MoEMethod)
    assert _CUSTOMIZED_METHOD_TO_QUANT_CONFIG["fp8"] is mod.IluFp8Config


def test_ilu_fp8_inherits_checkpoint_detection(monkeypatch):
    mod = _load_ilu_fp8(monkeypatch)

    config = mod.IluFp8Config.from_config(_BLOCK_CONFIG)

    assert isinstance(config, mod.IluFp8Config)
    assert config.is_checkpoint_fp8_serialized is True
    assert config.weight_block_size == [128, 128]
    assert config.get_name() == "fp8"


def test_get_quant_method_routes_block_routed_experts(monkeypatch):
    mod = _load_ilu_fp8(monkeypatch)

    sentinel = object()
    monkeypatch.setattr(mod, "RoutedExperts", _DummyRoutedExperts)
    monkeypatch.setattr(mod, "is_layer_skipped", lambda **kwargs: False)
    monkeypatch.setattr(mod, "IluFp8MoEMethod", lambda config, layer: sentinel)

    config = mod.IluFp8Config.from_config(_BLOCK_CONFIG)
    method = config.get_quant_method(
        _DummyRoutedExperts(), "model.layers.3.mlp.experts"
    )

    assert method is sentinel


def test_get_quant_method_delegates_per_tensor_fp8(monkeypatch):
    """Per-tensor FP8 (no ``weight_block_size``) must not hit our branch."""
    mod = _load_ilu_fp8(monkeypatch)

    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    sentinel = object()
    monkeypatch.setattr(mod, "RoutedExperts", _DummyRoutedExperts)
    monkeypatch.setattr(
        Fp8Config, "get_quant_method", lambda self, layer, prefix: sentinel
    )
    monkeypatch.setattr(
        mod,
        "IluFp8MoEMethod",
        lambda *a, **k: pytest.fail("per-tensor FP8 must delegate to super()"),
    )

    config = mod.IluFp8Config.from_config(_PER_TENSOR_CONFIG)
    method = config.get_quant_method(
        _DummyRoutedExperts(), "model.layers.3.mlp.experts"
    )

    assert method is sentinel


def test_get_quant_method_delegates_skipped_layer(monkeypatch):
    """A skipped (``modules_to_not_convert``) expert must delegate to super()."""
    mod = _load_ilu_fp8(monkeypatch)

    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    sentinel = object()
    monkeypatch.setattr(mod, "RoutedExperts", _DummyRoutedExperts)
    monkeypatch.setattr(mod, "is_layer_skipped", lambda **kwargs: True)
    monkeypatch.setattr(
        Fp8Config, "get_quant_method", lambda self, layer, prefix: sentinel
    )
    monkeypatch.setattr(
        mod,
        "IluFp8MoEMethod",
        lambda *a, **k: pytest.fail("skipped layer must delegate to super()"),
    )

    config = mod.IluFp8Config.from_config(_BLOCK_CONFIG)
    method = config.get_quant_method(
        _DummyRoutedExperts(), "model.layers.3.mlp.experts"
    )

    assert method is sentinel


def test_method_init_skips_backend_selection(monkeypatch):
    """``IluFp8MoEMethod.__init__`` must not call ``select_fp8_moe_backend``."""
    mod = _load_ilu_fp8(monkeypatch)

    import vllm.model_executor.layers.quantization.fp8 as community_fp8

    def _boom(*args, **kwargs):
        raise RuntimeError("select_fp8_moe_backend must not be called on Iluvatar")

    # If the parent __init__ ran, this raises; a clean construct proves the skip.
    monkeypatch.setattr(community_fp8, "select_fp8_moe_backend", _boom)
    # Decouple from the heavy base initialiser (env / distributed state).
    monkeypatch.setattr(
        mod.FusedMoEMethodBase, "__init__", lambda self, moe_config: None
    )

    config = mod.IluFp8Config.from_config(_BLOCK_CONFIG)
    layer = types.SimpleNamespace(moe_config=object())
    method = mod.IluFp8MoEMethod(config, layer)

    assert method.fp8_backend is None
    assert method.moe_kernel is None
    assert method.block_quant is True
    assert method.weight_scale_name == "weight_scale_inv"
    assert method.is_monolithic is False
    assert method.supports_eplb is False
    assert not hasattr(method, "experts_cls")


def test_preprocess_pads_k_to_ixmm_group_boundary(monkeypatch):
    mod = _load_ilu_fp8(monkeypatch)

    captured = {}

    def _quantize_fp4(x):
        captured["normalized"] = x
        return torch.zeros_like(x, dtype=torch.uint8)

    fake_gemm = types.SimpleNamespace(
        _quantize_fp4=_quantize_fp4,
        _pack_nibbles_ix30=lambda x: x[:, ::2, :],
        preprocess_group_scale_for_running_acc=lambda x: x,
    )
    fp4_module = types.ModuleType("ixformer.inference.functions.fp4")
    gemm_module = types.ModuleType("ixformer.inference.functions.fp4.gemm")
    gemm_module.gemm = fake_gemm
    monkeypatch.setitem(sys.modules, "ixformer.inference.functions.fp4", fp4_module)
    monkeypatch.setitem(
        sys.modules, "ixformer.inference.functions.fp4.gemm", gemm_module
    )

    weight = torch.ones((1, 2, 640), dtype=torch.float32)
    scale_inv = torch.ones((1, 1, 5), dtype=torch.float32)
    ixmm_weight, ixmm_scale = mod.preprocess_fp8_block_weight_for_ixmm(
        weight, scale_inv, (128, 128), allow_k_padding=True
    )

    assert ixmm_weight.shape == (1, 384, 2)
    assert ixmm_scale.shape == (1, 3, 2)
    assert captured["normalized"].shape == (1, 768, 2)
    assert torch.count_nonzero(captured["normalized"][:, 640:, :]) == 0
