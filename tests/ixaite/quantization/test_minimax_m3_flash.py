# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""Flash quant override, MXFP8/NVFP4 convert, fail-closed apply.

Avoids importing ``backends.ixaite.quantization.config`` so collection does not
require the baseline ``ixformer`` package (ixaite-only hosts may lack it).
Registration wiring is checked via AST against ``config.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import (
    get_quantization_config,
    register_quantization_config,
)

from vllm_iluvatar.backends.ixaite.platform import IxaitePlatform
from vllm_iluvatar.backends.ixaite.quantization.minimax_m3_flash import (
    METHOD_NAME,
    IluMxfp8ToW8A8LinearMethod,
    IluNvfp4ToW4A8MoEMethod,
    IxaiteMiniMaxM3FlashConfig,
)

_CONFIG_PY = (
    Path(__file__).resolve().parents[3]
    / "vllm_iluvatar"
    / "backends"
    / "ixaite"
    / "quantization"
    / "config.py"
)


@pytest.fixture(autouse=True)
def _register_flash():
    from vllm.model_executor.layers.quantization import (
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG,
    )

    register_quantization_config(METHOD_NAME)(IxaiteMiniMaxM3FlashConfig)
    yield
    _CUSTOMIZED_METHOD_TO_QUANT_CONFIG.pop(METHOD_NAME, None)


def _synthetic_mxfp8(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    bf16 = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    blocks = bf16.float().view(n, k // 32, 32)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-8)
    exp = torch.round(torch.log2(amax / 448.0)) + 127
    exp = exp.clamp(0, 254).to(torch.uint8)
    scale_f = torch.exp2(exp.to(torch.float32) - 127.0).unsqueeze(-1)
    q = (blocks / scale_f).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(n, k)
    return q, exp.contiguous()


def _make_nvfp4(n: int, k: int, *, global_scale: float = 1.0):
    from ixaite.ops.minimax_weight_convert_common import _FP4_E2M1_LUT, NVFP4_BLOCK

    assert k % NVFP4_BLOCK == 0 and k % 2 == 0
    bf16 = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    blocks = bf16.float().view(n, k // NVFP4_BLOCK, NVFP4_BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-8)
    local = ((amax * float(global_scale)) / 6.0).clamp(max=448.0)
    scale_f8 = local.to(torch.float8_e4m3fn)
    scale_f = scale_f8.float() / float(global_scale)
    q = (blocks / scale_f.unsqueeze(-1)).clamp(-6.0, 6.0)
    lut = _FP4_E2M1_LUT
    pos = lut[:8]
    dist = (q.unsqueeze(-1) - pos.view(1, 1, 1, 8)).abs()
    pos_code = dist.argmin(dim=-1)
    neg_dist = (q.unsqueeze(-1) - (-pos).view(1, 1, 1, 8)).abs()
    neg_code = neg_dist.argmin(dim=-1) + 8
    codes = torch.where(q < 0, neg_code, pos_code).to(torch.uint8).view(n, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    g = torch.tensor(global_scale, dtype=torch.float32)
    return packed, scale_f8.contiguous(), g


def test_override_claims_mxfp8_nvfp4():
    cfg = {
        "quant_method": "mxfp8",
        "moe_quant_algo": "NVFP4",
        "weight_block_size": [1, 32],
    }
    assert (
        IxaiteMiniMaxM3FlashConfig.override_quantization_method(cfg, None)
        == METHOD_NAME
    )
    assert (
        IxaiteMiniMaxM3FlashConfig.override_quantization_method(
            {"quant_method": "mxfp8"}, None
        )
        is None
    )


def test_registration_wired_in_ixaite_config_and_platform_accepts():
    source = _CONFIG_PY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    text = ast.dump(tree)
    assert "IxaiteMiniMaxM3FlashConfig" in text
    assert "ilu_minimax_m3_flash" in source or "METHOD_NAME" in source
    assert get_quantization_config(METHOD_NAME) is IxaiteMiniMaxM3FlashConfig
    IxaitePlatform.verify_quantization(METHOD_NAME)


def test_get_quant_method_routes_linear_and_moe():
    config = IxaiteMiniMaxM3FlashConfig()
    linear = object.__new__(LinearBase)
    assert isinstance(
        config.get_quant_method(linear, "model.layers.0.self_attn.qkv_proj"),
        IluMxfp8ToW8A8LinearMethod,
    )

    class RoutedExperts(torch.nn.Module):
        moe_config = SimpleNamespace()

    experts = RoutedExperts()
    assert isinstance(
        config.get_quant_method(experts, "model.layers.0.mlp.experts"),
        IluNvfp4ToW4A8MoEMethod,
    )


def test_mxfp8_linear_convert_and_reject_native_dtype(monkeypatch):
    import vllm.model_executor.parameter as parameter_module

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    method = IluMxfp8ToW8A8LinearMethod()
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[32],
        input_size=64,
        output_size=32,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale_inv.dtype == torch.uint8
    assert tuple(layer.weight_scale_inv.shape) == (32, 2)

    n, k = 32, 64
    q, exp = _synthetic_mxfp8(n, k)
    layer.weight = torch.nn.Parameter(q, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(exp, requires_grad=False)
    method.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.int8
    assert layer.weight.shape == (n, k)
    assert layer.weight_scale.shape[0] == n
    assert not hasattr(layer, "weight_scale_inv")

    bad = torch.nn.Module()
    bad.weight = torch.nn.Parameter(
        torch.empty(n, k, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    with pytest.raises(RuntimeError, match="refused native apply"):
        method.apply(bad, torch.randn(2, k, dtype=torch.bfloat16))


def test_nvfp4_moe_create_weights_and_reject_unconverted():
    method = IluNvfp4ToW4A8MoEMethod(SimpleNamespace())
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=64,
        intermediate_size_per_partition=32,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    assert layer.w13_weight_packed.dtype == torch.uint8
    assert layer.w13_weight_packed.shape == (2, 64, 32)
    assert layer.w2_weight_packed.shape == (2, 64, 16)
    assert not hasattr(layer, "w13_weight")
    assert not hasattr(layer, "_ilu_shared_mxfp8")

    with pytest.raises(RuntimeError, match="refused native apply"):
        method.apply(
            layer,
            torch.randn(2, 64, dtype=torch.bfloat16),
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.long),
            None,
            None,
        )


def test_nvfp4_moe_convert_reference_sets_int8_weights():
    method = IluNvfp4ToW4A8MoEMethod(SimpleNamespace())
    layer = torch.nn.Module()
    experts, hidden, intermediate = 1, 64, 32
    method.create_weights(
        layer,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )

    half = intermediate
    for h in (0, 1):
        packed, scale, g = _make_nvfp4(half, hidden, global_scale=0.5)
        layer.w13_weight_packed.data[0, h * half : (h + 1) * half] = packed
        layer.w13_weight_scale.data[0, h * half : (h + 1) * half] = scale
        layer.w13_weight_global_scale.data[0, h] = g
    packed2, scale2, g2 = _make_nvfp4(hidden, intermediate, global_scale=0.5)
    layer.w2_weight_packed.data[0] = packed2
    layer.w2_weight_scale.data[0] = scale2
    layer.w2_weight_global_scale.data[0] = g2

    method.process_weights_after_loading(layer)
    assert layer.w13_weight.dtype == torch.int8
    assert layer.w2_weight.dtype == torch.int8
    assert not hasattr(layer, "w13_weight_packed")
    assert not hasattr(layer, "w2_weight_packed")
