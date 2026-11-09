# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""expert/half streaming NVFP4→W4A8 convert drop order."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_iluvatar.backends.ixaite.quantization.minimax_m3_flash import (
    RECOMMENDED_LOAD_TP,
    IluNvfp4ToW4A8MoEMethod,
    stream_convert_nvfp4_moe_weights,
)


def _make_nvfp4(n: int, k: int, *, global_scale: float = 1.0):
    from ixaite.ops.minimax_weight_convert_common import _FP4_E2M1_LUT, NVFP4_BLOCK

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
    neg_code = (q.unsqueeze(-1) - (-pos).view(1, 1, 1, 8)).abs().argmin(dim=-1) + 8
    codes = torch.where(q < 0, neg_code, pos_code).to(torch.uint8).view(n, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return (
        packed,
        scale_f8.contiguous(),
        torch.tensor(global_scale, dtype=torch.float32),
    )


def _fill_layer(layer: torch.nn.Module, *, experts: int = 2) -> None:
    method = IluNvfp4ToW4A8MoEMethod(SimpleNamespace())
    hidden, intermediate = 64, 32
    method.create_weights(
        layer,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    half = intermediate
    for e in range(experts):
        for h in (0, 1):
            packed, scale, g = _make_nvfp4(half, hidden, global_scale=0.5)
            layer.w13_weight_packed.data[e, h * half : (h + 1) * half] = packed
            layer.w13_weight_scale.data[e, h * half : (h + 1) * half] = scale
            layer.w13_weight_global_scale.data[e, h] = g
        packed2, scale2, g2 = _make_nvfp4(hidden, intermediate, global_scale=0.5)
        layer.w2_weight_packed.data[e] = packed2
        layer.w2_weight_scale.data[e] = scale2
        layer.w2_weight_global_scale.data[e] = g2


def test_recommended_load_tp_matches_sglang_flash():
    assert RECOMMENDED_LOAD_TP == 8


def test_stream_convert_drops_w13_packs_before_w2_and_before_install():
    layer = torch.nn.Module()
    _fill_layer(layer, experts=2)

    phases: list[tuple[str, frozenset[str]]] = []

    def on_phase(name: str, current: torch.nn.Module) -> None:
        phases.append(
            (
                name,
                frozenset(
                    attr
                    for attr in (
                        "w13_weight_packed",
                        "w2_weight_packed",
                        "w13_weight",
                        "w2_weight",
                    )
                    if hasattr(current, attr)
                ),
            )
        )

    stream_convert_nvfp4_moe_weights(layer, reference=True, on_phase=on_phase)

    by_name = dict(phases)
    assert "w13_nvfp4_dropped" in by_name
    assert "w13_weight_packed" not in by_name["w13_nvfp4_dropped"]
    # w2 packs must still be on the module while w13 converts.
    assert "w2_weight_packed" in by_name["w13_nvfp4_dropped"]
    assert "w13_weight" not in by_name["w13_nvfp4_dropped"]

    assert "w13_weight" in by_name["w13_w4a8_installed"]
    assert "w13_weight_packed" not in by_name["w13_w4a8_installed"]
    assert "w2_weight_packed" in by_name["w13_w4a8_installed"]

    assert "w2_weight_packed" not in by_name["w2_nvfp4_dropped"]
    assert "w13_weight" in by_name["w2_nvfp4_dropped"]
    assert "w2_weight" not in by_name["w2_nvfp4_dropped"]

    assert by_name["w2_w4a8_installed"] == frozenset({"w13_weight", "w2_weight"})
    assert layer.w13_weight.dtype == torch.int8
    assert layer.w2_weight.dtype == torch.int8
    assert not hasattr(layer, "w13_weight_packed")
    assert not hasattr(layer, "w2_weight_packed")


def test_process_weights_uses_stream_convert():
    method = IluNvfp4ToW4A8MoEMethod(SimpleNamespace())
    layer = torch.nn.Module()
    _fill_layer(layer, experts=1)
    method.process_weights_after_loading(layer)
    assert layer.w13_weight.dtype == torch.int8
    assert layer.w2_weight.dtype == torch.int8
