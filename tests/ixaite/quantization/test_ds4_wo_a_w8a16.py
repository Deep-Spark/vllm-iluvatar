# SPDX-License-Identifier: Apache-2.0
"""0731 compressed-tensors wo_a must Marlin-repack, not stay on W8A8 2D GEMM."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_iluvatar.backends.ixaite.models.deepseek_v4.checkpoint import (
    is_ds4_wo_a_bmm_layer,
    maybe_repack_ds4_wo_a_w8a16,
)


def _wo_a_geom():
    return SimpleNamespace(
        n_local_groups=1,
        o_lora_rank=4,
        heads_per_group=2,
        head_dim=8,
    )


def _int8_wo_a_layer(*, geom=None):
    layer = torch.nn.Module()
    n = 1 * 4
    k = 2 * 8
    layer.weight = torch.nn.Parameter(
        torch.ones(n, k, dtype=torch.int8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones(n, 1, dtype=torch.float32), requires_grad=False
    )
    if geom is not None:
        layer.bmm_geom = geom
    return layer


def test_is_ds4_wo_a_rejects_glm_geom():
    layer = torch.nn.Module()
    layer.bmm_geom = SimpleNamespace(
        num_heads=8,
        qk_nope_head_dim=128,
        v_head_dim=128,
        kv_lora_rank=512,
    )
    assert is_ds4_wo_a_bmm_layer(layer) is False


def test_maybe_repack_skips_without_geom():
    layer = _int8_wo_a_layer()
    assert maybe_repack_ds4_wo_a_w8a16(layer) is False
    assert hasattr(layer, "weight")


def test_maybe_repack_skips_bf16_weight():
    layer = _int8_wo_a_layer(geom=_wo_a_geom())
    layer.weight = torch.nn.Parameter(
        torch.ones_like(layer.weight, dtype=torch.bfloat16),
        requires_grad=False,
    )
    assert maybe_repack_ds4_wo_a_w8a16(layer) is False
    assert hasattr(layer, "weight")


# The grouped guard chooses the private repacker before dense transforms can run.
def test_maybe_repack_int8_wo_a(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.models.deepseek_v4.checkpoint.repack_ds4_wo_a_w8a16",
        lambda layer, **kwargs: calls.append((layer, kwargs)),
    )
    layer = _int8_wo_a_layer(geom=_wo_a_geom())
    assert maybe_repack_ds4_wo_a_w8a16(layer) is True
    assert calls == [(layer, {"gemm_format": "k16n16"})]


def test_process_weights_wrapper_skips_w8a8_for_wo_a(monkeypatch):
    from vllm_iluvatar.backends.ixaite.quantization import config as ct

    called = {"orig": False}

    def fake_orig(self, layer):
        called["orig"] = True

    monkeypatch.setattr(ct.CompressedTensorsLinearMethod, "process_weights_after_loading", fake_orig)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.models.deepseek_v4.checkpoint.maybe_repack_ds4_wo_a_w8a16",
        lambda layer: True,
    )

    method = ct.IxaiteCompressedTensorsLinearMethod(ct.IxaiteCompressedTensorsConfig({}, [], "int-quantized"))
    method.process_weights_after_loading(torch.nn.Module())
    assert called["orig"] is False


def test_process_weights_wrapper_falls_through(monkeypatch):
    from vllm_iluvatar.backends.ixaite.quantization import config as ct

    called = {"orig": False}

    def fake_orig(self, layer):
        called["orig"] = True

    monkeypatch.setattr(ct.CompressedTensorsLinearMethod, "process_weights_after_loading", fake_orig)
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.models.deepseek_v4.checkpoint.maybe_repack_ds4_wo_a_w8a16",
        lambda layer: False,
    )

    method = ct.IxaiteCompressedTensorsLinearMethod(ct.IxaiteCompressedTensorsConfig({}, [], "int-quantized"))
    method.process_weights_after_loading(torch.nn.Module())
    assert called["orig"] is True
