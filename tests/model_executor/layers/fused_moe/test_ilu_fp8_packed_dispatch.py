# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the packed FP8 activation and routing layout."""

import pytest
import torch
import triton
from conftest import requires_corex30

from vllm_iluvatar import envs
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fp8_packed_dispatch import (
    _pack_quantized_routes,
    _quantize_and_pack_routes,
    _unpack_route_metadata,
    expand_prequantized_input,
)

HIDDEN_SIZE = 5120
TOP_K = 6
PACKET_STRIDE = HIDDEN_SIZE + 4 + TOP_K * 8


@pytest.mark.parametrize(
    "name",
    [
        "VLLM_ILU_V41_PACKED_FP8_DISPATCH",
        "VLLM_ILU_V41_PACKED_FP8_FUSED_QUANT",
        "VLLM_ILU_V41_DECODE_MOE_MODE1",
    ],
)
def test_switches_are_runtime_readers_with_legacy_precedence(name, monkeypatch):
    legacy = name.replace("VLLM_ILU_", "VLLM_", 1)
    monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(legacy, raising=False)
    assert getattr(envs, name) is False
    monkeypatch.setenv(legacy, "1")
    assert getattr(envs, name) is True
    assert getattr(envs, legacy) is True
    monkeypatch.setenv(name, "0")
    assert getattr(envs, name) is False
    assert getattr(envs, legacy) is False
    monkeypatch.setenv(name, "1")
    assert getattr(envs, name) is True


def routing(rows):
    indices = torch.arange(rows * TOP_K, device="cuda", dtype=torch.int32).view(
        rows, TOP_K
    )
    weights = torch.rand((rows, TOP_K), device="cuda", dtype=torch.float32)
    return indices, weights


def unpack(packet):
    rows = packet.shape[0]
    scales = torch.empty(rows, device="cuda", dtype=torch.float32)
    indices = torch.empty((rows, TOP_K), device="cuda", dtype=torch.int32)
    weights = torch.empty((rows, TOP_K), device="cuda", dtype=torch.float32)
    _unpack_route_metadata[(rows,)](
        packet,
        scales.view(torch.uint8),
        indices.view(torch.uint8),
        weights.view(torch.uint8),
        HIDDEN_SIZE,
        PACKET_STRIDE,
    )
    return scales, indices, weights


@requires_corex30
@pytest.mark.parametrize("rows", [1, 24, 96])
def test_packet_roundtrip(rows):
    quantized = torch.randn((rows, HIDDEN_SIZE), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(rows, device="cuda", dtype=torch.float32) + 0.01
    indices, weights = routing(rows)
    packet = torch.empty((rows, PACKET_STRIDE), device="cuda", dtype=torch.uint8)
    _pack_quantized_routes[(rows, triton.cdiv(PACKET_STRIDE, 1024))](
        quantized.view(torch.uint8),
        scales.view(torch.uint8),
        indices.view(torch.uint8),
        weights.view(torch.uint8),
        packet,
        HIDDEN_SIZE,
        PACKET_STRIDE,
        1024,
    )
    actual_scales, actual_indices, actual_weights = unpack(packet)
    assert torch.equal(packet[:, :HIDDEN_SIZE], quantized.view(torch.uint8))
    assert torch.equal(actual_scales, scales)
    assert torch.equal(actual_indices, indices)
    assert torch.equal(actual_weights, weights)


@requires_corex30
def test_fused_quantization_matches_fp32_reference():
    rows = 24
    hidden_states = torch.randn(
        (rows, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16
    )
    hidden_states[0].zero_()
    indices, weights = routing(rows)
    packet = torch.empty((rows, PACKET_STRIDE), device="cuda", dtype=torch.uint8)
    _quantize_and_pack_routes[(rows,)](
        hidden_states,
        indices,
        weights,
        packet,
        HIDDEN_SIZE,
        PACKET_STRIDE,
        8192,
        num_warps=8,
    )
    expected_scales = (hidden_states.float().abs().amax(-1) * (1.0 / 448.0)).clamp_min(
        1.0e-12
    )
    expected_quantized = (
        (hidden_states.float() / expected_scales[:, None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )
    actual_scales, actual_indices, actual_weights = unpack(packet)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=0, atol=0)
    assert torch.equal(packet[:, :HIDDEN_SIZE], expected_quantized.view(torch.uint8))
    assert torch.equal(actual_indices, indices)
    assert torch.equal(actual_weights, weights)


@requires_corex30
@pytest.mark.parametrize("strided", [False, True])
def test_expand_quantized_rows_preserves_routes_and_padding(strided):
    storage = torch.randn((4, PACKET_STRIDE), device="cuda").to(torch.float8_e4m3fn)
    quantized = storage[:, :HIDDEN_SIZE]
    if not strided:
        quantized = quantized.contiguous()
    scales = torch.tensor([0.5, 1.0, 2.0, 4.0], device="cuda", dtype=torch.float32)
    mapping = torch.tensor([17, -1, 0, 23, 7], device="cuda", dtype=torch.int32)
    output, output_scales = expand_prequantized_input(quantized, scales, mapping, TOP_K)
    tokens = mapping.clamp_min(0).long() // TOP_K
    expected = quantized.view(torch.uint8)[tokens]
    expected[mapping < 0] = 0
    expected_scales = torch.where(mapping >= 0, scales[tokens], 1.0)
    assert torch.equal(output.view(torch.uint8), expected)
    assert torch.equal(output_scales, expected_scales)
