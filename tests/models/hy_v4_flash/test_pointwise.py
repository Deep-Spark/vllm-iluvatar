# SPDX-License-Identifier: Apache-2.0
"""Flash fused residual normalization and draft output writes."""

from types import SimpleNamespace

import pytest
import torch
import triton

from tests.conftest import requires_cuda
from vllm_iluvatar.models.hy_v4_flash.dfly import (
    _pack_kv,
    _update_anchor,
    _write_draft_tokens,
)
from vllm_iluvatar.models.hy_v4_flash.norm import sum_rms_norm


@requires_cuda
@pytest.mark.parametrize("keep_sum", [False, True])
@torch.inference_mode()
def test_sum_rms_norm(keep_sum):
    torch.manual_seed(42)
    x = torch.randn(7, 3072, device="cuda", dtype=torch.bfloat16)
    res = torch.randn_like(x)
    weight = torch.rand_like(x[0])
    norm = SimpleNamespace(weight=weight, variance_epsilon=1e-6)
    output, summed = sum_rms_norm(x, res, norm, keep_sum=keep_sum)
    reference_sum = x + res
    ref = reference_sum.float()
    ref = (ref * (ref.square().mean(-1, keepdim=True) + 1e-6).rsqrt()) * weight
    torch.testing.assert_close(output.float(), ref, atol=0.015, rtol=0.005)
    if keep_sum:
        torch.testing.assert_close(summed, reference_sum, atol=0, rtol=0)
    else:
        assert summed is None


@requires_cuda
@torch.inference_mode()
def test_draft_anchor_and_strided_output():
    torch.manual_seed(42)
    rows, width = 129, 3072
    embeds = torch.randn(rows * 6, width, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(embeds[::6], memory_format=torch.contiguous_format)
    gate = torch.randn_like(embeds[0])
    expected = embeds.clone()
    expected[::6] = embeds[::6] + gate * v
    _update_anchor[(12, rows)](embeds, v, gate, width, 256, enable_fp_fusion=False)
    torch.testing.assert_close(embeds, expected, atol=0, rtol=0)

    tokens = torch.arange(rows, device="cuda", dtype=torch.int64)
    active = tokens.remainder(3) == 0
    out = torch.full((rows, 5), -7, device="cuda", dtype=torch.int64)
    expected = out.clone()
    expected[:, 2] = torch.where(active, tokens, -1)
    _write_draft_tokens[(2,)](tokens, active, out[:, 2], rows, out.stride(0))
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@requires_cuda
@pytest.mark.parametrize("fp8", [False, True])
@torch.inference_mode()
def test_draft_history_tiles(fp8):
    torch.manual_seed(42)
    row_width, width, tile_rows = (640, 2112, 4) if fp8 else (576, 2054, 1)
    cache = torch.randint(256, (4096, row_width), device="cuda", dtype=torch.uint8)
    if not fp8:
        cache = cache.to(torch.bfloat16)
    snapshot = torch.stack(
        [torch.randperm(4096, device="cuda")[:2048] for _ in range(3)]
    )
    snapshot[1, 1] = -1
    lengths = torch.tensor([7, 3, 2048], device="cuda", dtype=torch.int32)
    sources = torch.tensor([1, 0, 2], device="cuda", dtype=torch.int64)
    active = torch.tensor([True, False, True], device="cuda")
    new = cache[:18].clone()
    actual = torch.full((3, width, row_width), 17, device="cuda", dtype=cache.dtype)
    expected = actual.clone()

    for dst, src, length in ((0, 1, 3), (2, 2, 2048)):
        slots = snapshot[src, :length]
        expected[dst, :length] = cache[slots.clamp_min(0)]
        expected[dst, :length][slots < 0] = 0
        if not fp8:
            expected[dst, length : length + 6] = new[dst * 6 : (dst + 1) * 6]

    staged_rows = 2048 if fp8 else 2054
    _pack_kv[(triton.cdiv(staged_rows, tile_rows), 3)](
        cache,
        snapshot,
        lengths,
        sources,
        active,
        new,
        actual,
        width,
        row_width,
        fp8,
        tile_rows,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
