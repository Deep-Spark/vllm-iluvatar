# SPDX-License-Identifier: Apache-2.0
"""Numerical tests for the fused MRoPE (gather + rotate) kernel."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from conftest import requires_cuda

from vllm_iluvatar.backends.ixaite.layers.mrope import (
    ilu_fused_mrope,
    ilu_fused_mrope_supported,
)


def _community_reference(
    positions,
    query,
    key,
    cos_sin_cache,
    mrope_section,
    head_size,
    rotary_dim,
    mrope_interleaved,
):
    """Community path: gather + chunk + contiguous + triton rotation."""
    from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope

    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    q = query.clone()
    k = key.clone()
    q, k = triton_mrope(
        q,
        k,
        cos,
        sin,
        mrope_section,
        head_size,
        rotary_dim,
        mrope_interleaved,
        is_neox_style=True,
    )
    return q, k


@requires_cuda
@pytest.mark.parametrize(
    "mrope_interleaved,mrope_section", [(True, [11, 11, 10]), (False, [16, 24, 24])]
)
@pytest.mark.parametrize("num_tokens", [1, 7, 128])
@pytest.mark.parametrize("rotary_dim", [64])
def test_fused_mrope_matches_community(
    mrope_interleaved, mrope_section, num_tokens, rotary_dim
):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    head_size = 256
    n_qh, n_kh = 8, 1
    max_pos = 4096

    # Mimic vllm's persistent mrope buffer slice: [3, max_tokens] buffer cut
    # to [:, :num_tokens] — non-contiguous plane stride, unit token stride.
    pos_buf = torch.randint(0, max_pos, (3, max(num_tokens * 2, 8)), device=device)
    positions = pos_buf[:, :num_tokens]
    assert not positions.is_contiguous() or num_tokens * 2 <= 8
    query = torch.randn(num_tokens, n_qh * head_size, device=device, dtype=dtype)
    key = torch.randn(num_tokens, n_kh * head_size, device=device, dtype=dtype)
    cache = torch.randn(max_pos, rotary_dim, device=device, dtype=dtype)

    q_ref, k_ref = _community_reference(
        positions,
        query,
        key,
        cache,
        mrope_section,
        head_size,
        rotary_dim,
        mrope_interleaved,
    )

    q = query.clone()
    k = key.clone()
    assert ilu_fused_mrope_supported(positions, q, k, cache)
    ilu_fused_mrope(
        positions, q, k, cache, mrope_section, head_size, rotary_dim, mrope_interleaved
    )

    torch.testing.assert_close(q, q_ref, rtol=0, atol=0)
    torch.testing.assert_close(k, k_ref, rtol=0, atol=0)


@requires_cuda
def test_fused_mrope_supported_rejects_bad_inputs():
    device = "cuda"
    q = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    cache = torch.randn(128, 64, device=device, dtype=torch.bfloat16)

    pos_1d = torch.randint(0, 128, (4,), device=device)
    assert not ilu_fused_mrope_supported(pos_1d, q, k, cache)

    pos_2d = torch.randint(0, 128, (3, 4), device=device)
    assert ilu_fused_mrope_supported(pos_2d, q, k, cache)

    # Buffer-slice layout (non-contiguous plane stride, unit token stride).
    pos_slice = torch.randint(0, 128, (3, 8), device=device)[:, :4]
    assert not pos_slice.is_contiguous() and pos_slice.stride(1) == 1
    assert ilu_fused_mrope_supported(pos_slice, q, k, cache)

    # Transposed positions cannot enter the owned in-place kernel.
    pos_noncontig = torch.randint(0, 128, (4, 3), device=device).t()
    assert pos_noncontig.shape == (3, 4)
    assert not ilu_fused_mrope_supported(pos_noncontig, q, k, cache)

    cache_fp32 = cache.float()
    assert not ilu_fused_mrope_supported(pos_2d, q, k, cache_fp32)


# Unsupported layouts must leave inputs unchanged and never call the saved community path.
@pytest.mark.parametrize("kind", ["gpt_j", "1d", "no_key", "noncontiguous"])
def test_mrope_patch_rejects_unsupported_layouts_before_mutation(kind):
    from vllm_iluvatar.backends.ixaite.patches import mrope as patch_mrope

    q = torch.zeros(2, 4)
    k = torch.zeros(2, 4)
    positions = torch.zeros(3, 2, dtype=torch.long)
    rotary = SimpleNamespace(
        is_neox_style=kind != "gpt_j",
        _match_cos_sin_cache_dtype=lambda _: torch.zeros(4, 4),
    )
    if kind == "1d":
        positions = positions[0]
    if kind == "noncontiguous":
        q = torch.zeros(4, 2).t()
    with pytest.raises(RuntimeError, match="ixaite MRoPE"):
        patch_mrope._ilu_mrope_forward_cuda(
            rotary, positions, q, None if kind == "no_key" else k
        )
    assert torch.equal(q, torch.zeros_like(q))
    assert torch.equal(k, torch.zeros_like(k))


# The bound method must preserve exact results after changed-input graph replay.
@requires_cuda
@torch.inference_mode()
def test_mrope_patch_capture_uses_owned_kernel():
    from tests.ixaite.backend_isolation.ixformer_guard import forbid_ixformer_calls
    from vllm_iluvatar.backends.ixaite.patches import mrope as patch_mrope

    positions = torch.randint(0, 64, (3, 7), device="cuda")
    query = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    rotary = SimpleNamespace(
        is_neox_style=True,
        _match_cos_sin_cache_dtype=lambda _: cache,
        mrope_section=[16, 8, 8],
        head_size=64,
        rotary_dim=64,
        mrope_interleaved=False,
    )
    with forbid_ixformer_calls():
        patch_mrope._ilu_mrope_forward_cuda(rotary, positions, query, key)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            patch_mrope._ilu_mrope_forward_cuda(rotary, positions, query, key)
        query.normal_()
        key.normal_()
        positions.add_(1)
        expected_q, expected_k = _community_reference(
            positions, query, key, cache, rotary.mrope_section, 64, 64, False
        )
        graph.replay()
        torch.testing.assert_close(query, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(key, expected_k, rtol=0, atol=0)
