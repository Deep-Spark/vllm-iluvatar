# SPDX-License-Identifier: Apache-2.0
"""Dense geometry and numerical checks for the independent ixaite GEMM."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.ixaite.backend_isolation.ixformer_guard import forbid_ixformer_calls
from vllm_iluvatar.backends.ixaite import ops as ilu_ops
from vllm_iluvatar.backends.ixaite.layers import utils as layer_utils


# Preserve the existing reference and tolerance for this model geometry.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_qkv_proj_ixaite_wrapper_matches_torch(monkeypatch):
    torch.manual_seed(3)
    x = torch.randn((8, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((10240, 4096), device="cuda", dtype=torch.bfloat16)

    actual = layer_utils.default_unquantized_gemm(object(), x, weight, bias=None)
    expected = F.linear(x, weight, None)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


# Preserve the existing reference and tolerance for this model geometry.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_qwen3_8b_fused_qkv_matches_split_linears(monkeypatch):
    """Packed QKV (6144,4096) equals concat of q/k/v GEMMs on the ixaite path."""
    torch.manual_seed(7)
    hidden = 4096
    n_q, n_kv = 32 * 128, 8 * 128
    x = torch.randn((4, hidden), device="cuda", dtype=torch.bfloat16)
    wq = torch.randn((n_q, hidden), device="cuda", dtype=torch.bfloat16)
    wk = torch.randn((n_kv, hidden), device="cuda", dtype=torch.bfloat16)
    wv = torch.randn((n_kv, hidden), device="cuda", dtype=torch.bfloat16)
    fused_w = torch.cat([wq, wk, wv], dim=0)

    fused = layer_utils.default_unquantized_gemm(object(), x, fused_w, bias=None)
    split = torch.cat(
        [
            F.linear(x, wq, None),
            F.linear(x, wk, None),
            F.linear(x, wv, None),
        ],
        dim=-1,
    )
    torch.testing.assert_close(fused, split, rtol=2e-2, atol=2e-2)
    assert fused.shape[-1] == 6144


# Preserve the existing reference and tolerance for this model geometry.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_mlp_gate_up_ixaite_wrapper_matches_torch(monkeypatch):
    torch.manual_seed(3)
    x = torch.randn((8, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((24576, 4096), device="cuda", dtype=torch.bfloat16)

    actual = layer_utils.default_unquantized_gemm(object(), x, weight, bias=None)
    expected = F.linear(x, weight, None)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


# Preserve the existing reference and tolerance for this model geometry.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_lm_head_ixaite_wrapper_matches_torch(monkeypatch):
    torch.manual_seed(3)
    x = torch.randn((8, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((248320, 4096), device="cuda", dtype=torch.bfloat16)

    actual = layer_utils.default_unquantized_gemm(object(), x, weight, bias=None)
    expected = F.linear(x, weight, None)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


# Preserve the existing reference and tolerance for this model geometry.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_mlp_down_ixaite_wrapper_matches_torch(monkeypatch):
    torch.manual_seed(3)
    x = torch.randn((8, 12288), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((4096, 12288), device="cuda", dtype=torch.bfloat16)

    actual = layer_utils.default_unquantized_gemm(object(), x, weight, bias=None)
    expected = F.linear(x, weight, None)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


# Preserve the existing reference and tolerance for this model geometry.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize(
    ("n", "k"),
    (
        (8192, 5120),
        (4096, 5120),
        (48, 5120),
        (24, 5120),
        (5120, 3072),
        (5120, 1536),
        (124160, 5120),
        (62080, 5120),
        (7168, 5120),
        (3584, 5120),
        (17408, 5120),
        (8704, 5120),
        (5120, 8704),
        (5120, 4352),
        (5120, 2304),
        (5120, 1152),
    ),
)
def test_qwen36_27b_linear_ixaite_wrapper_matches_torch(
    monkeypatch, n: int, k: int
) -> None:
    torch.manual_seed(3)
    x = torch.randn((8, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    actual = layer_utils.default_unquantized_gemm(object(), x, weight, bias=None)
    expected = F.linear(x, weight, None)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


# The scalar shared-expert gate needs padding without ever selecting a vendor GEMM.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_linear_ixaite_routes_shared_expert_gate():
    torch.manual_seed(17)
    x = torch.randn((8, 2048), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((1, 2048), device="cuda", dtype=torch.bfloat16)
    with forbid_ixformer_calls():
        actual = layer_utils.default_unquantized_gemm(object(), x, weight)
    torch.testing.assert_close(actual, F.linear(x, weight), rtol=2e-2, atol=2e-2)


# Unknown and previously vendor-pinned shapes now fail before a kernel writes output.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize("n,k", [(8192, 1024), (538, 1152), (8192, 1056)])
def test_unsupported_linear_does_not_fall_back(n, k):
    x = torch.zeros((1, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros((n, k), device="cuda", dtype=torch.bfloat16)
    with forbid_ixformer_calls(), pytest.raises(ValueError):
        ilu_ops.linear(x, weight)
