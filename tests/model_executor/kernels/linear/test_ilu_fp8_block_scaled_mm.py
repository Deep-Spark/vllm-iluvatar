# SPDX-License-Identifier: Apache-2.0
"""On-device numeric tests for the Iluvatar FP8 block-scaled dense kernel.

Gated by ``requires_cuda`` + the ``cuda_device`` fixture (design doc §13.2): the
fp8 GEMM genuinely needs an Iluvatar device. Compares
``IluFp8BlockScaledMMKernel.apply_block_scaled_mm`` against an independent fp32
per-block dequant reference at DeepSeek-V4-like dense shapes, and checks the
on-device ``is_supported`` gate (arch forced to ivcore30) so the kernel is the
selected head of ``_POSSIBLE_FP8_BLOCK_KERNELS[CUDA]``.
"""

from __future__ import annotations

import types

import pytest
import torch
from conftest import requires_cuda

from vllm_iluvatar.model_executor.kernels.linear.scaled_mm import (
    ilu_fp8_block_scaled_mm as mod,
)
from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (  # noqa: E501
    IluFp8BlockScaledMMKernel,
)


def _make_kernel(gn: int, gk: int, out_dtype: torch.dtype) -> IluFp8BlockScaledMMKernel:
    k = IluFp8BlockScaledMMKernel.__new__(IluFp8BlockScaledMMKernel)
    k.weight_group_shape = (gn, gk)
    k.config = types.SimpleNamespace(out_dtype=out_dtype)
    # Exercise the full dequant-both (fp8 activation + As) math so this test
    # still validates the per-block scale broadcasting. The runtime default is
    # apply_input_quant=False (bf16 activation, As ignored) because Iluvatar has
    # no per_token_group_fp8_quant op; that path is covered by the serve smoke.
    k.apply_input_quant = True
    return k


def _ref_block_mm(A, B, As, Bs, gn, gk, out_dtype):
    a = A.to(torch.float32)
    b = B.to(torch.float32)
    As = As.to(torch.float32)
    Bs = Bs.to(torch.float32)
    M, K = a.shape
    N = b.shape[0]
    C = torch.zeros(M, N, dtype=torch.float32, device=a.device)
    k_tiles = (K + gk - 1) // gk
    n_tiles = (N + gn - 1) // gn
    for i in range(k_tiles):
        ks, ke = i * gk, min((i + 1) * gk, K)
        a_t = a[:, ks:ke]
        as_t = As[:, i : i + 1]
        for j in range(n_tiles):
            ns, ne = j * gn, min((j + 1) * gn, N)
            b_t = b[ns:ne, ks:ke]
            s = as_t * Bs[j, i]
            C[:, ns:ne] += torch.matmul(a_t, b_t.t()) * s
    return C.to(out_dtype)


@requires_cuda
@pytest.mark.parametrize(
    "M,N,K",
    [
        (16, 512, 7168),   # wo_b-like: K = DeepSeek-V4 hidden
        (8, 2048, 512),    # gate / shared-expert-like
    ],
)
def test_apply_block_scaled_mm_matches_reference(cuda_device, M, N, K):
    torch.manual_seed(0)
    gn = gk = 128
    k_tiles = K // gk
    n_tiles = N // gn

    A = (torch.randn(M, K, device=cuda_device) * 0.3).to(torch.float8_e4m3fn)
    B = (torch.randn(N, K, device=cuda_device) * 0.3).to(torch.float8_e4m3fn)
    As = torch.rand(M, k_tiles, device=cuda_device, dtype=torch.float32) * 0.5 + 0.1
    Bs = (
        torch.rand(n_tiles, k_tiles, device=cuda_device, dtype=torch.float32) * 0.5
        + 0.1
    )

    kernel = _make_kernel(gn, gk, torch.bfloat16)
    out = kernel.apply_block_scaled_mm(A, B, As, Bs)
    ref = _ref_block_mm(A, B, As, Bs, gn, gk, torch.bfloat16)

    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    # bf16 output: loose tolerance dominated by bf16 rounding, not the math.
    torch.testing.assert_close(
        out.to(torch.float32), ref.to(torch.float32), rtol=2e-2, atol=2e-2
    )


@requires_cuda
def test_is_supported_true_on_device(cuda_device, monkeypatch):
    # On a real Iluvatar device is_cuda_alike() is True; force the arch predicate
    # so the assertion does not depend on the concrete ixformer build.
    monkeypatch.setattr(mod, "_is_ivcore30_fn", lambda: True)
    ok, reason = IluFp8BlockScaledMMKernel.is_supported()
    assert ok is True
    assert reason is None
