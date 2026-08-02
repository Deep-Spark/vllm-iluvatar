# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar FP8 block-scaled dense linear kernel.

Covers the pure-Python / pure-torch mechanisms (runnable
without a GPU):

* the general patch inserts ``IluFp8BlockScaledMMKernel`` at the head of
  ``_POSSIBLE_FP8_BLOCK_KERNELS[CUDA]`` and is idempotent + arch-gated
  (corex30 / ivcore30 only);
* ``is_supported`` gates on the module-level ``_is_ivcore30_fn`` predicate;
* ``_apply_block_scaled_mm_torch`` (the retained pure-torch block reference, no
  longer on the runtime path) reproduces the block dequant + matmul reference
  (K/N group indexing, per-token/per-block scale broadcast, e8m0 decode and the
  non-divisible tail-block edge).

fp8 dequant runs via ``.to(torch.float32)`` which is available on CPU.
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm_iluvatar.model_executor.kernels.linear.scaled_mm import (
    ilu_fp8_block_scaled_mm as mod,
)
from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (  # noqa: E501
    IluFp8BlockScaledMMKernel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kernel(
    gn: int,
    gk: int,
    out_dtype: torch.dtype,
    apply_input_quant: bool = True,
) -> IluFp8BlockScaledMMKernel:
    """Build a kernel without running ``__init__`` (which needs a full config).

    ``_apply_block_scaled_mm_torch`` only reads ``weight_group_shape``,
    ``config.out_dtype`` and ``apply_input_quant``. The block reference tests here
    pass a per-block activation scale ``As``, so ``apply_input_quant`` defaults to
    True to exercise the full dequant-both math.
    """
    k = IluFp8BlockScaledMMKernel.__new__(IluFp8BlockScaledMMKernel)
    k.weight_group_shape = (gn, gk)
    k.config = types.SimpleNamespace(out_dtype=out_dtype)
    k.apply_input_quant = apply_input_quant
    return k


def _ref_block_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    gn: int,
    gk: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Independent per-block dequant+accumulate reference (nested tiles).

    Mirrors ``tests/kernels/quant_utils.py::native_w8a8_block_matmul`` and is
    intentionally written differently from the kernel's vectorized broadcast so
    it validates the group-indexing direction rather than duplicating it.
    """
    def _e8m0(scale: torch.Tensor) -> torch.Tensor:
        exp_bits = scale.view(torch.uint8).to(torch.int32)
        return (exp_bits << 23).view(torch.float32)

    if As.dtype == torch.float8_e8m0fnu:
        As = _e8m0(As)
    if Bs.dtype == torch.float8_e8m0fnu:
        Bs = _e8m0(Bs)
    a = A.to(torch.float32)
    b = B.to(torch.float32)
    As = As.to(torch.float32)
    Bs = Bs.to(torch.float32)
    M, K = a.shape
    N = b.shape[0]
    C = torch.zeros(M, N, dtype=torch.float32)
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


def _quant_fp8(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn)


def _cuda_list():
    import vllm.model_executor.kernels.linear as k
    from vllm.platforms import PlatformEnum

    return k._POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.CUDA]


# ---------------------------------------------------------------------------
# Registration / arch gate
# ---------------------------------------------------------------------------

def test_apply_inserts_at_head_and_idempotent(monkeypatch):
    import vllm_iluvatar.ilu_util.arch as arch_mod
    from vllm_iluvatar.ilu_patch.general_patches import (
        patch_fp8_block_scaled_kernel as patch,
    )

    monkeypatch.setattr(arch_mod, "is_ivcore30", lambda: True)

    cuda = _cuda_list()
    orig = list(cuda)
    try:
        while IluFp8BlockScaledMMKernel in cuda:
            cuda.remove(IluFp8BlockScaledMMKernel)

        patch._apply()
        assert cuda[0] is IluFp8BlockScaledMMKernel

        # Idempotent: a second _apply must not add a duplicate.
        patch._apply()
        assert cuda.count(IluFp8BlockScaledMMKernel) == 1
    finally:
        cuda[:] = orig


def test_apply_gated_off_on_non_ivcore30(monkeypatch):
    import vllm_iluvatar.ilu_util.arch as arch_mod
    from vllm_iluvatar.ilu_patch.general_patches import (
        patch_fp8_block_scaled_kernel as patch,
    )

    monkeypatch.setattr(arch_mod, "is_ivcore30", lambda: False)

    cuda = _cuda_list()
    orig = list(cuda)
    try:
        while IluFp8BlockScaledMMKernel in cuda:
            cuda.remove(IluFp8BlockScaledMMKernel)

        patch._apply()
        assert IluFp8BlockScaledMMKernel not in cuda
    finally:
        cuda[:] = orig


def test_register_applies_once(monkeypatch):
    import vllm_iluvatar.ilu_util.arch as arch_mod
    from vllm_iluvatar.ilu_patch import general_patch as gp
    from vllm_iluvatar.ilu_patch.general_patches import (
        patch_fp8_block_scaled_kernel as patch,
    )

    monkeypatch.setattr(arch_mod, "is_ivcore30", lambda: True)

    cuda = _cuda_list()
    orig = list(cuda)
    # register_once dedupes by module name; clear it so this test is hermetic.
    monkeypatch.setattr(gp, "_APPLIED", set(gp._APPLIED))
    gp._APPLIED.discard(patch.__name__)
    try:
        while IluFp8BlockScaledMMKernel in cuda:
            cuda.remove(IluFp8BlockScaledMMKernel)

        patch.register()
        assert cuda[0] is IluFp8BlockScaledMMKernel
        assert patch.__name__ in gp._APPLIED
    finally:
        cuda[:] = orig


# ---------------------------------------------------------------------------
# is_supported arch gate
# ---------------------------------------------------------------------------

def test_is_supported_true_on_ivcore30(monkeypatch):
    monkeypatch.setattr(mod, "_is_ivcore30_fn", lambda: True)
    ok, reason = IluFp8BlockScaledMMKernel.is_supported(90)
    assert ok is True
    assert reason is None


def test_is_supported_false_off_ivcore30(monkeypatch):
    monkeypatch.setattr(mod, "_is_ivcore30_fn", lambda: False)
    ok, reason = IluFp8BlockScaledMMKernel.is_supported(90)
    assert ok is False
    assert "ivcore30" in reason


# ---------------------------------------------------------------------------
# torch dequant correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "M,N,K,gn,gk",
    [
        (4, 256, 256, 128, 128),   # clean 128x128 blocks, multiple tiles
        (3, 128, 128, 128, 128),   # single block
        (5, 130, 200, 128, 128),   # non-divisible N and K (tail blocks)
    ],
)
def test_dequant_matches_reference(M, N, K, gn, gk):
    torch.manual_seed(0)
    k_tiles = (K + gk - 1) // gk
    n_tiles = (N + gn - 1) // gn

    A = _quant_fp8(torch.randn(M, K) * 0.3)
    B = _quant_fp8(torch.randn(N, K) * 0.3)
    As = torch.rand(M, k_tiles, dtype=torch.float32) * 0.5 + 0.1
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * 0.5 + 0.1

    kernel = _make_kernel(gn, gk, torch.float32)
    out = kernel._apply_block_scaled_mm_torch(A, B, As, Bs)
    ref = _ref_block_mm(A, B, As, Bs, gn, gk, torch.float32)

    assert out.shape == (M, N)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)


def test_dequant_e8m0_weight_scale():
    torch.manual_seed(1)
    M, N, K, gn, gk = 4, 128, 128, 128, 128
    A = _quant_fp8(torch.randn(M, K) * 0.3)
    B = _quant_fp8(torch.randn(N, K) * 0.3)
    As = torch.rand(M, 1, dtype=torch.float32) * 0.5 + 0.1
    # e8m0: exponent-only (power-of-two) scale stored as uint8 exponent bits.
    exps = torch.randint(120, 134, (1, 1), dtype=torch.uint8)
    Bs = exps.view(torch.float8_e8m0fnu)

    kernel = _make_kernel(gn, gk, torch.float32)
    out = kernel._apply_block_scaled_mm_torch(A, B, As, Bs)
    ref = _ref_block_mm(A, B, As, Bs, gn, gk, torch.float32)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)


def test_returns_configured_out_dtype():
    M, N, K, gn, gk = 2, 128, 128, 128, 128
    A = _quant_fp8(torch.randn(M, K) * 0.3)
    B = _quant_fp8(torch.randn(N, K) * 0.3)
    As = torch.rand(M, 1, dtype=torch.float32) * 0.5 + 0.1
    Bs = torch.rand(1, 1, dtype=torch.float32) * 0.5 + 0.1

    kernel = _make_kernel(gn, gk, torch.bfloat16)
    out = kernel._apply_block_scaled_mm_torch(A, B, As, Bs)
    assert out.dtype == torch.bfloat16
