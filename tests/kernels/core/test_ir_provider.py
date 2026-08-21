# SPDX-License-Identifier: Apache-2.0
"""Iluvatar IR provider: registration, priority, and numerics.

Covers the phase-2 ``"iluvatar"`` IR op provider (vllm_iluvatar/kernels/ilu_ops.py)
plus the IluPlatform overrides (import_ir_kernels / get_default_ir_op_priority):

- registration: importing the package registers "iluvatar" for both IR ops.
- priority: IluPlatform.get_default_ir_op_priority puts "iluvatar" first in
  eager mode and falls back to "native" under inductor compilation.
- numerics: the iluvatar impl matches the IR native reference (needs GPU + ixformer).

ixformer / vllm are mandatory deps to import the provider: if either is missing
the module fails to import (a hard error, not a skip). Only the numeric test is
additionally gated on an actual CUDA device being present.
"""

from __future__ import annotations

from types import SimpleNamespace

# ixformer + vllm are MANDATORY deps of this build: a missing one is an
# environment error (collection-time ImportError), never a silent skip.
import ixformer  # noqa: F401
import pytest
import torch
from vllm import ir

# Registers the "iluvatar" provider for both IR ops as an import side effect.
import vllm_iluvatar.kernels  # noqa: F401


def test_provider_registered() -> None:
    assert "iluvatar" in ir.ops.rms_norm.impls
    assert "iluvatar" in ir.ops.fused_add_rms_norm.impls


def _priority(*, inductor: bool):
    from vllm.config.compilation import CompilationMode

    from vllm_iluvatar.platforms.ilu_platform import IluPlatform

    cc = SimpleNamespace(
        backend="inductor" if inductor else "eager",
        mode=CompilationMode.VLLM_COMPILE if inductor else CompilationMode.NONE,
    )
    vllm_config = SimpleNamespace(compilation_config=cc)
    return IluPlatform.get_default_ir_op_priority(vllm_config)


def test_priority_eager_prefers_iluvatar() -> None:
    prio = _priority(inductor=False)
    assert prio.rms_norm[0] == "iluvatar"
    assert prio.fused_add_rms_norm[0] == "iluvatar"
    # rms_norm keeps vllm_c as a fallback; fused_add_rms_norm does not.
    assert "vllm_c" in prio.rms_norm
    assert "vllm_c" not in prio.fused_add_rms_norm
    assert prio.rms_norm[-1] == "native"
    assert prio.fused_add_rms_norm[-1] == "native"


def test_priority_inductor_uses_native() -> None:
    prio = _priority(inductor=True)
    assert prio.rms_norm == ["native"]
    assert prio.fused_add_rms_norm == ["native"]


# --- numeric correctness (requires Iluvatar/CUDA device) --------------------

if not torch.cuda.is_available():
    pytest.skip("Iluvatar/CUDA device required", allow_module_level=True)

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]
HIDDEN_SIZES = [768, 5120]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_rms_norm_matches_native(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(0)
    device = "cuda:0"

    weight = torch.empty(hidden_size, dtype=dtype, device=device).normal_(
        mean=1.0, std=0.1
    )
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device) * (
        1 / (2 * hidden_size)
    )
    epsilon = 1e-6

    ilu = ir.ops.rms_norm.impls["iluvatar"].impl_fn(x, weight, epsilon)
    ref = ir.ops.rms_norm.impls["native"].impl_fn(x, weight, epsilon)
    torch.testing.assert_close(ilu, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_fused_add_rms_norm_matches_native(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(0)
    device = "cuda:0"

    weight = torch.empty(hidden_size, dtype=dtype, device=device).normal_(
        mean=1.0, std=0.1
    )
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device) * (
        1 / (2 * hidden_size)
    )
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device) * (
        1 / hidden_size
    )
    epsilon = 1e-6

    # The iluvatar impl is in-place (inplace=True), so clone per call.
    ilu_out, ilu_res = ir.ops.fused_add_rms_norm.impls["iluvatar"].impl_fn(
        x.clone(), residual.clone(), weight, epsilon
    )
    ref_out, ref_res = ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
        x.clone(), residual.clone(), weight, epsilon
    )
    torch.testing.assert_close(ilu_out, ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ilu_res, ref_res, atol=1e-2, rtol=1e-2)
