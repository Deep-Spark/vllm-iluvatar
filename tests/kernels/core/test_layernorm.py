# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness of torch.ops._C.rms_norm (ixformer-backed).

Mirrors the rms_norm portion of upstream tests/kernels/core/test_layernorm.py,
but exercises the plugin's ``torch.ops._C.rms_norm`` directly (the operator the
reverted community vllm/kernels/vllm_c.py calls) and checks it against a plain
PyTorch reference. The native build is a mandatory dep (missing -> hard import
error); only the actual GPU device is gated via skip.

The plugin does NOT implement fused_add_rms_norm (the residual path is handled
by the IluRMSNorm OOT override via vllm._custom_ops), so only the no-residual
rms_norm op is covered here.
"""

from __future__ import annotations

import pytest

# Native extension is a MANDATORY dep (importing it registers
# torch.ops._C.rms_norm): a missing build is an environment error
# (collection-time ImportError), never a silent skip.
import torch
import torch.library
import vllm_iluvatar._C  # noqa: F401

if not torch.cuda.is_available():
    pytest.skip("Iluvatar/CUDA device required", allow_module_level=True)

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]
HIDDEN_SIZES = [8, 768, 769, 5120, 5125, 8192]
SEEDS = [0]


def _ref_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Reference matching vLLM RMSNorm.forward_native (no residual)."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(orig_dtype)
    return x * weight


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    strided_input: bool,
) -> None:
    torch.manual_seed(seed)
    device = "cuda:0"

    weight = torch.empty(hidden_size, dtype=dtype, device=device).normal_(
        mean=1.0, std=0.1
    )
    epsilon = 1e-6
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype, device=device)
    x = x[..., :hidden_size]
    assert x.is_contiguous() != strided_input
    x *= scale

    ref_out = _ref_rms_norm(x, weight, epsilon)

    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    torch.ops._C.rms_norm(out, x, weight, epsilon)

    # RMS/LayerNorm reductions carry larger numerical error; use a loose tol.
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_rms_norm_opcheck(dtype: torch.dtype) -> None:
    """torch.library.opcheck: schema/alias/fake consistency for rms_norm."""
    opcheck = torch.library.opcheck

    device = "cuda:0"
    hidden_size = 768
    weight = torch.empty(hidden_size, dtype=dtype, device=device).normal_(
        mean=1.0, std=0.1
    )
    x = torch.randn(83, hidden_size, dtype=dtype, device=device) * (
        1 / (2 * hidden_size)
    )
    out = torch.empty_like(x)
    opcheck(torch.ops._C.rms_norm, (out, x, weight, 1e-6))
