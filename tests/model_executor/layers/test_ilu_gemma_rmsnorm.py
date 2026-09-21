# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness of ``IluGemmaRMSNorm.forward_native``.

Community ``GemmaRMSNorm.forward_cuda`` only reaches the ixformer 2D kernel for
2D inputs; anything else falls back to ``forward_native``, so the native path is
a live production path and needs its own coverage against the ixformer golden.
"""

from __future__ import annotations

import ixformer.inference.functions as ixf_ops
import pytest
import torch

from vllm_iluvatar.model_executor.layers.layernorm import IluGemmaRMSNorm

DTYPES = [torch.float16, torch.bfloat16]
TOLERANCE = {
    torch.float16: dict(atol=5e-3, rtol=5e-3),
    torch.bfloat16: dict(atol=5e-2, rtol=3e-2),
}
SHAPES = [(7, 512), (2, 5, 512), (7, 8, 128)]


def _make_layer(hidden_size: int, dtype: torch.dtype, device: str) -> IluGemmaRMSNorm:
    layer = IluGemmaRMSNorm(hidden_size, eps=1e-6).to(device=device, dtype=dtype)
    layer.weight.data = torch.randn(hidden_size, dtype=dtype, device=device) * 0.1
    return layer


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_forward_native_with_residual_matches_reference(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    cuda_device: str,
    default_vllm_config,
) -> None:
    torch.manual_seed(42)
    layer = _make_layer(shape[-1], dtype, cuda_device)
    x = torch.randn(*shape, dtype=dtype, device=cuda_device)
    residual = torch.randn_like(x)

    out, out_residual = layer.forward_native(x.clone(), residual.clone())
    ref, ref_residual = ixf_ops.ref_gemma_rms_norm(
        x.clone(), layer.weight.data, layer.variance_epsilon, residual.clone()
    )

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref.float(), **tol)
    torch.testing.assert_close(out_residual.float(), ref_residual.float(), **tol)
    # The residual chain feeds the next decoder layer, so it must keep the
    # caller's dtype rather than leaking the internal fp32 accumulator.
    assert out.dtype == dtype
    assert out_residual.dtype == dtype


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_forward_native_without_residual_matches_reference(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    cuda_device: str,
    default_vllm_config,
) -> None:
    torch.manual_seed(42)
    layer = _make_layer(shape[-1], dtype, cuda_device)
    x = torch.randn(*shape, dtype=dtype, device=cuda_device)

    out = layer.forward_native(x.clone(), None)
    ref = ixf_ops.ref_gemma_rms_norm(
        x.clone(), layer.weight.data, layer.variance_epsilon, None
    )

    torch.testing.assert_close(out.float(), ref.float(), **TOLERANCE[dtype])
    assert out.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_forward_cuda_non_2d_fallback_matches_reference(
    dtype: torch.dtype,
    cuda_device: str,
    default_vllm_config,
) -> None:
    """forward_cuda delegates non-2D inputs to forward_native."""
    torch.manual_seed(42)
    shape = (2, 5, 512)
    layer = _make_layer(shape[-1], dtype, cuda_device)
    x = torch.randn(*shape, dtype=dtype, device=cuda_device)
    residual = torch.randn_like(x)

    out, out_residual = layer.forward_cuda(x.clone(), residual.clone())
    ref, ref_residual = ixf_ops.ref_gemma_rms_norm(
        x.clone(), layer.weight.data, layer.variance_epsilon, residual.clone()
    )

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref.float(), **tol)
    torch.testing.assert_close(out_residual.float(), ref_residual.float(), **tol)
