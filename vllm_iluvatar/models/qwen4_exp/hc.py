# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection kernels for Qwen4Exp.

The production path is exclusively ixformer CUDA on CoreX30. Reference
implementations belong in tests, not in this runtime adapter.
"""

from __future__ import annotations

import torch
from ixformer.inference.functions.qwen38 import hc as hc_ops


def _require_cuda(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise RuntimeError("Qwen4Exp HyperConnection requires ixformer CUDA tensors")


def _as_2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    if x.stride(-1) != 1:
        raise ValueError("hc input must have last-dim stride 1")
    if x.dim() == 2:
        return x, x.shape
    return x.reshape(-1, x.shape[-1]), x.shape


def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    _require_cuda(x)
    if not weight.is_contiguous():
        raise ValueError("hc weight must be contiguous")
    x2d, shape = _as_2d(x)
    scratch = torch.empty_like(x2d)
    xn = torch.empty_like(x2d)
    hc_ops.qwen_hc_combine_norm(
        x2d,
        None,
        None,
        None,
        weight,
        scratch,
        xn,
        eps,
        num_groups,
        backend="ixformer_native",
    )
    return xn.reshape(shape)


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    _require_cuda(x)
    x2d, shape = _as_2d(x)
    out = torch.empty_like(x2d)
    hc_ops.qwen_hc_silu(x2d, out, hc_count, backend="ixformer_native")
    return out.reshape(shape)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    _require_cuda(x)
    x2d, x_shape = _as_2d(x)
    g2d, _ = _as_2d(gate)
    dim = x2d.shape[-1] // hc_count
    out = torch.empty(x2d.shape[0], dim, dtype=x.dtype, device=x.device)
    hc_ops.qwen_hc_gate_mix(x2d, g2d, out, hc_count, backend="ixformer_native")
    return out.reshape(*x_shape[:-1], dim)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    _require_cuda(residual)
    r2d, shape = _as_2d(residual)
    b2d, _ = _as_2d(block_output)
    i2d, _ = _as_2d(injection_logits)
    hidden = torch.empty_like(r2d)
    hc_ops.qwen_hc_combine(r2d, b2d, i2d, hidden, hc_count, backend="ixformer_native")
    return hidden.reshape(shape)


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cuda(residual)
    if not norm_weight.is_contiguous():
        raise ValueError("hc weight must be contiguous")
    r2d, shape = _as_2d(residual)
    b2d, _ = _as_2d(block_output)
    i2d, _ = _as_2d(injection_logits)
    hidden = torch.empty_like(r2d)
    xn = torch.empty_like(r2d)
    hc_ops.qwen_hc_combine_norm(
        r2d,
        b2d,
        i2d,
        None,
        norm_weight,
        hidden,
        xn,
        eps,
        hc_count,
        backend="ixformer_native",
    )
    return hidden.reshape(shape), xn.reshape(shape)


__all__ = [
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_combine_norm",
    "hc_gate_mix",
    "hc_silu",
]
