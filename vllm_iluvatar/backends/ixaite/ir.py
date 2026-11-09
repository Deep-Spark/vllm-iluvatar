# Source: kernels/ilu_ops.py at 47fb804; ixaite IR cannot enter native ixformer.
# SPDX-License-Identifier: Apache-2.0
"""Register the ixaite IR implementations before kernel priority validation."""

from __future__ import annotations

import torch
from torch import Tensor
from vllm import ir
from vllm.platforms import current_platform

from vllm_iluvatar.backends.ixaite import ops

CUDA_ALIKE = current_platform.is_cuda_alike()

# These kernels normalize the full width with weights matching the input dtype.
_rms_no_var_size = lambda x, weight, epsilon, variance_size=None: (
    variance_size is None and (weight is None or weight.dtype == x.dtype)
)
_rms_add_no_var_size = lambda x, x_residual, weight, epsilon, variance_size=None: (
    variance_size is None and (weight is None or weight.dtype == x.dtype)
)


@ir.ops.rms_norm.register_impl(
    "ixaite", supports_args=_rms_no_var_size, supported=CUDA_ALIKE
)
# Match the community schema so compiled consumers use the same selected ABI.
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    if weight is None:
        # Kernel requires a weight tensor; pass ones.
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    assert variance_size is None
    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, epsilon)
    return out


@ir.ops.fused_add_rms_norm.register_impl(
    "ixaite",
    supports_args=_rms_add_no_var_size,
    supported=CUDA_ALIKE,
    inplace=True,
)
# Retain both aliased outputs required by the community residual contract.
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    if weight is None:
        # Kernel requires a weight tensor; pass ones.
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    assert variance_size is None
    return ops.fused_add_rms_norm(x, x_residual, weight, epsilon)
