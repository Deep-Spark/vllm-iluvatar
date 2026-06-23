# SPDX-License-Identifier: Apache-2.0
"""Iluvatar IR op provider — registers the ``"iluvatar"`` implementations.

Corresponds to: vllm/kernels/vllm_c.py (the community ``"vllm_c"`` provider).

The two community IR ops (vllm/ir/ops/layernorm.py) are backed here by ixformer
kernels directly (no torch.ops._C indirection). Registration happens as an
import side effect; ``IluPlatform.import_ir_kernels`` imports this module so the
provider is registered before ``KernelConfig.ir_op_priority`` is validated.

Provider name is ``"iluvatar"`` (used in priority lists / ``--ir-op-priority``).
Each impl's signature is byte-identical to the IR op it implements, otherwise
``IrOpImpl.__init__`` schema validation rejects it.
"""

from __future__ import annotations

import ixformer.inference.functions as ixf_ops  # type: ignore[import]
import torch
from torch import Tensor
from vllm import ir
from vllm.platforms import current_platform

CUDA_ALIKE = current_platform.is_cuda_alike()
"""ixformer kernels run on the CUDA-alike Iluvatar device."""

# The ixformer kernels (like the vLLM C kernels) require no variance_size
# override and a weight whose dtype matches the input.
_rms_no_var_size = lambda x, weight, epsilon, variance_size=None: (
    variance_size is None and (weight is None or weight.dtype == x.dtype)
)
_rms_add_no_var_size = lambda x, x_residual, weight, epsilon, variance_size=None: (
    variance_size is None and (weight is None or weight.dtype == x.dtype)
)


@ir.ops.rms_norm.register_impl(
    "iluvatar", supports_args=_rms_no_var_size, supported=CUDA_ALIKE
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    if weight is None:
        # Kernel requires a weight tensor; pass ones.
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    assert variance_size is None
    out = torch.empty_like(x)
    ixf_ops.rms_norm(x, weight, epsilon, out)
    return out


@ir.ops.fused_add_rms_norm.register_impl(
    "iluvatar",
    supports_args=_rms_add_no_var_size,
    supported=CUDA_ALIKE,
    inplace=True,
)
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
    # ixformer returns (normed_output, new_residual).
    return ixf_ops.residual_rms_norm(x, weight, epsilon, 1.0, x_residual)
