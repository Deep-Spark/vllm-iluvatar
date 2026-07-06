# SPDX-License-Identifier: Apache-2.0
"""Iluvatar overrides for ``vllm.model_executor.layers.utils``."""

from __future__ import annotations

import ixformer.inference.functions as ixf_ops
import torch


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    if (
        bias is None
        and x.dtype in (torch.half, torch.bfloat16)
        and weight.dtype == torch.float32
    ):
        return ixf_ops.mixed_type_linear(input=x, weight=layer.weight)
    if x.dtype == torch.float32:
        return torch.nn.functional.linear(x, weight, bias)
    return ixf_ops.linear(x, weight, bias)
