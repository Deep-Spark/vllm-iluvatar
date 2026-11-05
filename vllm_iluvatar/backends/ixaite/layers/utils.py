# Source: layers/utils.py at 47fb804; bind ixaite before copied GEMM references.
# SPDX-License-Identifier: Apache-2.0
"""Iluvatar overrides for ``vllm.model_executor.layers.utils``."""

from __future__ import annotations

import torch

from vllm_iluvatar.backends.ixaite import ops as ilu_ops


def process_unquantized_weights_after_loading(
    self: object,
    layer: torch.nn.Module,
) -> None:
    from vllm.model_executor.parameter import ModelWeightParameter

    weight = getattr(layer, "weight", None)
    if not isinstance(weight, ModelWeightParameter):
        return

    new_weight = torch.nn.Parameter(
        weight.as_subclass(torch.Tensor),
        requires_grad=weight.requires_grad,
    )
    new_weight.__dict__.update(getattr(weight, "__dict__", {}))
    layer.register_parameter("weight", new_weight)


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    if x.dtype == torch.float32:
        return torch.nn.functional.linear(x, weight, bias)
    return ilu_ops.linear(x, weight, bias)
