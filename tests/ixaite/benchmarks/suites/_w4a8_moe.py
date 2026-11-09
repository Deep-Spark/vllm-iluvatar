# SPDX-License-Identifier: Apache-2.0
"""Shared packed TN operands for MoE W4A8 L2 suites."""

from __future__ import annotations

import torch
from ixaite.ops.w4a8_layout import pack_logical_weight


def packed_grouped_operands(
    *,
    tokens_per_experts: list[int],
    n: int,
    k: int,
    device: torch.device,
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(x_q, weight, i_scales, w_scales, tokens)`` on ``device``.

    ``tokens`` stays CPU int32 (prefill group-GEMM ABI). Decode GEMV copies
    it to GPU at the call site.
    """
    num_experts = len(tokens_per_experts)
    total = sum(tokens_per_experts)
    generator = torch.Generator().manual_seed(seed)
    x_q = torch.randint(-16, 17, (total, k), dtype=torch.int8, generator=generator)
    logical = torch.randint(
        -8, 8, (num_experts, n, k), dtype=torch.int8, generator=generator
    )
    weight = torch.stack(
        [
            pack_logical_weight(logical[expert], format="TN")
            for expert in range(num_experts)
        ]
    )
    i_scales = torch.rand(total, dtype=torch.float32, generator=generator) * 0.01
    w_scales = (
        torch.rand(num_experts, n, dtype=torch.float32, generator=generator) * 0.01
    )
    tokens = torch.tensor(tokens_per_experts, dtype=torch.int32)
    return (
        x_q.to(device),
        weight.to(device),
        i_scales.to(device),
        w_scales.to(device),
        tokens,
    )
