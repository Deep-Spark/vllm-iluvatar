# SPDX-License-Identifier: Apache-2.0
"""Iluvatar top-k/top-p sampler override.

Why:
    The upstream Triton top-k/top-p fast path is unstable on the current
    Iluvatar sampling stack for larger batches. Keep the CPU behavior unchanged
    but force the CUDA path to use the PyTorch implementation.
Remove:
    When the Triton sampler path is validated on Iluvatar for the affected
    workloads.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.v1.sample.ops import topk_topp_sampler as _community


# [ILU override] apply_top_k_top_p - delta: disable the Triton GPU fast path.
def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    if current_platform.is_cpu():
        if _community.HAS_TRITON:
            return _community.apply_top_k_top_p_triton(logits, k, p)
        return _community.apply_top_k_top_p_pytorch(
            logits, k, p, allow_cpu_sync=True
        )

    return _community.apply_top_k_top_p_pytorch(logits, k, p)
