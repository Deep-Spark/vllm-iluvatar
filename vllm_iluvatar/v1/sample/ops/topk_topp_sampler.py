# SPDX-License-Identifier: Apache-2.0
"""Iluvatar top-k/top-p sampler override.

Why:
    The upstream Triton top-k/top-p fast path is unstable on the current
    Iluvatar sampling stack for larger batches. Keep the CPU behavior unchanged
    but force the default CUDA path to use the PyTorch implementation.
    VLLM_ILU_GLM_SAMPLING_MODE optionally selects native CQ sorted/radix sampling;
    unsupported contracts retain the original sampler.
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
        return _community.apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)

    return _community.apply_top_k_top_p_pytorch(logits, k, p)


# Keep the original method for unsupported contracts and the disabled mode.
_original_forward_native = _community.TopKTopPSampler.forward_native


def forward_native(self, logits, generators, k, p):
    """Optional CQ sampling; processed probabilities and FP64 keep upstream."""
    from vllm import envs
    from vllm.config.model import PROCESSED_LOGPROBS_MODES

    from vllm_iluvatar.ilu_util.arch import is_ivcore30

    if (
        envs.VLLM_ILU_GLM_SAMPLING_MODE == "off"
        or not is_ivcore30()
        or not logits.is_cuda
        or logits.dtype != torch.float32
        or not logits.is_contiguous()
        or self.use_fp64_gumbel
        or self.logprobs_mode in PROCESSED_LOGPROBS_MODES
        or k is not None
        or p is None
        or logits.ndim != 2
        or not 0 < logits.shape[0] <= 65535
        or not 0 < logits.numel() <= 2147483647
        or p.dtype != torch.float32
        or not p.is_contiguous()
        or p.ndim != 1
        or p.shape[0] != logits.shape[0]
        or p.device != logits.device
    ):
        return _original_forward_native(self, logits, generators, k, p)

    import ixformer.inference.functions as F

    # Preserve per-request generator draw order/state, including mixed batches.
    # Both modes fold the default generator into the native kernel.
    noise = None
    if generators:
        noise = torch.empty_like(logits)
        if len(generators) != logits.shape[0]:
            noise.exponential_()
        for row, generator in generators.items():
            noise[row].exponential_(generator=generator)
    if envs.VLLM_ILU_GLM_SAMPLING_MODE == "sorted":
        values, order = logits.sort(dim=-1, descending=False)
        return F.sampling_sorted_top_p(values, order, noise, p), None
    return F.sampling_radix_top_p(logits, noise, p), None
