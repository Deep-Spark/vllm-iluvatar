# SPDX-License-Identifier: Apache-2.0
"""Iluvatar GLM sampling override.

Why:
    VLLM_ILU_GLM_SAMPLING_MODE optionally selects native CQ sorted/radix
    sampling on ivcore30. apply_top_k_top_p stays on upstream, so the
    community dispatch (Triton at eight rows or more, the PyTorch sort
    below that) is unchanged. Unsupported contracts retain the original
    sampler.
Remove:
    When upstream provides this GLM sampling path.
"""

from __future__ import annotations

import torch
from vllm.v1.sample.ops import topk_topp_sampler as _community

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
        if (
            p is not None
            and p.ndim == 1
            and p.shape[0] == 1
            and logits.ndim == 2
            and logits.shape[0] != 1
        ):
            p = p.expand(logits.shape[0]).contiguous()
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
