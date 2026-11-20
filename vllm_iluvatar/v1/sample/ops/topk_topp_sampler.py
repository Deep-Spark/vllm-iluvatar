# SPDX-License-Identifier: Apache-2.0
"""Iluvatar GLM sampling overrides.

MR/CQ use the weighted-radix top-p mask for supported distributions.
The legacy CQ sampler remains optional; unsupported requests keep upstream paths.
"""

from __future__ import annotations

import torch
from vllm.v1.sample.ops import topk_topp_sampler as _community

# Keep the original methods for unsupported contracts.
_original_forward_native = _community.TopKTopPSampler.forward_native
_original_apply_top_k_top_p = _community.apply_top_k_top_p


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


def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    """Weighted-radix mask for supported V1 and V2 distributions."""
    if _native_top_p_available(logits, k, p):
        from ixformer.inference.functions import apply_top_p_radix

        return apply_top_p_radix(logits, p)
    return _original_apply_top_k_top_p(logits, k, p)


def _native_top_p_available(logits, k, p):
    # Keep Torch's unstable small-sort tie policy on the original path.
    # The radix mask follows the large-sort bit and token-ID ordering.
    if (
        k is not None
        or p is None
        or not logits.is_cuda
        or logits.dtype != torch.float32
        or logits.ndim != 2
        or not logits.is_contiguous()
        or logits.shape[1] <= 4096
        or not 0 < logits.shape[0] <= 65535
        or logits.numel() > 2**31 - 256
        or p.dtype != torch.float32
        or p.ndim != 1
        or p.shape[0] != logits.shape[0]
        or not p.is_contiguous()
        or p.device != logits.device
    ):
        return False
    from ixformer.inference import functions as F
    from ixformer.inference.backend_registry import current_cuda_arch

    return current_cuda_arch() in ("ivcore11", "ivcore30") and hasattr(
        F, "apply_top_p_radix"
    )
