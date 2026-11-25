# SPDX-License-Identifier: Apache-2.0
"""Iluvatar GLM sampling override.

Why:
    VLLM_ILU_GLM_SAMPLING_MODE optionally selects native CQ sorted/radix
    sampling on ivcore30. The independent TOP_P_RADIX flag enables the
    native probability mask for ordinary sampling and MTP verification in
    both runners. Disabled or unsupported contracts retain upstream dispatch.
Remove:
    When upstream provides this GLM sampling path.
"""

from __future__ import annotations

import torch
from vllm.v1.sample.ops import topk_topp_sampler as _community

# Keep the original method for unsupported contracts and the disabled mode.
_original_forward_native = _community.TopKTopPSampler.forward_native
_original_apply_top_k_top_p = _community.apply_top_k_top_p


def _native_top_p_available(logits, k, p):
    # Preserve upstream's small-sort tie policy. Native FP32 reductions may
    # still move near-cutoff probabilities; this path is explicitly opt-in.
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

    from vllm_iluvatar.ilu_util.arch import is_ivcore30

    return is_ivcore30() and hasattr(F, "apply_top_p_radix")


def apply_top_k_top_p(logits, k, p):
    """Opt-in CQ mask; no RNG draws and no change to disabled dispatch."""
    from vllm import envs

    if envs.VLLM_ILU_TOP_P_RADIX and _native_top_p_available(logits, k, p):
        from ixformer.inference.functions import apply_top_p_radix

        return apply_top_p_radix(logits, p)
    return _original_apply_top_k_top_p(logits, k, p)


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
