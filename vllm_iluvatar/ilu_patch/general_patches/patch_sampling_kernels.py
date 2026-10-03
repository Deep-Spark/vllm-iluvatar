# SPDX-License-Identifier: Apache-2.0
"""Route Gumbel and speculative rejection sampling to CUDA kernels."""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="Gumbel and rejection sampling use CUDA extension kernels",
    )


def _apply() -> None:
    from vllm.v1.worker.gpu.sample import gumbel
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler
    from vllm.v1.worker.gpu.spec_decode import (
        rejection_sampler_utils as rejection,
    )

    from vllm_iluvatar.v1.worker.gpu.spec_decode import sampling_kernels
    from vllm_iluvatar.v1.worker.gpu.spec_decode.greedy_rejection import (
        make_verify_with_greedy_cuda,
    )

    replacements = (
        (gumbel, "_gumbel_sample_kernel", sampling_kernels.gumbel_sample_kernel),
        (
            rejection,
            "_compute_local_logits_stats_kernel",
            sampling_kernels.compute_local_logits_stats_kernel,
        ),
        (rejection, "_rejection_kernel", sampling_kernels.rejection_kernel),
        (rejection, "_resample_kernel", sampling_kernels.resample_kernel),
        (
            rejection,
            "_insert_resampled_kernel",
            sampling_kernels.insert_resampled_kernel,
        ),
    )
    for module, name, replacement in replacements:
        p.rebind_module_func(module, name, replacement)

    p.rebind_module_func(
        rejection,
        "rejection_sample",
        sampling_kernels.rejection_sample_with_aligned_draft(
            rejection.rejection_sample
        ),
    )
    rejection_sampler.RejectionSampler._verify = make_verify_with_greedy_cuda(
        rejection_sampler.RejectionSampler._verify
    )
