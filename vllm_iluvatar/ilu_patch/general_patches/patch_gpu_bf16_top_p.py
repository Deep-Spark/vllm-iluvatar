# SPDX-License-Identifier: Apache-2.0
"""Opt-in V2 sampler BF16 sorting; unsupported requests use the original method."""

from functools import wraps

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(__name__, _apply, detail="exact BF16 keys-only V2 top-p fast path")


def _apply() -> None:
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    from vllm_iluvatar.v1.sample.ops.bf16_top_p import (
        apply_bf16_top_p,
        eligible,
    )

    original = Sampler.apply_sampling_params

    @wraps(original)
    def apply_sampling_params(
        self,
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        seq_lens_upper_bound_np,
        skip_top_k_top_p=False,
    ):
        if eligible(self, logits, idx_mapping_np, skip_top_k_top_p):
            return apply_bf16_top_p(self, logits, expanded_idx_mapping, idx_mapping_np)
        return original(
            self,
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            seq_lens_upper_bound_np,
            skip_top_k_top_p=skip_top_k_top_p,
        )

    Sampler.apply_sampling_params = apply_sampling_params
