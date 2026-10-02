# SPDX-License-Identifier: Apache-2.0
"""Greedy DSpark verification fast path backed by native CUDA."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
import torch


def make_verify_with_greedy_cuda(original: Callable[..., Any]) -> Callable[..., Any]:
    """Specialize the exact greedy contract without changing other samplers."""

    @wraps(original)
    def _verify(
        self: Any,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        seq_lens_upper_bound_np: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temperatures = self.sampler.sampling_states.temperature.np[idx_mapping_np]
        use_fast_path = (
            draft_logits is None
            and self.synthetic_conditional_rates is None
            and not self.use_block_verification
            and not self.enable_adaptive_verification
            and self.watermark_key is None
            and np.all(idx_mapping_np >= 0)
            and np.all(temperatures == 0.0)
        )
        if not use_fast_path:
            return original(
                self,
                logits,
                draft_logits,
                draft_sampled,
                pos,
                cu_num_logits,
                idx_mapping,
                idx_mapping_np,
                expanded_idx_mapping,
                expanded_local_pos,
                seq_lens_upper_bound_np,
            )

        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
            seq_lens_upper_bound_np,
        )
        sampled, num_sampled = torch.ops._C.greedy_rejection_sample(
            processed_logits,
            draft_sampled,
            cu_num_logits,
            self.num_speculative_steps,
        )
        return processed_logits, sampled, num_sampled

    return _verify
