# SPDX-License-Identifier: Apache-2.0
"""ModelRunnerV2 FLy verifier with the accepted target-probability gate."""

import numpy as np
import torch
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler as CommunityRejectionSampler,
)

from vllm_iluvatar.config.fly_config import SpeculativeConfig


class RejectionSampler(CommunityRejectionSampler):
    """Keep upstream chunking/logprobs and override only FLy verification."""

    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
        *,
        watermark_key: int | None = None,
    ):
        super().__init__(sampler, spec_config, device, watermark_key=watermark_key)
        self.fly_window_size = 0
        if spec_config.rejection_sample_method != "fly":
            return

        # Ordinary speculative decoding does not need the optional FLy operators.
        from vllm_iluvatar.v1.worker.gpu.spec_decode.fly_rejection import (
            rejection_sample,
        )

        self._fly_rejection_sample = rejection_sample
        assert spec_config.fly_window_size is not None
        self.fly_window_size = spec_config.fly_window_size
        self.fly_entropy_threshold = spec_config.fly_entropy_threshold
        self.fly_entropy_top_k = spec_config.fly_entropy_top_k
        self.fly_min_target_probability = spec_config.fly_min_target_probability

    def _verify(
        self,
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
        if self.fly_window_size == 0:
            return super()._verify(
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
        sampled, num_sampled = self._fly_rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
            fly_window_size=self.fly_window_size,
            fly_entropy_threshold=self.fly_entropy_threshold,
            fly_entropy_top_k=self.fly_entropy_top_k,
            fly_min_target_probability=self.fly_min_target_probability,
            **self._watermarking_kwargs(
                draft_sampled, expanded_idx_mapping, expanded_local_pos
            ),
        )
        return processed_logits, sampled, num_sampled
