# SPDX-License-Identifier: Apache-2.0
"""Wire compact sampling into V2; other distributions retain upstream paths."""

from functools import wraps

from vllm_iluvatar.ilu_patch import general_patch as p


def register():
    p.register_once(
        __name__, _apply, detail="compact BF16 nucleus sampling and verification"
    )


def _apply():
    import torch
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    from vllm_iluvatar.v1.sample.ops.top_p_summary import (
        eligible,
        sample_summary,
        verify_eligible,
    )

    original_init = Sampler.__init__
    original_sample = Sampler.sample
    original_verify = RejectionSampler._verify

    @wraps(original_init)
    def init(self, vllm_config, *args, **kwargs):
        original_init(self, vllm_config, *args, **kwargs)
        self._ilu_summary_replicated_batch = (
            type(self) is Sampler
            and not vllm_config.parallel_config.enable_batch_sharded_sampling
        )

    @wraps(original_sample)
    def sample(
        self,
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        seq_lens_upper_bound_np,
        return_logprobs=False,
    ):
        if eligible(self, logits, idx_mapping_np, return_logprobs=return_logprobs):
            n = logits.shape[0]
            cu = torch.arange(n + 1, device=logits.device, dtype=torch.int32)
            sampled, counts = sample_summary(
                self, logits, expanded_idx_mapping, cu, input_ids, pos, 1
            )
            # Ordinary sampling expects one token per completed prefill/decode
            # row. Match upstream Gumbel argmax's first-index tie on all -inf.
            return sampled[:, 0].masked_fill_(counts == 0, 0), logits
        return original_sample(
            self,
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            seq_lens_upper_bound_np,
            return_logprobs=return_logprobs,
        )

    @wraps(original_verify)
    def verify(
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
    ):
        if verify_eligible(self, logits, draft_logits, idx_mapping_np):
            sampled, counts = sample_summary(
                self.sampler,
                logits,
                expanded_idx_mapping,
                cu_num_logits,
                draft_sampled,
                pos,
                self.num_speculative_steps + 1,
            )
            # Processed-logprob consumers were gated out before this branch.
            return logits, sampled, counts
        return original_verify(
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

    Sampler.__init__ = init
    Sampler.sample = sample
    RejectionSampler._verify = verify
