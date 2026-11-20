# SPDX-License-Identifier: Apache-2.0
"""CQ vocabulary-parallel greedy selection using the existing split sampler.

Registered through the standard LogitsProcessor OOT extension point. The
projection, padding and TP reduction follow vLLM v0.31 (341bf7d4d1); only the
local maximum reduction is specialized. Ordinary logits processing is inherited.
"""

from __future__ import annotations

import torch
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

from vllm_iluvatar.ilu_util.arch import is_ivcore30
from vllm_iluvatar.model_executor.layers.greedy_tp import (
    greedy_local_pair,
    greedy_rank_winner,
)


def _use_split_argmax(logits: torch.Tensor) -> bool:
    return (
        is_ivcore30()
        and logits.is_cuda
        and logits.dtype in (torch.float32, torch.bfloat16)
        and logits.ndim == 2
        and logits.shape[0] > 0
        and logits.shape[1] > 0
        and logits.stride(1) == 1
    )


@LogitsProcessor.register_oot()
class IluLogitsProcessor(LogitsProcessor):
    def _get_greedy_metadata(self, logits: torch.Tensor):
        batch = logits.shape[0]
        capacity = 1 << (batch - 1).bit_length()
        key = (logits.device, capacity)
        cache = getattr(self, "_greedy_metadata", None)
        if cache is None:
            cache = self._greedy_metadata = {}
        metadata = cache.get(key)
        if metadata is None:
            metadata = (
                torch.arange(capacity, device=logits.device, dtype=torch.int32),
                torch.zeros(capacity, device=logits.device, dtype=torch.float32),
                torch.zeros(capacity, device=logits.device, dtype=torch.int64),
                torch.zeros(capacity, device=logits.device, dtype=torch.int64),
            )
            # Keep ordinary allocations alive for every captured shape. A cold
            # capture creates its own metadata nodes without caching graph-pool
            # tensors across independent captures.
            if not torch.cuda.is_current_stream_capturing():
                cache[key] = metadata
        return metadata

    def _local_max(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not _use_split_argmax(logits):
            return logits.max(dim=-1)
        batch = logits.shape[0]
        metadata = self._get_greedy_metadata(logits)
        mapping, temperature, seeds, positions = metadata
        indices = gumbel_sample(
            logits,
            mapping[:batch],
            temperature,
            seeds,
            positions[:batch],
            apply_temperature=False,
            is_drafting=False,
        )
        values = logits.gather(1, indices[:, None]).squeeze(1)
        return values, indices

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not is_ivcore30():
            return super().get_top_tokens(lm_head, hidden_states, embedding_bias)
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        if lm_head.tp_size > 1 and _use_split_argmax(logits):
            local_pair = greedy_local_pair(
                logits,
                self._get_greedy_metadata(logits),
                lm_head.shard_indices.org_vocab_start_index,
            )
            gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
            return greedy_rank_winner(
                gathered.view(hidden_states.shape[0], lm_head.tp_size, 2)
            )

        local_values, local_indices = self._local_max(logits)
        global_indices = local_indices + lm_head.shard_indices.org_vocab_start_index
        if lm_head.tp_size == 1:
            return global_indices

        local_pair = torch.stack((local_values.float(), global_indices.float()), dim=-1)
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        gathered = gathered.view(hidden_states.shape[0], lm_head.tp_size, 2)
        max_rank = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        return gathered[:, :, 1].gather(-1, max_rank).squeeze(-1).to(torch.int64)
