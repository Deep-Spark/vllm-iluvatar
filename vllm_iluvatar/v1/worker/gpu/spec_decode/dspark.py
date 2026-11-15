# SPDX-License-Identifier: Apache-2.0
"""Iluvatar DSpark vocab-sharded draft-logits optimizations.

The community DSpark speculator remains the owner of input preparation,
backbone execution and request state. This module replaces the sequential
draft-head stage so Iluvatar's local-head APIs can avoid redundant full-vocab
communication when the draft implements ``compute_draft_logits_local``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger

from .dspark_bias_sample import FusedBiasWorkspace, fused_bias_gumbel_sample

logger = init_logger("vllm." + __name__)


def vocab_parallel_argmax(
    local_logits: torch.Tensor,
    vocab_start: int,
    num_vocab_padding: int,
) -> torch.Tensor:
    """Return full-vocabulary argmax IDs without gathering full logits."""
    if num_vocab_padding > 0:
        local_logits[..., -num_vocab_padding:] = -float("inf")

    local_max_values, local_ids = local_logits.max(dim=-1)
    global_ids = local_ids + vocab_start
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return global_ids

    # Supported vocabulary IDs are exactly representable in fp32. TP ranks
    # follow increasing vocabulary order, so argmax over rank maxima preserves
    # torch.argmax's first-ID tie rule on the fully gathered vocabulary.
    local_pairs = torch.stack((local_max_values.float(), global_ids.float()), dim=-1)
    gathered = tensor_model_parallel_all_gather(local_pairs, dim=-1)
    gathered = gathered.view(*local_max_values.shape, tp_size, 2)
    max_rank = gathered[..., 0].argmax(dim=-1, keepdim=True)
    return gathered[..., 1].gather(-1, max_rank).squeeze(-1).to(torch.int64)


def gather_vocab_parallel_logits(
    local_logits: torch.Tensor,
    vocab_size: int,
    num_vocab_padding: int,
) -> torch.Tensor:
    """Gather one combined distribution and remove vocabulary padding."""
    if num_vocab_padding > 0:
        local_logits[..., -num_vocab_padding:] = -float("inf")
    if get_tensor_model_parallel_world_size() == 1:
        return local_logits[..., :vocab_size]
    logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
    return logits[..., :vocab_size]


def _draft_supports_local_logits(model: Any) -> bool:
    return all(
        hasattr(model, name)
        for name in (
            "compute_draft_logits_local",
            "markov_bias_local",
            "get_draft_vocab_shard",
        )
    )


def sample_sequential(self: Any, num_reqs: int, head_hidden: torch.Tensor) -> None:
    """Run the DSpark Markov head, using shard-local logits when available."""
    if self._draft_topk is not None or not _draft_supports_local_logits(self.model):
        return self._ilu_community_sample_sequential(num_reqs, head_hidden)

    n_spec = self.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[self.sample_indices[:num_sample]]
    base_logits = self.model.compute_draft_logits_local(sample_hidden)
    vocab_start, num_vocab_padding = self.model.get_draft_vocab_shard()
    local_vocab_size = base_logits.shape[-1]
    base_logits = base_logits.view(num_reqs, n_spec, local_vocab_size)

    if self._d2t_scatter_index is not None:
        draft_vocab_size = self._d2t_scatter_index.shape[0]
    else:
        draft_vocab_size = self.vocab_size

    idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
    sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
    confidence_markov_embeds: list[torch.Tensor] = []
    prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
    fuse_bias = (
        getattr(self, "_ilu_fused_bias_sample", False)
        and self.draft_logits is not None
        and self.acceptance_estimator is None
        and self._d2t_scatter_index is None
        and self.draft_watermarker is None
        and get_tensor_model_parallel_world_size() == 1
    )
    sample_vocab_size = min(draft_vocab_size, local_vocab_size)
    sample_padding = max(0, sample_vocab_size - (local_vocab_size - num_vocab_padding))
    fused_workspace = None
    if fuse_bias and hasattr(torch.ops._C, "ilu_dspark_bias_sample_v4"):
        fused_workspace = getattr(self, "_ilu_fused_bias_workspace", None)
        if fused_workspace is None:
            fused_workspace = FusedBiasWorkspace(
                self.draft_tokens.shape[0],
                sample_vocab_size,
                base_logits.device,
                self.use_fp64_gumbel,
            )
            self._ilu_fused_bias_workspace = fused_workspace
        if (
            fused_workspace.vocab_size != sample_vocab_size
            or fused_workspace.use_fp64 != self.use_fp64_gumbel
        ):
            raise RuntimeError("DSpark sampler workspace configuration changed")
    if fuse_bias and not getattr(self, "_ilu_fused_bias_sample_logged", False):
        logger.info(
            "DSpark fused bias sampler active: version=%d rows=%d vocab=%d dtype=%s; "
            "native RNG and draft-logits cache preserved",
            4
            if fused_workspace is not None
            else 2
            if hasattr(torch.ops._C, "ilu_dspark_bias_sample_v2")
            else 1,
            num_reqs,
            sample_vocab_size,
            base_logits.dtype,
        )
        self._ilu_fused_bias_sample_logged = True

    for i in range(n_spec):
        markov_embed = self.model.markov_embed(prev)
        if self.use_confidence_head:
            confidence_markov_embeds.append(markov_embed)
        bias = self.model.markov_bias_local(markov_embed)
        if fused_workspace is not None:
            draft_sampled_i = self.draft_tokens[:num_reqs, i]
            fused_workspace.sample_into(
                base_logits[:, i, :sample_vocab_size],
                bias[:, :sample_vocab_size],
                idx_map[:, i],
                self.temperature,
                self.seeds,
                sample_pos[:, i],
                self.draft_logits,
                self._step_cols[i],
                draft_sampled_i,
                sample_padding,
            )
            prev = draft_sampled_i
            continue
        if fuse_bias:
            draft_sampled_i = fused_bias_gumbel_sample(
                base_logits[:, i, :sample_vocab_size],
                bias[:, :sample_vocab_size],
                idx_map[:, i],
                self.temperature,
                self.seeds,
                sample_pos[:, i] - 1,
                self.draft_logits,
                self._step_cols[i],
                self.use_fp64_gumbel,
                sample_padding,
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i
            continue
        logits_i = base_logits[:, i] + bias
        if self.draft_logits is not None:
            logits_i = gather_vocab_parallel_logits(
                logits_i,
                draft_vocab_size,
                num_vocab_padding,
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
        elif self.use_local_argmax_reduction:
            draft_ids = vocab_parallel_argmax(
                logits_i,
                vocab_start,
                num_vocab_padding,
            )
            draft_sampled_i = self.model.map_draft_to_target(draft_ids)
        else:
            logits_i = gather_vocab_parallel_logits(
                logits_i,
                draft_vocab_size,
                num_vocab_padding,
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
        self.draft_tokens[:num_reqs, i] = draft_sampled_i
        prev = draft_sampled_i

    if self.use_confidence_head:
        confidence = self.model.compute_confidence(
            sample_hidden,
            torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
        )
        self.draft_token_confidence_probs[:num_reqs] = confidence.view(
            num_reqs, n_spec
        )


def make_init_cudagraph_manager(
    original: Callable[..., None],
) -> Callable[..., None]:
    """Select the independent DSpark eager or backbone-only graph policy."""

    def init_cudagraph_manager(self: Any, cudagraph_mode: Any) -> None:
        from vllm import envs
        from vllm.config.compilation import CUDAGraphMode
        from vllm.distributed import get_pp_group

        library = os.environ.get("VLLM_ILU_DSPARK_FUSED_BIAS_LIBRARY")
        if library and not hasattr(torch.ops._C, "ilu_dspark_bias_sample_v2"):
            torch.ops.load_library(library)
        state_library = os.environ.get("VLLM_ILU_DSPARK_FUSED_BIAS_STATE_LIBRARY")
        if state_library and not hasattr(torch.ops._C, "ilu_dspark_bias_sample_v4"):
            torch.ops.load_library(state_library)
        self._ilu_fused_bias_sample = envs.VLLM_ILU_DSPARK_FUSED_BIAS_SAMPLE
        if self._ilu_fused_bias_sample and not any(
            hasattr(torch.ops._C, name)
            for name in (
                "ilu_dspark_bias_sample_v4",
                "ilu_dspark_bias_sample_v2",
                "ilu_fused_bias_gumbel_sample",
            )
        ):
            raise RuntimeError("Rebuild vllm-iluvatar for DSpark bias fusion")

        backbone_only = envs.VLLM_ILU_DSPARK_BACKBONE_GRAPH
        if backbone_only and (
            get_tensor_model_parallel_world_size() != 1
            or get_pp_group().world_size != 1
            or envs.VLLM_ILU_DP_VOCAB_GROUP_SIZE < 2
            or cudagraph_mode.decode_mode() != CUDAGraphMode.FULL
            or self.vllm_config.lora_config is not None
        ):
            raise ValueError(
                "DSpark backbone graphs require TP1/PP1, DP vocabulary "
                "projection, full decode graphs, and no LoRA adapters"
            )
        if self.speculative_config.enforce_eager and not backbone_only:
            logger.info(
                "DSpark CUDA graph disabled by speculative_config.enforce_eager; "
                "target CUDA graph policy is unchanged."
            )
            cudagraph_mode = CUDAGraphMode.NONE
        original(self, cudagraph_mode)
        if backbone_only:
            from .dspark_backbone_graph import DSparkBackboneGraph

            state = getattr(self, "_ilu_backbone_graph", None)
            if state is None:
                state = DSparkBackboneGraph(self)
                self._ilu_backbone_graph = state
            state.bind_manager(self.query_cudagraph_manager)
            logger.info(
                "DSpark backbone CUDA graph enabled; distributed projection "
                "and Markov sampling stay eager."
            )

    return init_cudagraph_manager
