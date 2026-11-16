# SPDX-License-Identifier: Apache-2.0
"""Iluvatar DSpark vocab-sharded draft-logits optimizations.

The community DSpark speculator remains the owner of input preparation,
backbone execution and request state. This module replaces the sequential
draft-head stage so Iluvatar's local-head APIs can avoid redundant full-vocab
communication when the draft implements ``compute_draft_logits_local``.

Path A (``VLLM_ILU_DSPARK_FUSED_MARKOV=1``): fuse base+bias+gumbel into a
vectorized CUDA kernel via ixformer and keep the 3-layer backbone on its own
CUDA graph.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import Any

import torch
from ixformer.inference.functions.models import deepseek_v41 as dsv41
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _draft_supports_local_logits(model: Any) -> bool:
    return all(
        hasattr(model, name)
        for name in (
            "compute_draft_logits_local",
            "markov_bias_local",
            "get_draft_vocab_shard",
        )
    )


def _can_use_fused_markov(self: Any) -> bool:
    if os.environ.get("VLLM_ILU_DSPARK_FUSED_MARKOV", "1") != "1":
        return False
    if self._draft_topk is not None:
        return False
    if self.draft_logits is None:
        return False
    if self.use_local_argmax_reduction:
        return False
    if get_tensor_model_parallel_world_size() != 1:
        return False
    if not _draft_supports_local_logits(self.model):
        return False
    return hasattr(dsv41, "dspark_gumbel_bias_sample")


def _sample_sequential_fused(
    self: Any,
    num_reqs: int,
    head_hidden: torch.Tensor,
) -> None:
    """Fused Path-A Markov loop (TP=1 full-vocab probabilistic drafting)."""
    n_spec = self.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[self.sample_indices[:num_sample]]
    base_logits = self.model.compute_draft_logits_local(sample_hidden)
    vocab_start, num_vocab_padding = self.model.get_draft_vocab_shard()
    del vocab_start
    local_vocab_size = base_logits.shape[-1]
    base_logits = base_logits.view(num_reqs, n_spec, local_vocab_size)
    if num_vocab_padding > 0:
        base_logits[..., -num_vocab_padding:] = float("-inf")

    block_size = 1024
    num_blocks = (local_vocab_size + block_size - 1) // block_size
    device = base_logits.device

    bufs = getattr(self, "_ilu_fused_markov_bufs", None)
    if (
        bufs is None
        or bufs["local_argmax"].shape[0] < self.max_num_reqs
        or bufs["local_argmax"].shape[1] < num_blocks
    ):
        bufs = {
            "local_argmax": torch.empty(
                self.max_num_reqs, num_blocks, dtype=torch.int64, device=device
            ),
            "local_max": torch.empty(
                self.max_num_reqs, num_blocks, dtype=torch.float32, device=device
            ),
            "next_prev": torch.empty(
                self.max_num_reqs, dtype=torch.int64, device=device
            ),
            "me": torch.empty(
                self.max_num_reqs, 256, dtype=base_logits.dtype, device=device
            ),
        }
        self._ilu_fused_markov_bufs = bufs

    local_argmax = bufs["local_argmax"][:num_reqs, :num_blocks]
    local_max = bufs["local_max"][:num_reqs, :num_blocks]
    next_prev = bufs["next_prev"][:num_reqs]
    me_buf = bufs["me"][:num_reqs]

    idx_map_flat = self.sample_idx_mapping[:num_sample]
    sample_pos_flat = self.sample_pos[:num_sample]
    confidence_markov_embeds: list[torch.Tensor] = []

    next_prev.copy_(self.input_buffers.input_ids[self._anchor_idx[:num_reqs]])

    markov_w1 = self.model.model.markov_head.markov_w1.weight
    use_embed_kernel = (
        hasattr(dsv41, "dspark_markov_embed")
        and markov_w1.dtype == torch.bfloat16
        and markov_w1.shape[-1] == 256
        and markov_w1.is_contiguous()
    )

    for i in range(n_spec):
        if use_embed_kernel:
            dsv41.dspark_markov_embed(me_buf, markov_w1, next_prev, None)
            markov_embed = me_buf
        else:
            markov_embed = self.model.markov_embed(next_prev)
            me_buf.copy_(markov_embed)

        if self.use_confidence_head:
            confidence_markov_embeds.append(me_buf.clone())

        bias = self.model.markov_bias_local(markov_embed)
        if not bias.is_contiguous():
            bias = bias.contiguous()

        dsv41.dspark_gumbel_bias_sample(
            draft_tokens=self.draft_tokens[:num_reqs],
            next_prev=next_prev,
            local_argmax=local_argmax,
            local_max=local_max,
            logits_cache=self.draft_logits,
            base_logits=base_logits,
            bias=bias,
            sample_idx_mapping=idx_map_flat,
            sample_pos=sample_pos_flat,
            temperatures=self.temperature,
            seeds=self.seeds,
            n_spec=n_spec,
            step=i,
            block_size=block_size,
        )

    if self.use_confidence_head:
        confidence = self.model.compute_confidence(
            sample_hidden,
            torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
        )
        self.draft_token_confidence_probs[:num_reqs] = confidence.view(num_reqs, n_spec)


def make_init_cudagraph_manager(
    orig_init_cg: Callable[..., None],
) -> Callable[..., None]:
    """Honor ``speculative_config.enforce_eager`` on DFlash/DSpark speculators,
    or capture the 3-layer DSpark backbone in a CUDA graph while keeping
    sequential Markov sampling eager when VLLM_ILU_DSPARK_BACKBONE_CG=1."""

    @functools.wraps(orig_init_cg)
    def init_cudagraph_manager(self: Any, cudagraph_mode: CUDAGraphMode) -> None:
        is_dspark = self.__class__.__name__ == "DSparkSpeculator"
        backbone_cg = (
            is_dspark
            and os.environ.get("VLLM_ILU_DSPARK_BACKBONE_CG", "1") == "1"
            and get_pp_group().world_size == 1
        )
        if getattr(self.speculative_config, "enforce_eager", False) and not backbone_cg:
            logger.info(
                "%s speculator has enforce_eager=True; running the draft eagerly.",
                self._speculator_name,
            )
            cudagraph_mode = CUDAGraphMode.NONE
            orig_init_cg(self, cudagraph_mode)
            return

        if getattr(self.speculative_config, "enforce_eager", False) and backbone_cg:
            logger.info(
                "DSpark backbone-only CUDA graph enabled (3-layer backbone captured, "
                "Markov sequential sampling %s).",
                "fused" if _can_use_fused_markov(self) else "eager",
            )
            self._ilu_backbone_cg_only = True
            self._ilu_capturing_backbone_cg = False
            self._ilu_current_num_reqs = 1
            self._ilu_head_hidden_buf = torch.zeros(
                self.max_num_tokens,
                self.draft_model_config.get_hidden_size(),
                dtype=self.dtype,
                device=self.device,
            )
            orig_init_cg(self, cudagraph_mode)
            if self.query_cudagraph_manager is not None:
                orig_capture = self.query_cudagraph_manager.capture
                orig_run_fullgraph = self.query_cudagraph_manager.run_fullgraph

                def capture_backbone_only(*args: Any, **kwargs: Any) -> Any:
                    self._ilu_capturing_backbone_cg = True
                    try:
                        return orig_capture(*args, **kwargs)
                    finally:
                        self._ilu_capturing_backbone_cg = False

                def run_fullgraph_and_sample(batch_desc: Any) -> Any:
                    orig_run_fullgraph(batch_desc)
                    num_tokens_padded = batch_desc.num_tokens
                    num_reqs = getattr(
                        self,
                        "_ilu_current_num_reqs",
                        num_tokens_padded // self.num_query_per_req,
                    )
                    self._sample_sequential(
                        num_reqs, self._ilu_head_hidden_buf[:num_tokens_padded]
                    )

                self.query_cudagraph_manager.capture = capture_backbone_only
                self.query_cudagraph_manager.run_fullgraph = run_fullgraph_and_sample
            return

        orig_init_cg(self, cudagraph_mode)

    return init_cudagraph_manager


def make_capture(orig_capture: Callable[..., None]) -> Callable[..., None]:
    """Skip CUDA graph capture when the DFlash/DSpark speculator runs eagerly."""

    @functools.wraps(orig_capture)
    def capture(self: Any) -> None:
        assert self.query_cudagraph_manager is not None
        if self.query_cudagraph_manager.cudagraph_mode == CUDAGraphMode.NONE:
            logger.info(
                "Skipping CUDA graph capture for %s speculator (eager mode).",
                self._speculator_name,
            )
            return
        orig_capture(self)

    return capture


def make_generate_draft(
    original: Callable[..., None],
) -> Callable[..., None]:
    @functools.wraps(original)
    def _generate_draft(
        self: Any,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: Any = None,
    ) -> None:
        if getattr(self, "_ilu_capturing_backbone_cg", False):
            if cudagraph_runtime_mode is None:
                cudagraph_runtime_mode = CUDAGraphMode.NONE
            head_hidden = self._run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            self._ilu_head_hidden_buf[:num_tokens_padded].copy_(head_hidden)
            return
        return original(
            self,
            num_reqs,
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

    return _generate_draft


def make_build_uniform_attn_metadata(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    @functools.wraps(original)
    def _build_uniform_attn_metadata(self: Any, *args: Any, **kwargs: Any) -> Any:
        if "num_reqs" in kwargs:
            self._ilu_current_num_reqs = kwargs["num_reqs"]
        elif len(args) >= 2:
            self._ilu_current_num_reqs = args[1]
        return original(self, *args, **kwargs)

    return _build_uniform_attn_metadata


def load_draft_model(self, target_model, target_attn_layer_names):
    model = self._ilu_community_load_draft_model(target_model, target_attn_layer_names)
    self._ilu_target_context_direct = False
    parallel = self.vllm_config.parallel_config
    # The runner owns scheduling/stream lifetime; models own their projection
    # and auxiliary-state semantics. Unsupported models keep community behavior.
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.use_ubatching
        or self.vllm_config.lora_config is not None
    ):
        return model
    bind = getattr(model, "try_bind_target_context", None)
    if callable(bind):
        self._ilu_target_context_direct = bind(target_model, self.hidden_states)
    if self._ilu_target_context_direct:
        logger.info(
            "DSpark target writes projected auxiliary states into draft context"
        )
    return model


def propose(
    self,
    input_batch,
    attn_metadata,
    slot_mappings,
    last_hidden_states,
    aux_hidden_states,
    *args,
    **kwargs,
):
    if getattr(self, "_ilu_target_context_direct", False):
        if aux_hidden_states:
            raise ValueError(
                "DSpark direct context unexpectedly exported auxiliary states"
            )
        # Same storage and extent as the destination in community propose:
        # Tensor.copy_ is a no-op; Graph capture keeps this address fixed.
        last_hidden_states = self.hidden_states[: input_batch.num_tokens]
        aux_hidden_states = None
    return self._ilu_community_propose(
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        *args,
        **kwargs,
    )


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


def sample_sequential(self: Any, num_reqs: int, head_hidden: torch.Tensor) -> None:
    """Run the DSpark Markov head, using shard-local logits when available."""
    if self._draft_topk is not None or not _draft_supports_local_logits(self.model):
        return self._ilu_community_sample_sequential(num_reqs, head_hidden)

    if _can_use_fused_markov(self):
        return _sample_sequential_fused(self, num_reqs, head_hidden)

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

    idx_map_t = (
        self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec).t().contiguous()
    )
    sample_pos_t = self.sample_pos[:num_sample].view(num_reqs, n_spec).t().contiguous()
    confidence_markov_embeds: list[torch.Tensor] = []
    prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

    for i in range(n_spec):
        markov_embed = self.model.markov_embed(prev)
        if self.use_confidence_head:
            confidence_markov_embeds.append(markov_embed)
        bias = self.model.markov_bias_local(markov_embed)
        logits_i = base_logits[:, i] + bias
        if self.draft_logits is not None:
            logits_i = gather_vocab_parallel_logits(
                logits_i,
                draft_vocab_size,
                num_vocab_padding,
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map_t[i], sample_pos_t[i], i
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
                logits_i, idx_map_t[i], sample_pos_t[i], i
            )
        self.draft_tokens[:num_reqs, i] = draft_sampled_i
        prev = draft_sampled_i

    if self.use_confidence_head:
        confidence = self.model.compute_confidence(
            sample_hidden,
            torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
        )
        self.draft_token_confidence_probs[:num_reqs] = confidence.view(num_reqs, n_spec)
