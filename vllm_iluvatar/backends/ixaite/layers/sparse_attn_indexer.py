# Source: vllm_iluvatar/model_executor/layers/sparse_attn_indexer.py at 8932464; branch fixes belong only to ixaite.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar Sparse Attention Indexer custom op layer."""

from __future__ import annotations

import torch
from vllm import _custom_ops as ops
from vllm import envs
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention.pcp import maybe_gather_indexer_k
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_iluvatar.backends.ixaite import ops as ilu_ops
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV32IndexerMetadata,
)


def _mqa_logits(*args, **kwargs):
    if envs.VLLM_ILU_GLM_DSA_NATIVE_MQA_LOGITS:
        kwargs.setdefault("backend", "ixformer_native")
    return ilu_ops.dsa_indexer_mqa_logits_with_blocks_bf16(*args, **kwargs)


# Distinct from the vllm-fork schema ``vllm::ilu_sparse_attn_indexer``.
_PLUGIN_INDEXER_OP = "ilu_plugin_sparse_attn_indexer"


def ilu_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    # attn_metadata is unavailable during dummy/profiling runs.
    assert topk_indices_buffer is not None
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return ilu_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q,
            k,
            weights,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )

    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, IluDeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    # Grid follows slot_mapping's (CUDA-graph padded) length; truncate k if it
    # is longer. Padding rows carry slot_mapping == -1 and are skipped in the
    # kernel (see vllm_indexer_k_cache.cu ``if (slot_idx < 0) return;``).
    if not skip_k_cache_insert:
        num_tokens = slot_mapping.shape[0]
        use_pcp = attn_metadata.pcp_world_size > 1
        if use_pcp:
            num_tokens //= attn_metadata.pcp_world_size
        k = k[:num_tokens]
        k, slot_mapping = maybe_gather_indexer_k(
            k, slot_mapping, num_decode_tokens, use_pcp
        )
        ops.indexer_k_cache(k, kv_cache, slot_mapping)

    if has_prefill:
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata is not None
        for chunk in prefill_metadata.chunks:
            if chunk.max_context_len == 0 or chunk.token_start == chunk.token_end:
                continue
            logits = _mqa_logits(
                q[chunk.token_start : chunk.token_end],
                chunk.cu_seqlens_q,
                chunk.cu_seq_lens,
                kv_cache,
                chunk.block_table,
                weights[chunk.token_start : chunk.token_end],
                max_q_len=chunk.max_q_len,
                max_kv_len=chunk.max_kv_len,
                max_context_len=chunk.max_context_len,
                valid_lens=chunk.valid_lens,
            )
            ilu_ops.dsa_update_topk_indices(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_tokens,
                topk_indices_buffer[chunk.token_start : chunk.token_end],
            )

    if has_decode:
        decode_metadata = attn_metadata.decode
        assert decode_metadata is not None
        if decode_metadata.max_context_len == 0:
            # Early decode steps can have no KV entries yet: no top-k
            # candidates, and the logits kernel cannot materialize a matrix
            # with zero columns. Write -1 (no selection) and return.
            topk_indices_buffer[:num_decode_tokens].fill_(-1)
            return topk_indices_buffer

        # Ragged MTP decode batches (for example query lengths [4, 1]) are
        # represented by cu_seqlens_q plus one causal [ks, ke) span per query
        # token. The same variable-query logits kernel used for prefill can
        # therefore handle requires_padding=True without an explicit pack step.
        logits = _mqa_logits(
            q[:num_decode_tokens],
            decode_metadata.cu_seqlens_q,
            decode_metadata.cu_seqlens_kv,
            kv_cache,
            decode_metadata.block_table,
            weights[:num_decode_tokens],
            max_q_len=decode_metadata.max_q_len,
            max_kv_len=decode_metadata.max_kv_len,
            max_context_len=decode_metadata.max_context_len,
            valid_lens=decode_metadata.valid_lens,
        )

        ilu_ops.dsa_update_topk_indices(
            logits,
            decode_metadata.cu_seqlen_ks,
            decode_metadata.cu_seqlen_ke,
            topk_tokens,
            topk_indices_buffer[:num_decode_tokens],
        )

    return topk_indices_buffer


def ilu_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool = False,
) -> torch.Tensor:
    assert topk_indices_buffer is not None
    return topk_indices_buffer


direct_register_custom_op(
    op_name=_PLUGIN_INDEXER_OP,
    op_func=ilu_sparse_attn_indexer,
    mutates_args=["kv_cache", "topk_indices_buffer"],
    fake_impl=ilu_sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


class IluSparseAttnIndexer(SparseAttnIndexer):
    """Sparse Attention Indexer Custom Op Layer for Iluvatar DSA models."""

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        CustomOp.__init__(self)
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        skip_k_cache_insert: bool = False,
    ):
        if current_platform.is_cuda():
            return self.forward_cuda(
                hidden_states, q_quant, k, weights, skip_k_cache_insert
            )
        raise NotImplementedError(
            "SparseAttnIndexer native forward is only implemented for CUDA platforms."
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        skip_k_cache_insert: bool = False,
    ):
        return getattr(torch.ops.vllm, _PLUGIN_INDEXER_OP)(
            hidden_states,
            # This plugin op has a string schema; LayerName belongs to community ops.
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            q_quant,
            k,
            weights,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert or skip_k_cache_insert,
        )
