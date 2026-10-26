# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar Sparse Attention Indexer custom op layer."""

from __future__ import annotations

import ixformer.inference.functions as ixfops
import torch
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _encode_layer_name, direct_register_custom_op

from vllm_iluvatar.model_executor.kernels.attention.dsa.ilu_candidate_blocks import (
    apply_candidate_mask,
    apply_initialized_decode_candidate_mask,
    select_candidate_blocks,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV32IndexerMetadata,
)


def _two_level_filter(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    valid_lens: torch.Tensor,
    candidate_blocks: torch.Tensor | None,
    candidate_block_size: int,
    candidate_write: bool,
    token_start: int,
    token_end: int,
    *,
    initialized_decode: bool = False,
) -> None:
    """V4.1's coarse pass over the logits, in place, before the top-k.

    The layer at ``candidate_source_layer_id`` writes the shortlist; every
    index source above it masks against the shortlist that layer left behind.
    ``candidate_blocks`` is ``None`` on models without a candidate source and
    on the layers below it, which is what makes this a no-op there.
    """
    if candidate_blocks is None or not envs.VLLM_ILU_DS4_CANDIDATE_FILTER:
        return
    assert candidate_block_size > 0, (
        "a candidate source needs candidate_block_size; the model passed "
        f"{candidate_block_size}"
    )
    # ``cu_seqlen_ks`` is an offset into the concatenated KV space, not into
    # the logits: ``dsa_indexer_mqa_logits_with_blocks`` emits a row whose
    # column 0 is that request's first key, and ``dsa_update_topk_indices``
    # reads the span as ``ke - ks`` for exactly that reason. So the columns
    # start at zero and only the length carries over.
    # ``valid_lens`` is cu_seqlen_ke - cu_seqlen_ks
    rows = candidate_blocks[token_start:token_end]
    if candidate_write:
        select_candidate_blocks(
            logits,
            None,
            valid_lens,
            rows.shape[1],
            candidate_block_size,
            rows,
        )
    elif initialized_decode and candidate_block_size == 8:
        apply_initialized_decode_candidate_mask(logits, valid_lens, rows)
    else:
        apply_candidate_mask(
            logits,
            None,
            valid_lens,
            rows,
            candidate_block_size,
        )


def _mqa_logits(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    weights: torch.Tensor,
    *,
    max_q_len: int,
    max_kv_len: int,
    max_context_len: int,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    valid_lens: torch.Tensor,
    owned_logits: torch.Tensor | None = None,
    previous_extent: torch.Tensor | None = None,
    kv_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """One chunk's paged MQA logits, per the record the query carries.

    An INT8 query means an INT8 K cache, which only the ixinfer kernel reads.
    It derives each row's visibility from ``valid_lens`` where the span kernel
    takes the ``[ks, ke)`` pair; the two carry the same information, since the
    length is all either one uses (see ``_two_level_filter``).
    """
    if q.dtype is torch.int8:
        if kv_scales is not None:
            return ixfops.dsa_indexer_mqa_logits_split(
                q, cu_seqlens_q, cu_seqlens_kv, kv_cache, kv_scales,
                block_table, weights, max_q_len=max_q_len,
                max_kv_len=max_kv_len, max_context_len=max_context_len,
                valid_lens=valid_lens, logits=owned_logits,
                previous_extent=previous_extent,
            )
        if owned_logits is not None and previous_extent is not None:
            from ixformer.inference.functions.deepseek.indexer import (
                dsa_indexer_mqa_logits_ixinfer_owned,
            )
            return dsa_indexer_mqa_logits_ixinfer_owned(
                q, cu_seqlens_q, cu_seqlens_kv, kv_cache, block_table, weights,
                owned_logits, previous_extent, max_q_len=max_q_len,
                max_kv_len=max_kv_len, valid_lens=valid_lens)
        return ixfops.dsa_indexer_mqa_logits(
            q,
            cu_seqlens_q,
            cu_seqlens_kv,
            kv_cache,
            block_table,
            weights,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            max_context_len=max_context_len,
            valid_lens=valid_lens,
            backend="ixinfer",
        )
    return ixfops.dsa_indexer_mqa_logits_with_blocks(
        q,
        cu_seqlens_q,
        cu_seqlens_kv,
        kv_cache,
        block_table,
        weights,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        max_context_len=max_context_len,
        cu_seqlen_ks=cu_seqlen_ks,
        cu_seqlen_ke=cu_seqlen_ke,
    )


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
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
    owned_logits: torch.Tensor | None = None,
    previous_extent: torch.Tensor | None = None,
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

    cache_layer = get_forward_context().no_compile_layers.get(k_cache_prefix)
    kv_scales = (
        cache_layer.kv_cache_scales
        if getattr(cache_layer, "split_indexer", False) else None
    )
    if getattr(cache_layer, "split_indexer", False) and kv_scales is None:
        raise RuntimeError("Split indexer payload is missing its scale plane")
    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, IluDeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    # Grid follows slot_mapping's (CUDA-graph padded) length; truncate k if it
    # is longer. Padding rows carry slot_mapping == -1 and are skipped in the
    # kernel (see vllm_indexer_k_cache.cu ``if (slot_idx < 0) return;``).
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]
        if not skip_k_cache_insert:
            ops.indexer_k_cache(k, kv_cache, slot_mapping)
    elif not skip_k_cache_insert:
        raise RuntimeError("indexer k is required when skip_k_cache_insert is false")

    if has_prefill:
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata is not None
        for chunk in prefill_metadata.chunks:
            if chunk.max_context_len == 0:
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
                cu_seqlen_ks=chunk.cu_seqlen_ks,
                cu_seqlen_ke=chunk.cu_seqlen_ke,
                valid_lens=chunk.valid_lens,
                kv_scales=kv_scales,
            )
            _two_level_filter(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                chunk.valid_lens,
                candidate_blocks,
                candidate_block_size,
                candidate_write,
                chunk.token_start,
                chunk.token_end,
            )
            ixfops.dsa_update_topk_indices(
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

        use_owned = (
            not has_prefill and not decode_metadata.requires_padding
            and q.dtype == torch.int8 and q.shape[1:] == (32, 128)
            and num_decode_tokens in (6, 12, 18, 24)
            and decode_metadata.max_q_len == 6
            and decode_metadata.cu_seqlens_q.numel() == num_decode_tokens // 6 + 1
            and owned_logits is not None and previous_extent is not None
            and owned_logits.shape == (num_decode_tokens, decode_metadata.max_context_len)
            and owned_logits.dtype == torch.bfloat16
            and previous_extent.shape == (num_decode_tokens,)
            and previous_extent.dtype == torch.int32
            and owned_logits.device == previous_extent.device == q.device
        )

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
            cu_seqlen_ks=decode_metadata.cu_seqlen_ks,
            cu_seqlen_ke=decode_metadata.cu_seqlen_ke,
            valid_lens=decode_metadata.valid_lens,
            owned_logits=owned_logits if use_owned else None,
            previous_extent=previous_extent if use_owned else None,
            kv_scales=kv_scales,
        )

        _two_level_filter(
            logits,
            decode_metadata.cu_seqlen_ks,
            decode_metadata.cu_seqlen_ke,
            decode_metadata.valid_lens,
            candidate_blocks,
            candidate_block_size,
            candidate_write,
            0,
            num_decode_tokens,
            initialized_decode=q.dtype == torch.int8,
        )
        ixfops.dsa_update_topk_indices(
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
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
    owned_logits: torch.Tensor | None = None,
    previous_extent: torch.Tensor | None = None,
) -> torch.Tensor:
    assert topk_indices_buffer is not None
    return topk_indices_buffer


direct_register_custom_op(
    op_name="ilu_sparse_attn_indexer",
    op_func=ilu_sparse_attn_indexer,
    # The candidate source writes its shortlist into ``candidate_blocks``; the
    # layers above it only read, but the schema is shared.
    mutates_args=["topk_indices_buffer", "candidate_blocks", "owned_logits", "previous_extent"],
    fake_impl=ilu_sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@SparseAttnIndexer.register_oot()
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
        compress_ratio: int = 1,
        candidate_blocks: torch.Tensor | None = None,
        candidate_block_size: int = 0,
        candidate_write: bool = False,
        owned_decode_state: bool = False,
        max_num_decode_seqs: int = 2,
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
        self.use_fp4_cache = use_fp4_cache

        self.compress_ratio = compress_ratio
        self.candidate_blocks = candidate_blocks
        self.candidate_block_size = candidate_block_size
        self.candidate_write = candidate_write
        self.dense_mha_metadata_layer_name = ""
        # Width is already compressed; owners are outside graph-pool scratch.
        # Reserve only shapes allowed by the scheduler. Larger configurations
        # keep the public logits path without allocating owned decode state.
        self._owned_decode_tokens = (
            tuple(6 * batch for batch in range(1, max_num_decode_seqs + 1))
            if owned_decode_state and 1 <= max_num_decode_seqs <= 4 else ()
        )
        if self._owned_decode_tokens:
            tokens = self._owned_decode_tokens[-1]
            device = topk_indices_buffer.device
            stride = (max_model_len + 31) // 32 * 32
            logits = torch.empty((tokens, stride), dtype=torch.bfloat16,
                                 device=device)
            extent = torch.full((tokens,), -1, dtype=torch.int32, device=device)
            # Sequential graph shapes reuse physical rows. Exterior validity
            # therefore belongs to that storage, not to a graph's row count.
            # Register only owners; derive views after module/device transforms.
            self.register_buffer("_owned_decode_logits", logits, persistent=False)
            self.register_buffer("_owned_decode_extent", extent, persistent=False)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        raise NotImplementedError(
            "SparseAttnIndexer native forward is only implemented for CUDA platforms."
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        owned_logits = previous_extent = None
        if self._owned_decode_tokens:
            metadata = get_forward_context().attn_metadata
            if isinstance(metadata, dict):
                md = metadata.get(self.k_cache.prefix)
                if (isinstance(md, IluDeepseekV32IndexerMetadata)
                        and md.num_prefills == 0 and md.decode is not None
                        and not md.decode.requires_padding
                        and md.num_decode_tokens in self._owned_decode_tokens):
                    owned_logits = self._owned_decode_logits[
                        :md.num_decode_tokens, :self.max_model_len]
                    previous_extent = self._owned_decode_extent[:md.num_decode_tokens]
        return torch.ops.vllm.ilu_sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_quant,
            k,
            weights,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.candidate_blocks,
            self.candidate_block_size,
            self.candidate_write,
            owned_logits,
            previous_extent,
        )
