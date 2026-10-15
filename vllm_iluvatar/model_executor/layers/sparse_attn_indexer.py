# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar Sparse Attention Indexer custom op layer."""

from __future__ import annotations

import os

import ixformer.inference.functions as ixfops
import torch
from vllm import _custom_ops as ops
from vllm import envs
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.v1.attention.ops.pcp import maybe_gather_indexer_k
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_iluvatar.model_executor.kernels.attention.dsa.ilu_candidate_blocks import (
    apply_candidate_mask,
    select_candidate_blocks,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV32IndexerMetadata,
)


def _mqa_logits(*args, **kwargs):
    """Paged MQA logits for one chunk.

    Callers pass the same ``[cu_seqlen_ks, cu_seqlen_ke)`` span that top-k
    consumes. The logits kernels on this ixformer take that visibility as
    ``valid_lens`` (``ke - ks``): column 0 of a row is the request's first
    visible key, and the length is the only bound either side uses. INT8
    queries go to ``dsa_indexer_mqa_logits`` with ``backend="ixinfer"``; a
    uint8 cache with a bf16/fp8 query is the FP8 record; V4.1 BF16 cache
    uses ``ixformer_native`` (override with
    ``VLLM_ILU_V41_BF16_INDEXER_BACKEND``).
    """
    q = args[0]
    kv_cache = args[3] if len(args) > 3 else kwargs["kv_cache"]
    ks = kwargs.pop("cu_seqlen_ks", None)
    ke = kwargs.pop("cu_seqlen_ke", None)
    if kwargs.get("valid_lens") is None and ks is not None and ke is not None:
        kwargs["valid_lens"] = ke - ks
    if q.dtype is torch.int8:
        kwargs["backend"] = "ixinfer"
        return ixfops.dsa_indexer_mqa_logits(*args, **kwargs)
    if kv_cache.dtype == torch.uint8:
        # ixinfer supports only 64/256-row pages. Ratio1's 128 rows use native.
        kwargs.setdefault(
            "backend",
            os.environ.get("VLLM_ILU_V41_FP8_INDEXER_BACKEND", "ixformer_native"),
        )
        return ixfops.dsa_indexer_mqa_logits_with_blocks_fp8(*args, **kwargs)
    if envs.VLLM_ILU_GLM_DSA_NATIVE_MQA_LOGITS:
        kwargs.setdefault("backend", "ixformer_native")
    elif os.environ.get("VLLM_ILU_V41_INDEXER_KV_DTYPE", "").strip().lower() == "bf16":
        # Registry default is ixinfer (INT8 CQ). BF16 on ivcore11 needs the
        # native C4A kernel that honors valid_lens and the 64/128 page sizes.
        kwargs.setdefault(
            "backend",
            os.environ.get("VLLM_ILU_V41_BF16_INDEXER_BACKEND", "ixformer_native"),
        )
    return ixfops.dsa_indexer_mqa_logits(*args, **kwargs)


def _mqa_logits_for_candidates(candidate_blocks, *args, **kwargs):
    del candidate_blocks
    # Visibility has to reach the producer. The legacy with_blocks entry on
    # this ixformer ignores ``valid_lens`` and does not take the span pair, so
    # every record goes through ``_mqa_logits``.
    return _mqa_logits(*args, **kwargs)


def _two_level_filter(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    candidate_blocks: torch.Tensor | None,
    candidate_block_size: int,
    candidate_write: bool,
    token_start: int,
    token_end: int,
) -> None:
    if candidate_blocks is None:
        return
    assert candidate_block_size > 0
    row_ke = cu_seqlen_ke - cu_seqlen_ks
    rows = candidate_blocks[token_start:token_end]
    if candidate_write:
        select_candidate_blocks(
            logits,
            None,
            row_ke,
            rows.shape[1],
            candidate_block_size,
            rows,
        )
    else:
        apply_candidate_mask(
            logits,
            None,
            row_ke,
            rows,
            candidate_block_size,
        )


def _indexer_logits_and_topk(
    *,
    candidate_blocks: torch.Tensor | None,
    candidate_block_size: int,
    candidate_write: bool,
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    max_context_len: int,
    valid_lens: torch.Tensor,
    token_start: int,
    token_end: int,
    logits: torch.Tensor | None = None,
) -> None:
    """Score the indexer, mask to candidate blocks, then select the top-k.

    When the fp8 KV cache has no candidate-block stage (the common V4.1 path),
    scoring and top-k go through the fused ``dsv41_indexer_mqa_topk_fp8`` entry,
    which shares one logits buffer and leaves room for a single fused kernel.
    The candidate-block and bf16 paths keep the explicit logits -> mask -> top-k
    sequence, since the mask has to run between the two kernels.
    """
    if (
        candidate_blocks is None
        and kv_cache.dtype == torch.uint8
        and os.environ.get("VLLM_ILU_V41_INDEXER_KV_DTYPE") == "fp8"
    ):
        ixfops.dsv41_indexer_mqa_topk_fp8(
            q,
            cu_seqlens_q,
            cu_seqlens_kv,
            kv_cache,
            block_table,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            topk_tokens,
            topk_indices_buffer,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            max_context_len=max_context_len,
            valid_lens=valid_lens,
            logits=logits,
            backend=os.environ.get(
                "VLLM_ILU_V41_FP8_INDEXER_BACKEND", "ixformer_native"
            ),
        )
        return

    logits = _mqa_logits_for_candidates(
        candidate_blocks,
        q,
        cu_seqlens_q,
        cu_seqlens_kv,
        kv_cache,
        block_table,
        weights,
        logits=logits,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        max_context_len=max_context_len,
        valid_lens=valid_lens,
    )
    _two_level_filter(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        candidate_blocks,
        candidate_block_size,
        candidate_write,
        token_start,
        token_end,
    )
    ixfops.dsa_update_topk_indices(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_tokens,
        topk_indices_buffer,
    )


def ilu_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor | None,
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
            skip_k_cache_insert,
            candidate_blocks,
            candidate_block_size,
            candidate_write,
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
        assert k is not None
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
            _indexer_logits_and_topk(
                candidate_blocks=candidate_blocks,
                candidate_block_size=candidate_block_size,
                candidate_write=candidate_write,
                q=q[chunk.token_start : chunk.token_end],
                cu_seqlens_q=chunk.cu_seqlens_q,
                cu_seqlens_kv=chunk.cu_seq_lens,
                kv_cache=kv_cache,
                block_table=chunk.block_table,
                weights=weights[chunk.token_start : chunk.token_end],
                cu_seqlen_ks=chunk.cu_seqlen_ks,
                cu_seqlen_ke=chunk.cu_seqlen_ke,
                topk_tokens=topk_tokens,
                topk_indices_buffer=topk_indices_buffer[
                    chunk.token_start : chunk.token_end
                ],
                max_q_len=chunk.max_q_len,
                max_kv_len=chunk.max_kv_len,
                max_context_len=chunk.max_context_len,
                valid_lens=chunk.valid_lens,
                token_start=chunk.token_start,
                token_end=chunk.token_end,
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
        decode_logits = None
        cache_layer = get_forward_context().no_compile_layers.get(k_cache_prefix)
        logits_buffer = (
            getattr(cache_layer, "logits_buffer", None)
            if cache_layer is not None
            else None
        )
        if logits_buffer is not None:
            if num_decode_tokens > logits_buffer.shape[0]:
                raise ValueError(
                    "indexer decode exceeds preallocated logits rows"
                )
            decode_logits = logits_buffer[:num_decode_tokens, :decode_metadata.max_context_len]
        _indexer_logits_and_topk(
            candidate_blocks=candidate_blocks,
            candidate_block_size=candidate_block_size,
            candidate_write=candidate_write,
            q=q[:num_decode_tokens],
            cu_seqlens_q=decode_metadata.cu_seqlens_q,
            cu_seqlens_kv=decode_metadata.cu_seqlens_kv,
            kv_cache=kv_cache,
            block_table=decode_metadata.block_table,
            weights=weights[:num_decode_tokens],
            cu_seqlen_ks=decode_metadata.cu_seqlen_ks,
            cu_seqlen_ke=decode_metadata.cu_seqlen_ke,
            topk_tokens=topk_tokens,
            topk_indices_buffer=topk_indices_buffer[:num_decode_tokens],
            max_q_len=decode_metadata.max_q_len,
            max_kv_len=decode_metadata.max_kv_len,
            max_context_len=decode_metadata.max_context_len,
            valid_lens=decode_metadata.valid_lens,
            token_start=0,
            token_end=num_decode_tokens,
            logits=decode_logits,
        )

    return topk_indices_buffer


def ilu_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor | None,
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
) -> torch.Tensor:
    assert topk_indices_buffer is not None
    return topk_indices_buffer


direct_register_custom_op(
    op_name="ilu_sparse_attn_indexer",
    op_func=ilu_sparse_attn_indexer,
    mutates_args=["kv_cache", "topk_indices_buffer", "candidate_blocks"],
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
        self.compress_ratio = compress_ratio
        self.candidate_blocks = candidate_blocks
        self.candidate_block_size = candidate_block_size
        self.candidate_write = candidate_write
        self.skip_k_cache_insert = skip_k_cache_insert

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
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
        k: torch.Tensor | None,
        weights: torch.Tensor,
        skip_k_cache_insert: bool = False,
    ):
        return torch.ops.vllm.ilu_sparse_attn_indexer(
            hidden_states,
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
            self.candidate_blocks,
            self.candidate_block_size,
            self.candidate_write,
        )
