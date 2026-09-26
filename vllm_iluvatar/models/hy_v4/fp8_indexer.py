# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8 lightning indexer for Hunyuan V4 (grouped fp8 K cache).

The BF16 indexer in ``attention.py`` keeps K in the checkpoint dtype and feeds
the cuinfer bf16 mqa-logits kernel. This variant runs the same three steps on
the fp8 lane that DeepSeek V4 uses -- ``ds4_fused_indexer_q_rope_fp8_quant``,
a grouped fp8 K insert, and ``dsa_indexer_mqa_logits_with_blocks_fp8`` --
which measures 2.2~2.6x faster on hy4's prefill shapes and up to 9x on decode,
and halves the indexer K cache (132 B/token vs 256 B).

Three things make the reuse possible despite hy4 running ``index_n_heads=32``
against ds4's 64:

* ``index_head_dim`` is 128 on both, and that is the only compile-time
  constant the fp8 logits kernel bakes in. H reaches it as a tensor-descriptor
  field, which is why the ``ixinfer`` backend is pinned below -- the
  ``ixformer_native`` kernel *does* hard-code H=64 in its epilogue reduction.
* Both models build rotary embeddings with ``is_neox_style=False`` and keep
  the RoPE band in the last ``rope_dim`` dims, matching the fused q op's
  contract exactly.
* hy4 has no compressor, so the K side is a plain quantized scatter
  (``hy4_indexer_k_fp8_insert``) rather than ds4's fused pool/norm/rope insert.
"""

from __future__ import annotations

import ixformer.inference.functions as ixfops
import torch
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV32IndexerBackend,
    IluDeepseekV32IndexerMetadata,
)

# 4-byte fp32 ue8m0 scale per token (quant_block_size == index_head_dim).
_INDEXER_FP8_SCALE_BYTES = 4


class HYV4Fp8IndexerCache(torch.nn.Module, AttentionLayerBase):
    """Grouped fp8 indexer K cache: ``uint8 [num_blocks, block_size, D + 4]``.

    Within a block the ``[block_size, D]`` e4m3 K region and the
    ``[block_size]`` fp32 ue8m0 scale region are stored *grouped* rather than
    interleaved per token, so every K row stays 128B/16-aligned for the ixmma
    matrix loader.
    """

    def __init__(self, head_dim: int, prefix: str, cache_config: CacheConfig) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim + _INDEXER_FP8_SCALE_BYTES
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = torch.uint8
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluDeepseekV32IndexerBackend


def hy4_fp8_indexer_forward(
    indexer,
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: torch.nn.Module,
) -> torch.Tensor:
    """fp8 replacement for ``Indexer.forward``; ``indexer`` is that module."""
    q, _ = indexer.wq_b(qr)
    q = q.view(-1, indexer.n_head, indexer.head_dim)

    kw, _ = indexer.wk_weights_proj(hidden_states)
    k = indexer.k_norm(kw[:, : indexer.head_dim])

    # Only K needs the standalone rotary call now: the fused q op applies RoPE
    # itself. ``k_pe`` is [num_tokens, 1, rope_dim] (MQA) and is rotated in
    # place in the tail of k.
    nope_dim = indexer.head_dim - indexer.rope_dim
    k_pe = k[:, nope_dim:].unsqueeze(1)
    k_pe_out, _ = rotary_emb(positions, k_pe)
    if k_pe_out is not k_pe:
        # The CustomOp fallback returns fresh tensors instead of rotating in
        # place.
        k_pe.copy_(k_pe_out.reshape(k_pe.shape))

    # Fused forward-RoPE + per-(token, head) ue8m0 fp8 quant. The op folds
    # softmax_scale, head_scale and the per-row q_scale into ``weights``, so
    # ``weights_in`` is the raw projection slice, *not* the pre-scaled
    # ``weights`` the bf16 path builds.
    q_fp8, weights = ixfops.ds4_fused_indexer_q_rope_fp8_quant(
        q,
        positions,
        rotary_emb.cos_sin_cache,
        kw[:, indexer.head_dim :].contiguous(),
        indexer.softmax_scale,
        indexer.n_head_scale,
        indexer.rope_dim,
    )

    topk_indices_buffer = indexer.topk_indices_buffer
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return topk_indices_buffer
    md = attn_metadata.get(indexer.k_cache.prefix)
    if md is None:
        return topk_indices_buffer
    assert isinstance(md, IluDeepseekV32IndexerMetadata)

    kv_cache = indexer.k_cache.kv_cache
    slot_mapping = md.slot_mapping
    # Grid follows slot_mapping's (CUDA-graph padded) length; padding rows
    # carry slot_mapping == -1 and are skipped inside the kernel.
    num_tokens = slot_mapping.shape[0]
    ixfops.hy4_indexer_k_fp8_insert(k[:num_tokens], kv_cache, slot_mapping)

    topk_tokens = indexer.topk_tokens

    if md.num_prefills > 0:
        assert md.prefill is not None
        for chunk in md.prefill.chunks:
            if chunk.max_context_len == 0:
                continue
            logits = ixfops.dsa_indexer_mqa_logits_with_blocks_fp8(
                q_fp8[chunk.token_start : chunk.token_end],
                chunk.cu_seqlens_q,
                chunk.cu_seq_lens,
                kv_cache,
                chunk.block_table,
                weights[chunk.token_start : chunk.token_end],
                max_q_len=chunk.max_q_len,
                max_kv_len=chunk.max_kv_len,
                max_context_len=chunk.max_context_len,
                valid_lens=chunk.valid_lens,
                backend="ixinfer",
            )
            ixfops.dsa_update_topk_indices(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_tokens,
                topk_indices_buffer[chunk.token_start : chunk.token_end],
            )

    if md.num_decodes > 0:
        assert md.decode is not None
        num_decode_tokens = md.num_decode_tokens
        if md.decode.max_context_len == 0:
            # Early decode steps can have no KV entries yet: no top-k
            # candidates, and the logits kernel cannot materialize a matrix
            # with zero columns.
            topk_indices_buffer[:num_decode_tokens].fill_(-1)
            return topk_indices_buffer

        logits = ixfops.dsa_indexer_mqa_logits_with_blocks_fp8(
            q_fp8[:num_decode_tokens],
            md.decode.cu_seqlens_q,
            md.decode.cu_seqlens_kv,
            kv_cache,
            md.decode.block_table,
            weights[:num_decode_tokens],
            max_q_len=md.decode.max_q_len,
            max_kv_len=md.decode.max_kv_len,
            max_context_len=md.decode.max_context_len,
            valid_lens=md.decode.valid_lens,
            backend="ixinfer",
        )
        ixfops.dsa_update_topk_indices(
            logits,
            md.decode.cu_seqlen_ks,
            md.decode.cu_seqlen_ke,
            topk_tokens,
            topk_indices_buffer[:num_decode_tokens],
        )

    return topk_indices_buffer
