# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8 lightning indexer for Hunyuan V4 (grouped fp8 K cache).

The BF16 indexer in ``attention.py`` keeps K in the checkpoint dtype and feeds
the cuinfer bf16 mqa-logits kernel. This variant runs the same three steps on
the fp8 lane that DeepSeek V4 uses -- ``ds4_fused_indexer_q_rope_fp8_quant``,
a grouped fp8 K insert, and ``dsa_indexer_mqa_logits_with_kvcache_fp8`` --
which measures 2.2~2.6x faster on hy4's prefill shapes and up to 9x on decode,
and halves the indexer K cache (132 B/token vs 256 B).

Three things make the reuse possible despite hy4 running ``index_n_heads=32``
against ds4's 64:

* ``index_head_dim`` is 128 on both. The shared FP8 facade dispatches to
  native H32/H64 specializations and honors explicit registry overrides.
  HY4 retains all 32 indexer heads on every TP rank.
* Both models build rotary embeddings with ``is_neox_style=False`` and keep
  the RoPE band in the last ``rope_dim`` dims, matching the fused q op's
  contract exactly.
* hy4 has no compressor. K is LayerNorm + tail RoPE + a grouped fp8
  scatter, fused as ``hy4_indexer_k_norm_rope_fp8_insert`` when that op is
  present (``VLLM_ILU_HYV4_FUSE_INDEXER_K=0`` keeps the three-kernel path).
"""

from __future__ import annotations

import os

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

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache.squeeze(1)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluDeepseekV32IndexerBackend


def _mixed_mrope_cos_sin(
    mrope_emb: torch.nn.Module,
    positions: torch.Tensor,
    dtype_ref: torch.Tensor,
) -> torch.Tensor:
    """Per-token ``cos || sin`` for a ``[3, T]`` M-RoPE position.

    The fused indexer kernels index a 1-D table and apply GPT-J RoPE. Feeding
    them ``arange(T)`` plus this table is the same rotation ``MRotaryEmbedding``
    would apply to the three rows.
    """
    from vllm.model_executor.layers.rotary_embedding.mrope import (
        apply_interleaved_rope,
    )

    cache = mrope_emb._match_cos_sin_cache_dtype(dtype_ref)
    gathered = cache[positions.long()]
    half = gathered.shape[-1] // 2
    cos = gathered[..., :half]
    sin = gathered[..., half:]
    section = [int(v) for v in mrope_emb.mrope_section]
    if mrope_emb.mrope_interleaved:
        cos = apply_interleaved_rope(cos, section)
        sin = apply_interleaved_rope(sin, section)
    else:
        cos_parts = []
        sin_parts = []
        offset = 0
        for row, width in enumerate(section):
            cos_parts.append(cos[row, ..., offset : offset + width])
            sin_parts.append(sin[row, ..., offset : offset + width])
            offset += width
        cos = torch.cat(cos_parts, dim=-1)
        sin = torch.cat(sin_parts, dim=-1)
    return torch.cat((cos, sin), dim=-1).contiguous()


def _indexer_rope_table(
    indexer,
    positions: torch.Tensor,
    rotary_emb: torch.nn.Module,
    dtype_ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(positions, cos_sin)`` the fused indexer kernels can consume.

    Three-row positions always use a mixed per-token table. This is also
    equivalent for text's identical rows and avoids a host read during capture.
    """
    if positions.ndim != 2:
        return positions.contiguous(), rotary_emb._match_cos_sin_cache_dtype(dtype_ref)
    mrope_emb = getattr(indexer, "mrope_emb", None)
    if mrope_emb is None or positions.shape[0] != 3:
        raise ValueError("HYV4 2-D positions require three rows and an M-RoPE module")
    section = getattr(mrope_emb, "_ixf_mrope_section", None)
    if positions.is_cuda and positions.shape[1] <= 16 and section is not None:
        return ixfops.mrope_cos_sin_table(
            positions,
            mrope_emb._match_cos_sin_cache_dtype(dtype_ref),
            section,
            mrope_emb.mrope_interleaved,
        )
    table = _mixed_mrope_cos_sin(mrope_emb, positions, dtype_ref)
    token_pos = torch.arange(table.shape[0], device=table.device, dtype=positions.dtype)
    return token_pos, table


def hy4_fp8_indexer_forward(
    indexer,
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: torch.nn.Module,
) -> torch.Tensor:
    """fp8 replacement for ``Indexer.forward``; ``indexer`` is that module."""
    indexer_positions, cos_sin = _indexer_rope_table(
        indexer, positions, rotary_emb, hidden_states
    )
    q, _ = indexer.wq_b(qr)
    q = q.view(-1, indexer.n_head, indexer.head_dim)

    kw, _ = indexer.wk_weights_proj(hidden_states)

    # Fused forward-RoPE + per-(token, head) ue8m0 fp8 quant. The op folds
    # softmax_scale, head_scale and the per-row q_scale into ``weights``, so
    # ``weights_in`` is the raw projection slice, *not* the pre-scaled
    # ``weights`` the bf16 path builds.
    q_fp8, weights = ixfops.fused_indexer_q_rope_quant(
        q,
        indexer_positions,
        cos_sin,
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
    if getattr(md, "pcp_world_size", 1) > 1:
        raise NotImplementedError(
            "HY4 FP8 indexer requires PCP=1 on this vLLM baseline"
        )
    # Grid follows slot_mapping's (CUDA-graph padded) length; padding rows
    # carry slot_mapping == -1 and are skipped inside the kernel.
    num_tokens = slot_mapping.shape[0]
    n = min(num_tokens, kw.shape[0], int(indexer_positions.numel()))
    k_raw = kw[:n, : indexer.head_dim]
    pos = indexer_positions[:n].contiguous()
    use_k_fuse = os.environ.get("VLLM_ILU_HYV4_FUSE_INDEXER_K", "1") != "0" and hasattr(
        ixfops, "hy4_indexer_k_norm_rope_fp8_insert"
    )
    if use_k_fuse:
        slots = slot_mapping.reshape(-1)[:n].contiguous()
        ixfops.hy4_indexer_k_norm_rope_fp8_insert(
            k_raw,
            indexer.k_norm.weight,
            indexer.k_norm.bias,
            pos,
            cos_sin,
            kv_cache,
            slots,
            float(indexer.k_norm.eps),
        )
    else:
        k = indexer.k_norm(k_raw)
        nope_dim = indexer.head_dim - indexer.rope_dim
        mrope_emb = getattr(indexer, "mrope_emb", None)
        if positions.ndim == 2:
            assert mrope_emb is not None  # validated by _indexer_rope_table
            # Same three-row rotation as the fused insert. ``rotary_emb`` here
            # is the 1-D module and would ignore the mixed table.
            k_pe = k[:, nope_dim:]
            k_out, _ = mrope_emb(positions[:, :n], k_pe, k_pe)
            k = k.clone()
            k[:, nope_dim:] = k_out.reshape(k_pe.shape)
        else:
            k_pe = k[:, nope_dim:].unsqueeze(1)
            k_pe_out, _ = rotary_emb(pos, k_pe)
            if k_pe_out is not k_pe:
                k_pe.copy_(k_pe_out.reshape(k_pe.shape))
        slots = slot_mapping.reshape(-1)[:n].contiguous()
        ixfops.hy4_indexer_k_fp8_insert(k, kv_cache, slots)

    topk_tokens = indexer.topk_tokens

    if md.num_prefills > 0:
        assert md.prefill is not None
        for chunk in md.prefill.chunks:
            if chunk.max_context_len == 0:
                continue
            logits = ixfops.dsa_indexer_mqa_logits_with_kvcache_fp8(
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
                context_lower_bound=chunk.context_lower_bound,
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

        logits = ixfops.dsa_indexer_mqa_logits_with_kvcache_fp8(
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
        )
        ixfops.dsa_update_topk_indices(
            logits,
            md.decode.cu_seqlen_ks,
            md.decode.cu_seqlen_ke,
            topk_tokens,
            topk_indices_buffer[:num_decode_tokens],
        )

    return topk_indices_buffer
