# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# Upstream reference: vLLM v0.27.1 (6e448d0),
# vllm/models/minimax_m3/common/indexer.py.
# Modified by Shanghai Iluvatar CoreX Semiconductor Co., Ltd. for the ixaite backend.
"""Ixaite MiniMax-M3 lightning indexer (no ixformer / Triton MSA)."""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import CacheConfig
from vllm.config.attention import IndexerKVDType
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerCache,
    MiniMaxM3IndexerMetadata,
)

SUPPORTED_INDEXER_KV_DTYPES = ("bfloat16", "bf16", "auto")
_SUPPORTED_SCORE_TYPES = ("max",)


def decode_indexer_topk(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int = 128,
    topk: int = 16,
    sm_scale: float | None = None,
    init_blocks: int = 0,
    local_blocks: int = 1,
    reference: bool | None = None,
) -> torch.Tensor:
    """Decode indexer top-k via ``ixaite.ops.minimax_sparse_indexer``.

    ``q`` / ``k_cache`` follow the ixaite contract; ``reference`` defaults to
    True off CUDA so unit tests stay device-free.
    """
    use_ref = (not torch.cuda.is_available()) if reference is None else reference
    from ixaite.ops.minimax_sparse_indexer import decode_indexer_topk as _decode

    return _decode(
        q,
        k_cache,
        seq_lens,
        block_size=block_size,
        topk=topk,
        sm_scale=sm_scale,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        reference=use_ref,
    )


def require_supported_indexer(
    *, score_type: str = "max", indexer_kv_dtype: str = "bf16"
) -> None:
    if indexer_kv_dtype not in SUPPORTED_INDEXER_KV_DTYPES:
        raise NotImplementedError(
            "ixaite MiniMax indexer requires indexer_kv_dtype in "
            f"{SUPPORTED_INDEXER_KV_DTYPES}, got {indexer_kv_dtype!r}"
        )
    if score_type not in _SUPPORTED_SCORE_TYPES:
        raise NotImplementedError(
            "ixaite MiniMax indexer only implements score_type in "
            f"{_SUPPORTED_SCORE_TYPES}, got {score_type!r}"
        )


def gather_index_k_paged(
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int | None = 128,
) -> torch.Tensor:
    """Gather key-only pages ``[blocks, page, D]`` into dense ``[B, S, 1, D]``.

    Vectorized. When ``block_size`` is set, pad ``S`` to a multiple of it.
    """
    if index_cache.ndim != 3:
        raise ValueError(
            "ixaite MiniMax indexer cache expects [blocks, page, D], "
            f"got {tuple(index_cache.shape)}"
        )
    page = int(index_cache.shape[1])
    head_dim = int(index_cache.shape[2])
    batch = int(block_table.shape[0])
    max_seq = 1 if seq_lens.numel() == 0 else max(int(seq_lens.max().item()), 1)
    if block_size is not None:
        bs = max(int(block_size), 1)
        max_seq = ((max_seq + bs - 1) // bs) * bs
    out = torch.zeros(
        batch,
        max_seq,
        1,
        head_dim,
        device=index_cache.device,
        dtype=index_cache.dtype,
    )
    if max_seq <= 0 or batch <= 0:
        return out
    tok = torch.arange(max_seq, device=index_cache.device, dtype=torch.long)
    page_idx = torch.div(tok, page, rounding_mode="floor")
    page_off = tok - page_idx * page
    bt_w = int(block_table.shape[1])
    page_idx_c = page_idx.clamp(max=max(bt_w - 1, 0))
    phys = block_table[:, page_idx_c].to(dtype=torch.long).clamp(min=0)
    po = page_off.unsqueeze(0).expand(batch, -1)
    out[:, :, 0, :] = index_cache[phys, po]
    mask = tok.unsqueeze(0) < seq_lens.to(
        device=index_cache.device, dtype=torch.long
    ).unsqueeze(1)
    out = out.masked_fill(~mask.unsqueeze(-1).unsqueeze(-1), 0)
    return out


# Back-compat alias used by older call sites / tests.
_gather_index_k_paged = gather_index_k_paged


def _insert_index_k(
    index_k: torch.Tensor,
    index_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter ``[T, D]`` index keys into paged ``[blocks, page, D]``."""
    if index_cache.numel() == 0 or index_k.numel() == 0:
        return
    page = int(index_cache.shape[1])
    # Graph-safe: clamp PAD slots instead of boolean filtering / host sync.
    slots = slot_mapping.long().clamp(min=0)
    blocks = torch.div(slots, page, rounding_mode="floor")
    offs = slots % page
    index_cache[blocks, offs] = index_k


def _decode_indexer_topk_fused(
    q: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int,
    topk: int,
    sm_scale: float,
    init_blocks: int,
    local_blocks: int,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Prefer paged indexer when MQA; else vectorized dense gather."""
    page = int(index_cache.shape[1])
    from vllm_iluvatar.backends.ixaite.models.minimax_m3.sparse_attention import (
        _bin_seqlen,
    )

    if max_seq_len is not None and int(max_seq_len) > 0:
        max_seq = _bin_seqlen(int(max_seq_len), block_size)
    else:
        max_seq = _bin_seqlen(int(seq_lens.max().item()), block_size)
    table_cap = int(block_table.shape[1]) * page
    if table_cap > 0:
        max_seq = min(
            max_seq, ((table_cap + block_size - 1) // block_size) * block_size
        )

    if q.is_cuda and int(q.shape[1]) == 1:
        try:
            from ixaite.ops.minimax_sparse_indexer import decode_indexer_topk_paged

            from vllm_iluvatar.backends.ixaite.models.minimax_m3.sparse_attention import (
                block_table_to_req_locs,
            )

            k_pool = index_cache.reshape(-1, 1, int(index_cache.shape[-1]))
            locs = block_table_to_req_locs(block_table, max_seq, page)
            return decode_indexer_topk_paged(
                q,
                k_pool,
                locs,
                seq_lens.to(dtype=torch.int32),
                block_size=block_size,
                topk=topk,
                sm_scale=sm_scale,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
            )
        except (ImportError, ValueError, RuntimeError):
            pass

    k_dense = gather_index_k_paged(
        index_cache, block_table, seq_lens, block_size=block_size
    )
    if k_dense.shape[2] == 1 and q.shape[1] > 1:
        k_dense = k_dense.expand(-1, -1, int(q.shape[1]), -1).contiguous()
    return decode_indexer_topk(
        q,
        k_dense,
        seq_lens.to(dtype=torch.int32),
        block_size=block_size,
        topk=topk,
        sm_scale=sm_scale,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        reference=False if q.is_cuda else None,
    )


class IxaiteMiniMaxM3Indexer(nn.Module):
    """Owns the index-key side cache and fills the shared top-k buffer."""

    def __init__(
        self,
        *,
        num_kv_heads: int,
        scale: float,
        topk_blocks: int,
        sparse_block_size: int,
        num_index_heads: int,
        index_head_dim: int,
        prefix: str,
        init_blocks: int = 0,
        local_blocks: int = 1,
        score_type: str = "max",
        cache_config: CacheConfig | None = None,
        indexer_kv_dtype: IndexerKVDType = "bf16",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        require_supported_indexer(
            score_type=score_type, indexer_kv_dtype=str(indexer_kv_dtype)
        )
        self.num_kv_heads = int(num_kv_heads)
        self.scale = float(scale)
        self.topk_blocks = int(topk_blocks)
        self.block_size = int(sparse_block_size)
        self.init_blocks = int(init_blocks)
        self.local_blocks = int(local_blocks)
        self.num_index_heads = int(num_index_heads)
        self.index_head_dim = int(index_head_dim)
        self.topk_indices_buffer = topk_indices_buffer
        # Community cache registers ``{prefix}.index_cache`` for slot_mapping.
        self.index_cache = MiniMaxM3IndexerCache(
            head_dim=self.index_head_dim,
            prefix=f"{prefix}.index_cache",
            cache_config=cache_config,
            indexer_kv_dtype=indexer_kv_dtype,
        )

    def insert_keys(self, index_k: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        _insert_index_k(index_k, self.index_cache.kv_cache, slot_mapping)

    def _write_topk(
        self,
        topk_hbk: torch.Tensor,
        *,
        token_offset: int,
        num_tokens: int,
    ) -> None:
        """Write ixaite ``[H, B, K]`` topk into token-major ``[T, H, K]`` buffer."""
        buf = self.topk_indices_buffer
        if buf is None:
            raise RuntimeError(
                "ixaite MiniMax indexer requires a shared topk_indices_buffer"
            )
        # [H, B, K] -> [B, H, K]
        token_major = topk_hbk.permute(1, 0, 2).contiguous()
        if token_major.shape[0] != num_tokens:
            raise RuntimeError(
                f"topk batch {token_major.shape[0]} != num_tokens {num_tokens}"
            )
        end = token_offset + num_tokens
        if end > buf.shape[0]:
            raise RuntimeError(
                f"topk buffer too small: need {end}, have {buf.shape[0]}"
            )
        buf[token_offset:end].copy_(token_major.to(dtype=buf.dtype))

    def forward(
        self, index_query: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None
        index_md = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, MiniMaxM3IndexerMetadata)
        num_tokens = index_md.num_actual_tokens
        num_decode = index_md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache
        decode_topk = None
        prefill_topk = None

        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            if d.decode_query_len != 1:
                raise RuntimeError(
                    "ixaite MiniMax indexer decode currently requires "
                    f"decode_query_len==1, got {d.decode_query_len}"
                )
            q = iq[:num_decode]
            topk = _decode_indexer_topk_fused(
                q,
                kv,
                d.block_table,
                d.seq_lens,
                block_size=self.block_size,
                topk=self.topk_blocks,
                sm_scale=self.scale,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                max_seq_len=int(d.max_seq_len),
            )
            self._write_topk(topk, token_offset=0, num_tokens=num_decode)
            decode_topk = topk

        if index_md.num_prefills > 0:
            p = index_md.prefill
            assert p is not None
            np_ = index_md.num_prefill_tokens
            q = iq[num_decode:num_tokens]
            if p.max_query_len == 1:
                topk = _decode_indexer_topk_fused(
                    q,
                    kv,
                    p.block_table,
                    p.seq_lens,
                    block_size=self.block_size,
                    topk=self.topk_blocks,
                    sm_scale=self.scale,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                    max_seq_len=int(p.max_seq_len),
                )
                self._write_topk(topk, token_offset=num_decode, num_tokens=np_)
                prefill_topk = topk
            else:
                # Multi-q: sparse attend recomputes causal top-k from the
                # stashed index tensors. Still fill a placeholder so the
                # shared buffer length stays consistent.
                if self.topk_indices_buffer is not None:
                    self.topk_indices_buffer[num_decode : num_decode + np_].fill_(-1)
                prefill_topk = None

        return decode_topk, prefill_topk


__all__ = [
    "SUPPORTED_INDEXER_KV_DTYPES",
    "IxaiteMiniMaxM3Indexer",
    "decode_indexer_topk",
    "gather_index_k_paged",
    "require_supported_indexer",
]
