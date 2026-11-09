# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# Upstream reference: vLLM v0.27.1 (6e448d0),
# vllm/models/minimax_m3/common/sparse_attention.py.
# Modified by Shanghai Iluvatar CoreX Semiconductor Co., Ltd. for the ixaite backend.
"""Ixaite MiniMax-M3 sparse attend (no ixformer / NVIDIA MSA).

Community metadata and its builder stay unchanged. This module owns kernel
dispatch onto ``ixaite.ops.minimax_sparse_*`` and fails closed when the
FlyDSL path cannot run.

Decode prefers the SGLang-aligned fused path:
``paged_selected_kv_gather`` + ``sparse_decode_attn_selected``. Prefill with
``max_query_len>1`` tiles queries over the same selected gather/attend.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import torch
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend as _CommunitySparseBackend,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseDecodeMetadata,
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseMetadataBuilder,
    MiniMaxM3SparsePrefillMetadata,
)
from vllm.v1.attention.backend import AttentionLayer

logger = logging.getLogger(__name__)

__all__ = [
    "SUPPORTED_KV_CACHE_DTYPES",
    "SUPPORTED_KV_CACHE_TORCH_DTYPES",
    "MiniMaxM3SparseBackend",
    "MiniMaxM3SparseDecodeMetadata",
    "MiniMaxM3SparseIxaiteImpl",
    "MiniMaxM3SparseImpl",
    "MiniMaxM3SparseMetadata",
    "MiniMaxM3SparseMetadataBuilder",
    "MiniMaxM3SparsePrefillMetadata",
    "gather_paged_kv_nhd",
    "minimax_m3_sparse_attn_decode",
    "minimax_m3_sparse_attn_prefill",
]

SUPPORTED_KV_CACHE_DTYPES: tuple[CacheDType, ...] = ("bfloat16",)
SUPPORTED_KV_CACHE_TORCH_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16,)

_BACKEND_LOGGED = False


def _pad_seq(max_seq: int, block_size: int) -> int:
    bs = max(int(block_size), 1)
    max_seq = max(int(max_seq), 1)
    return ((max_seq + bs - 1) // bs) * bs


def _bin_seqlen(max_seq: int, block_size: int) -> int:
    """Snap to a coarse bin so FlyDSL keys stay stable across decode steps."""
    padded = _pad_seq(max_seq, block_size)
    for bin_sz in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
        if padded <= bin_sz:
            return bin_sz
    return padded


def gather_paged_kv_nhd(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_kv_heads: int,
    block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather vLLM ``[blocks, 2, page, Hkv, D]`` into dense ``[B, S, Hkv, D]``.

    Vectorized (no host ``.item()`` loops). When ``block_size`` is set, ``S`` is
    rounded up so ixaite dense sparse-decode contracts hold.
    """
    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise ValueError(
            "ixaite MiniMax sparse expects KV cache "
            f"[blocks, 2, page, Hkv, D], got {tuple(kv_cache.shape)}"
        )
    page = int(kv_cache.shape[2])
    head_dim = int(kv_cache.shape[-1])
    if int(kv_cache.shape[3]) != int(num_kv_heads):
        raise ValueError(
            f"KV head mismatch: cache Hkv={kv_cache.shape[3]} vs {num_kv_heads}"
        )
    batch = int(block_table.shape[0])
    if seq_lens.numel() == 0:
        max_seq = 1
    else:
        max_seq = int(seq_lens.max().item())
        max_seq = max(max_seq, 1)
    if block_size is not None:
        max_seq = _pad_seq(max_seq, block_size)
    device = kv_cache.device
    dtype = kv_cache.dtype
    k_out = torch.zeros(
        batch, max_seq, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    v_out = torch.zeros_like(k_out)
    if max_seq <= 0 or batch <= 0:
        return k_out, v_out

    tok = torch.arange(max_seq, device=device, dtype=torch.long)
    page_idx = torch.div(tok, page, rounding_mode="floor")
    page_off = tok - page_idx * page
    # Clamp page_idx into the block_table width.
    bt_w = int(block_table.shape[1])
    page_idx_c = page_idx.clamp(max=max(bt_w - 1, 0))
    # [B, S] physical block ids
    phys = block_table[:, page_idx_c].to(dtype=torch.long).clamp(min=0)
    # Advanced index: phys / page_off broadcast over heads.
    k_out[:, :, :, :] = kv_cache[phys, 0, page_off.unsqueeze(0).expand(batch, -1)]
    v_out[:, :, :, :] = kv_cache[phys, 1, page_off.unsqueeze(0).expand(batch, -1)]
    # Zero tokens past seq_lens (and padded tail).
    mask = tok.unsqueeze(0) < seq_lens.to(device=device, dtype=torch.long).unsqueeze(1)
    k_out = k_out.masked_fill(~mask.unsqueeze(-1).unsqueeze(-1), 0)
    v_out = v_out.masked_fill(~mask.unsqueeze(-1).unsqueeze(-1), 0)
    return k_out, v_out


def block_table_to_req_locs(
    block_table: torch.Tensor,
    max_seqlen: int,
    page: int,
) -> torch.Tensor:
    """Build SGLang-style ``req_to_token`` locs ``[B, S]`` from a block table.

    Physical token id = ``block_id * page + offset`` so a flat
    ``[blocks*page, ...]`` pool can be indexed without host syncs.
    """
    batch = int(block_table.shape[0])
    device = block_table.device
    tok = torch.arange(max_seqlen, device=device, dtype=torch.long)
    page_idx = torch.div(tok, page, rounding_mode="floor")
    page_off = tok - page_idx * page
    bt_w = int(block_table.shape[1])
    page_idx_c = page_idx.clamp(max=max(bt_w - 1, 0))
    phys = block_table[:, page_idx_c].to(dtype=torch.long).clamp(min=0)
    return (phys * int(page) + page_off.unsqueeze(0).expand(batch, -1)).to(
        dtype=torch.int32
    )


def _topk_to_hkv_major(topk_idx: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Map shared token-major ``[Q, H?, K]`` topk to ``[Hkv, B, topk]``."""
    if topk_idx.ndim != 3:
        raise ValueError(f"topk_idx must be 3-D, got {tuple(topk_idx.shape)}")
    if topk_idx.shape[1] == num_kv_heads:
        return topk_idx.permute(1, 0, 2).contiguous()
    if topk_idx.shape[0] == num_kv_heads:
        return topk_idx.contiguous()
    raise ValueError(
        f"topk_idx layout {tuple(topk_idx.shape)} does not match Hkv={num_kv_heads}"
    )


def _align_topk_heads(
    topk_idx: torch.Tensor, num_q_heads: int, num_kv_heads: int, topk: int
) -> torch.Tensor:
    """Expand indexer ``[Hq, B, K]`` to ``[Hkv, B, K]`` when Hq != Hkv."""
    if topk_idx.shape[0] == num_kv_heads:
        return topk_idx
    if topk_idx.shape[0] == num_q_heads and num_q_heads % num_kv_heads == 0:
        # Take the first head of each GQA group (indexer often MQA).
        group = num_q_heads // num_kv_heads
        return topk_idx[::group].contiguous()
    if topk_idx.shape[0] == 1 and num_kv_heads > 1:
        return topk_idx.expand(num_kv_heads, -1, -1).contiguous()
    if topk_idx.shape[0] != num_kv_heads:
        raise ValueError(
            f"cannot align topk heads {topk_idx.shape[0]} to Hkv={num_kv_heads} "
            f"(Hq={num_q_heads}, topk={topk})"
        )
    return topk_idx


def _kv_nhd_views(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """K/V as ``[blocks, page, Hkv, D]``. Prefer views; contiguous only if needed."""
    k_nhd = kv_cache[:, 0]
    v_nhd = kv_cache[:, 1]
    if not k_nhd.is_contiguous():
        k_nhd = k_nhd.contiguous()
    if not v_nhd.is_contiguous():
        v_nhd = v_nhd.contiguous()
    return k_nhd, v_nhd


def _selected_decode_cuda(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_hkv: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int,
    sm_scale: float,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Fused selected gather + attend (SGLang decode path)."""
    from ixaite.ops.minimax_sparse_attn import sparse_decode_attn_selected
    from ixaite.ops.minimax_sparse_gather import paged_selected_kv_gather

    page = int(kv_cache.shape[2])
    batch = int(seq_lens.shape[0])
    # Prefer host-side metadata max_seq_len (CUDA-graph safe). Fall back to a
    # sync only when the caller has no metadata (unit tests / dense fallback).
    if max_seq_len is not None and int(max_seq_len) > 0:
        max_seq = _bin_seqlen(int(max_seq_len), block_size)
    else:
        max_seq = _bin_seqlen(int(seq_lens.max().item()), block_size)
    table_cap = int(block_table.shape[1]) * page
    if table_cap > 0:
        max_seq = min(max_seq, _pad_seq(table_cap, block_size))
    req = block_table_to_req_locs(block_table, max_seq, page)
    slot_ids = torch.arange(batch, device=q.device, dtype=torch.int32)
    k_nhd, v_nhd = _kv_nhd_views(kv_cache)
    k_sel, v_sel, valid = paged_selected_kv_gather(
        k_nhd,
        v_nhd,
        req,
        slot_ids,
        seq_lens.to(dtype=torch.int32),
        topk_hkv,
        block_size=block_size,
        page_size=page,
    )
    return sparse_decode_attn_selected(
        q,
        k_sel,
        v_sel,
        valid,
        sm_scale=sm_scale,
    )


def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    *,
    block_size: int = 128,
    max_seq_len: int | None = None,
    reference: bool | None = None,
) -> None:
    """Decode sparse attend via ixaite (reference on CPU / when requested)."""
    del decode_query_len
    if kv_cache.dtype not in SUPPORTED_KV_CACHE_TORCH_DTYPES:
        raise NotImplementedError(
            "ixaite MiniMax sparse requires KV in "
            f"{SUPPORTED_KV_CACHE_TORCH_DTYPES}, got {kv_cache.dtype}"
        )
    use_ref = (not torch.cuda.is_available()) if reference is None else reference
    if q.ndim != 3:
        raise ValueError(f"q must be [T, Hq, D], got {tuple(q.shape)}")
    batch = int(seq_lens.shape[0])
    if q.shape[0] != batch:
        raise ValueError(
            f"decode q tokens {q.shape[0]} != batch {batch}; "
            "only decode_query_len==1 is wired for ixaite gather"
        )
    topk = _topk_to_hkv_major(topk_idx, num_kv_heads)

    if not use_ref and q.is_cuda:
        try:
            out = _selected_decode_cuda(
                q,
                kv_cache,
                topk,
                block_table,
                seq_lens,
                block_size=block_size,
                sm_scale=sm_scale,
                max_seq_len=max_seq_len,
            )
            output.copy_(out)
            return
        except (ImportError, ValueError, RuntimeError) as exc:
            logger.debug("fused selected decode fell back to dense: %s", exc)

    from ixaite.ops.minimax_sparse_attn import sparse_decode_attn

    k_cache, v_cache = gather_paged_kv_nhd(
        kv_cache,
        block_table,
        seq_lens,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
    )
    out = sparse_decode_attn(
        q,
        k_cache,
        v_cache,
        seq_lens.to(dtype=torch.int32),
        topk,
        block_size=block_size,
        sm_scale=sm_scale,
        reference=use_ref,
    )
    output.copy_(out)


def _causal_indexer_topk(
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    causal: torch.Tensor,
    *,
    block_size: int,
    topk: int,
    sm_scale: float | None,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Per-token top-k over one shared K (matches SGLang prefill indexer)."""
    from ixaite.ops.minimax_sparse_indexer_common import INIT_SCORE, LOCAL_SCORE

    if idx_k.ndim == 4:
        idx_k = idx_k[0]
    queries, num_q_heads, head_dim = idx_q.shape
    seq_len, num_kv_heads, _ = idx_k.shape
    if num_q_heads % num_kv_heads:
        raise ValueError(
            f"index heads {num_q_heads} not divisible by kv heads {num_kv_heads}"
        )
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    group = num_q_heads // num_kv_heads
    q_f = idx_q.float().view(queries, num_kv_heads, group, head_dim)
    qk = torch.einsum("thgd,shd->thgs", q_f, idx_k.float()) * float(sm_scale)
    causal = causal.to(device=idx_q.device, dtype=torch.long)
    n_blocks_total = (seq_len + block_size - 1) // block_size
    pad = n_blocks_total * block_size - seq_len
    if pad:
        qk = torch.nn.functional.pad(qk, (0, pad), value=float("-inf"))
    score = qk.view(queries, num_q_heads, n_blocks_total, block_size).amax(dim=-1)
    n_blocks = (causal + block_size - 1) // block_size
    boundary = (n_blocks - 1).clamp(min=0)
    rows = torch.arange(queries, device=idx_q.device)
    heads = torch.arange(num_q_heads, device=idx_q.device)
    qk_blocks = qk.view(queries, num_q_heads, n_blocks_total, block_size)
    bnd = qk_blocks[rows.unsqueeze(1), heads, boundary.unsqueeze(1)]
    bnd_pos = boundary.unsqueeze(-1) * int(block_size) + torch.arange(
        block_size, device=idx_q.device
    )
    bnd = bnd.masked_fill(
        bnd_pos.unsqueeze(1) >= causal.view(queries, 1, 1), float("-inf")
    )
    score[rows.unsqueeze(1), heads, boundary.unsqueeze(1)] = bnd.amax(dim=-1)
    if init_blocks > 0:
        score[:, :, :init_blocks] = INIT_SCORE
    block_id = torch.arange(n_blocks_total, device=idx_q.device)
    if local_blocks > 0:
        is_local = (block_id.unsqueeze(0) >= (n_blocks.unsqueeze(1) - local_blocks)) & (
            block_id.unsqueeze(0) < n_blocks.unsqueeze(1)
        )
        score = score.masked_fill(is_local.unsqueeze(1), LOCAL_SCORE)
    invalid = block_id.unsqueeze(0) >= n_blocks.unsqueeze(1)
    score = score.masked_fill(invalid.unsqueeze(1), float("-inf"))
    take = min(int(topk), n_blocks_total)
    _, indices = torch.topk(score, k=take, dim=-1)
    indices = torch.where(
        indices < n_blocks.view(queries, 1, 1),
        indices,
        torch.full_like(indices, -1),
    )
    if take < int(topk):
        filler = torch.full(
            (queries, num_q_heads, int(topk) - take),
            -1,
            device=idx_q.device,
            dtype=indices.dtype,
        )
        indices = torch.cat((indices, filler), dim=-1)
    return indices.permute(1, 0, 2).contiguous().to(torch.int32)


def minimax_m3_sparse_attn_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_buffer: torch.Tensor | None,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    *,
    block_size: int = 128,
    topk_blocks: int = 16,
    idx_q: torch.Tensor | None = None,
    idx_k_cache: torch.Tensor | None = None,
    idx_sm_scale: float | None = None,
    init_blocks: int = 0,
    local_blocks: int = 1,
) -> None:
    """Multi-query sparse prefill tiled like SGLang ``_sparse_prefill_queries``.

    When ``idx_q`` / ``idx_k_cache`` are provided, recomputes per-token causal
    top-k. Otherwise consumes the shared ``topk_buffer`` (q=1 compatible).
    """
    if kv_cache.dtype not in SUPPORTED_KV_CACHE_TORCH_DTYPES:
        raise NotImplementedError(
            "ixaite MiniMax sparse requires KV in "
            f"{SUPPORTED_KV_CACHE_TORCH_DTYPES}, got {kv_cache.dtype}"
        )
    num_reqs = int(seq_lens.shape[0])
    page = int(kv_cache.shape[2])
    for r in range(num_reqs):
        q0 = int(cu_seqlens_q[r].item())
        q1 = int(cu_seqlens_q[r + 1].item())
        if q1 <= q0:
            continue
        q_tok = q[q0:q1]
        tq = q1 - q0
        prefix = int(context_lens[r].item())
        final_len = int(seq_lens[r].item())
        bt_row = block_table[r : r + 1]
        if idx_q is not None and idx_k_cache is not None:
            from vllm_iluvatar.backends.ixaite.models.minimax_m3.indexer import (
                gather_index_k_paged,
            )

            idx_k = gather_index_k_paged(
                idx_k_cache,
                bt_row,
                torch.tensor([final_len], device=q.device, dtype=torch.int32),
                block_size=None,
            )
            if idx_k.shape[2] == 1 and idx_q.shape[1] > 1:
                idx_k = idx_k.expand(-1, -1, int(idx_q.shape[1]), -1)
            causal = (
                torch.arange(1, tq + 1, device=q.device, dtype=torch.int32) + prefix
            )
            topk_idx = _causal_indexer_topk(
                idx_q[q0:q1],
                idx_k,
                causal,
                block_size=block_size,
                topk=topk_blocks,
                sm_scale=idx_sm_scale,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
            )
            topk_idx = _align_topk_heads(
                topk_idx, int(idx_q.shape[1]), num_kv_heads, topk_blocks
            )
            # Tile selected attend; treat each query as its own batch row with
            # causal length as seq_lens for validity.
            from ixaite.ops.minimax_sparse_attn import sparse_decode_attn_selected
            from ixaite.ops.minimax_sparse_gather import paged_selected_kv_gather

            k_nhd, v_nhd = _kv_nhd_views(kv_cache)
            # Prefill: pad to final_len once (host value from this request).
            max_seq = _pad_seq(final_len, block_size)
            req_full = block_table_to_req_locs(bt_row, max_seq, page)
            tile = max(1, min(tq, 64))
            for s in range(0, tq, tile):
                e = min(tq, s + tile)
                rows = e - s
                topk_tile = topk_idx[:, s:e, :]
                causal_tile = causal[s:e]
                req = req_full.expand(rows, -1)
                slot_ids = torch.zeros(rows, device=q.device, dtype=torch.int32)
                k_sel, v_sel, valid = paged_selected_kv_gather(
                    k_nhd,
                    v_nhd,
                    req,
                    slot_ids,
                    causal_tile,
                    topk_tile,
                    block_size=block_size,
                    page_size=page,
                )
                output[q0 + s : q0 + e] = sparse_decode_attn_selected(
                    q_tok[s:e],
                    k_sel,
                    v_sel,
                    valid,
                    sm_scale=sm_scale,
                    softmax_in_bf16=True,
                )
        else:
            # Shared buffer already filled (typically q=1 chunks).
            assert topk_buffer is not None
            minimax_m3_sparse_attn_decode(
                q_tok,
                kv_cache,
                topk_buffer[q0:q1],
                bt_row.expand(tq, -1) if tq > 1 else bt_row,
                (
                    torch.full((tq,), final_len, device=q.device, dtype=torch.int32)
                    if tq > 1
                    else seq_lens[r : r + 1]
                ),
                num_kv_heads,
                sm_scale,
                output[q0:q1],
                1,
                block_size=block_size,
            )


class MiniMaxM3SparseBackend(_CommunitySparseBackend):
    """Community sparse backend with the ixaite K/V-split NHD gather path."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = list(
        SUPPORTED_KV_CACHE_DTYPES
    )
    supported_dtypes: ClassVar[list[torch.dtype]] = list(
        SUPPORTED_KV_CACHE_TORCH_DTYPES
    )

    @staticmethod
    def get_name() -> str:
        return "ILA_MINIMAX_M3_SPARSE_IXAITE"

    @staticmethod
    def get_impl_cls() -> type[MiniMaxM3SparseIxaiteImpl]:
        return MiniMaxM3SparseIxaiteImpl

    @staticmethod
    def get_builder_cls():
        return MiniMaxM3SparseMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str not in ("auto", *SUPPORTED_KV_CACHE_DTYPES):
            raise NotImplementedError(
                f"ixaite M3 sparse attention does not support {cache_dtype_str} "
                f"KV; supported: {('auto', *SUPPORTED_KV_CACHE_DTYPES)}"
            )
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            raise NotImplementedError("MiniMax-M3 does not use cross-layer KV blocks")
        # Physical (2, blocks, page, Hkv, D) so kv_cache[:, 0/1] are contiguous
        # byte ranges — required by paged_selected_kv_gather without a full
        # K/V copy every forward (matches AITER MiniMax stride_order).
        return (1, 0, 2, 3, 4)


class MiniMaxM3SparseIxaiteImpl(MiniMaxM3SparseImpl):
    """Ixaite block-sparse attend over the indexer-selected blocks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global _BACKEND_LOGGED
        if not _BACKEND_LOGGED:
            logger.info(
                "[IluMiniMaxSparse] backend=ixaite FlyDSL fused-selected "
                "block=%s topk=%s",
                getattr(self, "block_size", None),
                getattr(self, "topk_blocks", None),
            )
            _BACKEND_LOGGED = True

    def reserve_prefill_workspace(self, total_q: int) -> None:
        del total_q

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        query_fp8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del query_fp8
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            self.reserve_prefill_workspace(query.shape[0])
            return output
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)

        num_decode = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        topk_buffer = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk_buffer is not None
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        block_size = int(getattr(self, "block_size", 128) or 128)
        topk_blocks = int(getattr(self, "topk_blocks", 16) or 16)

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None
            assert isinstance(d, MiniMaxM3SparseDecodeMetadata)
            if d.decode_query_len != 1:
                raise RuntimeError(
                    "ixaite MiniMax sparse decode currently requires "
                    f"decode_query_len==1, got {d.decode_query_len}"
                )
            minimax_m3_sparse_attn_decode(
                q[:num_decode],
                kv_cache,
                topk_buffer[:num_decode],
                d.block_table,
                d.seq_lens,
                self.num_kv_heads,
                self.scale,
                out[:num_decode],
                d.decode_query_len,
                block_size=block_size,
                max_seq_len=int(main_md.max_seq_len),
            )

        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None
            assert isinstance(p, MiniMaxM3SparsePrefillMetadata)
            if p.max_query_len == 1:
                minimax_m3_sparse_attn_decode(
                    q[num_decode:],
                    kv_cache,
                    topk_buffer[num_decode:num_tokens],
                    p.block_table,
                    p.seq_lens,
                    self.num_kv_heads,
                    self.scale,
                    out[num_decode:],
                    1,
                    block_size=block_size,
                    max_seq_len=int(p.max_seq_len),
                )
            else:
                # Multi-q: prefer recomputing causal top-k when indexer cache
                # is reachable via the layer; else use the shared buffer.
                idx_q = getattr(layer, "_ilu_idx_q", None)
                idx_cache = getattr(layer, "_ilu_idx_k_cache", None)
                minimax_m3_sparse_attn_prefill(
                    q[num_decode:num_tokens],
                    kv_cache,
                    topk_buffer[num_decode:num_tokens],
                    p.block_table,
                    p.seq_lens,
                    p.context_lens,
                    p.cu_seqlens_q,
                    self.num_kv_heads,
                    self.scale,
                    out[num_decode:num_tokens],
                    block_size=block_size,
                    topk_blocks=topk_blocks,
                    idx_q=idx_q,
                    idx_k_cache=idx_cache,
                    idx_sm_scale=getattr(self, "idx_scale", None),
                    init_blocks=int(getattr(self, "init_blocks", 0) or 0),
                    local_blocks=int(getattr(self, "local_blocks", 1) or 1),
                )
        return output
