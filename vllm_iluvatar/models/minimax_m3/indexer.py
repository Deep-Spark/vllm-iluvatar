# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MiniMax-M3 lightning indexer: ixformer score + top-k.

Side cache, metadata, and metadata builders come from
``vllm.models.minimax_m3.common.indexer`` unchanged. What is local here is the
kernel dispatch itself: :class:`MiniMaxM3IndexerImpl` (community) declares
``forward`` abstract precisely so each kernel family implements its own, so this
module subclasses that base rather than the community Triton impl.

The ixformer top-k kernels want an exact contiguous ``[H,Q,K]`` output while
vLLM hands over a transposed, Q-padded view of its persistent buffer, so the
adapters below let ixformer allocate and then publish into the active prefix.
The indexer cache is BF16-only: ixformer has no FP8 score path.
"""

from __future__ import annotations

from typing import ClassVar

import ixformer.inference.functions as ixf_ops
import torch
from vllm.config.attention import IndexerKVDType
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend as _CommunityIndexerBackend,
)
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerCache,
    MiniMaxM3IndexerDecodeMetadata,
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerMetadataBuilder,
    MiniMaxM3IndexerPrefillMetadata,
    MiniMaxM3IndexerTritonMetadataBuilder,
)
from vllm.v1.attention.backend import AttentionBackend

# Re-exported, not just imported: the side cache, metadata and builders are
# community's, reused verbatim. Naming them here is what keeps this module a
# drop-in for ``common.indexer`` and is asserted by
# ``test_minimax_m3_reuses_community_common_graph``. ``__all__`` (rather than a
# bare noqa) is what stops the unused-import fixer from deleting them.
__all__ = [
    "SUPPORTED_INDEXER_CACHE_DTYPES",
    "SUPPORTED_INDEXER_KV_DTYPES",
    "SUPPORTED_INDEXER_TORCH_DTYPES",
    "MiniMaxM3IndexerBackend",
    "MiniMaxM3IndexerCache",
    "MiniMaxM3IndexerDecodeMetadata",
    "MiniMaxM3IndexerImpl",
    "MiniMaxM3IndexerIxformerImpl",
    "MiniMaxM3IndexerMetadata",
    "MiniMaxM3IndexerMetadataBuilder",
    "MiniMaxM3IndexerPrefillMetadata",
    "MiniMaxM3IndexerTritonMetadataBuilder",
    "minimax_m3_index_decode",
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
]

# Single source of truth for the indexer side cache, mirroring
# ``sparse_attention.SUPPORTED_KV_CACHE_DTYPES``. ``SUPPORTED_INDEXER_KV_DTYPES``
# is the ``--attention-config indexer_kv_dtype`` spelling checked by the impl and
# by ``models.minimax_m3.model``; ``SUPPORTED_INDEXER_CACHE_DTYPES`` is the
# ``CacheDType`` spelling vLLM's cache-dtype resolution reads off the backend.
SUPPORTED_INDEXER_KV_DTYPES: tuple[IndexerKVDType, ...] = ("bf16",)
SUPPORTED_INDEXER_CACHE_DTYPES: tuple[CacheDType, ...] = ("bfloat16",)
SUPPORTED_INDEXER_TORCH_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16,)
# ixformer's score kernel reduces with max over the index heads; ``sum`` would
# need a different kernel, so it must fail rather than silently score wrong.
_SUPPORTED_SCORE_TYPES = ("max",)


def _publish_topk(result: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    """Publish an ixformer dense ``[H,Q,K]`` result into vLLM's output view.

    vLLM owns one persistent token-major ``[Qmax,H,K]`` buffer and passes its
    transposed ``[H,Qmax,K]`` view (or a suffix of that view) as ``out``.
    ixformer requires an exact contiguous ``[H,Q,K]`` output, so we let it
    allocate and copy into the active prefix of the persistent view.
    """
    if out is None:
        return result
    if out.ndim != 3:
        raise ValueError(f"MiniMax-M3 top-k out must be 3-D, got {out.ndim}")
    expected_prefix = (result.shape[0], result.shape[1], result.shape[2])
    if (
        out.shape[0] != expected_prefix[0]
        or out.shape[1] < expected_prefix[1]
        or out.shape[2] != expected_prefix[2]
    ):
        raise ValueError(
            "MiniMax-M3 top-k out must contain an [H,Q,K] prefix of "
            f"{expected_prefix}, got {tuple(out.shape)}"
        )
    out_prefix = out[:, : result.shape[1], :]
    out_prefix.copy_(result)
    return out_prefix


def minimax_m3_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
) -> torch.Tensor:
    return ixf_ops.minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        max_seq_len,
        num_kv_heads,
    )


def minimax_m3_index_topk(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    result = ixf_ops.minimax_m3_index_topk(
        score,
        cu_seqlens_q,
        prefix_lens,
        max_query_len,
        topk,
        init_blocks,
        local_blocks,
        None,
    )
    return _publish_topk(result, out)


def minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    out: torch.Tensor | None = None,
    score_out: torch.Tensor | None = None,
) -> torch.Tensor:
    if score_out is not None:
        raise NotImplementedError("ixformer M3 index decode has no score_out path")
    result = ixf_ops.minimax_m3_index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_query_len,
        max_decode_query_len,
        None,
    )
    return _publish_topk(result, out)


class MiniMaxM3IndexerBackend(_CommunityIndexerBackend):
    """Community indexer backend restricted to the BF16 side cache.

    ``get_builder_cls`` is inherited, so the builder is community's
    ``MiniMaxM3IndexerTritonMetadataBuilder``. Despite the name it is the
    generic metadata builder -- nothing in it is Triton-specific -- and it is
    reused verbatim. Its ``num_valid_pages_buffer`` is only read by the SM100
    MSA top-k path and stays unused here.
    """

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = list(
        SUPPORTED_INDEXER_CACHE_DTYPES
    )
    # As in ``sparse_attention``: community's base also advertises float16 as a
    # model dtype, which no ixformer indexer kernel here runs.
    supported_dtypes: ClassVar[list[torch.dtype]] = list(SUPPORTED_INDEXER_TORCH_DTYPES)

    @staticmethod
    def get_impl_cls() -> type[MiniMaxM3IndexerIxformerImpl]:
        return MiniMaxM3IndexerIxformerImpl


class MiniMaxM3IndexerIxformerImpl(MiniMaxM3IndexerImpl):
    """ixformer score + top-k for both prefill and decode."""

    indexer_backend_cls: ClassVar[type[AttentionBackend]] = MiniMaxM3IndexerBackend

    # Named rather than swallowed by ``**kwargs`` so the two guards below cannot
    # be bypassed and so the signature stays introspectable; everything else is
    # forwarded to the community base unchanged.
    def __init__(
        self,
        *,
        score_type: str = "max",
        indexer_kv_dtype: IndexerKVDType = "bf16",
        **kwargs,
    ) -> None:
        if indexer_kv_dtype not in SUPPORTED_INDEXER_KV_DTYPES:
            raise NotImplementedError(
                "Iluvatar MiniMax-M3 indexer requires indexer_kv_dtype in "
                f"{SUPPORTED_INDEXER_KV_DTYPES}, got {indexer_kv_dtype!r}."
            )
        if score_type not in _SUPPORTED_SCORE_TYPES:
            raise NotImplementedError(
                "Iluvatar MiniMax-M3 indexer only implements score_type in "
                f"{_SUPPORTED_SCORE_TYPES}, got {score_type!r}."
            )
        super().__init__(
            score_type=score_type,
            indexer_kv_dtype=indexer_kv_dtype,
            **kwargs,
        )

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        index_md = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, MiniMaxM3IndexerMetadata)
        num_tokens = index_md.num_actual_tokens
        nd = index_md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache

        # Both sides write into the single shared persistent topk_indices_buffer
        # (decode at [:, :nd], prefill at [:, nd:]) and return views into it.
        # None -> allocate fresh.
        buf = self.topk_indices_buffer
        buf_htk = None if buf is None else buf.transpose(0, 1)
        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None
        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            decode_topk = minimax_m3_index_decode(
                iq[:nd],
                kv,
                d.block_table,
                d.seq_lens,
                d.max_seq_len,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                self.num_kv_heads,
                d.decode_query_len,
                d.max_decode_query_len,
                out=buf_htk,
            )
        if index_md.num_prefills > 0:
            p = index_md.prefill
            assert p is not None
            score = minimax_m3_index_score(
                iq[nd:],
                kv,
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                p.context_lens,
                p.max_query_len,
                p.max_seq_len,
                self.num_kv_heads,
            )
            prefill_topk = minimax_m3_index_topk(
                score,
                p.cu_seqlens_q,
                p.context_lens,
                p.max_query_len,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                out=buf_htk[:, nd:, :] if buf_htk is not None else None,
            )
        return decode_topk, prefill_topk
