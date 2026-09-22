# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MiniMax-M3 mxfp8 sparse attend.

Metadata and its builder come from
``vllm.models.minimax_m3.common.sparse_attention`` unchanged. What is local
here is the kernel dispatch: :class:`MiniMaxM3SparseImpl` (community) declares
``forward`` abstract precisely so each kernel family implements its own, so this
module subclasses that base rather than the community Triton impl.

Two Iluvatar deltas drive the code below: ixformer wants a contiguous
token-major ``[Q,H,K]`` top-k (the same layout as the shared buffer), and it
reads a K/V-split NHD cache instead of the community packed layout. FP8 KV has
no ixformer path, so the backend advertises BF16 only.
"""

from __future__ import annotations

from typing import ClassVar

import ixformer.inference.functions as ixf_ops
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
from vllm.v1.worker.workspace import current_workspace_manager

# Re-exported, not just imported: the metadata and its builder are community's,
# reused verbatim, which is what keeps this module a drop-in for
# ``common.sparse_attention`` and is asserted by
# ``test_minimax_m3_reuses_community_common_graph``. ``__all__`` (rather than a
# bare noqa) is what stops the unused-import fixer from deleting them.
__all__ = [
    "SUPPORTED_KV_CACHE_DTYPES",
    "SUPPORTED_KV_CACHE_TORCH_DTYPES",
    "MiniMaxM3SparseBackend",
    "MiniMaxM3SparseDecodeMetadata",
    "MiniMaxM3SparseImpl",
    "MiniMaxM3SparseIxformerImpl",
    "MiniMaxM3SparseMetadata",
    "MiniMaxM3SparseMetadataBuilder",
    "MiniMaxM3SparsePrefillMetadata",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
]

# Single source of truth for "which KV dtypes does the ixformer M3 sparse path
# accept". ``MiniMaxM3SparseBackend.supported_kv_cache_dtypes`` is what vLLM's
# cache-dtype resolution reads, ``get_kv_cache_shape`` gates the allocation,
# ``_check_unquantized_kv`` gates the kernel call, and
# ``mxfp8.model`` reuses the same list for its init-time guard --
# all four derive from here so they cannot drift apart.
SUPPORTED_KV_CACHE_DTYPES: tuple[CacheDType, ...] = ("bfloat16",)
SUPPORTED_KV_CACHE_TORCH_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16,)
_M3_SPARSE_PREFILL_BACKEND = "ixformer_perf"


def _dense_topk(topk_idx: torch.Tensor) -> torch.Tensor:
    """Return the dense token-major ``[Q,H,K]`` layout ixformer kernels want."""
    return topk_idx if topk_idx.is_contiguous() else topk_idx.contiguous()


def _check_unquantized_kv(
    kv_cache: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    if kv_cache.dtype not in SUPPORTED_KV_CACHE_TORCH_DTYPES:
        raise NotImplementedError(
            "ixformer M3 sparse attention requires KV in "
            f"{sorted(str(d) for d in SUPPORTED_KV_CACHE_TORCH_DTYPES)}, "
            f"got {kv_cache.dtype}"
        )
    if k_scale is not None or v_scale is not None:
        raise NotImplementedError("ixformer M3 sparse attention has no FP8 KV scales")


def _minimax_m3_sparse_attn_ixformer(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> None:
    _check_unquantized_kv(kv_cache, k_scale, v_scale)
    ixf_ops.minimax_m3_sparse_attn(
        q,
        kv_cache,
        _dense_topk(topk_idx),
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        num_kv_heads,
        sm_scale,
        output,
        workspace=workspace,
        backend=_M3_SPARSE_PREFILL_BACKEND,
    )


def minimax_m3_sparse_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    _minimax_m3_sparse_attn_ixformer(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        num_kv_heads,
        sm_scale,
        output,
        k_scale,
        v_scale,
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
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    _check_unquantized_kv(kv_cache, k_scale, v_scale)
    ixf_ops.minimax_m3_sparse_attn_decode(
        q,
        kv_cache,
        _dense_topk(topk_idx),
        block_table,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
        decode_query_len,
    )


class MiniMaxM3SparseBackend(_CommunitySparseBackend):
    """Community sparse backend with the ixformer K/V-split NHD cache."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = list(
        SUPPORTED_KV_CACHE_DTYPES
    )
    # Community's base also advertises float16 as a model dtype. Narrowing it
    # here means ``--dtype float16`` is rejected as "this backend does not run
    # float16" instead of surfacing later as a KV-cache-dtype error.
    supported_dtypes: ClassVar[list[torch.dtype]] = list(
        SUPPORTED_KV_CACHE_TORCH_DTYPES
    )

    @staticmethod
    def get_impl_cls() -> type[MiniMaxM3SparseIxformerImpl]:
        return MiniMaxM3SparseIxformerImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # ``"auto"`` means "follow the model dtype", which for MiniMax-M3 is
        # BF16; anything else must name a dtype this backend actually runs.
        if cache_dtype_str not in ("auto", *SUPPORTED_KV_CACHE_DTYPES):
            raise NotImplementedError(
                f"ixformer M3 sparse attention does not support {cache_dtype_str} "
                f"KV; supported: {('auto', *SUPPORTED_KV_CACHE_DTYPES)}"
            )
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            raise NotImplementedError("MiniMax-M3 does not use cross-layer KV blocks")
        return (0, 1, 2, 3, 4)


class MiniMaxM3SparseIxformerImpl(MiniMaxM3SparseImpl):
    """ixformer block-sparse attend over the indexer-selected blocks."""

    def reserve_prefill_workspace(self, total_q: int) -> torch.Tensor | None:
        if _M3_SPARSE_PREFILL_BACKEND != "ixformer_perf":
            return None
        workspace_bytes = ixf_ops.minimax_m3_sparse_attn_tcu_workspace_size(
            total_q, self.num_kv_heads, self.topk_blocks
        )
        (workspace,) = current_workspace_manager().get_simultaneous(
            ((workspace_bytes,), torch.uint8),
        )
        return workspace

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        query_fp8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend to the indexer-selected blocks. ``query_fp8`` is part of the
        community signature and unused: ix30 has no FP8-query attend."""
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            # Profiling reserves the maximum scratch once; every M3 layer then
            # borrows the same workspace-manager slot during real prefill.
            self.reserve_prefill_workspace(query.shape[0])
            return output  # profiling run; caches unbound
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)

        num_decode = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        # Indexer top-k from the shared token-major buffer: decode [:num_decode],
        # prefill packed immediately after.
        topk_buffer = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk_buffer is not None
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)

        # Decode [:num_decode]: split-K over the selected blocks
        # (request-major chunks).
        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None
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
            )

        # Prefill [num_decode:]: cu_seqlens_q already rebased to 0.
        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None
            prefill_topk = topk_buffer[num_decode:num_tokens]
            if p.max_query_len == 1:
                # Match the indexer's q=1 dispatch and reuse decode split-K.
                minimax_m3_sparse_attn_decode(
                    q[num_decode:],
                    kv_cache,
                    prefill_topk,
                    p.block_table,
                    p.seq_lens,
                    self.num_kv_heads,
                    self.scale,
                    out[num_decode:],
                    1,
                )
            else:
                workspace = self.reserve_prefill_workspace(q[num_decode:].shape[0])
                _minimax_m3_sparse_attn_ixformer(
                    q[num_decode:],
                    kv_cache,
                    prefill_topk,
                    p.block_table,
                    p.cu_seqlens_q,
                    p.seq_lens,
                    p.context_lens,
                    p.max_query_len,
                    self.num_kv_heads,
                    self.scale,
                    out[num_decode:],
                    workspace=workspace,
                )
        return output
