# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MiniMax-M3 W4A8 block-sparse attention.

Sparse attn kernels via ``ixformer.inference.functions`` (same pattern as
DeepSeek-V4 ``ds4_flash_mla_*`` / ``ds4_paged_sparse_attn_fwd`` call sites).

BF16 / mxfp8 use ``mxfp8/sparse_attention.py`` (token-major ``[Q,H,K]`` top-k).
This module is the W4A8 token-major ``[Q,H,K]`` path.
"""

from dataclasses import dataclass
from typing import ClassVar

import ixformer.inference.functions as ixfops
import torch
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, is_quantized_kv_cache

# One sparse block == one KV page (matches ixformer / upstream MiniMax-M3).
SPARSE_BLOCK_SIZE = 128
_SPARSE_PREFILL_BACKEND = "ixformer_perf"


def resolve_sparse_attention_cfg(config) -> dict[str, int]:
    """Single source for topk/init/local/block_size used by buffer and impl."""
    sparse_cfg = config.sparse_attention_config
    topk_blocks = int(sparse_cfg["sparse_topk_blocks"])
    init_blocks = int(sparse_cfg["sparse_init_block"])
    local_blocks = int(sparse_cfg["sparse_local_block"])
    block_size = int(sparse_cfg["sparse_block_size"])
    if topk_blocks <= 0:
        raise ValueError(
            f"MiniMax-M3 sparse_topk_blocks must be > 0, got {topk_blocks}"
        )
    if init_blocks < 0 or local_blocks < 0:
        raise ValueError(
            "MiniMax-M3 sparse_init_block and sparse_local_block must be "
            f">= 0, got init={init_blocks} local={local_blocks}"
        )
    if init_blocks + local_blocks > topk_blocks:
        raise ValueError(
            "MiniMax-M3 sparse_init_block + sparse_local_block "
            f"({init_blocks}+{local_blocks}) exceeds sparse_topk_blocks="
            f"{topk_blocks}"
        )
    if block_size != SPARSE_BLOCK_SIZE:
        raise ValueError(
            f"MiniMax-M3 sparse_block_size must be {SPARSE_BLOCK_SIZE}, "
            f"got {block_size}"
        )
    return {
        "topk_blocks": topk_blocks,
        "init_blocks": init_blocks,
        "local_blocks": local_blocks,
        "block_size": block_size,
        "num_index_heads": int(sparse_cfg["sparse_num_index_heads"]),
        "index_dim": int(sparse_cfg["sparse_index_dim"]),
    }


class IluMiniMaxM3SparseBackend(AttentionBackend):
    """Block-sparse GQA backend for MiniMax M3 sparse attention layers."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # Native sparse kernels are BF16/FP16 only; do not advertise FP8 until
    # ixformer implements FP8 dispatch + scale metadata.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "MINIMAX_M3_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["IluMiniMaxM3SparseImpl"]:
        return IluMiniMaxM3SparseImpl

    @staticmethod
    def get_builder_cls() -> type["IluMiniMaxM3SparseMetadataBuilder"]:
        return IluMiniMaxM3SparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Page size == sparse block size (one sparse block per KV page).
        return [128]

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            raise NotImplementedError  # no cross-layer KV blocks in M3
        # Native sparse kernels require NHD [B, 2, block, H, D] (size(2)==block).
        # Identity stride order is correct for that layout; reject HND from the
        # resolved layout (env override or connector default), not only the env.
        if get_kv_cache_layout() == "HND":
            raise ValueError(
                "Iluvatar MiniMax-M3 sparse attention requires NHD KV cache "
                "layout; HND is not supported"
            )
        return (0, 1, 2, 3, 4)


@dataclass
class IluMiniMaxM3SparsePrefillMetadata:
    """Per-prefill state for block-sparse attend."""

    cu_seqlens_q: torch.Tensor  # [num_prefills + 1] int32, rebased to 0
    seq_lens: torch.Tensor  # [num_prefills] int32, total KV lengths
    context_lens: torch.Tensor  # [num_prefills] int32 (cached/context tokens)
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int


@dataclass
class IluMiniMaxM3SparseDecodeMetadata:
    """Per-decode state (cudagraph-safe). ``decode_query_len`` is uniform
    per-request query length (1, or 1 + num_speculative_tokens)."""

    seq_lens: torch.Tensor  # [num_decodes] int32
    block_table: torch.Tensor
    decode_query_len: int


@dataclass
class IluMiniMaxM3SparseMetadata(AttentionMetadata):
    """Sparse-attention metadata, split into prefill and decode sub-metadata."""

    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    num_actual_tokens: int  # total query tokens (decode-first batch)

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    prefill: IluMiniMaxM3SparsePrefillMetadata | None = None
    decode: IluMiniMaxM3SparseDecodeMetadata | None = None


class IluMiniMaxM3SparseMetadataBuilder(
    AttentionMetadataBuilder[IluMiniMaxM3SparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.context_len_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> IluMiniMaxM3SparseMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )
        if num_decodes + num_prefills != num_reqs:
            raise ValueError(
                f"decode+prefill reqs {num_decodes + num_prefills} != "
                f"num_reqs {num_reqs}"
            )
        if num_decode_tokens + num_prefill_tokens != num_tokens:
            raise ValueError(
                f"decode+prefill tokens "
                f"{num_decode_tokens + num_prefill_tokens} != "
                f"num_tokens {num_tokens}"
            )

        context_lens = self.context_len_buffer[:num_reqs]
        context_lens.copy_(
            common_attn_metadata.compute_num_computed_tokens(), non_blocking=True
        )

        prefill_metadata: IluMiniMaxM3SparsePrefillMetadata | None = None
        if num_prefills > 0:
            prefill_metadata = IluMiniMaxM3SparsePrefillMetadata(
                cu_seqlens_q=(query_start_loc[num_decodes:] - num_decode_tokens).to(
                    torch.int32
                ),
                seq_lens=seq_lens[num_decodes:],
                context_lens=context_lens[num_decodes:],
                block_table=block_table[num_decodes:],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
            )

        decode_metadata: IluMiniMaxM3SparseDecodeMetadata | None = None
        if num_decodes > 0:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            if decode_query_len <= 0:
                raise ValueError(
                    f"decode_query_len must be > 0, got {decode_query_len}"
                )
            if not torch.all(
                (query_lens_cpu == decode_query_len) | (query_lens_cpu == 0)
            ):
                raise ValueError(
                    "MiniMax-M3 sparse decode requires a uniform query length"
                )
            if num_decode_tokens != num_decodes * decode_query_len:
                raise ValueError(
                    f"num_decode_tokens {num_decode_tokens} != "
                    f"num_decodes {num_decodes} * decode_query_len "
                    f"{decode_query_len}"
                )
            decode_metadata = IluMiniMaxM3SparseDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
                decode_query_len=decode_query_len,
            )

        return IluMiniMaxM3SparseMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )


class IluMiniMaxM3SparseImpl(AttentionImplBase[IluMiniMaxM3SparseMetadata]):
    """Iluvatar block-sparse GQA attend (ixformer kernels).

    The layer pre-inserts K/V and runs the indexer into the shared
    token-major ``topk_indices_buffer``; this impl slices that buffer.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_blocks: int,
        sparse_block_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.use_fp8_kv = is_quantized_kv_cache(kv_cache_dtype)
        if self.use_fp8_kv:
            raise NotImplementedError(
                f"Iluvatar MiniMax-M3 sparse attention does not support "
                f"kv_cache_dtype={kv_cache_dtype!r} yet (native kernels are "
                f"BF16/FP16 only)."
            )
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size
        if sparse_block_size != SPARSE_BLOCK_SIZE:
            raise ValueError(
                f"Iluvatar MiniMax-M3 requires sparse_block_size == "
                f"{SPARSE_BLOCK_SIZE}, got {sparse_block_size}"
            )

    def reserve_tcu_workspace(self, ref: torch.Tensor) -> None:
        """Make the TCU densify peak visible to vLLM memory profiling.

        The kernel allocates this scratch itself on every call, so the
        throwaway tensor here only has to raise the profiled peak before KV
        cache sizing (same trick as the GLM/DS dummy-run reservations).
        """
        if ref.numel() == 0:
            return
        nbytes = ixfops.minimax_m3_sparse_attn_tcu_workspace_size(
            ref.shape[0],
            self.num_kv_heads,
            self.topk_blocks,
        )
        torch.empty(nbytes, dtype=torch.uint8, device=ref.device)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        if not isinstance(main_md, IluMiniMaxM3SparseMetadata):
            raise TypeError(f"expected IluMiniMaxM3SparseMetadata, got {type(main_md)}")

        num_decode = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)

        indexer = layer.indexer  # type: ignore[attr-defined]
        topk_buf = indexer.topk_indices_buffer
        np_ = main_md.num_prefill_tokens
        decode_topk = topk_buf[:num_decode] if main_md.num_decodes > 0 else None
        prefill_topk = (
            topk_buf[num_decode : num_decode + np_]
            if main_md.num_prefills > 0
            else None
        )

        if main_md.num_decodes > 0:
            d = main_md.decode
            if d is None:
                raise RuntimeError(
                    "MiniMax-M3 sparse decode metadata missing decode sub-state"
                )
            if decode_topk is None:
                raise RuntimeError(
                    "MiniMax-M3 sparse decode missing topk_indices_buffer slice"
                )
            ixfops.minimax_m3_sparse_attn_decode(
                q[:num_decode],
                kv_cache,
                decode_topk,
                d.block_table,
                d.seq_lens,
                self.num_kv_heads,
                self.scale,
                out[:num_decode],
                d.decode_query_len,
            )

        if main_md.num_prefills > 0:
            p = main_md.prefill
            if p is None:
                raise RuntimeError(
                    "MiniMax-M3 sparse prefill metadata missing prefill sub-state"
                )
            np_ = main_md.num_prefill_tokens
            if prefill_topk is None:
                raise RuntimeError(
                    "MiniMax-M3 sparse prefill missing topk_indices_buffer slice"
                )
            ixfops.minimax_m3_sparse_attn(
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
                backend=_SPARSE_PREFILL_BACKEND,
            )
        return output
