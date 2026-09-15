# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MiniMax-M3 W4A8 lightning indexer.

Indexer kernels via ``ixformer.inference.functions`` (same pattern as
DeepSeek-V4 ``dsa_indexer_*`` call sites). mxfp8 uses
``mxfp8/indexer.py``.
"""

from dataclasses import dataclass
from typing import ClassVar

import ixformer.inference.functions as ixfops
import torch
from torch import nn
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.attention import IndexerKVDType
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
)

logger = init_logger(__name__)


class IluMiniMaxM3IndexerBackend(AttentionBackend):
    """Indexer side-cache backend (key-only)."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # Native index kernels interpret cache as query dtype; index cache is BF16
    # only — do not advertise FP16 (would silently misread BF16 cache bytes).
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "MINIMAX_M3_SPARSE_INDEXER"

    @staticmethod
    def get_impl_cls() -> type:
        # Side cache only needs the metadata builder; forward is on
        # ``IluMiniMaxM3Indexer`` (module), not an AttentionImpl.
        return IluMiniMaxM3Indexer

    @staticmethod
    def get_builder_cls() -> type["IluMiniMaxM3IndexerMetadataBuilder"]:
        return IluMiniMaxM3IndexerMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]

    @classmethod
    def is_sparse(cls) -> bool:
        return True


class IluMiniMaxM3IndexerCache(nn.Module, AttentionLayerBase):
    """Side KV cache for the indexer's per-token index keys (key-only)."""

    def __init__(
        self,
        head_dim: int,
        prefix: str,
        cache_config: CacheConfig | None = None,
        indexer_kv_dtype: IndexerKVDType = "bf16",
    ) -> None:
        super().__init__()
        if indexer_kv_dtype == "bf16":
            cache_dtype = torch.bfloat16
        else:
            raise NotImplementedError(
                f"indexer_kv_dtype={indexer_kv_dtype!r} is not supported on "
                "Iluvatar MiniMax-M3 (native index kernels are BF16/FP16 only; "
                "use 'bf16')."
            )
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.indexer_kv_dtype = indexer_kv_dtype
        self.dtype = cache_dtype
        self.prefix = prefix
        self.cache_config = cache_config
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Key-only: MLAAttentionSpec budgets one vector/token.
        # Page size must equal sparse block size (== main KV page size).
        block_size = vllm_config.cache_config.block_size
        if block_size != 128:
            raise ValueError(
                f"MiniMax-M3 indexer cache requires block_size == 128, got {block_size}"
            )
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def forward(self) -> None: ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluMiniMaxM3IndexerBackend


@dataclass
class IluMiniMaxM3IndexerPrefillMetadata:
    cu_seqlens_q: torch.Tensor  # [num_prefills + 1] int32, rebased to 0
    seq_lens: torch.Tensor  # [num_prefills] int32
    context_lens: torch.Tensor  # [num_prefills] int32
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int


@dataclass
class IluMiniMaxM3IndexerDecodeMetadata:
    seq_lens: torch.Tensor  # [num_decodes] int32
    block_table: torch.Tensor
    max_seq_len: int
    decode_query_len: int
    max_decode_query_len: int


@dataclass
class IluMiniMaxM3IndexerMetadata(AttentionMetadata):
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    num_actual_tokens: int

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    prefill: IluMiniMaxM3IndexerPrefillMetadata | None = None
    decode: IluMiniMaxM3IndexerDecodeMetadata | None = None


class IluMiniMaxM3IndexerMetadataBuilder(
    AttentionMetadataBuilder[IluMiniMaxM3IndexerMetadata]
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
        hf_config = vllm_config.model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        sparse_cfg = text_config.sparse_attention_config
        total_index_heads = sparse_cfg["sparse_num_index_heads"]
        tp_size = get_tensor_model_parallel_world_size()
        if total_index_heads >= tp_size:
            if total_index_heads % tp_size != 0:
                raise ValueError(
                    f"sparse_num_index_heads={total_index_heads} must be "
                    f"divisible by tensor_parallel_size={tp_size}"
                )
        elif tp_size % total_index_heads != 0:
            raise ValueError(
                f"tensor_parallel_size={tp_size} must be divisible by "
                f"sparse_num_index_heads={total_index_heads}"
            )
        self.num_index_heads = max(1, total_index_heads // tp_size)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        assert self.reorder_batch_threshold is not None
        self.max_decode_query_len = self.reorder_batch_threshold

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
    ) -> IluMiniMaxM3IndexerMetadata:
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

        prefill_metadata: IluMiniMaxM3IndexerPrefillMetadata | None = None
        if num_prefills > 0:
            prefill_metadata = IluMiniMaxM3IndexerPrefillMetadata(
                cu_seqlens_q=(query_start_loc[num_decodes:] - num_decode_tokens).to(
                    torch.int32
                ),
                seq_lens=seq_lens[num_decodes:],
                context_lens=context_lens[num_decodes:],
                block_table=block_table[num_decodes:],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
            )

        decode_metadata: IluMiniMaxM3IndexerDecodeMetadata | None = None
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
                raise ValueError("MiniMax-M3 decode requires a uniform query length")
            if num_decode_tokens != num_decodes * decode_query_len:
                raise ValueError(
                    f"num_decode_tokens {num_decode_tokens} != "
                    f"num_decodes {num_decodes} * decode_query_len "
                    f"{decode_query_len}"
                )
            decode_metadata = IluMiniMaxM3IndexerDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
                max_seq_len=common_attn_metadata.max_seq_len,
                decode_query_len=decode_query_len,
                max_decode_query_len=self.max_decode_query_len,
            )

        return IluMiniMaxM3IndexerMetadata(
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


class IluMiniMaxM3Indexer(nn.Module):
    """Lightning indexer: owns the index-key cache and ixformer score/topk kernels.

    Mirrors DeepSeek-V4's flat ``IluDeepseekV4Indexer`` style (no impl
    selection layer).
    """

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
        local_blocks: int = 0,
        cache_config: CacheConfig | None = None,
        indexer_kv_dtype: IndexerKVDType = "bf16",
        topk_indices_buffer: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.num_index_heads = num_index_heads
        self.index_head_dim = index_head_dim
        self.indexer_kv_dtype = indexer_kv_dtype
        if topk_indices_buffer is None:
            raise ValueError(
                "MiniMax-M3 indexer requires a shared topk_indices_buffer "
                "allocated by the model"
            )
        # Shared model-level [max_T, H, K] storage. Native kernels pack
        # densely, and token-major layout makes ``buf[offset:offset+T]`` a
        # contiguous [T, H, K] workspace (decode then prefill).
        self.topk_indices_buffer = topk_indices_buffer
        self.index_cache = IluMiniMaxM3IndexerCache(
            head_dim=index_head_dim,
            prefix=f"{prefix}.index_cache",
            cache_config=cache_config,
            indexer_kv_dtype=indexer_kv_dtype,
        )

    def _dense_topk_out(
        self, num_tokens: int, *, token_offset: int = 0
    ) -> torch.Tensor | None:
        """Return a contiguous ``[num_tokens, H, K]`` view into the shared buffer.

        Decode uses offset 0; prefill is packed immediately after decode so
        both stay alive for the subsequent sparse kernels.
        """
        if num_tokens <= 0:
            return None
        buf = self.topk_indices_buffer
        if buf is None:
            raise RuntimeError(
                "MiniMax-M3 indexer requires a shared topk_indices_buffer "
                "allocated by the model"
            )
        start = int(token_offset)
        end = start + num_tokens
        if (
            buf.dim() != 3
            or buf.size(1) != self.num_index_heads
            or buf.size(2) != self.topk_blocks
        ):
            raise RuntimeError(
                "MiniMax-M3 topk buffer must be token-major "
                f"[max_T, H={self.num_index_heads}, K={self.topk_blocks}], "
                f"got {tuple(buf.shape)}"
            )
        if end > buf.size(0):
            raise RuntimeError(
                "MiniMax-M3 topk buffer is too small: "
                f"need {end} tokens, have {buf.size(0)}"
            )
        return buf[start:end]

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        index_md = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, IluMiniMaxM3IndexerMetadata)
        num_tokens = index_md.num_actual_tokens
        num_decode = index_md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None
        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            # Contiguous [num_decode, H, K] prefix of the shared
            # token-major buffer.
            decode_out = self._dense_topk_out(num_decode, token_offset=0)
            decode_topk = ixfops.minimax_m3_index_decode(
                iq[:num_decode],
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
                out=decode_out,
            )
        if index_md.num_prefills > 0:
            p = index_md.prefill
            assert p is not None
            np_ = index_md.num_prefill_tokens
            score = ixfops.minimax_m3_index_score(
                iq[num_decode:],
                kv,
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                p.context_lens,
                p.max_query_len,
                p.max_seq_len,
                self.num_kv_heads,
            )
            # Limit out to actual prefill tokens.
            prefill_out = self._dense_topk_out(np_, token_offset=num_decode)
            prefill_topk = ixfops.minimax_m3_index_topk(
                score,
                p.cu_seqlens_q,
                p.context_lens,
                p.max_query_len,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                out=prefill_out,
            )
        return decode_topk, prefill_topk
