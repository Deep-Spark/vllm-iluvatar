# SPDX-License-Identifier: Apache-2.0
"""Iluvatar sliding-window-attention cache + backend for DeepSeek-V4.

Mirrors :class:`DeepseekV4SWACache` / :class:`DeepseekSparseSWABackend` from
[vllm/v1/attention/backends/mla/sparse_swa.py](sparse_swa.py).

Supports both bf16 and fp8_ds_mla KV cache formats:

- **bf16**: ``(num_blocks, block_size, head_dim)`` with ``dtype=bfloat16``.
  Page size = ``64 * 512 * 2 = 65536`` bytes.
- **fp8_ds_mla**: ``(num_blocks, block_size, 584)`` with ``dtype=uint8``.
  Per-token layout: 448B FP8 NoPE + 64*2B bf16 RoPE + 8B UE8M0 scale.
  Page size = ``64 * 584 = 37376`` bytes.

The metadata builder is subclassed as ``IluDeepseekSparseSWAMetadataBuilder``
to support ``VLLM_ILU_DS4_PREFILL_AS_DECODE``: when enabled, the SWA kernel
range is extended from decode-only to decode+prefill tokens so that each
prefill query token gets its own per-token SWA window indices/lens (reusing
the decode kernel ``ds4_flash_mla_with_kvcache`` for prefill).
"""

from __future__ import annotations

import logging
from typing import ClassVar

import ixformer.inference.functions as ixfops
import torch
import vllm.envs as envs

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec

logger = logging.getLogger(__name__)


class IluDeepseekV4SWACache(torch.nn.Module, AttentionLayerBase):
    """SWA cache layer for DeepSeek-V4 in ilu vllm.

    Supports bf16 (contiguous head_dim) and fp8_ds_mla (584-byte packed)
    layouts, selected via ``cache_config.cache_dtype``.
    """

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
    ) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.window_size = window_size
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.block_size = 64

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        cache_dtype = self.cache_config.cache_dtype
        is_fp8 = cache_dtype in ("fp8_ds_mla", "fp8")

        if is_fp8:
            return SlidingWindowMLASpec(
                block_size=self.block_size,
                num_kv_heads=1,
                head_size=self.head_dim,
                head_size_v=0,
                dtype=torch.uint8,
                sliding_window=self.window_size,
                cache_dtype_str="fp8_ds_mla",
                model_version="deepseek_v4",
                # Ilu fp8 attention kernels require 64B-aligned block bases.
                alignment=64,
            )
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            head_size_v=0,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            model_version=None,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluDeepseekSparseSWABackend


class IluDeepseekSparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    """Extends the community SWA metadata builder for PREFILL_AS_DECODE.

    Delegates to ``super().build()`` for the standard decode-side SWA
    computation, then post-processes: when ``VLLM_ILU_DS4_PREFILL_AS_DECODE``
    is enabled and prefill tokens are present, extends the SWA kernel range
    to cover the prefill token range and populates
    ``prefill_swa_indices`` / ``prefill_swa_lens`` on the returned metadata.

    This keeps the override minimal — upstream changes to the base
    ``build()`` are inherited automatically; only the prefill-as-decode
    delta is maintained here.
    """

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
    ) -> DeepseekSparseSWAMetadata:
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )

        if not envs.VLLM_ILU_DS4_PREFILL_AS_DECODE:
            return metadata

        num_prefill_tokens = metadata.num_prefill_tokens
        if num_prefill_tokens == 0:
            return metadata

        num_decode_tokens = metadata.num_decode_tokens
        num_swa_tokens = num_decode_tokens + num_prefill_tokens

        # Extend SWA computation to cover the prefill token range.
        # ds4_compute_swa_decode_indices_and_lens is purely position-based,
        # so extending the processed range yields correct causal SWA windows
        # for prefill tokens too.  The decode portion (0:num_decode_tokens)
        # is recomputed with identical results; the prefill portion
        # (num_decode_tokens:num_swa_tokens) is newly populated.
        self.decode_swa_lens[num_swa_tokens:] = 0
        ixfops.ds4_compute_swa_decode_indices_and_lens(
            swa_indices=self.decode_swa_indices,
            swa_lens=self.decode_swa_lens,
            window_size=self.window_size,
            query_start_loc=metadata.query_start_loc,
            seq_lens=metadata.seq_lens,
            token_to_req_indices=self.token_to_req_indices,
            is_valid_token=self.is_valid_token,
            block_table=metadata.block_table,
            block_size=self.block_size,
            num_decode_tokens=num_swa_tokens,
        )

        # Override prefill SWA fields with the computed slices.
        metadata.prefill_swa_indices = (
            self.decode_swa_indices[num_decode_tokens:num_swa_tokens]
        )
        metadata.prefill_swa_lens = (
            self.decode_swa_lens[num_decode_tokens:num_swa_tokens]
        )

        return metadata


class IluDeepseekSparseSWABackend(DeepseekSparseSWABackend):
    """SWA backend supporting bf16 and fp8_ds_mla cache."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto", "bfloat16", "fp8_ds_mla", "fp8",
    ]

    @staticmethod
    def get_name() -> str:
        return "ILU_DEEPSEEK_SPARSE_SWA"

    @staticmethod
    def get_builder_cls() -> type[DeepseekSparseSWAMetadataBuilder]:
        return IluDeepseekSparseSWAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        if cache_dtype_str == "fp8_ds_mla":
            return (num_blocks, block_size, 584)
        return (num_blocks, block_size, head_size)
