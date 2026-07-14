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

The metadata builder (``DeepseekSparseSWAMetadataBuilder``) is reused as-is
because its build/forward path is dtype-agnostic.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec


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
        return DeepseekSparseSWAMetadataBuilder

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
