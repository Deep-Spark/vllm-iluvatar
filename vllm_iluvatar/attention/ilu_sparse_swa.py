# SPDX-License-Identifier: Apache-2.0
"""Iluvatar bf16 sliding-window-attention cache + backend for DeepSeek-V4.

Mirrors :class:`DeepseekV4SWACache` / :class:`DeepseekSparseSWABackend` from
[vllm/v1/attention/backends/mla/sparse_swa.py](sparse_swa.py) with three
behavioural changes:

1. ``IluDeepseekV4SWACache.dtype`` is :class:`torch.bfloat16` (not the
   uint8-bytes layout used by ``fp8_ds_mla``); the assertion is updated
   accordingly.
2. :meth:`get_kv_cache_spec` keeps ``model_version=None`` (the upstream
   ``"deepseek_v4"`` value is fp8-only and the spec asserts on any other
   string) and pins ``head_size_v=0`` so ``SlidingWindowSpec`` falls
   through to the single-latent bf16 size: ``block_size * head_size *
   dtype_size`` (``64 * 512 * 2 = 65536`` bytes/page, padded to ``65664``
   under ``alignment=576``). That figure deliberately matches
   ``IluDeepseekV4MLAAttention``'s C4A page so SWA + main-MLA layers
   continue to share a single ``KVCacheGroup`` / physical tensor.
3. :class:`IluDeepseekSparseSWABackend` removes ``fp8_ds_mla`` from
   ``supported_kv_cache_dtypes`` and routes ``get_kv_cache_shape`` through
   the parent's plain-bf16 branch only.

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
    """BF16 SWA cache layer for DeepSeek-V4 in ilu vllm.

    Same physical block layout as the upstream V4 SWA cache (block_size=64
    tokens to align with C4A's [256/4, head_dim] block), but stores raw
    BF16 values instead of the 584-byte fp8_ds_mla packed format.
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

        # Same constraint as upstream V4 SWA: SWA blocks share their physical
        # tensor with C4A KV blocks, which fixes block_size at 64.
        self.block_size = 64
        assert (
            self.dtype == torch.bfloat16
        ), f"IluDeepseekV4SWACache requires bfloat16, got {self.dtype}"

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # NOTE: ``model_version`` is left at None here so we fall through to
        # ``SlidingWindowSpec.real_page_size_bytes`` (single-latent bf16 K
        # store). Setting ``model_version="ilu_deepseek_v4"`` would trip the
        # upstream assertion in ``SlidingWindowMLASpec.real_page_size_bytes``
        # (which only special-cases the upstream ``"deepseek_v4"`` fp8 layout).
        # We pin ``head_size_v=0`` because MLA stores a single shared latent;
        # without this, ``SlidingWindowSpec.__post_init__`` defaults
        # ``head_size_v`` to ``head_size``, doubling the page allocation.
        #
        # NOTE: no ``alignment``. The upstream V4 SWA pads pages to 576B for
        # FlashMLA, but the ilu fork dispatches all SWA reads/writes through
        # ixformer kernels (``ds4_fused_qnorm_rope_kv_insert``,
        # ``ds4_flash_mla_with_kvcache`` and friends), which expect a
        # *contiguous* paged layout (see the strict ``is_contiguous()``
        # ``TORCH_CHECK`` in ``ds4_fused_qnorm_rope_kv_insert.cpp``).
        # Setting ``alignment=576`` triggers ``page_size_padded`` and the
        # ``as_strided`` branch in ``_reshape_kv_cache_tensors``, which leaves
        # gaps between blocks (per-block stride 65664 bf16 elems vs. natural
        # 65536) and makes the 2D view non-contiguous. Dropping the alignment
        # leaves the natural ``64 * 512 * 2 = 65536`` bytes/page; this still
        # matches ``IluDeepseekV4MLAAttention`` C4A's natural page size, so
        # SWA + main-MLA layers continue to share a single ``KVCacheGroup`` /
        # physical tensor.
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
    """BF16-only SWA backend; rejects fp8_ds_mla cache dtype."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

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
            raise ValueError(
                "IluDeepseekSparseSWABackend does not support "
                "cache_dtype_str=fp8_ds_mla; use --kv-cache-dtype bfloat16"
            )
        return (num_blocks, block_size, head_size)
