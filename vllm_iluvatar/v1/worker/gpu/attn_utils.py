# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/worker/gpu/attn_utils.py

ILU wrapper implementation for V2 KV-cache view alignment. Runtime wiring,
source fingerprint, and signature checks live in
``ilu_patch.general_patches.patch_attn_utils`` and are auto-discovered during
``register_general``.

Target   : vllm.v1.worker.gpu.attn_utils._align_mixed_attention_kv_cache_views
Baseline : main_6e448d0
Why      : Hybrid attention/Mamba models can place Mamba state pages and full
           attention KV pages in the same raw tensor. Attention backends may
           expose a K/V-first logical view, while Mamba always interprets a
           block as one contiguous page. Without normalizing the attention view,
           the same block ID maps to different byte ranges and shared-cache
           state can be overwritten.
How      : keep the community helper for non-Mamba cases; when Mamba is present,
           restride K/V-first attention views to block-first storage.
Scope    : V2 hybrid attention/Mamba KV cache initialization on Iluvatar.
Remove   : when upstream carries equivalent hybrid Mamba shared-cache layout
           alignment in vllm.v1.worker.gpu.attn_utils.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)

AlignFn = Callable[
    [
        Iterable[AttentionGroup],
        dict[str, Any],
        list[int],
        str,
        KVCacheConfig,
    ],
    None,
]


def make_align_mixed_attention_kv_cache_views(orig_align: AlignFn) -> AlignFn:
    @wraps(orig_align)
    def _align_mixed_attention_kv_cache_views(
        attn_groups: Iterable[AttentionGroup],
        kv_caches: dict[str, Any],
        kernel_block_sizes: list[int],
        cache_dtype: str,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        attn_groups = list(attn_groups)
        if not any(isinstance(g.kv_cache_spec, MambaSpec) for g in attn_groups):
            orig_align(
                attn_groups,
                kv_caches,
                kernel_block_sizes,
                cache_dtype,
                kv_cache_config,
            )
            return

        block_dims_by_layer = _get_attention_block_dims_by_layer(
            attn_groups,
            kv_caches,
            kernel_block_sizes,
            cache_dtype,
        )
        for layer_name, block_dim in block_dims_by_layer.items():
            if block_dim == 0:
                continue
            assert block_dim == 1
            _restride_kv_first_kv_cache_to_blocks_first_storage(kv_caches[layer_name])

    return _align_mixed_attention_kv_cache_views


def _get_attention_block_dims_by_layer(
    attn_groups: Iterable[AttentionGroup],
    kv_caches: dict[str, Any],
    kernel_block_sizes: list[int],
    cache_dtype: str,
) -> dict[str, int]:
    block_dims_by_layer: dict[str, int] = {}
    for group in attn_groups:
        kv_cache_spec = group.kv_cache_spec
        if not isinstance(kv_cache_spec, AttentionSpec):
            continue
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue
        block_dim = group.backend.get_kv_cache_block_dim(
            kernel_block_sizes[group.kv_cache_group_id],
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype,
        )
        for layer_name in group.layer_names:
            if layer_name in kv_caches:
                block_dims_by_layer[layer_name] = block_dim
    return block_dims_by_layer


def _restride_kv_first_kv_cache_to_blocks_first_storage(
    kv_cache: torch.Tensor,
) -> None:
    assert kv_cache.ndim >= 3
    assert kv_cache.shape[0] == 2
    hidden_size = kv_cache.shape[2:].numel()
    expected_tail_stride = torch.empty(kv_cache.shape[2:]).stride()
    if kv_cache.stride()[2:] != expected_tail_stride:
        logger.warning_once(
            "Skipping hybrid Mamba KV-cache layout alignment for a non-NHD "
            "K/V-first attention view with stride %s.",
            kv_cache.stride(),
        )
        return
    kv_cache.as_strided_(
        size=kv_cache.shape,
        stride=(hidden_size, 2 * hidden_size, *expected_tail_stride),
    )
