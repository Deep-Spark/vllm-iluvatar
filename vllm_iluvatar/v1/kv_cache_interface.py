# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/kv_cache_interface.py

ILU override implementation for Attention quantization support.

Target   : AttentionSpec.state_content_size_bytes (int8 per-cell for all specs)
           AttentionSpec.num_heads (disallow num_head_slots for int8)
           KVCacheTensor (scale_size fields)
Baseline : v0.31.0 (db9527a468)
Why      : Iluvatar attention quantization Int8 support.
How      : state_content_size_bytes drives int8 per-cell sizing through the
           community formula ``unpadded_page_size_bytes = num_heads *
           num_states * state_content_size_bytes`` (and its alias
           ``real_page_size_bytes``, plus ``compute_logical_shape``'s C dim).
           * KVINT8 / MLA-INT8: K and V both int8 -> mutate self.dtype to
             torch.int8 (frozen via object.__setattr__) so every reader of
             self.dtype sees int8; the community formula
             ``(head_size + head_size_v) * get_dtype_size(dtype)`` then yields
             the int8 per-cell. MLA is single-sided (head_size_v == 0 by class
             default), so this reduces to head_size -- correct for MLA.
           * KINT8: K int8 (1 byte) + V keeps the original dtype -> mixed
             per-cell, self.dtype untouched (head_size_v == 0 for MLA).
           SlidingWindowMLASpec may also carry flash int8, so no spec is
           guarded out. ``state_content_bytes`` (nvfp4 / fp8_ds_mla / other
           backends) is always honored via the fallback. The K scale lives in
           a separate standalone tensor (``scale_page_size_bytes`` /
           not in the page.
Scope    : always enabled if attention quantization is enabled.
Remove   : When upstream supports int8 kv cache quantization.
"""

from __future__ import annotations

import torch
import vllm.envs as envs
from vllm.v1.kv_cache_interface import get_dtype_size


def attention_num_heads(self) -> int:
    """Patch for community ``AttentionSpec.num_heads``.

    int8 quantization assumes one slot per KV head; a non-trivial
    ``num_head_slots`` (published by some backends) is incompatible with the
    ILU int8 layout/sizing, so reject it explicitly.
    """
    if self.num_head_slots is not None:
        raise ValueError("num_head_slots is not supported for int8 quantization")
    return self.num_kv_heads


def attention_state_content_size_bytes(self) -> int:
    """Patch for community ``AttentionSpec.state_content_size_bytes``.

    Drives int8 KV-cache sizing through the community per-cell formula so that
    ``unpadded_page_size_bytes = num_heads * num_states *
    state_content_size_bytes`` (and its alias ``real_page_size_bytes``, plus
    ``compute_logical_shape``'s C dim) all yield int8 sizing consistently.

    Applies to every AttentionSpec subclass: FullAttention / SlidingWindow
    (flash int8) and MLAAttentionSpec / SlidingWindowMLASpec (MLA int8). MLA
    is single-sided (``head_size_v == 0``), so ``(head_size + head_size_v)``
    collapses to ``head_size`` -- the correct single-latent per-cell.

    * KVINT8 / MLA-INT8: K and V both int8 -> mutate ``self.dtype`` to
      ``torch.int8`` so every reader of ``self.dtype`` sees int8 and the
      community per-cell formula yields int8.
    * KINT8: K int8 (1 byte) stored ALONE in the V2 buffer as a dense
      ``[B, H, N, head_size]`` int8 view (C = head_size); ``self.dtype`` is
      mutated to int8 so create_kv_cache_views' ``.view(int8)`` is an identity.
      V is NOT interleaved into C -- it is a separate dense bf16 buffer
      allocated per layer by allocate_kv_cache_scale and bound to the
      attention layer (see model_runner.py / ilu_flash_attn.py).

    Idempotency: this property has a side effect (mutating the frozen
    ``dtype`` to int8). It is called more than once on the same spec --
    get_kv_cache_config_from_groups sizes the backing buffer / block_stride
    first, then create_kv_cache_views computes dense_page_size later. If the
    size were recomputed on the second call, ``get_dtype_size(self.dtype)``
    would see the already-mutated int8 (size 1) and return a different
    (smaller) C, so block_stride != dense_page_size and the kernel-block-split
    check fails. Cache the computed value on the instance on the first call
    and return it ever after; the dtype mutation also runs only once.
    ``state_content_bytes`` (nvfp4 / fp8_ds_mla / ...) is rejected.
    """
    cached = self.__dict__.get("_ilu_state_content_size_bytes")
    if cached is not None:
        return cached
    if self.state_content_bytes is not None:
        raise ValueError("state_content_bytes is not supported for int8 quantization")
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MLAAttentionSpec,
        SlidingWindowMLASpec,
        SlidingWindowSpec,
    )

    is_mla = isinstance(self, (MLAAttentionSpec, SlidingWindowMLASpec))
    is_flash = isinstance(self, (FullAttentionSpec, SlidingWindowSpec))
    # NOTE >>> iluvatar attention quantization
    if is_flash and envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE in ("KINT8", "KVINT8"):
        # K int8 (1 byte) is stored ALONE in the V2 buffer as a dense
        # [B, H, N, head_size] int8 view. With C = head_size the LBNHC
        # per-layer view has token stride = head_size and block stride =
        # H*head_size*block_size -- exactly the fixed-dense stride the ILU
        # int8 kernels (reshape_and_cache_with_scale / vllm_paged_attention_int8
        # / cuInferPageAttentionV7) assume, and the view is contiguous, so the
        # kernels' is_contiguous() checks pass.
        # V is NOT interleaved into C (that interleaving made the K view
        # non-contiguous with token stride = head_size + head_size_v,
        # mismatching the kernels' dense assumption and producing garbage
        # decode). V is allocated as a separate dense buffer per layer by
        # allocate_kv_cache_scale (see model_runner.py) and bound to the
        # attention layer. For KINT8 V is bf16; for KVINT8 V is int8 (with a
        # static per-head V-scale). K-scale (per-token float32) and, for
        # KVINT8, V-scale (static float32) are also allocated separately.
        # Mutate dtype to int8 so create_kv_cache_views' .view(int8) is an
        # identity (K is int8).
        object.__setattr__(self, "dtype", torch.int8)
        val = self.head_size
    elif (
        is_mla
        and envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == "INT8"
        and self.cache_dtype_str != "origin"
    ):
        # MLA is single-sided (head_size_v == 0), so this is just head_size.
        # Mutate the frozen dtype so every reader of self.dtype sees int8; the
        # community per-cell formula then yields int8.
        object.__setattr__(self, "dtype", torch.int8)
        val = (self.head_size + self.head_size_v) * get_dtype_size(self.dtype)
    else:
        raise ValueError(
            f"Invalid attention spec: {type(self)} and quantization type: {envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE if is_flash else envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE}"
        )
    object.__setattr__(self, "_ilu_state_content_size_bytes", val)
    return val


def _full_attention_scale_page_size_bytes(self) -> int:
    return self.num_states * self.num_heads * get_dtype_size(torch.float32)


def _sliding_window_scale_page_size_bytes(self) -> int:
    return self.num_states * self.num_heads * get_dtype_size(torch.float32)


def _mla_attention_scale_page_size_bytes(self) -> int:
    if envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == "INT8" and self.cache_dtype_str != "origin":
        return self.num_states * self.num_heads * 2 * get_dtype_size(torch.float32)
    return 0
