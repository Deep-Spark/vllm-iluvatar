# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/kv_cache_interface.py

ILU override implementation for Attention quantization support.

Target   : FullAttentionSpec / MLAAttentionSpec / KVCacheTensor
Baseline : main_6e448d0
Why      : Iluvatar attention quantization Int8 support.
           Provide meta information for allocating kv cache scales and adjust the num gpu blocks.
How      : FullAttentionSpec: add two properties and replace one property for kv cache calculation.
           MLAAttentionSpec: add two properties and replace one property for kv cache calculation.
           KVCacheTensor: add two properties for kv cache calculation.
Scope    : always enabled if attention quantization is enabled.
Remove   : When upstream support int8 kv cache quantization.
"""

from __future__ import annotations

import torch
import vllm.envs as envs


# Iluvatar Full Attention scale part
def _full_attention_real_page_size_bytes(self) -> int:
    import vllm.v1.kv_cache_interface as _m
    from vllm.v1.kv_cache_interface import KVQuantMode

    # NOTE >>> iluvatar attention quantization
    if envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE == "KVINT8":
        return (
            self.block_size * self.num_kv_heads * self.head_size
            + self.block_size * self.num_kv_heads * self.head_size_v
        )
    elif envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE == "KINT8":
        return (
            self.block_size * self.num_kv_heads * self.head_size
            + self.block_size
            * self.num_kv_heads
            * self.head_size_v
            * _m.get_dtype_size(self.dtype)
        )
    # <<< iluvatar attention quantization
    if self.kv_quant_mode.is_nvfp4:
        # Packed layout per head: fp4 data + fp8 block scales.
        # fp4 data: head_size//2 bytes (2 fp4 values per byte)
        # fp8 block scale: head_size//16 bytes (1 scale per 16 elements)
        last_dim = _m.nvfp4_kv_cache_full_dim(
            self.head_size
        ) + _m.nvfp4_kv_cache_full_dim(self.head_size_v)
    elif self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        last_dim = self.head_size // 2 + self.head_size_v // 2
    else:
        last_dim = self.head_size + self.head_size_v
    return (
        self.block_size * self.num_kv_heads * last_dim * _m.get_dtype_size(self.dtype)
    )


def _full_attention_scale_page_size_bytes(self) -> int:
    import vllm.v1.kv_cache_interface as _m

    return self.block_size * self.num_kv_heads * _m.get_dtype_size(torch.float32)


def _full_attention_fixed_size_bytes(self) -> int:
    import vllm.v1.kv_cache_interface as _m

    if envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE == "KVINT8":
        return self.num_kv_heads * self.head_size_v * _m.get_dtype_size(torch.float32)
    return 0


# Iluvatar Sliding Window Attention scale part
def _sliding_window_real_page_size_bytes(self) -> int:
    import vllm.v1.kv_cache_interface as _m

    # NOTE >>> iluvatar attention quantization
    if envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE == "KVINT8":
        return (
            self.block_size * self.num_kv_heads * self.head_size
            + self.block_size * self.num_kv_heads * self.head_size_v
        )
    elif envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE == "KINT8":
        return (
            self.block_size * self.num_kv_heads * self.head_size
            + self.block_size
            * self.num_kv_heads
            * self.head_size_v
            * _m.get_dtype_size(self.dtype)
        )
    # <<< iluvatar attention quantization
    if self.kv_quant_mode.is_nvfp4:
        last_dim = _m.nvfp4_kv_cache_full_dim(
            self.head_size
        ) + _m.nvfp4_kv_cache_full_dim(self.head_size_v)
        return (
            self.block_size
            * self.num_kv_heads
            * last_dim
            * _m.get_dtype_size(self.dtype)
        )
    return (
        self.block_size
        * self.num_kv_heads
        * (self.head_size + self.head_size_v)
        * _m.get_dtype_size(self.dtype)
    )


def _sliding_window_scale_page_size_bytes(self) -> int:
    import vllm.v1.kv_cache_interface as _m

    return self.block_size * self.num_kv_heads * _m.get_dtype_size(torch.float32)


def _sliding_window_fixed_size_bytes(self) -> int:
    import vllm.v1.kv_cache_interface as _m

    if envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE == "KVINT8":
        return self.num_kv_heads * self.head_size_v * _m.get_dtype_size(torch.float32)
    return 0


# Iluvatar MLA Attention scale part
def _mla_attention_real_page_size_bytes(self) -> int:
    import vllm.v1.kv_cache_interface as _m
    from vllm.v1.kv_cache_interface import KVQuantMode

    # NOTE >>> iluvatar attention quantization
    if envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == "INT8" and self.cache_dtype_str != "origin":
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * _m.get_dtype_size(torch.int8)
        )
    # <<< iluvatar attention quantization
    if self.cache_dtype_str == "fp8_ds_mla":
        if self.model_version == "deepseek_v4":
            # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token.
            # head_size stays semantic (512); bytes are determined here.
            return self.storage_block_size * 584
        # V3.2 main MLA: 656-byte custom layout (kv_lora_rank=512 +
        # qk_rope_head_dim=64, head_size=576). See flashmla_sparse.py.
        return self.block_size * 656
    if self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        head_dim = self.head_size // 2
    else:
        head_dim = self.head_size
    return (
        self.storage_block_size
        * self.num_kv_heads
        * head_dim
        * _m.get_dtype_size(self.dtype)
    )


def _mla_attention_scale_page_size_bytes(self) -> int:
    import vllm.v1.kv_cache_interface as _m

    if envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE == "INT8" and self.cache_dtype_str != "origin":
        return (
            self.block_size * self.num_kv_heads * 2 * _m.get_dtype_size(torch.float32)
        )
    return 0


def _mla_attention_fixed_size_bytes(self) -> int:
    return 0
