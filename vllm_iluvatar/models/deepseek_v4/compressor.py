# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/models/deepseek_v4/compressor.py

ILU override implementation for
CompressorStateCache.get_kv_cache_spec.  This module holds ONLY the
override implementation; the wiring (register/_apply, signature guards,
binding) lives in
``ilu_patch.general_patches.patch_compressor`` and is auto-discovered during
``register_general``.

Target   : vllm.models.deepseek_v4.compressor.CompressorStateCache.get_kv_cache_spec
Baseline : main_6e448d0
Why      : The community ``get_kv_cache_spec`` uses an ``alignment`` kwarg
           tied to the fp8_ds_mla paged layout; the Iluvatar bf16 path needs a
           different block_size derivation and no alignment constraint.
How      : direct class-method replacement via ``setattr``.
Scope    : DeepSeek-V4 compressor state-cache on Iluvatar.
Remove   : when upstream exposes a configurable block-size strategy.
"""

from __future__ import annotations

# Community types bound by patch_compressor._apply().
VllmConfig = None
KVCacheSpec = None
SlidingWindowMLASpec = None


# [ILU override] get_kv_cache_spec — delta: Iluvatar bf16 path uses a
# different block_size derivation (fp8: 16384-byte bucket alignment;
# bf16: 2x default) and drops the ``alignment`` kwarg entirely.
def _ilu_get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
    is_fp8 = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
    block_size = 16384 // (self.state_dim * 4) if is_fp8 else self.block_size * 2
    return SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=self.state_dim,
        dtype=self.dtype,
        sliding_window=self.sliding_window,
    )
