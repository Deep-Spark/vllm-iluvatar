# SPDX-License-Identifier: Apache-2.0
"""Wiring for Iluvatar compressor state-cache override.

Corresponds to: vllm/models/deepseek_v4/compressor.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

Target   : vllm.models.deepseek_v4.compressor.CompressorStateCache.get_kv_cache_spec
Baseline : main_88e5e2c57be8
Why      : The community ``get_kv_cache_spec`` uses an ``alignment`` kwarg
           tied to the fp8_ds_mla paged layout; the Iluvatar bf16 path needs a
           different block_size derivation and no alignment constraint.
How      : direct class-method replacement via ``setattr``.
Scope    : DeepSeek-V4 compressor state-cache on Iluvatar.
Remove   : when upstream exposes a configurable block-size strategy.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register Iluvatar compressor get_kv_cache_spec override."""
    p.register_once(
        __name__,
        _apply,
        detail="CompressorStateCache.get_kv_cache_spec override",
    )


def _apply() -> None:
    import types

    from vllm.config import VllmConfig
    from vllm.models.deepseek_v4.compressor import CompressorStateCache
    from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec

    # [ILU override] get_kv_cache_spec — delta: Iluvatar bf16 path uses a
    # different block_size derivation (fp8: 16384-byte bucket alignment;
    # bf16: 2x default) and drops the ``alignment`` kwarg entirely.
    def _ilu_get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        is_fp8 = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        if is_fp8:
            block_size = 16384 // (self.state_dim * 4)
        else:
            block_size = self.block_size * 2
        return SlidingWindowMLASpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
        )

    _ilu_mod = types.SimpleNamespace(get_kv_cache_spec=_ilu_get_kv_cache_spec)
    p.check_signatures(CompressorStateCache, _ilu_mod, ["get_kv_cache_spec"], strict=True)

    CompressorStateCache.get_kv_cache_spec = _ilu_get_kv_cache_spec
