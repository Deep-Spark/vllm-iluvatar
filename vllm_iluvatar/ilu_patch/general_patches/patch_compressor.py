# SPDX-License-Identifier: Apache-2.0
"""Wiring for Iluvatar compressor state-cache override.

Corresponds to: vllm/models/deepseek_v4/compressor.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

Target   : vllm.models.deepseek_v4.compressor.CompressorStateCache.get_kv_cache_spec
Baseline : main_61ac3680210
Why      : The community ``get_kv_cache_spec`` uses an ``alignment`` kwarg
           tied to the fp8_ds_mla paged layout; the Iluvatar bf16 path needs a
           different block_size derivation and no alignment constraint.
How      : direct class-method replacement via ``setattr``.
Scope    : DeepSeek-V4 compressor state-cache on Iluvatar.
Remove   : when upstream exposes a configurable block-size strategy.
Fingerprint: get_kv_cache_spec community source is hashed at register-time
           against the GET_KV_CACHE_SPEC baseline below; gated by
           VLLM_ILU_PATCH_CHECK_FINGERPRINT (default on) so startup fails
           fast if upstream changed the body.
"""

from __future__ import annotations

import inspect
import types

from vllm_iluvatar.ilu_patch import general_patch as p

# Verbatim community-baseline source of
# CompressorStateCache.get_kv_cache_spec (4-space method indent),
# used as the drift anchor by the fingerprint guard.
#
# Baseline: main_61ac3680210
GET_KV_CACHE_SPEC = """\
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # fp8_ds_mla is the UE8M0 paged layout and needs 576B alignment. Plain
        # full-cache rows share state pages with contiguous KV pages, so padding
        # would break page matching.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )
"""


def register() -> None:
    """Register Iluvatar compressor get_kv_cache_spec override."""
    p.register_once(
        __name__,
        _apply,
        detail="CompressorStateCache.get_kv_cache_spec override",
    )


def _apply() -> None:
    from vllm.config import VllmConfig
    from vllm.models.deepseek_v4.compressor import CompressorStateCache
    from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec

    import vllm_iluvatar.models.deepseek_v4.compressor as impl

    # Drift guard for the full-method copy (VLLM_ILU_PATCH_CHECK_FINGERPRINT).
    p.check_fingerprint(
        "CompressorStateCache.get_kv_cache_spec",
        inspect.getsource(CompressorStateCache.get_kv_cache_spec),
        GET_KV_CACHE_SPEC,
        resync_hint=(
            "Re-sync _ilu_get_kv_cache_spec in "
            "vllm_iluvatar/models/deepseek_v4/compressor.py and the "
            "GET_KV_CACHE_SPEC baseline in "
            "ilu_patch/general_patches/patch_compressor.py to the new "
            "community source of CompressorStateCache.get_kv_cache_spec."
        ),
    )

    # Bind community types onto the impl module.
    impl.VllmConfig = VllmConfig
    impl.KVCacheSpec = KVCacheSpec
    impl.SlidingWindowMLASpec = SlidingWindowMLASpec

    # Strict signature validation.
    _ilu_ns = types.SimpleNamespace(
        get_kv_cache_spec=impl._ilu_get_kv_cache_spec,
    )
    p.check_signatures(
        CompressorStateCache,
        _ilu_ns,
        ["get_kv_cache_spec"],
        strict=True,
    )

    CompressorStateCache.get_kv_cache_spec = impl._ilu_get_kv_cache_spec
