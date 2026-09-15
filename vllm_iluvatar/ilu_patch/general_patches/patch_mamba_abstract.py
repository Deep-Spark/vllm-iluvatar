# SPDX-License-Identifier: Apache-2.0
"""Install ``bind_kv_cache_scale`` on Mamba-like attention layers.

Corresponds to: vllm/model_executor/layers/mamba/abstract.py
Target   : vllm.model_executor.layers.mamba.abstract.MambaBase
Baseline : main_a67969c5a0
Why      : ILU ``bind_kv_cache_scale`` (free function in
           vllm_iluvatar.v1.worker.gpu.model_runner) mirrors the community
           ``bind_kv_cache`` free function and calls
           ``layer.bind_kv_cache_scale(...)`` on every layer that owns a KV
           cache entry. Hybrid models (e.g. Qwen3-GDN) co-allocate Mamba-like
           layers (``QwenGatedDeltaNetAttention`` subclasses ``MambaBase``) in
           the same ``forward_context`` as attention layers, so ``MambaBase``
           must expose ``bind_kv_cache_scale`` just as it already exposes
           ``bind_kv_cache``; otherwise the uniform call raises
           ``AttributeError``.
How      : store the per-layer ``[k_scale, v_scale]`` views (``[None, None]`` for
           Mamba, which carries no K/V scale) on ``self.kv_cache_scale``,
           mirroring ``bind_kv_cache`` storing ``self.kv_cache``. Mamba forward
           never reads ``kv_cache_scale``, so this is a harmless no-op store.
Scope    : all Mamba-like layers (MambaBase subclasses) on Iluvatar.
Remove   : when ILU quantized KV cache no longer co-allocates Mamba layers in the
           same scale-binding pass.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the MambaBase ``bind_kv_cache_scale`` hook."""
    p.register_once(
        __name__,
        _apply,
        detail="MambaBase.bind_kv_cache_scale mirror of bind_kv_cache",
    )


def _apply() -> None:
    from vllm.model_executor.layers.mamba.abstract import (
        MambaBase,  # type: ignore[reportMissingImports]
    )

    # [ILU override] MambaBase.bind_kv_cache_scale - delta: store the per-layer
    # [k_scale, v_scale] views the way bind_kv_cache stores the KV view. Mamba
    # carries no K/V scale (allocate_kv_cache_scale passes [None, None]), so this
    # is a harmless store that lets the bind_kv_cache_scale free function call
    # every forward_context layer uniformly.
    def _bind_kv_cache_scale(self, kv_scale_cache):
        self.kv_cache_scale = kv_scale_cache

    MambaBase.bind_kv_cache_scale = _bind_kv_cache_scale
