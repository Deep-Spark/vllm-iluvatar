# SPDX-License-Identifier: Apache-2.0
"""Install generic CoreX attention compilation compatibility wrappers.

Corresponds to: vllm/model_executor/layers/attention/attention.py
Target   : vllm.model_executor.layers.attention.attention.Attention.__init__,
           vllm.model_executor.layers.attention.attention.Attention.forward,
           Attention.process_weights_after_loading
Baseline : v0.31.0 (db9527a468)
Why      : CoreX TorchDynamo cannot trace the community ``torch.Size`` call
           used by Attention's default output-shape allocation path. Ixformer
           torch ops also require the community opaque attention custom-op
           boundary during piecewise torch.compile.
How      : wrappers supply the semantically identical tuple output shape and
           restore ``use_direct_call=False`` for ixformer torch ops.
Scope    : all community Attention instances on Iluvatar.
Remove   : when CoreX Dynamo supports ``torch.Size`` construction and safely
           traces ixformer attention state mutations.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the generic attention compilation compatibility wrappers."""
    p.register_once(
        __name__,
        _apply,
        detail="Attention opaque ixformer boundary and tuple output-shape",
    )


def _apply() -> None:
    from vllm.model_executor.layers.attention.attention import (  # type: ignore[reportMissingImports]
        Attention,
    )

    from vllm_iluvatar.model_executor.layers.attention.attention import (
        make_forward,
        make_init,
        make_process_weights_after_loading,
    )

    # [ILU override] Attention.__init__ - delta: preserve the opaque custom-op
    # boundary needed by ixformer stateful attention in piecewise graphs.
    Attention.__init__ = make_init(Attention.__init__)
    # [ILU override] Attention.forward - delta: avoid torch.Size in traced path.
    Attention.forward = make_forward(Attention.forward)
    # [ILU override] Bind externally calibrated FP8 Q/K/V scales after loading.
    Attention.process_weights_after_loading = make_process_weights_after_loading(
        Attention.process_weights_after_loading
    )

    # [ILU override] Attention.bind_kv_cache_scale - delta: store the per-layer
    # [k_scale, v_scale] views (carved from the shared KV backing buffer by
    # allocate_kv_cache_scale) the same way bind_kv_cache stores the KV view.
    # ILU attention backends (ilu_flash_attn / mla_attention) read
    # ``layer.kv_cache_scale`` at forward time. Mirrors the community
    # ``bind_kv_cache`` layer hook so subclasses can unpack the raw scale
    # allocation the way Mamba unpacks the raw KV allocation.
    def _bind_kv_cache_scale(self, kv_scale_cache):
        self.kv_cache_scale = kv_scale_cache

    Attention.bind_kv_cache_scale = _bind_kv_cache_scale

    # [ILU override] MLAAttention.bind_kv_cache_scale - delta: same store as
    # Attention above. MLAAttention is a separate nn.Module (not Attention), so
    # the Attention patch does not reach it; the ILU MLA forward reads
    # ``self.kv_cache_scale`` (see model_executor/layers/attention/mla_attention.py
    # forward_full) when use_int8_mla is set, and allocate_kv_cache_scale calls
    # ``bind_kv_cache_scale`` on every layer that owns a KV scale, so MLAAttention
    # must expose the same hook.
    from vllm.model_executor.layers.attention.mla_attention import (  # type: ignore[reportMissingImports]
        MLAAttention,
    )

    MLAAttention.bind_kv_cache_scale = _bind_kv_cache_scale
