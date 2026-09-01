# SPDX-License-Identifier: Apache-2.0
"""Wiring for Iluvatar V2 hybrid attention/Mamba KV-cache layout alignment.

Corresponds to: vllm/v1/worker/gpu/attn_utils.py

Target   : vllm.v1.worker.gpu.attn_utils._align_mixed_attention_kv_cache_views
Baseline : main_6e448d0
Why      : Hybrid attention/Mamba models can share one raw KV-cache tensor, but
           attention backends may expose K/V-first logical views while Mamba
           interprets each block as one contiguous page. The V2 worker must
           normalize those shared views before block IDs are used.
How      : Wrap the community mixed-layout helper. Non-Mamba cases keep the
           community path; Mamba hybrid cases restride K/V-first attention views
           to block-first storage.
Scope    : V2 hybrid attention/Mamba KV cache initialization on Iluvatar.
Remove   : when upstream carries equivalent hybrid Mamba shared-cache layout
           alignment.
Fingerprint: _align_mixed_attention_kv_cache_views community source is hashed
           at register-time against the baseline snapshot below; gated by
           VLLM_ILU_PATCH_CHECK_FINGERPRINT.
"""

from __future__ import annotations

import inspect
import types

from vllm_iluvatar.ilu_patch import general_patch as p

ALIGN_MIXED_ATTENTION_KV_CACHE_VIEWS = '''\
def _align_mixed_attention_kv_cache_views(
    attn_groups: Iterable[AttentionGroup],
    kv_caches: dict[str, Any],
    kernel_block_sizes: list[int],
    cache_dtype: str,
    kv_cache_config: KVCacheConfig,
) -> None:
    """Align shared attention KV views when backends disagree on layout.

    Encoder-decoder models can share one raw allocation between decoder
    self-attention (K/V-first ROCM_ATTN, block dim 1) and cross-attention
    (blocks-first backends, block dim 0). Keep the physical storage in the
    K/V-first layout expected by ROCM_ATTN, and restride the blocks-first
    logical views so block IDs address the same bytes.
    """
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

    for kv_tensor in kv_cache_config.kv_cache_tensors:
        if kv_tensor.block_stride > 0:
            continue
        shared_block_dims = {
            block_dims_by_layer[layer_name]
            for layer_name in kv_tensor.shared_by
            if layer_name in block_dims_by_layer
        }
        if 0 not in shared_block_dims or 1 not in shared_block_dims:
            continue

        for layer_name in kv_tensor.shared_by:
            if block_dims_by_layer.get(layer_name) == 0:
                _restride_blocks_first_kv_cache_to_kv_first_storage(
                    kv_caches[layer_name]
                )
'''


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="V2 hybrid attention/Mamba shared-cache layout alignment",
    )


def _apply() -> None:
    from vllm.v1.worker.gpu import attn_utils as _m

    import vllm_iluvatar.v1.worker.gpu.attn_utils as impl

    p.check_fingerprint(
        "vllm.v1.worker.gpu.attn_utils._align_mixed_attention_kv_cache_views",
        inspect.getsource(_m._align_mixed_attention_kv_cache_views),
        ALIGN_MIXED_ATTENTION_KV_CACHE_VIEWS,
        resync_hint=(
            "Re-sync vllm_iluvatar/v1/worker/gpu/attn_utils.py and the "
            "ALIGN_MIXED_ATTENTION_KV_CACHE_VIEWS baseline in "
            "ilu_patch/general_patches/patch_attn_utils.py to the new "
            "community source of _align_mixed_attention_kv_cache_views."
        ),
    )
    align = impl.make_align_mixed_attention_kv_cache_views(
        _m._align_mixed_attention_kv_cache_views
    )
    ilu_funcs = types.SimpleNamespace(_align_mixed_attention_kv_cache_views=align)
    p.check_signatures(
        _m,
        ilu_funcs,
        ["_align_mixed_attention_kv_cache_views"],
        strict=True,
    )

    p.rebind_module_func(_m, "_align_mixed_attention_kv_cache_views", align)
