# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 KV cache tensor layout, replacing the community packed slab.

Target   : vllm.v1.core.kv_cache_utils._get_kv_cache_config_packed
           vllm.v1.core.kv_cache_utils._pool_bytes_per_block
Baseline : main_61ac3680210
Why      : The packed layout restarts every cache group's byte offsets at 0
           inside one block slab, so groups whose page sizes differ nest into
           each other. For DeepSeek-V4 (pages of 1168 / 37376 / 149504 B) a
           block's bytes only have a meaning once you know which group owns that
           block ID, and prefix caching corrupts generations that rely on it.
How      : Rebind the packed planner and the capacity divisor that must agree
           with it. Both defer to community for every non-DeepSeek-V4 layout,
           including an explicit enable_cross_layers_blocks opt-in.
Scope    : always installed; the DeepSeek-V4 predicate is evaluated per worker
           at plan time.
Remove   : when upstream stops planning DeepSeek-V4 through the packed layout.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p

# Community source this patch is pinned against. Only the branch that routes
# DeepSeek-V4 into the packed planner matters here: if it disappears, rebinding
# the planner silently stops taking effect.
_USE_PACKED_BASELINE = '''
def _use_packed_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> bool:
    is_dsv4 = all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    )
    kv_transfer_config = vllm_config.kv_transfer_config
    extra_config = (
        kv_transfer_config.kv_connector_extra_config
        if kv_transfer_config is not None
        else {}
    )
    # NOTE: enable_cross_layers_blocks is an experimental API and subject to change with
    # https://github.com/vllm-project/vllm/issues/42082
    enable_cross_layers = (
        str(extra_config.get("enable_cross_layers_blocks", "False")).lower() == "true"
    )
    return is_dsv4 or (enable_cross_layers and len(kv_cache_groups) > 1)
'''


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="DeepSeek-V4 KV cache layout replaces the packed block slab",
    )


def _apply() -> None:
    import inspect

    from vllm.v1.core import kv_cache_utils as _m

    from vllm_iluvatar.v1.core import kv_cache_utils as _ilu

    p.check_fingerprint(
        "_use_packed_kv_cache_config",
        inspect.getsource(_m._use_packed_kv_cache_config),
        _USE_PACKED_BASELINE,
        resync_hint=(
            "Confirm DeepSeek-V4 still reaches _get_kv_cache_config_packed, then "
            "refresh _USE_PACKED_BASELINE in "
            "vllm_iluvatar/ilu_patch/general_patches/patch_dsv4_kv_cache_layout.py."
        ),
    )
    p.check_signatures(
        _m,
        _ilu,
        ["_get_kv_cache_config_packed", "_pool_bytes_per_block"],
    )

    p.rebind_module_func(
        _m, "_get_kv_cache_config_packed", _ilu._get_kv_cache_config_packed
    )
    p.rebind_module_func(_m, "_pool_bytes_per_block", _ilu._pool_bytes_per_block)
