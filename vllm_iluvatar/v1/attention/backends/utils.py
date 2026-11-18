# SPDX-License-Identifier: Apache-2.0
"""KV cache layout negotiation for heterogeneous pipeline stages."""

from __future__ import annotations

from collections import Counter
from functools import wraps

from vllm.v1.attention.backends import utils as upstream
from vllm.v1.kv_cache_layout import KVCacheLayout


def make_resolve_kv_cache_layout(original):
    """Intersect worker support, then retain upstream layout validation."""

    @wraps(original)
    def resolve_kv_cache_layout(vllm_config, supported_layouts, kv_cache_specs=None):
        # Keep the upstream cached-result and missing-worker error paths.
        if (
            vllm_config.cache_config.kv_cache_layout is not None
            or not supported_layouts
            or not all(supported_layouts)
            or all(names == supported_layouts[0] for names in supported_layouts)
        ):
            return original(vllm_config, supported_layouts, kv_cache_specs)
        shared = set.intersection(*map(set, supported_layouts))
        if not shared:
            raise ValueError(
                f"Workers have no common supported KV cache layout: {supported_layouts}."
            )
        # A pure SWA stage reports the fallback order and has no preference.
        default = [layout.name for layout in upstream._DEFAULT_LAYOUT_PREFERENCE]
        votes = Counter(
            next(name for name in names if name in shared)
            for names in supported_layouts
            if list(names) != default
        )
        # Stable sort: equal votes retain enum order, as in upstream.
        common = sorted(
            (layout.name for layout in KVCacheLayout if layout.name in shared),
            key=votes.__getitem__,
            reverse=True,
        )
        return original(
            vllm_config, [common for _ in supported_layouts], kv_cache_specs
        )

    return resolve_kv_cache_layout
