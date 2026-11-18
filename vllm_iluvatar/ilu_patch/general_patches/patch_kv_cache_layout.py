# SPDX-License-Identifier: Apache-2.0
"""Negotiate KV layouts across heterogeneous PP stages without changing vLLM."""

from __future__ import annotations

import inspect

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(__name__, _apply, detail="common KV layout across PP workers")


def _apply() -> None:
    import vllm.v1.attention.backends.utils as community

    from vllm_iluvatar.v1.attention.backends.utils import (
        make_resolve_kv_cache_layout,
    )

    original = community.resolve_kv_cache_layout
    expected = ("vllm_config", "supported_layouts", "kv_cache_specs")
    if tuple(inspect.signature(original).parameters) != expected:
        raise RuntimeError(
            "ILU KV layout patch: upstream resolver signature changed; "
            "reconcile the wrapper before using this vLLM version."
        )
    p.rebind_module_func(
        community, "resolve_kv_cache_layout", make_resolve_kv_cache_layout(original)
    )
