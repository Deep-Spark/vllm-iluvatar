# SPDX-License-Identifier: Apache-2.0
"""Enable divergent local hybrid hits for FA+SWA (not only Mamba).

Corresponds to: vllm/v1/core/kv_cache_manager.py
Target      : KVCacheManager.get_computed_blocks_for_connector
Baseline    : v0.31.0 (db9527a468)
Why         : Community gates the connector divergent-hit path on
              ``has_mamba_layers``. DeepSeek-V4 is FA+SWA hybrid: after a
              remote KV load, Decode hashes the full FA prefix but SWA only
              retains the final window, so reconciled lookup hits 0 and every
              request re-pulls the full remote KV.
How         : Preserve the community Mamba path and additionally admit only
              hybrids whose cache groups are exclusively FA/MLA and SWA/SWA-MLA.
Scope       : FA+SWA hybrids (including DeepSeek-V4) with a connector that
              advertises ``supports_divergent_local_hybrid_hits``.
Fingerprint : the complete community method is SHA-checked at register-time;
              VLLM_ILU_PATCH_CHECK_FINGERPRINT controls the fail-fast guard.
"""

from __future__ import annotations

import inspect
import types

from vllm_iluvatar.ilu_patch import general_patch as p

_COMMUNITY_METHOD = (
    "vllm.v1.core.kv_cache_manager.KVCacheManager."
    "get_computed_blocks_for_connector"
)
_COMMUNITY_METHOD_SOURCE_SHA = (
    "e221a88f1656fcf55dc00973a47b5ca009c48140154ea2d4841688bff730eaa4"
)


def _check_community_method(method: object) -> None:
    if not p._enabled(p.ENV_CHECK_FINGERPRINT):
        return
    current_sha = p.source_sha(inspect.getsource(method))
    if current_sha != _COMMUNITY_METHOD_SOURCE_SHA:
        raise RuntimeError(
            f"ILU patch: community {_COMMUNITY_METHOD} source changed vs "
            f"baseline v0.31.0. Expected {_COMMUNITY_METHOD_SOURCE_SHA}, got "
            f"{current_sha}. Re-sync get_computed_blocks_for_connector in "
            "patch_connector_divergent_hybrid_hits.py with the new community "
            "method, then refresh the baseline SHA."
        )


def get_computed_blocks_for_connector(
    self, request: Request
) -> tuple[KVCacheBlocks, int, int, bool]:
    """Local prefix-cache lookup for a request scheduled with a KV connector.

    Hybrid (Mamba + full-attention) models can have per-group prefix hits
    diverge under block pressure: the full-attention tail may be evicted
    while a deeper Mamba state survives, or vice versa. Report the
    full-attention hit as the local prefix - the connector transfers the
    remaining suffix and the Mamba state is transferred unconditionally by
    nixl's ``_apply_prefix_caching`` - and flag when that hit ran deeper
    than a lagging group. Such a hit only has a valid Mamba state at its
    boundary if the connector supplies it, so the caller must fall back to
    ``get_computed_blocks`` to reconcile when no external tokens are found.

    Non-hybrid models and already-convergent hits use ``get_computed_blocks``.

    Returns:
        The ``get_computed_blocks`` triple (blocks, number of local computed
        tokens, shared-prefix boundary) plus ``hit_diverged``.

    """
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    coordinator = self.coordinator
    # >>> ILU MODIFICATION >>>
    # Community enables this path only for Mamba hybrids. Also admit a
    # non-Mamba hybrid when every cache group is FA/MLA or SWA/SWA-MLA and at
    # least one SWA group is present. This deliberately excludes other sparse
    # layouts such as ChunkedLocalAttention.
    from vllm.v1.kv_cache_interface import (
        KVCacheSpecKind,
        get_kv_cache_spec_kind,
    )

    is_fa_swa_hybrid = False
    if (
        not self.kv_cache_config.has_mamba_layers
        and isinstance(coordinator, HybridKVCacheCoordinator)
        and coordinator.full_attention_group_id is not None
    ):
        spec_kinds = {
            get_kv_cache_spec_kind(group.spec)
            for group in coordinator.attention_groups
        }
        full_attention_kinds = {
            KVCacheSpecKind.FULL_ATTENTION,
            KVCacheSpecKind.MLA_ATTENTION,
        }
        sliding_window_kinds = {
            KVCacheSpecKind.SLIDING_WINDOW,
            KVCacheSpecKind.SLIDING_WINDOW_MLA,
        }
        is_fa_swa_hybrid = (
            bool(spec_kinds & full_attention_kinds)
            and bool(spec_kinds & sliding_window_kinds)
            and spec_kinds <= full_attention_kinds | sliding_window_kinds
        )

    if not (
        (self.kv_cache_config.has_mamba_layers or is_fa_swa_hybrid)
        and isinstance(coordinator, HybridKVCacheCoordinator)
        and coordinator.full_attention_group_id is not None
    ):
        return *self.get_computed_blocks(request), False
    # <<< ILU MODIFICATION <<<

    if not self.prefix_cache_lookup_enabled(request):
        return self.empty_kv_cache_blocks, 0, 0, False

    fa_group_id = coordinator.full_attention_group_id
    computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(
        request.block_hashes, request.num_tokens - 1
    )
    if any(hit > per_group_hits[fa_group_id] for hit in per_group_hits):
        # A lagging group hit deeper than full attention means its
        # full-attention blocks were evicted; use the reconciled boundary
        # that every group agrees on.
        return *self.get_computed_blocks(request), False

    num_local = per_group_hits[fa_group_id]
    blocks = self.create_kv_cache_blocks(computed)
    # Per-group lookups do not detect an uncached shared prefix (boundary 0).
    return blocks, num_local, 0, min(per_group_hits) < num_local


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="Connector divergent hybrid hits for FA+SWA",
    )


def _apply() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    _check_community_method(KVCacheManager.get_computed_blocks_for_connector)

    replacement = get_computed_blocks_for_connector
    p.check_signatures(
        KVCacheManager,
        types.SimpleNamespace(get_computed_blocks_for_connector=replacement),
        ["get_computed_blocks_for_connector"],
        strict=True,
    )
    KVCacheManager.get_computed_blocks_for_connector = replacement
