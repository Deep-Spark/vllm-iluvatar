# SPDX-License-Identifier: Apache-2.0
"""Enable divergent local hybrid hits for FA+SWA (not only Mamba).

Corresponds to: vllm/v1/core/kv_cache_manager.py
Target      : KVCacheManager.get_computed_blocks_for_connector
              KVCacheManager.get_zeroing_block_ids_in_range
Baseline    : v0.31.0 (db9527a468)
Why         : Community gates the connector divergent-hit path on
              ``has_mamba_layers``. DeepSeek-V4 is FA+SWA hybrid: after a
              remote KV load, Decode hashes the full FA prefix but SWA only
              retains the final window, so reconciled lookup hits 0 and every
              request re-pulls the full remote KV. Additionally, when FA+SWA
              takes the divergent path with ``load_kv_async=True`` and
              ``needs_kv_cache_zeroing=True``, ``get_zeroing_block_ids_in_range``
              slices SWA groups from FA's ``num_new_local_computed_tokens``
              instead of the SWA window start, leaving newly allocated SWA
              window blocks in ``new_block_ids_to_zero`` to race async RDMA.
How         : 1. Preserve the community Mamba path and additionally admit
                 hybrids whose prefix-cacheable groups are exclusively FA/MLA
                 and SWA/SWA-MLA (allowing non-cacheable ``CircularBufferSpec``
                 groups), requiring all FA/MLA groups to match
                 ``fa_group_id``'s hit length and computing ``hit_diverged``
                 across prefix-cacheable groups only.
              2. Start ``get_zeroing_block_ids_in_range`` at
                 ``min(start_token // mgr.block_size, mgr.num_cached_block.get(...))``
                 so all externally allocated SWA window blocks are skipped
                 from worker zeroing.
Scope       : FA+SWA hybrids (including DeepSeek-V4 / DeepSeek-V4.1) with a
              connector that advertises ``supports_divergent_local_hybrid_hits``.
Fingerprint : the complete community methods are SHA-checked at register-time;
              VLLM_ILU_PATCH_CHECK_FINGERPRINT controls the fail-fast guard.
"""

from __future__ import annotations

import inspect
import types
from typing import TYPE_CHECKING

from vllm.utils.math_utils import cdiv

from vllm_iluvatar.ilu_patch import general_patch as p

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

_COMMUNITY_METHOD = (
    "vllm.v1.core.kv_cache_manager.KVCacheManager.get_computed_blocks_for_connector"
)
_COMMUNITY_METHOD_SOURCE_SHA = (
    "e221a88f1656fcf55dc00973a47b5ca009c48140154ea2d4841688bff730eaa4"
)

_COMMUNITY_ZEROING_METHOD = (
    "vllm.v1.core.kv_cache_manager.KVCacheManager.get_zeroing_block_ids_in_range"
)
_COMMUNITY_ZEROING_METHOD_SOURCE_SHA = (
    "0e29476c6ccea81ff5ba8d0d6276695361ee6ffe260f440c26ec3f0e22366f42"
)


def _check_community_method(
    method: object,
    expected_sha: str | None = None,
    symbol: str = _COMMUNITY_METHOD,
) -> None:
    if not p._enabled(p.ENV_CHECK_FINGERPRINT):
        return
    if expected_sha is None:
        expected_sha = _COMMUNITY_METHOD_SOURCE_SHA
    current_sha = p.source_sha(inspect.getsource(method))
    if current_sha != expected_sha:
        raise RuntimeError(
            f"ILU patch: community {symbol} source changed vs "
            f"baseline v0.31.0. Expected {expected_sha}, got "
            f"{current_sha}. Re-sync {symbol.rsplit('.', 1)[-1]} in "
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
    # non-Mamba hybrid when every prefix-cacheable group is FA/MLA or
    # SWA/SWA-MLA (with at least one of each) and any non-cacheable group is a
    # CircularBufferSpec. This deliberately excludes other sparse layouts such
    # as ChunkedLocalAttention.
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
        KVCacheSpecKind,
        get_kv_cache_spec_kind,
    )

    full_attention_kinds = {
        KVCacheSpecKind.FULL_ATTENTION,
        KVCacheSpecKind.MLA_ATTENTION,
    }
    sliding_window_kinds = {
        KVCacheSpecKind.SLIDING_WINDOW,
        KVCacheSpecKind.SLIDING_WINDOW_MLA,
    }
    is_fa_swa_hybrid = False
    cacheable_group_ids: list[int] = []
    fa_group_ids: list[int] = []
    if (
        not self.kv_cache_config.has_mamba_layers
        and isinstance(coordinator, HybridKVCacheCoordinator)
        and coordinator.full_attention_group_id is not None
    ):
        kv_cache_groups = getattr(self.kv_cache_config, "kv_cache_groups", None)
        if kv_cache_groups:
            spec_kinds: set[KVCacheSpecKind] = set()
            all_groups_supported = True
            for gid, group in enumerate(kv_cache_groups):
                spec = group.kv_cache_spec
                if not spec.prefix_cacheable:
                    if not isinstance(spec, CircularBufferSpec):
                        all_groups_supported = False
                        break
                    continue
                kind = get_kv_cache_spec_kind(spec)
                spec_kinds.add(kind)
                cacheable_group_ids.append(gid)
                if kind in full_attention_kinds:
                    fa_group_ids.append(gid)
            is_fa_swa_hybrid = (
                all_groups_supported
                and bool(spec_kinds & full_attention_kinds)
                and bool(spec_kinds & sliding_window_kinds)
                and spec_kinds <= full_attention_kinds | sliding_window_kinds
            )
        else:
            spec_kinds = {
                get_kv_cache_spec_kind(group.spec)
                for group in coordinator.attention_groups
            }
            for group in coordinator.attention_groups:
                kind = get_kv_cache_spec_kind(group.spec)
                group_ids = getattr(group, "group_ids", ())
                cacheable_group_ids.extend(group_ids)
                if kind in full_attention_kinds:
                    fa_group_ids.extend(group_ids)
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

    # >>> ILU MODIFICATION >>>
    if is_fa_swa_hybrid and any(
        per_group_hits[gid] != per_group_hits[fa_group_id] for gid in fa_group_ids
    ):
        # Multiple FA/MLA groups (e.g. EAGLE/MTP drafter with drop_eagle_block)
        # disagreed on the prefix hit; only SWA groups may lag fa_group_id.
        return *self.get_computed_blocks(request), False

    num_local = per_group_hits[fa_group_id]
    blocks = self.create_kv_cache_blocks(computed)
    if is_fa_swa_hybrid and cacheable_group_ids:
        min_hit = min(per_group_hits[gid] for gid in cacheable_group_ids)
    else:
        min_hit = min(per_group_hits)
    # Per-group lookups do not detect an uncached shared prefix (boundary 0).
    return blocks, num_local, 0, min_hit < num_local
    # <<< ILU MODIFICATION <<<


def get_zeroing_block_ids_in_range(
    self, request_id: str, start_token: int, end_token: int
) -> list[int]:
    """The request's block ids covering [start_token, end_token), from
    the groups whose new blocks are zeroed by the worker."""
    ids: list[int] = []
    for mgr in self.coordinator.single_type_managers:
        if mgr.records_new_block_ids:
            # >>> ILU MODIFICATION >>>
            # For SWA groups on a divergent FA+SWA hit, add_local_computed_blocks
            # records the cached/skipped prefix boundary in num_cached_block,
            # while allocate_external_computed_blocks allocates window blocks
            # starting at num_cached_block (which can precede start_token //
            # block_size when FA hit deeper than the SWA window start).
            default_start_idx = start_token // mgr.block_size
            num_cached_block = getattr(mgr, "num_cached_block", None)
            if isinstance(num_cached_block, dict):
                start_idx = min(
                    default_start_idx,
                    num_cached_block.get(request_id, default_start_idx),
                )
            else:
                start_idx = default_start_idx
            # <<< ILU MODIFICATION <<<
            end_idx = cdiv(end_token, mgr.block_size)
            blocks = mgr.req_to_blocks[request_id]
            ids.extend(blk.block_id for blk in blocks[start_idx:end_idx])
    return ids


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="Connector divergent hybrid hits for FA+SWA",
    )


def _apply() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    _check_community_method(
        KVCacheManager.get_computed_blocks_for_connector,
        _COMMUNITY_METHOD_SOURCE_SHA,
        _COMMUNITY_METHOD,
    )
    _check_community_method(
        KVCacheManager.get_zeroing_block_ids_in_range,
        _COMMUNITY_ZEROING_METHOD_SOURCE_SHA,
        _COMMUNITY_ZEROING_METHOD,
    )

    p.check_signatures(
        KVCacheManager,
        types.SimpleNamespace(
            get_computed_blocks_for_connector=get_computed_blocks_for_connector,
            get_zeroing_block_ids_in_range=get_zeroing_block_ids_in_range,
        ),
        [
            "get_computed_blocks_for_connector",
            "get_zeroing_block_ids_in_range",
        ],
        strict=True,
    )
    KVCacheManager.get_computed_blocks_for_connector = get_computed_blocks_for_connector
    KVCacheManager.get_zeroing_block_ids_in_range = get_zeroing_block_ids_in_range
