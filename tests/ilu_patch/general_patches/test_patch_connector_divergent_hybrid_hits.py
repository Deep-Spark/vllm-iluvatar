# SPDX-License-Identifier: Apache-2.0
"""Divergent hybrid hit path must not require Mamba layers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
)
from vllm_iluvatar.ilu_patch.general_patches import (
    patch_connector_divergent_hybrid_hits as patch,
)
from vllm_iluvatar.ilu_patch.general_patches.patch_connector_divergent_hybrid_hits import (
    get_computed_blocks_for_connector,
)


def _spec_group(spec_cls):
    return SimpleNamespace(spec=MagicMock(spec=spec_cls))


def test_fa_swa_divergent_hit_without_mamba():
    empty = object()
    # MagicMock(spec=...) satisfies isinstance(..., HybridKVCacheCoordinator).
    coordinator = MagicMock(spec=HybridKVCacheCoordinator)
    coordinator.full_attention_group_id = 0
    coordinator.attention_groups = (
        _spec_group(FullAttentionSpec),
        _spec_group(SlidingWindowSpec),
    )
    coordinator.find_longest_cache_hit_per_group.return_value = (
        ("fa-blocks", "swa-blocks"),
        (230 * 256, 0),
    )
    manager = SimpleNamespace(
        coordinator=coordinator,
        kv_cache_config=SimpleNamespace(has_mamba_layers=False),
        empty_kv_cache_blocks=empty,
        prefix_cache_lookup_enabled=lambda request: True,
        get_computed_blocks=MagicMock(return_value=(empty, 0, 0)),
        create_kv_cache_blocks=lambda computed: computed,
    )
    request = SimpleNamespace(block_hashes=["h"], num_tokens=65536)

    blocks, num_local, boundary, hit_diverged = get_computed_blocks_for_connector(
        manager, request
    )

    assert blocks == ("fa-blocks", "swa-blocks")
    assert num_local == 230 * 256
    assert boundary == 0
    assert hit_diverged is True
    manager.get_computed_blocks.assert_not_called()


def test_non_swa_hybrid_falls_back_to_reconciled_lookup():
    empty = object()
    reconciled = (empty, 0, 0)
    coordinator = MagicMock(spec=HybridKVCacheCoordinator)
    coordinator.full_attention_group_id = 0
    coordinator.attention_groups = (
        _spec_group(FullAttentionSpec),
        _spec_group(ChunkedLocalAttentionSpec),
    )
    manager = SimpleNamespace(
        coordinator=coordinator,
        kv_cache_config=SimpleNamespace(has_mamba_layers=False),
        empty_kv_cache_blocks=empty,
        get_computed_blocks=MagicMock(return_value=reconciled),
    )
    request = SimpleNamespace(block_hashes=[], num_tokens=16)

    assert get_computed_blocks_for_connector(manager, request) == (
        empty,
        0,
        0,
        False,
    )
    manager.get_computed_blocks.assert_called_once_with(request)
    coordinator.find_longest_cache_hit_per_group.assert_not_called()


def test_community_mamba_path_is_preserved():
    empty = object()
    coordinator = MagicMock(spec=HybridKVCacheCoordinator)
    coordinator.full_attention_group_id = 0
    coordinator.attention_groups = ()
    coordinator.find_longest_cache_hit_per_group.return_value = (
        ("fa-blocks", "mamba-blocks"),
        (1024, 1024),
    )
    manager = SimpleNamespace(
        coordinator=coordinator,
        kv_cache_config=SimpleNamespace(has_mamba_layers=True),
        empty_kv_cache_blocks=empty,
        prefix_cache_lookup_enabled=lambda request: True,
        get_computed_blocks=MagicMock(return_value=(empty, 0, 0)),
        create_kv_cache_blocks=lambda computed: computed,
    )
    request = SimpleNamespace(block_hashes=["h"], num_tokens=2048)

    assert get_computed_blocks_for_connector(manager, request) == (
        ("fa-blocks", "mamba-blocks"),
        1024,
        0,
        False,
    )
    manager.get_computed_blocks.assert_not_called()


def test_non_hybrid_falls_back_to_reconciled_lookup():
    empty = object()
    reconciled = (empty, 0, 0)
    manager = SimpleNamespace(
        coordinator=SimpleNamespace(full_attention_group_id=None),
        kv_cache_config=SimpleNamespace(has_mamba_layers=False),
        empty_kv_cache_blocks=empty,
        get_computed_blocks=MagicMock(return_value=reconciled),
    )
    request = SimpleNamespace(block_hashes=[], num_tokens=16)

    assert get_computed_blocks_for_connector(manager, request) == (
        empty,
        0,
        0,
        False,
    )
    manager.get_computed_blocks.assert_called_once_with(request)


def test_baseline_matches_installed_community_method():
    patch._check_community_method(
        KVCacheManager.get_computed_blocks_for_connector
    )


def test_community_method_drift_fails(monkeypatch):
    monkeypatch.setattr(patch.p, "_enabled", lambda env: True)
    monkeypatch.setattr(patch, "_COMMUNITY_METHOD_SOURCE_SHA", "0" * 64)

    with pytest.raises(RuntimeError, match="community .* source changed"):
        patch._check_community_method(
            KVCacheManager.get_computed_blocks_for_connector
        )


def test_fingerprint_switch_skips_check(monkeypatch):
    monkeypatch.setattr(patch.p, "_enabled", lambda env: False)
    monkeypatch.setattr(patch, "_COMMUNITY_METHOD_SOURCE_SHA", "0" * 64)

    patch._check_community_method(
        KVCacheManager.get_computed_blocks_for_connector
    )
