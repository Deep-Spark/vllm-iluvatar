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
    CircularBufferSpec,
    FullAttentionSpec,
    KpoolTailSpec,
    SlidingWindowSpec,
)
from vllm_iluvatar.ilu_patch.general_patches import (
    patch_connector_divergent_hybrid_hits as patch,
)
from vllm_iluvatar.ilu_patch.general_patches.patch_connector_divergent_hybrid_hits import (
    get_computed_blocks_for_connector,
    get_zeroing_block_ids_in_range,
)


def _spec_group(spec_cls, group_ids=(0,)):
    spec = MagicMock(spec=spec_cls)
    spec.prefix_cacheable = spec_cls not in (CircularBufferSpec, KpoolTailSpec)
    return SimpleNamespace(spec=spec, group_ids=list(group_ids))


def _kv_cache_group(spec_cls):
    spec = MagicMock(spec=spec_cls)
    spec.prefix_cacheable = spec_cls not in (CircularBufferSpec, KpoolTailSpec)
    return SimpleNamespace(kv_cache_spec=spec)


def test_fa_swa_divergent_hit_without_mamba():
    empty = object()
    # MagicMock(spec=...) satisfies isinstance(..., HybridKVCacheCoordinator).
    coordinator = MagicMock(spec=HybridKVCacheCoordinator)
    coordinator.full_attention_group_id = 0
    coordinator.attention_groups = (
        _spec_group(FullAttentionSpec, (0,)),
        _spec_group(SlidingWindowSpec, (1,)),
    )
    coordinator.find_longest_cache_hit_per_group.return_value = (
        ("fa-blocks", "swa-blocks"),
        (230 * 256, 0),
    )
    manager = SimpleNamespace(
        coordinator=coordinator,
        kv_cache_config=SimpleNamespace(
            has_mamba_layers=False,
            kv_cache_groups=(
                _kv_cache_group(FullAttentionSpec),
                _kv_cache_group(SlidingWindowSpec),
            ),
        ),
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


def test_fa_swa_with_circular_buffer_converged_not_diverged():
    """CircularBufferSpec slot stays 0 in per_group_hits without forcing hit_diverged."""
    empty = object()
    coordinator = MagicMock(spec=HybridKVCacheCoordinator)
    coordinator.full_attention_group_id = 0
    coordinator.attention_groups = (
        _spec_group(FullAttentionSpec, (0,)),
        _spec_group(SlidingWindowSpec, (1,)),
    )
    coordinator.find_longest_cache_hit_per_group.return_value = (
        ("fa-blocks", "swa-blocks", []),
        (230 * 256, 230 * 256, 0),
    )
    manager = SimpleNamespace(
        coordinator=coordinator,
        kv_cache_config=SimpleNamespace(
            has_mamba_layers=False,
            kv_cache_groups=(
                _kv_cache_group(FullAttentionSpec),
                _kv_cache_group(SlidingWindowSpec),
                _kv_cache_group(CircularBufferSpec),
            ),
        ),
        empty_kv_cache_blocks=empty,
        prefix_cache_lookup_enabled=lambda request: True,
        get_computed_blocks=MagicMock(return_value=(empty, 0, 0)),
        create_kv_cache_blocks=lambda computed: computed,
    )
    request = SimpleNamespace(block_hashes=["h"], num_tokens=65536)

    blocks, num_local, boundary, hit_diverged = get_computed_blocks_for_connector(
        manager, request
    )

    assert blocks == ("fa-blocks", "swa-blocks", [])
    assert num_local == 230 * 256
    assert boundary == 0
    assert hit_diverged is False
    manager.get_computed_blocks.assert_not_called()


def test_multiple_fa_groups_disagreeing_falls_back_to_reconciled_lookup():
    """A lagging FA group (e.g. EAGLE drafter) must fall back to reconciled lookup."""
    empty = object()
    reconciled = ("reconciled-blocks", 9 * 256, 0)
    coordinator = MagicMock(spec=HybridKVCacheCoordinator)
    coordinator.full_attention_group_id = 0
    coordinator.attention_groups = (
        _spec_group(FullAttentionSpec, (0,)),
        _spec_group(FullAttentionSpec, (1,)),
        _spec_group(SlidingWindowSpec, (2,)),
    )
    coordinator.find_longest_cache_hit_per_group.return_value = (
        ("fa0-blocks", "fa1-eagle-blocks", "swa-blocks"),
        (10 * 256, 9 * 256, 0),
    )
    manager = SimpleNamespace(
        coordinator=coordinator,
        kv_cache_config=SimpleNamespace(
            has_mamba_layers=False,
            kv_cache_groups=(
                _kv_cache_group(FullAttentionSpec),
                _kv_cache_group(FullAttentionSpec),
                _kv_cache_group(SlidingWindowSpec),
            ),
        ),
        empty_kv_cache_blocks=empty,
        prefix_cache_lookup_enabled=lambda request: True,
        get_computed_blocks=MagicMock(return_value=reconciled),
        create_kv_cache_blocks=lambda computed: computed,
    )
    request = SimpleNamespace(block_hashes=["h"], num_tokens=4096)

    assert get_computed_blocks_for_connector(manager, request) == (
        "reconciled-blocks",
        9 * 256,
        0,
        False,
    )
    manager.get_computed_blocks.assert_called_once_with(request)


def test_get_zeroing_block_ids_in_range_includes_swa_window_blocks():
    """SWA window blocks allocated from num_cached_block (< start_idx) are included."""
    fa_blocks = [SimpleNamespace(block_id=100 + i) for i in range(256)]
    swa_blocks = [SimpleNamespace(block_id=0)] * 240 + [
        SimpleNamespace(block_id=500 + i) for i in range(16)
    ]
    fa_mgr = SimpleNamespace(
        records_new_block_ids=True,
        block_size=256,
        num_cached_block={"req-1": 254},
        req_to_blocks={"req-1": fa_blocks},
    )
    swa_mgr = SimpleNamespace(
        records_new_block_ids=True,
        block_size=256,
        num_cached_block={"req-1": 240},
        req_to_blocks={"req-1": swa_blocks},
    )
    manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=(fa_mgr, swa_mgr))
    )

    ids = get_zeroing_block_ids_in_range(
        manager, "req-1", 254 * 256, 256 * 256
    )

    expected_fa = [100 + 254, 100 + 255]
    expected_swa = [500 + i for i in range(16)]
    assert ids == expected_fa + expected_swa


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
        KVCacheManager.get_computed_blocks_for_connector,
        patch._COMMUNITY_METHOD_SOURCE_SHA,
        patch._COMMUNITY_METHOD,
    )
    patch._check_community_method(
        KVCacheManager.get_zeroing_block_ids_in_range,
        patch._COMMUNITY_ZEROING_METHOD_SOURCE_SHA,
        patch._COMMUNITY_ZEROING_METHOD,
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
