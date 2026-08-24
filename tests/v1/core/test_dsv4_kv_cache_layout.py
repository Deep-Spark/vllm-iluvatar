# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the DeepSeek-V4 KV cache layout override."""

from __future__ import annotations

import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

from vllm_iluvatar.ilu_patch import general_patch as p
from vllm_iluvatar.ilu_patch.general_patches import patch_dsv4_kv_cache_layout
from vllm_iluvatar.v1.core import kv_cache_utils as ilu

# Page sizes of the three DeepSeek-V4-Flash cache groups, in bytes.
_PAGE_RATIO_128 = 1168
_PAGE_RATIO_4 = 37376
_PAGE_RATIO_DENSE = 149504


def _group(page_size: int, layer_names: list[str]) -> KVCacheGroupSpec:
    specs = {
        name: SimpleNamespace(page_size_bytes=page_size, block_size=256)
        for name in layer_names
    }
    return KVCacheGroupSpec(
        layer_names=layer_names,
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=specs),
    )


def _dsv4_groups() -> list[KVCacheGroupSpec]:
    return [
        _group(_PAGE_RATIO_4, [f"r4.{i}" for i in range(21)]),
        _group(_PAGE_RATIO_DENSE, [f"dense.{i}" for i in range(3)]),
        _group(_PAGE_RATIO_128, [f"r128.{i}" for i in range(20)]),
    ]


def _config(*, override=None, cross_layers=False) -> SimpleNamespace:
    kv_transfer_config = None
    if cross_layers:
        kv_transfer_config = SimpleNamespace(
            kv_connector_extra_config={"enable_cross_layers_blocks": "True"}
        )
    return SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=override),
        kv_transfer_config=kv_transfer_config,
    )


def test_predicate_selects_multi_group_uniform_specs():
    assert ilu._use_dsv4_kv_cache_layout(_config(), _dsv4_groups())


def test_predicate_rejects_single_group():
    groups = [_group(_PAGE_RATIO_4, ["r4.0"])]
    assert not ilu._use_dsv4_kv_cache_layout(_config(), groups)


def test_predicate_rejects_non_uniform_specs():
    groups = _dsv4_groups()
    groups[1] = KVCacheGroupSpec(
        layer_names=["plain.0"],
        kv_cache_spec=SimpleNamespace(page_size_bytes=_PAGE_RATIO_4),
    )
    assert not ilu._use_dsv4_kv_cache_layout(_config(), groups)


def test_predicate_leaves_explicit_cross_layers_opt_in_on_packed():
    """An explicit opt-in asked for the packed slab, so it keeps getting it."""
    assert not ilu._use_dsv4_kv_cache_layout(
        _config(cross_layers=True), _dsv4_groups()
    )


def test_layers_of_equal_page_size_share_a_slot_across_groups():
    groups = [
        _group(100, ["a.0", "a.1"]),
        _group(100, ["b.0"]),
    ]
    buckets = ilu._bucket_layers_by_page_size(groups)

    assert buckets == {100: [["a.0", "b.0"], ["a.1"]]}
    # One slot per widest group, not one per layer.
    assert ilu._dsv4_bytes_per_block(groups) == 200


def test_bytes_per_block_is_the_sum_over_slots():
    expected = (
        _PAGE_RATIO_4 * 21 + _PAGE_RATIO_DENSE * 3 + _PAGE_RATIO_128 * 20
    )
    assert ilu._dsv4_bytes_per_block(_dsv4_groups()) == expected
    # Strictly more than the packed slab's stride, which is the widest group
    # alone because groups there nest inside one another.
    assert expected > _PAGE_RATIO_4 * 21


def test_every_tensor_is_its_own_allocation():
    """The whole point: no tensor aliases another at a byte offset."""
    groups = _dsv4_groups()
    num_blocks, tensors = ilu._get_kv_cache_config_packed(
        _config(), groups, available_memory=1 << 34
    )

    assert num_blocks > 0
    assert len(tensors) == 21 + 3 + 20
    for tensor in tensors:
        assert tensor.offset == 0
        assert tensor.block_stride == 0

    # Layers may only share a tensor when their pages are the same size, so a
    # shared tensor is carved identically whichever group indexes it.
    page_size_of = {
        name: spec.page_size_bytes
        for group in groups
        for name, spec in group.kv_cache_spec.kv_cache_specs.items()
    }
    for tensor in tensors:
        sizes = {page_size_of[name] for name in tensor.shared_by}
        assert len(sizes) == 1
        assert tensor.size == sizes.pop() * num_blocks


def test_num_blocks_agrees_with_the_capacity_divisor():
    groups = _dsv4_groups()
    available_memory = 1 << 34

    num_blocks, _ = ilu._get_kv_cache_config_packed(
        _config(), groups, available_memory
    )
    bytes_per_block = ilu._pool_bytes_per_block(_config(), groups)

    assert num_blocks == available_memory // bytes_per_block


def test_num_gpu_blocks_override_is_honoured():
    num_blocks, tensors = ilu._get_kv_cache_config_packed(
        _config(override=8), _dsv4_groups(), available_memory=1 << 34
    )
    assert num_blocks == 8
    assert all(tensor.size % 8 == 0 for tensor in tensors)


def test_planner_defers_to_community_for_non_dsv4(monkeypatch):
    groups = [_group(_PAGE_RATIO_4, ["r4.0"])]
    sentinel = (123, ["community-tensors"])
    monkeypatch.setattr(
        ilu, "origin_get_kv_cache_config_packed", lambda *args: sentinel
    )

    assert ilu._get_kv_cache_config_packed(_config(), groups, 1 << 30) is sentinel


def test_divisor_defers_to_community_for_non_dsv4(monkeypatch):
    groups = [_group(_PAGE_RATIO_4, ["r4.0"])]
    monkeypatch.setattr(ilu, "origin_pool_bytes_per_block", lambda *args: 4321)

    assert ilu._pool_bytes_per_block(_config(), groups) == 4321


def test_embedded_baseline_matches_live_community_source():
    from vllm.v1.core import kv_cache_utils as community

    assert p.source_sha(
        inspect.getsource(community._use_packed_kv_cache_config)
    ) == p.source_sha(patch_dsv4_kv_cache_layout._USE_PACKED_BASELINE), (
        "community _use_packed_kv_cache_config drifted; re-check that DeepSeek-V4 "
        "still reaches _get_kv_cache_config_packed before refreshing the baseline"
    )


def _stand_in_predicate(vllm_config, kv_cache_groups):
    """Give the fingerprint guard a real function to read source from."""
    return True


@pytest.mark.parametrize(
    "name",
    ["_get_kv_cache_config_packed", "_pool_bytes_per_block"],
)
def test_apply_rebinds_community_symbols(monkeypatch, name):
    monkeypatch.setattr(p, "check_fingerprint", lambda *a, **k: None)
    monkeypatch.setattr(p, "check_signatures", lambda *a, **k: None)

    from vllm.v1 import core

    community = ModuleType("vllm.v1.core.kv_cache_utils")
    community._use_packed_kv_cache_config = _stand_in_predicate
    community._get_kv_cache_config_packed = lambda *args: "original packed"
    community._pool_bytes_per_block = lambda *args: "original divisor"
    monkeypatch.setattr(core, "kv_cache_utils", community, raising=False)
    monkeypatch.setitem(sys.modules, community.__name__, community)

    patch_dsv4_kv_cache_layout._apply()

    assert getattr(community, name) is getattr(ilu, name)
