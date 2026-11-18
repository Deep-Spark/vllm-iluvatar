from __future__ import annotations

from types import SimpleNamespace

import pytest
import vllm.v1.attention.backends.utils as community
from vllm.config import CacheConfig

from vllm_iluvatar.v1.attention.backends.utils import make_resolve_kv_cache_layout


@pytest.mark.parametrize(
    ("supported", "expected"),
    [
        ([["LBNHC", "LBHNC"], ["LBHNC"]], "LBHNC"),
        (
            [
                [x.name for x in community._DEFAULT_LAYOUT_PREFERENCE],
                ["BLHNC", "BLNHC"],
            ],
            "BLHNC",
        ),
        ([["BLNHC", "BLHNC"], ["BLHNC", "BLNHC"]], "BLHNC"),
        ([["BLHNC", "BLNHC"], ["BLNHC", "BLHNC"], ["BLNHC", "BLHNC"]], "BLNHC"),
        ([["LBNHC", "LBHNC"], ["LBNHC", "LBHNC"]], "LBNHC"),
    ],
)
def test_shared_layout_follows_constrained_workers(monkeypatch, supported, expected):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    monkeypatch.setattr(community, "get_kv_connector_cache_layout", lambda _: None)
    config = SimpleNamespace(cache_config=CacheConfig())
    original_inputs = [names.copy() for names in supported]

    layout = make_resolve_kv_cache_layout(community.resolve_kv_cache_layout)(
        config, supported
    )

    assert layout.name == config.cache_config.kv_cache_layout == expected
    assert supported == original_inputs


def test_no_shared_layout_fails_before_recording_result():
    config = SimpleNamespace(cache_config=CacheConfig())
    with pytest.raises(ValueError, match="no common supported KV cache layout"):
        make_resolve_kv_cache_layout(community.resolve_kv_cache_layout)(
            config, [["LBNHC"], ["LBHNC"]]
        )
    assert config.cache_config.kv_cache_layout is None


def test_explicit_layout_must_be_shared_by_every_worker(monkeypatch):
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "LBNHC")
    config = SimpleNamespace(cache_config=CacheConfig())
    with pytest.raises(ValueError, match="does not satisfy every supported set"):
        make_resolve_kv_cache_layout(community.resolve_kv_cache_layout)(
            config, [["LBNHC", "LBHNC"], ["LBHNC"]]
        )


def test_original_receives_specs_and_still_checks_constraints():
    specs = object()
    config = SimpleNamespace(cache_config=SimpleNamespace(kv_cache_layout=None))

    def original(vllm_config, supported_layouts, kv_cache_specs=None):
        assert vllm_config is config
        assert supported_layouts == [["LBHNC"], ["LBHNC"]]
        assert kv_cache_specs is specs
        raise ValueError("upstream mixed-spec layout constraint")

    with pytest.raises(ValueError, match="upstream mixed-spec layout constraint"):
        make_resolve_kv_cache_layout(original)(
            config, [["LBNHC", "LBHNC"], ["LBHNC"]], specs
        )


def test_cached_result_keeps_upstream_shortcut(monkeypatch):
    config = SimpleNamespace(cache_config=CacheConfig())
    config.cache_config.kv_cache_layout = "LBNHC"
    layout = make_resolve_kv_cache_layout(community.resolve_kv_cache_layout)(config, [])
    assert layout.name == "LBNHC"


@pytest.mark.parametrize("supported", [[], [[]], [["LBNHC"], []]])
def test_missing_worker_support_keeps_upstream_failure(supported):
    config = SimpleNamespace(cache_config=CacheConfig())
    with pytest.raises(AssertionError, match="No worker reported"):
        make_resolve_kv_cache_layout(community.resolve_kv_cache_layout)(
            config, supported
        )
