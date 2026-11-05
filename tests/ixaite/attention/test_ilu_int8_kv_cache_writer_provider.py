# SPDX-License-Identifier: Apache-2.0
"""Independent INT8 writer binding must never publish a native fallback."""

from unittest.mock import Mock, call

import pytest
import torch

from tests.ixaite.backend_isolation.ixformer_guard import forbid_ixformer_calls
from vllm_iluvatar.backends.ixaite.attention import kv_cache


@pytest.fixture(autouse=True)
def _reset_int8_bindings():
    kv_cache._RESHAPE_WITH_SCALE_IMPL = None
    kv_cache._RESHAPE_MIX_IMPL = None
    kv_cache._QUANT_IMPL = None
    yield
    kv_cache._RESHAPE_WITH_SCALE_IMPL = None
    kv_cache._RESHAPE_MIX_IMPL = None
    kv_cache._QUANT_IMPL = None


# Both cache formats bind their own Python ABI once at startup.
def test_binds_both_python_entries(monkeypatch):
    with_scale, mix = Mock(), Mock()
    loader = Mock(side_effect=[(with_scale, "supported"), (mix, "supported")])
    monkeypatch.setattr(kv_cache, "load_ixaite_python_op", loader)
    kv_cache.configure_int8_kv_cache_writer_providers()
    assert kv_cache._RESHAPE_WITH_SCALE_IMPL is with_scale
    assert kv_cache._RESHAPE_MIX_IMPL is mix
    assert loader.call_args_list == [
        call("reshape_and_cache_with_scale", "reshape_and_cache_with_scale"),
        call("reshape_and_cache_mix", "reshape_and_cache_mix"),
    ]


def test_configure_binds_quant_and_writers(monkeypatch):
    quant, with_scale, mix = Mock(), Mock(), Mock()
    loader = Mock(
        side_effect=[
            (quant, "supported"),
            (with_scale, "supported"),
            (mix, "supported"),
        ]
    )
    monkeypatch.setattr(kv_cache, "load_ixaite_python_op", loader)
    kv_cache.configure_int8_attention_providers()
    assert kv_cache._QUANT_IMPL is quant
    assert kv_cache._RESHAPE_WITH_SCALE_IMPL is with_scale
    assert kv_cache._RESHAPE_MIX_IMPL is mix
    assert loader.call_args_list == [
        call("scaled_int8_quant_for_attn", "scaled_int8_quant_for_attn"),
        call("reshape_and_cache_with_scale", "reshape_and_cache_with_scale"),
        call("reshape_and_cache_mix", "reshape_and_cache_mix"),
    ]


def test_missing_quant_fails_before_writers(monkeypatch):
    monkeypatch.setattr(
        kv_cache,
        "load_ixaite_python_op",
        lambda name, _: (None, "missing quant")
        if name == "scaled_int8_quant_for_attn"
        else (object(), "ok"),
    )
    with pytest.raises(RuntimeError, match="scaled_int8_quant_for_attn"):
        kv_cache.configure_int8_attention_providers()
    assert kv_cache._QUANT_IMPL is None
    assert kv_cache._RESHAPE_WITH_SCALE_IMPL is None


# A missing entry fails before publishing an incomplete pair.
@pytest.mark.parametrize(
    "missing", ["reshape_and_cache_with_scale", "reshape_and_cache_mix"]
)
def test_unavailable_pair_fails_before_binding(monkeypatch, missing):
    old = object()
    monkeypatch.setattr(kv_cache, "_RESHAPE_WITH_SCALE_IMPL", old)
    monkeypatch.setattr(kv_cache, "_RESHAPE_MIX_IMPL", old)
    monkeypatch.setattr(
        kv_cache,
        "load_ixaite_python_op",
        lambda name, _: (None if name == missing else object(), "not available"),
    )
    with pytest.raises(RuntimeError, match=missing):
        kv_cache.configure_int8_kv_cache_writer_providers()
    assert kv_cache._RESHAPE_WITH_SCALE_IMPL is old
    assert kv_cache._RESHAPE_MIX_IMPL is old


# Reuse exact numerical cache comparison for both standalone writer contracts.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize("mixed", [False, True])
def test_gpu_writer_matches_original(mixed):
    from tests.ixaite.benchmarks.core.harness import clone_workload
    from tests.ixaite.benchmarks.suites.reshape_and_cache_int8 import (
        ReshapeAndCacheInt8Case,
    )

    name = "reshape_and_cache_mix" if mixed else "reshape_and_cache_with_scale"
    case = ReshapeAndCacheInt8Case("writer", name, 17, 4, 256, 16, mixed, ())
    reference = case.make_workload(torch.device("cuda"), torch.bfloat16)
    actual = clone_workload(reference)
    case.run(getattr(torch.ops._C, name), reference)
    with forbid_ixformer_calls():
        kv_cache.configure_int8_kv_cache_writer_providers()
        fn = kv_cache._RESHAPE_MIX_IMPL if mixed else kv_cache._RESHAPE_WITH_SCALE_IMPL
        case.run(fn, actual)
        case.check(reference, actual)
