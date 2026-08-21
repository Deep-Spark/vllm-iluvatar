# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the KV cache interface patch wiring."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from vllm_iluvatar.ilu_patch.general_patches import patch_kv_cache_interface


def _setup_modules(monkeypatch, *, flash_quant_type: str, mla_quant_type: str):
    import vllm.envs as envs
    import vllm.v1 as vllm_v1

    monkeypatch.setattr(
        envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", flash_quant_type, raising=False
    )
    monkeypatch.setattr(
        envs, "VLLM_ILU_MLA_ATTN_QUANT_TYPE", mla_quant_type, raising=False
    )

    full_original = object()
    sliding_original = object()
    mla_original = object()

    class FullAttentionSpec:
        real_page_size_bytes = full_original

    class SlidingWindowSpec:
        real_page_size_bytes = sliding_original

    class MLAAttentionSpec:
        real_page_size_bytes = mla_original

    @dataclass
    class KVCacheTensor:
        size: int

    original_tensor_init = KVCacheTensor.__init__
    community = ModuleType("vllm.v1.kv_cache_interface")
    community.FullAttentionSpec = FullAttentionSpec
    community.SlidingWindowSpec = SlidingWindowSpec
    community.MLAAttentionSpec = MLAAttentionSpec
    community.KVCacheTensor = KVCacheTensor

    def full_real(self):
        return "full-real"

    def full_scale(self):
        return "full-scale"

    def full_fixed(self):
        return "full-fixed"

    def sliding_real(self):
        return "sliding-real"

    def sliding_scale(self):
        return "sliding-scale"

    def sliding_fixed(self):
        return "sliding-fixed"

    def mla_real(self):
        return "mla-real"

    def mla_scale(self):
        return "mla-scale"

    def mla_fixed(self):
        return "mla-fixed"

    implementation = ModuleType("vllm_iluvatar.v1.kv_cache_interface")
    implementation._full_attention_real_page_size_bytes = full_real
    implementation._full_attention_scale_page_size_bytes = full_scale
    implementation._full_attention_fixed_size_bytes = full_fixed
    implementation._sliding_window_real_page_size_bytes = sliding_real
    implementation._sliding_window_scale_page_size_bytes = sliding_scale
    implementation._sliding_window_fixed_size_bytes = sliding_fixed
    implementation._mla_attention_real_page_size_bytes = mla_real
    implementation._mla_attention_scale_page_size_bytes = mla_scale
    implementation._mla_attention_fixed_size_bytes = mla_fixed

    monkeypatch.setattr(vllm_v1, "kv_cache_interface", community, raising=False)
    monkeypatch.setitem(sys.modules, community.__name__, community)
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)

    return SimpleNamespace(
        full=FullAttentionSpec,
        sliding=SlidingWindowSpec,
        mla=MLAAttentionSpec,
        tensor=KVCacheTensor,
        full_original=full_original,
        sliding_original=sliding_original,
        mla_original=mla_original,
        original_tensor_init=original_tensor_init,
        implementation=implementation,
    )


def test_patch_is_noop_when_quantization_is_disabled(monkeypatch):
    modules = _setup_modules(monkeypatch, flash_quant_type="", mla_quant_type="")

    patch_kv_cache_interface._apply()

    assert modules.full.real_page_size_bytes is modules.full_original
    assert modules.sliding.real_page_size_bytes is modules.sliding_original
    assert modules.mla.real_page_size_bytes is modules.mla_original
    for cls in (modules.full, modules.sliding, modules.mla):
        assert not hasattr(cls, "scale_page_size_bytes")
        assert not hasattr(cls, "fixed_size_bytes")
    assert set(modules.tensor.__dataclass_fields__) == {"size"}
    assert modules.tensor.__init__ is modules.original_tensor_init


@pytest.mark.parametrize(
    ("flash_quant_type", "mla_quant_type", "patched_classes"),
    [
        ("KVINT8", "", {"full", "sliding"}),
        ("KINT8", "", {"full", "sliding"}),
        ("", "INT8", {"mla"}),
        ("KVINT8", "INT8", {"full", "sliding", "mla"}),
        ("KINT8", "INT8", {"full", "sliding", "mla"}),
    ],
)
def test_patch_replaces_all_enabled_attention_specs_and_tensor(
    monkeypatch,
    flash_quant_type,
    mla_quant_type,
    patched_classes,
):
    modules = _setup_modules(
        monkeypatch,
        flash_quant_type=flash_quant_type,
        mla_quant_type=mla_quant_type,
    )

    patch_kv_cache_interface._apply()

    prefixes = {
        "full": "_full_attention",
        "sliding": "_sliding_window",
        "mla": "_mla_attention",
    }
    for class_name, prefix in prefixes.items():
        cls = getattr(modules, class_name)
        if class_name in patched_classes:
            for property_name in (
                "real_page_size_bytes",
                "scale_page_size_bytes",
                "fixed_size_bytes",
            ):
                patched_property = getattr(cls, property_name)
                expected = getattr(modules.implementation, f"{prefix}_{property_name}")
                assert isinstance(patched_property, property)
                assert patched_property.fget is expected
        else:
            original = getattr(modules, f"{class_name}_original")
            assert cls.real_page_size_bytes is original
            assert not hasattr(cls, "scale_page_size_bytes")
            assert not hasattr(cls, "fixed_size_bytes")

    assert {"fixed_size", "scale_size"} <= set(modules.tensor.__dataclass_fields__)
    tensor = modules.tensor(size=128, fixed_size=16, scale_size=32)
    assert tensor.fixed_size == 16
    assert tensor.scale_size == 32
