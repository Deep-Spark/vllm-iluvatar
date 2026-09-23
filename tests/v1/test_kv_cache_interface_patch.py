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
    state_content_original = object()
    num_heads_original = object()

    class AttentionSpec:
        state_content_size_bytes = state_content_original
        num_heads = num_heads_original

    class FullAttentionSpec(AttentionSpec):
        real_page_size_bytes = full_original

    class SlidingWindowSpec(AttentionSpec):
        real_page_size_bytes = sliding_original

    class MLAAttentionSpec(FullAttentionSpec):
        real_page_size_bytes = mla_original

    class SlidingWindowMLASpec(SlidingWindowSpec):
        pass

    @dataclass
    class KVCacheTensor:
        size: int

    original_tensor_init = KVCacheTensor.__init__
    community = ModuleType("vllm.v1.kv_cache_interface")
    community.AttentionSpec = AttentionSpec
    community.FullAttentionSpec = FullAttentionSpec
    community.SlidingWindowSpec = SlidingWindowSpec
    community.MLAAttentionSpec = MLAAttentionSpec
    community.SlidingWindowMLASpec = SlidingWindowMLASpec
    community.KVCacheTensor = KVCacheTensor

    def full_scale(self):
        return "full-scale"

    def sliding_scale(self):
        return "sliding-scale"

    def mla_scale(self):
        return "mla-scale"

    def state_content(self):
        return "state-content"

    def num_heads_fn(self):
        return "num-heads"

    implementation = ModuleType("vllm_iluvatar.v1.kv_cache_interface")
    implementation.attention_state_content_size_bytes = state_content
    implementation.attention_num_heads = num_heads_fn
    implementation._full_attention_scale_page_size_bytes = full_scale
    implementation._sliding_window_scale_page_size_bytes = sliding_scale
    implementation._mla_attention_scale_page_size_bytes = mla_scale

    monkeypatch.setattr(vllm_v1, "kv_cache_interface", community, raising=False)
    monkeypatch.setitem(sys.modules, community.__name__, community)
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)

    return SimpleNamespace(
        base=AttentionSpec,
        full=FullAttentionSpec,
        sliding=SlidingWindowSpec,
        mla=MLAAttentionSpec,
        tensor=KVCacheTensor,
        full_original=full_original,
        sliding_original=sliding_original,
        mla_original=mla_original,
        state_content_original=state_content_original,
        num_heads_original=num_heads_original,
        original_tensor_init=original_tensor_init,
        implementation=implementation,
    )


def test_patch_is_noop_when_quantization_is_disabled(monkeypatch):
    modules = _setup_modules(monkeypatch, flash_quant_type="", mla_quant_type="")

    patch_kv_cache_interface._apply()

    assert modules.base.state_content_size_bytes is modules.state_content_original
    assert modules.base.num_heads is modules.num_heads_original
    for cls in (modules.full, modules.sliding, modules.mla):
        assert not hasattr(cls, "scale_page_size_bytes")
    assert set(modules.tensor.__dataclass_fields__) == {"size"}
    assert modules.tensor.__init__ is modules.original_tensor_init


@pytest.mark.parametrize(
    ("flash_quant_type", "mla_quant_type"),
    [
        ("KVINT8", ""),
        ("KINT8", ""),
        ("", "INT8"),
        ("KVINT8", "INT8"),
        ("KINT8", "INT8"),
    ],
)
def test_patch_wires_properties(monkeypatch, flash_quant_type, mla_quant_type):
    modules = _setup_modules(
        monkeypatch,
        flash_quant_type=flash_quant_type,
        mla_quant_type=mla_quant_type,
    )

    patch_kv_cache_interface._apply()

    flash_enabled = flash_quant_type != ""
    mla_enabled = mla_quant_type != ""

    # state_content_size_bytes and num_heads on the base class are overridden
    # whenever any int8 quant is enabled.
    assert isinstance(modules.base.state_content_size_bytes, property)
    assert (
        modules.base.state_content_size_bytes.fget
        is modules.implementation.attention_state_content_size_bytes
    )
    assert isinstance(modules.base.num_heads, property)
    assert modules.base.num_heads.fget is modules.implementation.attention_num_heads

    # real_page_size_bytes is never overridden (community alias drives it via
    # state_content_size_bytes).
    assert modules.full.real_page_size_bytes is modules.full_original
    assert modules.sliding.real_page_size_bytes is modules.sliding_original
    assert modules.mla.real_page_size_bytes is modules.mla_original

    for cls in (modules.full, modules.sliding):
        if flash_enabled:
            assert isinstance(cls.scale_page_size_bytes, property)
        else:
            assert not hasattr(cls, "scale_page_size_bytes")

    if mla_enabled:
        assert isinstance(modules.mla.scale_page_size_bytes, property)
        assert (
            modules.mla.scale_page_size_bytes.fget
            is modules.implementation._mla_attention_scale_page_size_bytes
        )
    elif flash_enabled:
        # mla int8 disabled but flash int8 enabled: MLAAttentionSpec subclasses
        # FullAttentionSpec in the community, so it inherits Full's
        # scale_page_size_bytes property (fget is the full-attention scale fn).
        assert isinstance(modules.mla.scale_page_size_bytes, property)
        assert (
            modules.mla.scale_page_size_bytes.fget
            is modules.implementation._full_attention_scale_page_size_bytes
        )
    else:
        assert not hasattr(modules.mla, "scale_page_size_bytes")

    assert "scale_size" in set(modules.tensor.__dataclass_fields__)
    tensor = modules.tensor(size=128, scale_size=32)
    assert tensor.scale_size == 32
