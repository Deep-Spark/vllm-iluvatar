# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the KV cache utility patch wiring."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_iluvatar.ilu_patch.general_patches import patch_kv_cache_utils


def _setup_modules(monkeypatch, *, flash_quant_type: str, mla_quant_type: str):
    import vllm.envs as envs
    from vllm.v1 import core

    monkeypatch.setattr(
        envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", flash_quant_type, raising=False
    )
    monkeypatch.setattr(
        envs, "VLLM_ILU_MLA_ATTN_QUANT_TYPE", mla_quant_type, raising=False
    )

    def original_from_groups(*args, **kwargs):
        return args, kwargs

    def original_configs(*args, **kwargs):
        return args, kwargs

    community = ModuleType("vllm.v1.core.kv_cache_utils")
    community.get_kv_cache_config_from_groups = original_from_groups
    community.get_kv_cache_configs = original_configs

    def replacement_from_groups(*args, **kwargs):
        return args, kwargs

    def replacement_configs(*args, **kwargs):
        return args, kwargs

    implementation = ModuleType("vllm_iluvatar.v1.core.kv_cache_utils")
    implementation.get_kv_cache_config_from_groups = replacement_from_groups
    implementation.get_kv_cache_configs = replacement_configs

    monkeypatch.setattr(core, "kv_cache_utils", community, raising=False)
    monkeypatch.setitem(sys.modules, community.__name__, community)
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)

    return SimpleNamespace(
        community=community,
        original_from_groups=original_from_groups,
        original_configs=original_configs,
        replacement_from_groups=replacement_from_groups,
        replacement_configs=replacement_configs,
    )


def test_patch_is_noop_when_quantization_is_disabled(monkeypatch):
    modules = _setup_modules(monkeypatch, flash_quant_type="", mla_quant_type="")

    patch_kv_cache_utils._apply()

    assert (
        modules.community.get_kv_cache_config_from_groups
        is modules.original_from_groups
    )
    assert modules.community.get_kv_cache_configs is modules.original_configs


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
def test_patch_replaces_functions_when_quantization_is_enabled(
    monkeypatch, flash_quant_type, mla_quant_type
):
    modules = _setup_modules(
        monkeypatch,
        flash_quant_type=flash_quant_type,
        mla_quant_type=mla_quant_type,
    )

    patch_kv_cache_utils._apply()

    assert (
        modules.community.get_kv_cache_config_from_groups
        is modules.replacement_from_groups
    )
    assert modules.community.get_kv_cache_configs is modules.replacement_configs
