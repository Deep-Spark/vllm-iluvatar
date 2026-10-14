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

    from vllm_iluvatar.v1 import core as ilu_core

    monkeypatch.setattr(
        envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", flash_quant_type, raising=False
    )
    monkeypatch.setattr(
        envs, "VLLM_ILU_MLA_ATTN_QUANT_TYPE", mla_quant_type, raising=False
    )

    def original_from_groups(vllm_config, kv_cache_groups, available_memory):
        return vllm_config, kv_cache_groups, available_memory

    def original_configs(vllm_config, kv_cache_specs, available_memory):
        return vllm_config, kv_cache_specs, available_memory

    def original_scheduler(kv_cache_configs):
        return kv_cache_configs

    def original_pool_bytes_per_block(kv_cache_groups):
        return kv_cache_groups

    community = ModuleType("vllm.v1.core.kv_cache_utils")
    community.get_kv_cache_config_from_groups = original_from_groups
    community._pool_bytes_per_block = original_pool_bytes_per_block
    community.get_kv_cache_configs = original_configs
    community.generate_scheduler_kv_cache_config = original_scheduler

    def replacement_from_groups(vllm_config, kv_cache_groups, available_memory):
        return vllm_config, kv_cache_groups, available_memory

    def replacement_pool_bytes_per_block(kv_cache_groups):
        return kv_cache_groups

    def wrap(base):
        def wrapped(*args, **kwargs):
            return base(*args, **kwargs)

        wrapped.base = base
        return wrapped

    implementation = ModuleType("vllm_iluvatar.v1.core.kv_cache_utils")
    implementation.get_kv_cache_config_from_groups = replacement_from_groups
    implementation._ilu_pool_bytes_per_block = replacement_pool_bytes_per_block
    implementation.make_get_kv_cache_config_from_groups = wrap
    implementation.make_get_kv_cache_configs = wrap
    implementation.make_generate_scheduler_kv_cache_config = wrap

    monkeypatch.setattr(core, "kv_cache_utils", community, raising=False)
    monkeypatch.setattr(ilu_core, "kv_cache_utils", implementation, raising=False)
    monkeypatch.setitem(sys.modules, community.__name__, community)
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)

    return SimpleNamespace(
        community=community,
        original_from_groups=original_from_groups,
        original_pool_bytes_per_block=original_pool_bytes_per_block,
        original_configs=original_configs,
        original_scheduler=original_scheduler,
        replacement_from_groups=replacement_from_groups,
        replacement_pool_bytes_per_block=replacement_pool_bytes_per_block,
    )


def test_patch_wraps_community_planners_when_quantization_is_disabled(monkeypatch):
    modules = _setup_modules(monkeypatch, flash_quant_type="", mla_quant_type="")

    patch_kv_cache_utils._apply()

    assert (
        modules.community.get_kv_cache_config_from_groups.base
        is modules.original_from_groups
    )
    assert (
        modules.community._pool_bytes_per_block is modules.original_pool_bytes_per_block
    )
    assert modules.community.get_kv_cache_configs.base is modules.original_configs
    assert (
        modules.community.generate_scheduler_kv_cache_config.base
        is modules.original_scheduler
    )


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
        modules.community.get_kv_cache_config_from_groups.base
        is modules.replacement_from_groups
    )
    assert (
        modules.community._pool_bytes_per_block
        is modules.replacement_pool_bytes_per_block
    )
    assert modules.community.get_kv_cache_configs.base is modules.original_configs
    assert (
        modules.community.generate_scheduler_kv_cache_config.base
        is modules.original_scheduler
    )
