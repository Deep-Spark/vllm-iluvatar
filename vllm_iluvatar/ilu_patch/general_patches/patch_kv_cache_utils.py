# SPDX-License-Identifier: Apache-2.0
"""Wiring for Ilu quantized and FP8 wire-cache planning.

Corresponds to: vllm/v1/core/kv_cache_utils.py

Target   : _pool_bytes_per_block, get_kv_cache_config_from_groups,
           get_kv_cache_configs, generate_scheduler_kv_cache_config
Why      : preserve the v0.31 Ilu INT8 shared-buffer planner while reserving
           and describing a producer-only packed FP8 transfer cache.
Scope    : quantized planning is gated by the attention quantization
           environment; FP8 wire planning is gated to enabled IluNixl
           producer connectors.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from vllm_iluvatar.ilu_patch import general_patch as p

_EXPECTED_PARAMETERS = {
    "get_kv_cache_config_from_groups": (
        "vllm_config",
        "kv_cache_groups",
        "available_memory",
    ),
    "get_kv_cache_configs": (
        "vllm_config",
        "kv_cache_specs",
        "available_memory",
    ),
    "generate_scheduler_kv_cache_config": ("kv_cache_configs",),
}


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="Ilu quantized cache and FP8 wire-cache planning",
    )


def _check_parameters(name: str, function: Callable[..., object]) -> None:
    actual = tuple(inspect.signature(function).parameters)
    expected = _EXPECTED_PARAMETERS[name]
    if actual != expected:
        raise RuntimeError(
            f"ILU patch: vllm.v1.core.kv_cache_utils.{name} signature "
            f"changed from {expected} to {actual}; re-sync the FP8 planning "
            "wrapper before using this vLLM version"
        )


def _apply() -> None:
    import vllm.envs as envs
    import vllm.v1.core.kv_cache_utils as community

    import vllm_iluvatar.v1.core.kv_cache_utils as impl

    originals = {name: getattr(community, name) for name in _EXPECTED_PARAMETERS}
    for name, function in originals.items():
        _check_parameters(name, function)

    quantized_cache = (
        envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE != ""
        or envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE != ""
    )
    base_from_groups = (
        impl.get_kv_cache_config_from_groups
        if quantized_cache
        else originals["get_kv_cache_config_from_groups"]
    )

    replacements = {
        "get_kv_cache_config_from_groups": (
            impl.make_get_kv_cache_config_from_groups(base_from_groups)
        ),
        "get_kv_cache_configs": impl.make_get_kv_cache_configs(
            originals["get_kv_cache_configs"]
        ),
        "generate_scheduler_kv_cache_config": (
            impl.make_generate_scheduler_kv_cache_config(
                impl.generate_scheduler_kv_cache_config
            )
        ),
    }
    for name, replacement in replacements.items():
        p.rebind_module_func(community, name, replacement)
    if quantized_cache:
        p.rebind_module_func(
            community, "_pool_bytes_per_block", impl._ilu_pool_bytes_per_block
        )
