# SPDX-License-Identifier: Apache-2.0
"""
Iluvatar-specific environment variable definitions and community-default
overrides.  Implementation layer — no side effects on import.

Corresponds to: vllm/envs.py
Applied by:      vllm_iluvatar.register_platform()

Naming policy:
  - Canonical name: VLLM_ILU_<suffix> (Iluvatar namespace, kept apart from
    real community vars).
  - Legacy alias:   VLLM_<suffix> (the original, non-standard fork name).
  - Both `envs.VLLM_ILU_X` and `envs.VLLM_X` resolve to the same value, and
    both OS env keys are honoured (VLLM_ILU_X preferred, then VLLM_X, then
    default).  Existing call sites (envs.VLLM_X) and launch scripts keep
    working unchanged; new code should migrate to VLLM_ILU_X.
  - NIXL community vars are used directly with their community names.
  - Iluvatar-only NIXL vars use VLLM_ILU_NIXL_* only and do not register
    legacy aliases.

Contents:
  1. _ILU_DEFAULTS    — community vars whose default differs for Iluvatar
                        (NOT renamed; they are real community vars).
  2. _ILU_ENV_VARS    — new Iluvatar env vars with legacy aliases.
  3. _ILU_CANONICAL_ONLY_ENV_VARS — Iluvatar-only vars without aliases.
  4. _ILU_ENV_ALIASES — legacy VLLM_* names mapped to aliased readers.
  5. apply()          — inject defaults + register env readers.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

_TRUE_VALS = ("1", "true", "yes", "on")


def _bool_reader(name: str, default: str) -> Callable[[], bool]:
    return lambda: os.environ.get(name, default).strip().lower() in _TRUE_VALS


# Community default -> Iluvatar default. These are real community vars, so they
# are NOT renamed; we only change their default via os.environ.setdefault.
#   VLLM_USE_FLASHINFER_SAMPLER       : "1"   -> "0"
#   VLLM_DISABLE_SHARED_EXPERTS_STREAM: "0"   -> "1"
#   VLLM_USE_V2_MODEL_RUNNER          : unset -> "0"
_ILU_DEFAULTS: dict[str, str] = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "0",
}


def _get(new: str, old: str, default: str) -> str:
    """Read OS env, preferring the canonical VLLM_ILU_* name, then the legacy
    VLLM_* name, then the default."""
    val = os.environ.get(new)
    if val is not None:
        return val
    return os.environ.get(old, default)


if TYPE_CHECKING:
    # Canonical names (new code should use these).
    VLLM_ILU_W8A8_MOE_USE_W4A8: bool = False
    VLLM_ILU_WNA16_MOE_USE_W4A8: bool = False
    VLLM_ILU_W4A8_FORMAT: str = "TN"
    VLLM_ILU_W4A8_VERSION: int = 2
    VLLM_ILU_W8A8_FORMAT: str = "TN"
    VLLM_ILU_USE_INT8_MLA: bool = False
    VLLM_ILU_MLA_CUSTOMIZE: bool = True
    VLLM_ILU_MIX_QUANTIZATION_TYPE: str = ""
    VLLM_ILU_ATTN_OPT_LEVEL: int = 0
    VLLM_ILU_ATTN_STATIC_QUANT_SCALE_FILE_PATH: str = ""
    VLLM_ILU_USE_LORA_FUSION: bool = False
    VLLM_ILU_VIT_ATTN_OPT_LEVEL: int = 0
    VLLM_ILU_VIT_ATTN_LAYER: str = ""
    VLLM_ILU_GDN_ATTN_BACKEND: bool = False
    VLLM_ILU_DS4_PREFILL_OPT: bool = False
    VLLM_ILU_NIXL_WRITE_USE_HOST_BUFFER: bool = False
    VLLM_ILU_NIXL_HOST_DTYPE: str = "auto"
    VLLM_ILU_NIXL_PROXY_ADDR: str = ""
    VLLM_ILU_NIXL_SCHEDULER_ADDR: str = ""
    VLLM_ILU_NIXL_FORCE_PER_RANK_SEND: bool = False
    VLLM_ILU_NIXL_SERVER_ADDR: str = ""
    VLLM_ILU_NIXL_DS32_QUANT_AFTER_TRANSFER: bool = False
    VLLM_ILU_LINEAR_BACKEND: str = ""

# Canonical (VLLM_ILU_*) readers. Each lambda preserves the exact parsing the
# fork used in vllm/envs.py, and reads VLLM_ILU_* first then the legacy VLLM_*.
_ILU_ENV_VARS: dict[str, Callable[[], Any]] = {
    "VLLM_ILU_W8A8_MOE_USE_W4A8": lambda: (
        _get("VLLM_ILU_W8A8_MOE_USE_W4A8", "VLLM_W8A8_MOE_USE_W4A8", "0").lower()
        in ("1", "true")
    ),
    "VLLM_ILU_WNA16_MOE_USE_W4A8": lambda: (
        _get("VLLM_ILU_WNA16_MOE_USE_W4A8", "VLLM_WNA16_MOE_USE_W4A8", "0").lower()
        in ("1", "true")
    ),
    "VLLM_ILU_W4A8_FORMAT": lambda: _get(
        "VLLM_ILU_W4A8_FORMAT", "VLLM_W4A8_FORMAT", "TN"
    ).upper(),
    "VLLM_ILU_W4A8_VERSION": lambda: int(
        _get("VLLM_ILU_W4A8_VERSION", "VLLM_W4A8_VERSION", "2")
    ),
    "VLLM_ILU_W8A8_FORMAT": lambda: _get(
        "VLLM_ILU_W8A8_FORMAT", "VLLM_W8A8_FORMAT", "TN"
    ).upper(),
    "VLLM_ILU_USE_INT8_MLA": lambda: (
        _get("VLLM_ILU_USE_INT8_MLA", "VLLM_USE_INT8_MLA", "0").lower()
        in ("1", "true", "yes")
    ),
    "VLLM_ILU_MLA_CUSTOMIZE": lambda: (
        _get("VLLM_ILU_MLA_CUSTOMIZE", "VLLM_MLA_CUSTOMIZE", "1").lower()
        in ("1", "true", "yes")
    ),
    "VLLM_ILU_MIX_QUANTIZATION_TYPE": lambda: _get(
        "VLLM_ILU_MIX_QUANTIZATION_TYPE", "VLLM_MIX_QUANTIZATION_TYPE", ""
    ).upper(),
    "VLLM_ILU_ATTN_OPT_LEVEL": lambda: int(
        _get("VLLM_ILU_ATTN_OPT_LEVEL", "VLLM_ATTN_OPT_LEVEL", "0")
    ),
    "VLLM_ILU_ATTN_STATIC_QUANT_SCALE_FILE_PATH": lambda: _get(
        "VLLM_ILU_ATTN_STATIC_QUANT_SCALE_FILE_PATH",
        "VLLM_ATTN_STATIC_QUANT_SCALE_FILE_PATH",
        "",
    ),
    "VLLM_ILU_USE_LORA_FUSION": lambda: (
        _get("VLLM_ILU_USE_LORA_FUSION", "VLLM_USE_LORA_FUSION", "0").lower()
        in ("1", "true")
    ),
    # ViT attention quantization level
    "VLLM_ILU_VIT_ATTN_OPT_LEVEL": lambda: int(
        _get("VLLM_ILU_ATTN_OPT_LEVEL", "VLLM_VIT_ATTN_OPT_LEVEL", "0")
    ),
    # Comma-separated ViT block indices on which VLLM_VIT_ATTN_OPT_LEVEL
    # takes effect (e.g. "0,1,2,5,10").
    "VLLM_ILU_VIT_ATTN_LAYER": lambda: _get(
        "VLLM_ILU_VIT_ATTN_LAYER", "VLLM_VIT_ATTN_LAYER", ""
    ).upper(),
    # Gate the Iluvatar GDN attention backend. Community GDN remains the default
    # unless the deployment explicitly opts into the plugin metadata/kernel path.
    "VLLM_ILU_GDN_ATTN_BACKEND": lambda: (
        _get("VLLM_ILU_GDN_ATTN_BACKEND", "VLLM_GDN_ATTN_BACKEND", "0").lower()
        in _TRUE_VALS
    ),
    "VLLM_ILU_DS4_PREFILL_OPT": lambda: (
        _get("VLLM_ILU_DS4_PREFILL_OPT", "VLLM_DS4_PREFILL_OPT", "0").lower()
        in ("1", "true")
    ),
}


_ILU_CANONICAL_ONLY_ENV_VARS: dict[str, Callable[[], Any]] = {
    "VLLM_ILU_NIXL_WRITE_USE_HOST_BUFFER": _bool_reader(
        "VLLM_ILU_NIXL_WRITE_USE_HOST_BUFFER", "0"
    ),
    "VLLM_ILU_NIXL_HOST_DTYPE": lambda: os.environ.get(
        "VLLM_ILU_NIXL_HOST_DTYPE", "auto"
    ).lower(),
    "VLLM_ILU_NIXL_PROXY_ADDR": lambda: os.environ.get("VLLM_ILU_NIXL_PROXY_ADDR", ""),
    "VLLM_ILU_NIXL_SCHEDULER_ADDR": lambda: os.environ.get(
        "VLLM_ILU_NIXL_SCHEDULER_ADDR", ""
    ),
    "VLLM_ILU_NIXL_FORCE_PER_RANK_SEND": _bool_reader(
        "VLLM_ILU_NIXL_FORCE_PER_RANK_SEND", "0"
    ),
    "VLLM_ILU_NIXL_SERVER_ADDR": lambda: os.environ.get(
        "VLLM_ILU_NIXL_SERVER_ADDR", ""
    ),
    "VLLM_ILU_NIXL_DS32_QUANT_AFTER_TRANSFER": _bool_reader(
        "VLLM_ILU_NIXL_DS32_QUANT_AFTER_TRANSFER", "0"
    ),
    "VLLM_ILU_LINEAR_BACKEND": lambda: os.environ.get("VLLM_ILU_LINEAR_BACKEND", "")
    .strip()
    .lower()
    .replace("-", "_"),
}


def _canonical_to_legacy(name: str) -> str:
    """VLLM_ILU_FOO -> VLLM_FOO."""
    return "VLLM_" + name[len("VLLM_ILU_") :]


# Legacy VLLM_* aliases -> same reader as the canonical name. Keeps existing
# `envs.VLLM_X` call sites working.
_ILU_ENV_ALIASES: dict[str, Callable[[], Any]] = {
    _canonical_to_legacy(canonical): reader
    for canonical, reader in _ILU_ENV_VARS.items()
}


# Iluvatar plugin control vars that are read directly via os.environ where they
# are used (NOT through this module) — registered here only so that
# vllm.envs.validate_environ() recognises them and does not warn (or, with
# fail_on_environ_validation, hard-fail) about "Unknown vLLM environment
# variable". Sources:
#   VLLM_ILU_PATCH_CHECK_SIGNATURE / _CHECK_FINGERPRINT / _VERBOSE
#       -> vllm_iluvatar/ilu_patch/general_patch.py
#   VLLM_ILUVATAR_ENABLE_STARTUP_PATCH
#       -> vllm_iluvatar/__init__.py (STARTUP_PATCH_ENV)
_ILU_GUARD_ENV_VARS: dict[str, Callable[[], Any]] = {
    "VLLM_ILU_PATCH_CHECK_SIGNATURE": _bool_reader(
        "VLLM_ILU_PATCH_CHECK_SIGNATURE", "1"
    ),
    "VLLM_ILU_PATCH_CHECK_FINGERPRINT": _bool_reader(
        "VLLM_ILU_PATCH_CHECK_FINGERPRINT", "1"
    ),
    "VLLM_ILU_PATCH_VERBOSE": _bool_reader("VLLM_ILU_PATCH_VERBOSE", "0"),
    "VLLM_ILUVATAR_ENABLE_STARTUP_PATCH": _bool_reader(
        "VLLM_ILUVATAR_ENABLE_STARTUP_PATCH", "0"
    ),
}


def apply() -> None:
    """Inject Iluvatar env defaults and register env readers."""
    for key, value in _ILU_DEFAULTS.items():
        os.environ.setdefault(key, value)
        logger.debug("ILU envs: setdefault %s=%s", key, value)

    import vllm.envs as envs

    envs.environment_variables.update(_ILU_ENV_VARS)
    envs.environment_variables.update(_ILU_ENV_ALIASES)
    envs.environment_variables.update(_ILU_CANONICAL_ONLY_ENV_VARS)
    envs.environment_variables.update(_ILU_GUARD_ENV_VARS)
    logger.debug(
        "ILU envs: registered %d canonical + %d legacy-alias + "
        "%d canonical-only + %d guard env vars",
        len(_ILU_ENV_VARS),
        len(_ILU_ENV_ALIASES),
        len(_ILU_CANONICAL_ONLY_ENV_VARS),
        len(_ILU_GUARD_ENV_VARS),
    )
