# SPDX-License-Identifier: Apache-2.0
"""
Iluvatar-specific environment variable definitions and community-default
overrides.  Implementation layer — no side effects on import.

Corresponds to: vllm/envs.py
Applied by:      vllm_iluvatar.register_platform()

Naming policy (dual-name for backward compatibility):
  - Canonical name: VLLM_ILU_<suffix> (Iluvatar namespace, kept apart from
    real community vars).
  - Legacy alias:   VLLM_<suffix> (the original, non-standard fork name).
  - Both `envs.VLLM_ILU_X` and `envs.VLLM_X` resolve to the same value, and
    both OS env keys are honoured (VLLM_ILU_X preferred, then VLLM_X, then
    default).  Existing call sites (envs.VLLM_X) and launch scripts keep
    working unchanged; new code should migrate to VLLM_ILU_X.

Contents:
  1. _ILU_DEFAULTS    — community vars whose default differs for Iluvatar
                        (NOT renamed; they are real community vars).
  2. _ILU_ENV_VARS    — new Iluvatar-only env vars, keyed by canonical name.
  3. _ILU_ENV_ALIASES — legacy VLLM_* names mapped to the same readers.
  4. apply()          — inject defaults + register canonical/alias readers.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

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
    VLLM_ILU_ENFORCE_CUDA_GRAPH: bool = False
    VLLM_ILU_MOE_OPT_LEVEL: int = 0
    VLLM_ILU_LINEAR_OPT_LEVEL: int = 0
    VLLM_ILU_OPT_EXCLUDE_LAYERS: str = ""
    VLLM_ILU_LINEAR_SPECIFIED_LAYERS: str = ""
    VLLM_ILU_LINEAR_SPECIFIED_KEYS: str = ""
    VLLM_ILU_LINEAR_SPECIFIED_OPT_LEVEL: int = 0
    VLLM_ILU_ATTN_OPT_LEVEL: int = 0
    VLLM_ILU_ATTN_STATIC_QUANT_SCALE_FILE_PATH: str = ""
    VLLM_ILU_USE_LORA_FUSION: bool = False
    VLLM_ILU_VIT_ATTN_OPT_LEVEL: int = 0
    VLLM_ILU_VIT_ATTN_LAYER: str = ""
    VLLM_ILU_DS4_PREFILL_OPT: bool = False


# Canonical (VLLM_ILU_*) readers. Each lambda preserves the exact parsing the
# fork used in vllm/envs.py, and reads VLLM_ILU_* first then the legacy VLLM_*.
_ILU_ENV_VARS: dict[str, Callable[[], Any]] = {
    "VLLM_ILU_W8A8_MOE_USE_W4A8": lambda: _get(
        "VLLM_ILU_W8A8_MOE_USE_W4A8", "VLLM_W8A8_MOE_USE_W4A8", "0"
    ).lower()
    in ("1", "true"),
    "VLLM_ILU_WNA16_MOE_USE_W4A8": lambda: _get(
        "VLLM_ILU_WNA16_MOE_USE_W4A8", "VLLM_WNA16_MOE_USE_W4A8", "0"
    ).lower()
    in ("1", "true"),
    "VLLM_ILU_W4A8_FORMAT": lambda: _get(
        "VLLM_ILU_W4A8_FORMAT", "VLLM_W4A8_FORMAT", "TN"
    ).upper(),
    "VLLM_ILU_W4A8_VERSION": lambda: int(
        _get("VLLM_ILU_W4A8_VERSION", "VLLM_W4A8_VERSION", "2")
    ),
    "VLLM_ILU_W8A8_FORMAT": lambda: _get(
        "VLLM_ILU_W8A8_FORMAT", "VLLM_W8A8_FORMAT", "TN"
    ).upper(),
    "VLLM_ILU_USE_INT8_MLA": lambda: _get(
        "VLLM_ILU_USE_INT8_MLA", "VLLM_USE_INT8_MLA", "0"
    ).lower()
    in ("1", "true", "yes"),
    "VLLM_ILU_MLA_CUSTOMIZE": lambda: _get(
        "VLLM_ILU_MLA_CUSTOMIZE", "VLLM_MLA_CUSTOMIZE", "1"
    ).lower()
    in ("1", "true", "yes"),
    "VLLM_ILU_MIX_QUANTIZATION_TYPE": lambda: _get(
        "VLLM_ILU_MIX_QUANTIZATION_TYPE", "VLLM_MIX_QUANTIZATION_TYPE", ""
    ).upper(),
    "VLLM_ILU_ENFORCE_CUDA_GRAPH": lambda: bool(
        int(_get("VLLM_ILU_ENFORCE_CUDA_GRAPH", "VLLM_ENFORCE_CUDA_GRAPH", "0"))
    ),
    "VLLM_ILU_MOE_OPT_LEVEL": lambda: int(
        _get("VLLM_ILU_MOE_OPT_LEVEL", "VLLM_MOE_OPT_LEVEL", "0")
    ),
    "VLLM_ILU_LINEAR_OPT_LEVEL": lambda: int(
        _get("VLLM_ILU_LINEAR_OPT_LEVEL", "VLLM_LINEAR_OPT_LEVEL", "0")
    ),
    "VLLM_ILU_OPT_EXCLUDE_LAYERS": lambda: _get(
        "VLLM_ILU_OPT_EXCLUDE_LAYERS", "VLLM_OPT_EXCLUDE_LAYERS", ""
    ).upper(),
    "VLLM_ILU_LINEAR_SPECIFIED_LAYERS": lambda: _get(
        "VLLM_ILU_LINEAR_SPECIFIED_LAYERS", "VLLM_LINEAR_SPECIFIED_LAYERS", ""
    ).upper(),
    "VLLM_ILU_LINEAR_SPECIFIED_KEYS": lambda: _get(
        "VLLM_ILU_LINEAR_SPECIFIED_KEYS", "VLLM_LINEAR_SPECIFIED_KEYS", ""
    ).lower(),
    "VLLM_ILU_LINEAR_SPECIFIED_OPT_LEVEL": lambda: int(
        _get(
            "VLLM_ILU_LINEAR_SPECIFIED_OPT_LEVEL",
            "VLLM_LINEAR_SPECIFIED_OPT_LEVEL",
            "0",
        )
    ),
    "VLLM_ILU_ATTN_OPT_LEVEL": lambda: int(
        _get("VLLM_ILU_ATTN_OPT_LEVEL", "VLLM_ATTN_OPT_LEVEL", "0")
    ),
    "VLLM_ILU_ATTN_STATIC_QUANT_SCALE_FILE_PATH": lambda: _get(
        "VLLM_ILU_ATTN_STATIC_QUANT_SCALE_FILE_PATH",
        "VLLM_ATTN_STATIC_QUANT_SCALE_FILE_PATH",
        "",
    ),
    "VLLM_ILU_USE_LORA_FUSION": lambda: _get(
        "VLLM_ILU_USE_LORA_FUSION", "VLLM_USE_LORA_FUSION", "0"
    ).lower()
    in ("1", "true"),
    # ViT attention quantization level
    "VLLM_ILU_VIT_ATTN_OPT_LEVEL": lambda: int(
        _get("VLLM_ILU_ATTN_OPT_LEVEL", "VLLM_VIT_ATTN_OPT_LEVEL", "0")
    ),
    # Comma-separated ViT block indices on which VLLM_VIT_ATTN_OPT_LEVEL
    # takes effect (e.g. "0,1,2,5,10").
    "VLLM_ILU_VIT_ATTN_LAYER": lambda: _get(
        "VLLM_ILU_VIT_ATTN_LAYER", "VLLM_VIT_ATTN_LAYER", ""
    ).upper(),
    "VLLM_ILU_DS4_PREFILL_OPT": lambda: _get(
        "VLLM_ILU_DS4_PREFILL_OPT", "VLLM_DS4_PREFILL_OPT", "0"
    ).lower()
    in ("1", "true"),
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


def apply() -> None:
    """Inject Iluvatar env defaults and register canonical + legacy readers."""
    for key, value in _ILU_DEFAULTS.items():
        os.environ.setdefault(key, value)
        logger.debug("ILU envs: setdefault %s=%s", key, value)

    import vllm.envs as envs

    envs.environment_variables.update(_ILU_ENV_VARS)
    envs.environment_variables.update(_ILU_ENV_ALIASES)
    logger.debug(
        "ILU envs: registered %d canonical + %d legacy-alias env vars",
        len(_ILU_ENV_VARS),
        len(_ILU_ENV_ALIASES),
    )
