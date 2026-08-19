# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/vllm_flash_attn/__init__.py

Module stand-in injected as ``sys.modules["vllm.vllm_flash_attn"]`` at startup
(see ``vllm_iluvatar.ilu_patch.import_shims.install_all``).

Target  : sys.modules["vllm.vllm_flash_attn"]
Baseline: main_6e448d0
Why     : IluPlatform is cuda-like (``is_cuda()`` is True), so community
          ``vllm.v1.attention.backends.fa_utils`` takes its ``is_cuda()`` branch
          and runs a module-level
          ``from vllm.vllm_flash_attn import
          compile_flash_attn_varlen_func_from_specs, flash_attn_varlen_func,
          get_scheduler_metadata``. On corex that package ``__init__`` raises
          ``ImportError`` (no ``_vllm_fa2_C`` / ``_vllm_fa3_C``). This is an
          import-time binding, so a general patch (``register_general``) is too
          late; we stand in an ixformer-backed module BEFORE any deep
          ``import vllm``.
How     : expose ixformer ``flash_attn_varlen_func``, a compile-only no-op
          stub (ixformer has no FA4 CuTeDSL precompile entry), and a
          ``get_scheduler_metadata`` stub (returns None). ``reshape_and_cache_flash``
          comes from ``vllm._custom_ops`` (covered by the custom-ops import hook),
          so it is not provided here.
Scope   : Iluvatar (cuda-like) runs that import community fa_utils.
Remove  : when the plugin ships its own attention backend that no longer relies
          on the community fa_utils cuda branch (converge to a self-owned FA
          stack).
"""

from __future__ import annotations

import logging
import types

logger = logging.getLogger(__name__)

SHIM_KEY = "vllm.vllm_flash_attn"


def _missing_flash_attn_varlen_func(*_args, **_kwargs):
    raise RuntimeError(
        "ixformer flash_attn is unavailable; the vllm.vllm_flash_attn shim "
        "cannot service flash_attn_varlen_func on this host."
    )


def _get_scheduler_metadata_stub(*_args, **_kwargs):
    # ixformer flash_attn does not use upstream FA3 scheduler metadata.
    return None


def _compile_flash_attn_varlen_func_from_specs_stub(*_args, **_kwargs):
    # ixformer flash_attn has no FA4 CuTeDSL compile-only warmup entry. Returning
    # None matches the compile API's side-effect-only contract while keeping the
    # community fa_utils CUDA import path satisfied.
    return None


def build_module() -> types.ModuleType:
    """Build the ixformer-backed stand-in for ``vllm.vllm_flash_attn``."""
    module = types.ModuleType(SHIM_KEY)

    try:
        from ixformer.contrib.vllm_flash_attn import flash_attn_varlen_func
    except Exception:  # pragma: no cover - dev/CI hosts without ixformer
        logger.debug(
            "ixformer flash_attn unavailable; vllm.vllm_flash_attn shim uses a "
            "lazy-raising flash_attn_varlen_func placeholder."
        )
        flash_attn_varlen_func = _missing_flash_attn_varlen_func

    module.flash_attn_varlen_func = flash_attn_varlen_func
    module.compile_flash_attn_varlen_func_from_specs = (
        _compile_flash_attn_varlen_func_from_specs_stub
    )
    module.get_scheduler_metadata = _get_scheduler_metadata_stub
    module.__all__ = [
        "compile_flash_attn_varlen_func_from_specs",
        "flash_attn_varlen_func",
        "get_scheduler_metadata",
    ]
    return module
