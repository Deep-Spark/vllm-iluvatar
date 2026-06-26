# SPDX-License-Identifier: Apache-2.0
"""Iluvatar patch layer.

Two kinds of patches live here:

  1. Early import-compat shims + import-hook registration
     (``patch_import_compat.py`` / ``patch_custom_ops.py``).  These are applied
     at startup via ``vllm_iluvatar._load_import_compat_patch()`` (main process
     + spawned workers), BEFORE deep ``import vllm`` — they are NOT imported by
     this package ``__init__`` to keep startup cheap.  Required for community
     symbols called at *import time* (e.g. ``vllm._custom_ops``).

  2. Late-phase (general_plugins) patches that only need an initialised vLLM
     (module/class objects present).  These live as ``*_patch.py`` modules
     exposing ``register()`` in the mirror tree (see ``_PATCH_PACKAGES``); the
     override implementation itself stays in the upstream-named sibling module
     (e.g. ``parallel_state.py`` for ``parallel_state_patch.py``).  They are
     auto-discovered + applied by ``apply_general_patches()`` from
     ``register_general`` (the ``vllm.general_plugins`` entry point).

To add a late-phase patch: put the override implementation in the
upstream-named module and drop a sibling ``<name>_patch.py`` exposing
``def register() -> None`` at the top level of a package listed in
``_PATCH_PACKAGES`` (default ``vllm_iluvatar.distributed``).  Modules without
``register()`` (the impl siblings) are scanned but skipped — no edit here.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)

# Packages whose top-level modules are scanned for ``register()`` callables.
# A module here that defines ``register()`` is treated as a general-plugins
# patch and applied during ``register_general``.
_PATCH_PACKAGES = [
    "vllm_iluvatar.distributed",
    "vllm_iluvatar.model_executor.layers.fused_moe",
]

_applied = False


def apply_general_patches() -> None:
    """Auto-discover and apply all general-plugins patches (idempotent).

    Scans each package in ``_PATCH_PACKAGES`` (non-recursively, so sibling
    sub-packages such as ``device_communicators`` are skipped) and calls
    ``register()`` on every top-level module that defines it.
    """
    global _applied
    if _applied:
        return
    _applied = True

    for pkg_name in _PATCH_PACKAGES:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception:
            logger.exception("ILU: failed to import patch package %s", pkg_name)
            continue
        for info in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
            if info.ispkg:
                continue
            module = importlib.import_module(info.name)
            register = getattr(module, "register", None)
            if callable(register):
                register()
