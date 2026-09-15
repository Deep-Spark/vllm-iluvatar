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
     (module/class objects present).  Wiring modules live in
     ``ilu_patch.general_patches`` and expose ``register()``; override
     implementations stay in upstream-mirrored packages such as
     ``vllm_iluvatar.distributed``. They are auto-discovered + applied by
     ``apply_general_patches()`` from ``register_general`` (the
     ``vllm.general_plugins`` entry point).

To add a late-phase patch: put the override implementation in the
upstream-mirrored module, then add one small wiring module under
``ilu_patch.general_patches`` exposing ``def register() -> None``. The wiring
module should perform lazy imports, guards, and rebinding only.
"""

from __future__ import annotations

import importlib
import pkgutil

from vllm.logger import init_logger

logger = init_logger(__name__)

# Packages whose top-level modules are scanned for ``register()`` callables.
# Centralising wiring here keeps upstream-mirrored implementation directories
# from accumulating ``*_patch.py`` files as the plugin grows.
_PATCH_PACKAGES = [
    "vllm_iluvatar.ilu_patch.general_patches",
]

_applied = False


def apply_general_patches() -> None:
    """Auto-discover and apply all general-plugins patches (idempotent).

    Scans each package in ``_PATCH_PACKAGES`` non-recursively and calls
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
