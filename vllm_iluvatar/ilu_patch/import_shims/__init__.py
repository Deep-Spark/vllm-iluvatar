# SPDX-License-Identifier: Apache-2.0
"""Startup-phase *module stand-in* shims for Iluvatar.

Unlike the small attribute-level compat shims inlined in
``patch_import_compat.py`` (``torch.accelerator`` mapping, ``_symmetric_memory``
stub, ...), each module here provides a COMPLETE stand-in for a community module
that would otherwise be imported at *import time* and fail / pull NV-specific
code on corex.

Each ``*.py`` module in this package:

  * mirrors the upstream module name (e.g. ``vllm_flash_attn.py`` <->
    ``vllm/vllm_flash_attn/``) so it is easy to compare against community;
  * exposes ``SHIM_KEY`` (the ``sys.modules`` key it stands in for) and
    ``build_module() -> ModuleType``.

``install_all()`` is called early by ``patch_import_compat.apply()`` (before any
deep ``import vllm``). It injects each stand-in into ``sys.modules`` and, best
effort, binds it as an attribute on the already-imported parent package, so the
community import-time code transparently picks up the Iluvatar implementation.

To add a new stand-in: drop a mirror-named module here exposing ``SHIM_KEY`` +
``build_module()``; no edit to this orchestration is needed.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import sys

logger = logging.getLogger(__name__)


def install_all() -> None:
    """Discover sibling shim modules and inject them into ``sys.modules``.

    Idempotent: a shim whose ``SHIM_KEY`` is already present in ``sys.modules``
    is skipped. Each shim is installed defensively so one failure never blocks
    the others (or interpreter / vLLM startup).
    """
    for info in pkgutil.iter_modules(__path__, __name__ + "."):
        if info.ispkg:
            continue
        try:
            module = importlib.import_module(info.name)
            key = getattr(module, "SHIM_KEY", None)
            build = getattr(module, "build_module", None)
            if not key or not callable(build):
                continue
            if key in sys.modules:
                continue
            shim = build()
            sys.modules[key] = shim
            parent, _, child = key.rpartition(".")
            if parent and parent in sys.modules:
                setattr(sys.modules[parent], child, shim)
            logger.debug("ILU import shim installed: %s", key)
        except Exception:
            logger.exception("ILU: failed to install import shim %s", info.name)
