# SPDX-License-Identifier: Apache-2.0
"""
import_hook — post-import splat hooks via sys.meta_path.

Runs an ILU "splat" callback at the EXACT moment a target community module
finishes importing.  This is the right tool for community modules whose ILU
overrides are needed at *import time* (module-level calls) — e.g.
``vllm._custom_ops`` functions invoked while ``w8a8_utils`` is being imported
inside a spawned worker subprocess.

Why not the general_plugins phase?
    general_plugins only run after an explicit ``load_general_plugins()`` call,
    which happens late in engine init — AFTER the spawn-time import chain has
    already executed the community ``torch.ops._C`` call and crashed on corex.

Why not bootstrap-direct-import (``import vllm._custom_ops`` in bootstrap)?
    That would force ``current_platform`` resolution + ixformer + platform
    construction at startup time (even in processes that never use the op), and
    risks half-initialised / circular imports.

Mechanism:
    1. ``register(target_fullname, callback)`` records the callback and installs
       a single finder at the front of ``sys.meta_path`` (no module is imported
       here — registration is cheap).
    2. When anything imports ``target_fullname``, the finder steps aside to let
       the real finders build the spec, then wraps ``spec.loader.exec_module``.
    3. After the community module body has executed (functions defined) and
       before the importer uses it, the callback fires with the loaded module.
    4. Fallback: if the target is already in ``sys.modules`` at registration
       time, the callback fires immediately.

Callbacks receive the loaded module and MUST be idempotent (splat via
``setattr``); each target fires its callbacks exactly once per process.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import logging
import sys
from collections.abc import Callable

logger = logging.getLogger(__name__)

_Callback = Callable[[object], None]

_registry: dict[str, list[_Callback]] = {}
_fired: set[str] = set()


def _fire(fullname: str, module: object) -> None:
    """Run (once) all callbacks registered for ``fullname``."""
    if fullname in _fired:
        return
    _fired.add(fullname)
    for cb in _registry.get(fullname, []):
        try:
            cb(module)
        except Exception:
            logger.exception("ILU post-import hook for %s failed", fullname)
            raise


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Fires registered callbacks right after a target module is executed."""

    # Names currently being resolved by this finder, so the nested
    # ``find_spec`` delegation does not recurse into us again.
    _resolving: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _registry or fullname in self._resolving:
            return None

        # Step aside and let the real finders resolve the actual spec/loader.
        self._resolving.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._resolving.discard(fullname)

        if spec is None or spec.loader is None:
            return None

        real_exec = spec.loader.exec_module

        def exec_module(module, _real_exec=real_exec, _name=fullname):
            _real_exec(module)
            _fire(_name, module)

        spec.loader.exec_module = exec_module  # type: ignore[assignment]
        return spec


_finder = _PostImportFinder()


def _ensure_finder_installed() -> None:
    if _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)


def register(target_fullname: str, callback: _Callback) -> None:
    """Run ``callback(module)`` right after ``target_fullname`` is imported.

    If ``target_fullname`` is already imported (loaded before this hook was
    installed), the callback fires immediately as a fallback.
    """
    _registry.setdefault(target_fullname, []).append(callback)
    _ensure_finder_installed()

    existing = sys.modules.get(target_fullname)
    if existing is not None:
        # Target already loaded — exec_module won't fire again, so run now.
        _fire(target_fullname, existing)
