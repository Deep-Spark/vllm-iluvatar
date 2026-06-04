"""Import-time redirection of stock vLLM modules to iluvatar overrides.

This installs a single :class:`importlib.abc.MetaPathFinder` at the front of
``sys.meta_path``. The finder serves replacement / brand-new ``vllm.*``
modules from a **mirror tree** stored under
:mod:`vllm_iluvatar._vllm_overrides`.

Why a mirror tree (instead of a hand-maintained flat dict)
----------------------------------------------------------
The full iluvatar fork overlays ~60 modules onto community vLLM. A mirror
tree makes the ``vllm.a.b.c`` -> file mapping derivable from the path
(``_vllm_overrides/vllm/a/b/c.py``) and lets us add files without touching
this module. The override file keeps its original ``vllm....`` qualified
name (set as the spec ``name``), so any ``from vllm.x import y`` inside it
resolves through the normal import machinery -- we never rewrite imports.

Three cases the finder handles for ``fullname`` (only ``vllm.*`` is touched):

1. **Leaf override** -- ``_vllm_overrides/<path>.py`` exists -> serve it as a
   module. Works for both files that exist in community (full-file
   replacement) and brand-new files community never had.
2. **New package** -- ``_vllm_overrides/<path>/`` is a directory that does
   NOT exist in the installed ``vllm`` package -> serve it as a (namespace
   or regular) package so its children resolve.
3. **Existing-package __init__ override** -- ``_vllm_overrides/<path>/__init__.py``
   exists -> serve that ``__init__`` but point ``submodule_search_locations``
   at the *real* package dir so non-overridden siblings still load from
   community vLLM.

A bare container directory in the mirror tree (no ``__init__.py``) that also
exists in real vLLM is NOT shadowed: the finder returns ``None`` for the
package itself (real vLLM provides it) but still serves the deeper leaf via
its own fullname, because meta-path finders are consulted per fullname.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import logging
import os
import sys
from pathlib import Path

_PREFIX = "vllm."

logger = logging.getLogger("vllm_iluvatar")


def _overrides_root() -> Path:
    return Path(__file__).resolve().parent.parent / "_vllm_overrides"


def _real_vllm_root() -> Path | None:
    """Locate the installed ``vllm`` package dir WITHOUT importing vllm.

    Used only for Case-3 (existing-package ``__init__`` override) which always
    happens late; computing it via ``find_spec`` keeps the bootstrap path
    (interpreter startup, before ``import vllm``) free of a vllm import.
    """
    try:
        spec = importlib.util.find_spec("vllm")
    except Exception:
        return None
    if spec is None:
        return None
    locs = list(getattr(spec, "submodule_search_locations", None) or [])
    if locs:
        return Path(locs[0]).resolve()
    origin = getattr(spec, "origin", None)
    if origin:
        return Path(origin).resolve().parent
    return None


class _IluRedirectFinder(importlib.abc.MetaPathFinder):
    """Serve overridden / new ``vllm.*`` modules from the mirror tree."""

    def __init__(self) -> None:
        self._root = _overrides_root()
        self._vllm_root = self._root / "vllm"
        self._real_root_cached = False
        self._real_root: Path | None = None

    # -- helpers ---------------------------------------------------------
    def _rel(self, fullname: str) -> str:
        # "vllm.a.b" -> "a/b"
        return fullname[len(_PREFIX) :].replace(".", "/")

    @property
    def _real_root_lazy(self) -> Path | None:
        # Resolved lazily / cached: avoids importing or even locating vllm at
        # interpreter-startup bootstrap time (only needed for Case 3, late).
        if not self._real_root_cached:
            self._real_root = _real_vllm_root()
            self._real_root_cached = True
        return self._real_root

    def _real_dir_exists(self, rel: str) -> bool:
        root = self._real_root_lazy
        if root is None:
            return False
        return (root / rel).is_dir()

    # -- MetaPathFinder API ----------------------------------------------
    def find_spec(self, fullname, path=None, target=None):  # noqa: D401
        if not fullname.startswith(_PREFIX):
            return None

        rel = self._rel(fullname)
        py = self._vllm_root / (rel + ".py")
        pkg_dir = self._vllm_root / rel

        # Case 1: leaf module override.
        if py.is_file():
            return importlib.util.spec_from_file_location(fullname, os.fspath(py))

        if pkg_dir.is_dir():
            init = pkg_dir / "__init__.py"
            has_init = init.is_file()
            real_exists = self._real_dir_exists(rel)

            # Case 3: explicit __init__ override for an EXISTING real package.
            if has_init and real_exists:
                search = [os.fspath(pkg_dir), os.fspath(self._real_root_lazy / rel)]
                spec = importlib.util.spec_from_file_location(
                    fullname, os.fspath(init), submodule_search_locations=search
                )
                return spec

            # Case 2a: brand-new regular package (has its own __init__).
            if has_init and not real_exists:
                return importlib.util.spec_from_file_location(
                    fullname,
                    os.fspath(init),
                    submodule_search_locations=[os.fspath(pkg_dir)],
                )

            # Case 2b: brand-new namespace package (dir, no __init__).
            if not has_init and not real_exists:
                spec = importlib.machinery.ModuleSpec(
                    fullname, loader=None, is_package=True
                )
                spec.submodule_search_locations = [os.fspath(pkg_dir)]
                return spec

            # Bare container dir over an existing real package: don't shadow.
            return None

        return None


_INSTALLED = False


def _iter_override_module_names(root: Path):
    """Yield ``vllm....`` fullnames present as files in the mirror tree."""
    vroot = root / "vllm"
    if not vroot.is_dir():
        return
    for dirpath, _dirs, files in os.walk(vroot):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            full = Path(dirpath) / fn
            rel = full.relative_to(root).with_suffix("")
            parts = rel.parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            yield ".".join(parts)


# Distributed comm stack: process-global singleton state (the TP/PP groups)
# and custom-op impls whose module identity must stay consistent. The finder
# MAY serve their overlay -- but only safely when it is served as the *first
# and only* import (i.e. the .pth bootstrap installed the finder before
# ``import vllm``). If the finder is installed late (entry-point fallback) and
# the community copy is already imported, evicting + re-importing the overlay
# would re-register ops and split the global state, so we must NOT evict these.
_NO_EVICT = {
    "vllm.distributed.parallel_state",
    "vllm.distributed.device_communicators.cuda_communicator",
    "vllm.distributed.device_communicators.base_device_communicator",
    "vllm.distributed.device_communicators.all2all",
}


def is_installed() -> bool:
    return _INSTALLED


def install_module_redirects() -> None:
    """Idempotently install the redirect finder.

    Callable from BOTH the interpreter-startup ``.pth`` bootstrap (before
    ``import vllm``) and the late ``vllm.general_plugins`` entry point. It must
    therefore NOT import vllm.

    Eviction: we evict already-imported overridden modules so their next
    ``import`` yields our overlay. This is required for early-imported-but-safe
    modules such as ``vllm.utils.torch_utils`` (overlay routes
    ``weak_ref_tensor`` to ixformer instead of the non-existent
    ``torch.ops._C``). We DO NOT evict the distributed comm stack
    (``_NO_EVICT``): when the bootstrap ran first those modules are not yet
    imported and are served fresh as overlay (ixformer comm active); when only
    the late entry-point ran they are already community and must be kept to
    avoid op re-registration / split global state.

    Re-registration safety is additionally backstopped by
    :mod:`vllm_iluvatar.patch.patch_op_registration` and the overlay
    ``torch_utils`` guard.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    finder = _IluRedirectFinder()
    sys.meta_path.insert(0, finder)

    names = sorted(set(_iter_override_module_names(_overrides_root())))
    evicted = 0
    for modname in names:
        if modname in _NO_EVICT:
            continue
        if modname in sys.modules:
            del sys.modules[modname]
            evicted += 1

    logger.info(
        "vllm-iluvatar: installed override finder for %d modules "
        "(evicted %d already-imported)",
        len(names),
        evicted,
    )
