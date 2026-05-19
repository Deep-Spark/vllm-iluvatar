"""Plugin-wide bootstrap helpers.

The single public symbol is :func:`adapt_patch`. It orchestrates every
runtime monkey-patch and ``sys.modules`` redirect required by the
DeepSeek-V4 W4A8 path. All patches are idempotent so the function can be
called multiple times (vLLM is allowed to load plugins more than once per
process tree per the ``load_general_plugins`` docstring).
"""

from __future__ import annotations

_PATCHED = False


def adapt_patch() -> None:
    """Apply every plugin patch exactly once.

    Order is important:

    * ``sys.modules`` redirects happen FIRST so any subsequent ``import``
      of one of the redirected module paths gets the replacement.
    * Attribute-level patches (``setattr`` / class method swaps) come
      AFTER, because they target modules that may already be imported.
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # 1. Full-file sys.modules redirects (must run before downstream import).
    from .patch.patch_sys_modules import install_module_redirects

    install_module_redirects()

    # 2. Attribute-level patches on already-imported modules.
    from .patch.patch_custom_ops import apply as _patch_custom_ops
    from .patch.patch_kv_cache_utils import apply as _patch_kv_cache_utils
    from .patch.patch_model_config import apply as _patch_model_config

    _patch_custom_ops()
    _patch_kv_cache_utils()
    _patch_model_config()
