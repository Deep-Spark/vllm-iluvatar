# SPDX-License-Identifier: Apache-2.0
"""Wiring for the fa_utils ILU overrides (auto-discovered and applied via
``register()`` from ``vllm_iluvatar.ilu_patch.apply_general_patches`` during
``register_general``).

The override implementations live in the sibling ``fa_utils.py``; this module
only does the register-time wiring: a strict signature guard plus a propagating
rebind of the two helpers onto the community module.

Target   : vllm.v1.attention.backends.fa_utils.{get_flash_attn_version,
           flash_attn_supports_sinks}
Baseline : main_e0871ad2259
How      : full replacement (see fa_utils.py); installed with
           ``general_patch.rebind_module_func`` so already-imported
           ``from fa_utils import <name>`` references in community attention
           modules (flash_attn.py, MLA backends, ...) are rebound too — not just
           the ``fa_utils`` module attribute. Strict signature validation
           (prefix-compatible) is kept; there is no full-method copy, so no
           source-drift fingerprint guard is needed.
Scope    : all TP/PP runs on Iluvatar.
Remove   : together with fa_utils.py (see its ``Remove``).
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p

_OVERRIDES = ["get_flash_attn_version", "flash_attn_supports_sinks"]


def register() -> None:
    """Register the fa_utils ILU overrides.

    Delegates to ``general_patch.register_once``; auto-discovered and invoked by
    ``apply_general_patches()``.
    """
    p.register_once(
        __name__,
        _apply,
        detail="fa_utils flash-attn capability overrides (propagating rebind)",
    )


def _apply() -> None:
    import vllm.v1.attention.backends.fa_utils as _m

    import vllm_iluvatar.v1.attention.backends.fa_utils as impl

    # Strict (prefix-compatible) signature validation
    # (VLLM_ILU_PATCH_CHECK_SIGNATURE).
    p.check_signatures(_m, impl, _OVERRIDES, strict=True)

    # Propagating rebind: also fixes community modules that did
    # ``from fa_utils import <name>`` before this patch ran.
    for name in _OVERRIDES:
        p.rebind_module_func(_m, name, getattr(impl, name))
