# SPDX-License-Identifier: Apache-2.0
"""Wiring for the create_tcp_store ILU override (auto-discovered and applied via
``register()`` from ``vllm_iluvatar.ilu_patch.apply_general_patches`` during
``register_general``).

The override implementation lives in the sibling ``utils.py``; this module only
does the register-time wiring.

Target   : vllm.distributed.utils.create_tcp_store
Baseline : main_e0871ad2259
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the create_tcp_store wrapper.

    Delegates to ``general_patch.register_once``; auto-discovered and invoked by
    ``apply_general_patches()``.
    """
    p.register_once(__name__, _apply, detail="create_tcp_store wrapper")


def _apply() -> None:
    import vllm.distributed.utils as _m

    import vllm_iluvatar.distributed.utils as impl

    _m.create_tcp_store = impl._make_create_tcp_store(_m.create_tcp_store)
