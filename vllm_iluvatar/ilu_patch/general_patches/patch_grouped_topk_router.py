# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar grouped-topk router override.

Corresponds to: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

Target   : vllm.model_executor.layers.fused_moe.router.grouped_topk_router.grouped_topk
Baseline : main_6e448d0
Why      : community vLLM has no backend hook for the grouped-topk routing
           function; Iluvatar needs the ixformer grouped-topk kernel.
How      : strict signature check, then ``rebind_module_func`` on the
           module-level function so new ``GroupedTopk`` instances capture the
           Iluvatar implementation in ``native_impl``.
Scope    : grouped-topk MoE routing on Iluvatar.
Remove   : when upstream exposes a grouped-topk backend hook or native
           Iluvatar support.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p

_OVERRIDES = ["grouped_topk"]


def register() -> None:
    """Register the Iluvatar grouped-topk router override."""
    p.register_once(
        __name__,
        _apply,
        detail="grouped_topk router ixformer implementation",
    )


def _apply() -> None:
    import vllm.model_executor.layers.fused_moe.router.grouped_topk_router as _m

    import vllm_iluvatar.model_executor.layers.fused_moe.router.ilu_grouped_topk_router as impl

    p.check_signatures(_m, impl, _OVERRIDES, strict=True)

    # [ILU override] grouped_topk - delta: module-global rebind to ixformer
    # implementation; GroupedTopk.__init__ reads this global into native_impl.
    p.rebind_module_func(_m, "grouped_topk", impl.grouped_topk)
