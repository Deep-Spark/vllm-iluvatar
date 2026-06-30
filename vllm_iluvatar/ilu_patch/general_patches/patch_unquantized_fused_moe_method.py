# SPDX-License-Identifier: Apache-2.0
"""Wiring for the unquantized fused-MoE ILUVATAR_W16A16 methods.

Corresponds to:
    vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The implementation lives in
``vllm_iluvatar.model_executor.layers.fused_moe.ilu_unquantized_fused_moe_method``;
this module only validates signatures and installs the Iluvatar methods onto
the community class.

Target   : vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method.
           UnquantizedFusedMoEMethod.apply / forward_native
Baseline : main_e0871ad2259
Why      : community vLLM does not know the fork-only ILUVATAR_W16A16
           ``moe_kernel.ilu_apply`` path.
How      : class-level rebind of two methods. This plugin runs on Iluvatar and
           supports only the ILUVATAR_W16A16 backend here.
Scope    : unquantized MoE model runs on Iluvatar.
Remove   : when upstream provides an official vendor hook for unquantized MoE
           expert backends.
"""

from __future__ import annotations

import importlib

from vllm_iluvatar.ilu_patch import general_patch as p

_TARGET_MODULE = "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
_IMPL_MODULE = (
    "vllm_iluvatar.model_executor.layers.fused_moe."
    "ilu_unquantized_fused_moe_method"
)
_OVERRIDES = ["apply", "forward_native"]


def register() -> None:
    """Register ILUVATAR_W16A16 unquantized MoE methods."""
    p.register_once(
        __name__,
        _apply,
        detail="UnquantizedFusedMoEMethod ILUVATAR_W16A16 methods",
    )


def _apply() -> None:
    community = importlib.import_module(_TARGET_MODULE)
    impl = importlib.import_module(_IMPL_MODULE)
    cls = community.UnquantizedFusedMoEMethod

    p.check_signatures(cls, impl.IluUnquantizedFusedMoEMethod, _OVERRIDES, strict=True)

    cls.apply = impl.IluUnquantizedFusedMoEMethod.apply
    cls.forward_native = impl.IluUnquantizedFusedMoEMethod.forward_native
