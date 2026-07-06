# SPDX-License-Identifier: Apache-2.0
"""Wiring for Iluvatar layer utility overrides.

Corresponds to: vllm/model_executor/layers/utils.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementation lives in
``vllm_iluvatar.model_executor.layers.utils``; this module only performs
register-time wiring.

Target   : vllm.model_executor.layers.utils.default_unquantized_gemm
Baseline : main_4875b4456b5
Why      : unquantized GEMM on Iluvatar should use ixformer kernels for FP16/BF16
           inputs, while preserving the community torch path for FP32.
How      : rebind the module-level helper with
           ``general_patch.rebind_module_func`` so callers that imported the
           function by name also see the override.
Scope    : unquantized LinearBase forward path on Iluvatar.
Remove   : when upstream exposes a platform-specific unquantized GEMM hook.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p

_OVERRIDES = ["default_unquantized_gemm"]


def register() -> None:
    """Register Iluvatar layer utility overrides."""
    p.register_once(
        __name__,
        _apply,
        detail="layer utils default_unquantized_gemm override",
    )


def _apply() -> None:
    import vllm.model_executor.layers.utils as _m

    import vllm_iluvatar.model_executor.layers.utils as impl

    p.check_signatures(_m, impl, _OVERRIDES, strict=True)
    for name in _OVERRIDES:
        # [ILU override] default_unquantized_gemm - delta: dispatch BF16/FP16
        # unquantized linear to ixformer kernels on Iluvatar.
        p.rebind_module_func(_m, name, getattr(impl, name))
