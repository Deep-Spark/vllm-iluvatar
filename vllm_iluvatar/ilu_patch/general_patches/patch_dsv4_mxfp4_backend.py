# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar MXFP4 MoE backend-selection override (patch P1).

Corresponds to: vllm/model_executor/layers/fused_moe/oracle/mxfp4.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementation lives in
``vllm_iluvatar.model_executor.layers.fused_moe.oracle.ilu_mxfp4``; this module
only performs register-time wiring.

Target   : vllm.model_executor.layers.fused_moe.oracle.mxfp4.select_deepseek_v4_mxfp4_moe_backend
Baseline : main_9c12a81ac
Why      : community selection never yields ``Mxfp4MoeBackend.NONE`` off-ROCm
           (it raises ``NotImplementedError("No MXFP4 MoE backend supports the
           deployment configuration.")`` instead), while DeepSeek-V4 FP8 Stage 1
           needs NONE so ``Mxfp4MoEMethod.process_weights_after_loading``
           early-returns and the routed MXFP4 params stay byte-identical to the
           checkpoint.
How      : delegation wrapper (no method-body copy, so no fingerprint guard) via
           ``make_select_deepseek_v4_mxfp4_moe_backend``, installed with
           ``rebind_module_func`` on the defining module. The rebind is required
           because ``quantization/mxfp4.py:25`` is a ``from ... import
           select_deepseek_v4_mxfp4_moe_backend`` consumer and ``:484``
           (``Mxfp4MoEMethod.__init__``) calls that copied reference — a plain
           ``setattr`` on the oracle module would be a no-op for the only call
           site that matters.
Scope    : all routed-MXFP4 FusedMoE construction on Iluvatar (DeepSeek-V4 FP8
           checkpoint loading).
Remove   : when an Iluvatar MXFP4 MoE kernel lands and a real backend can be
           selected (Stage 2+).

Timing   : ``Mxfp4MoEMethod.__init__`` calls the target during ``FusedMoE``
           construction, so the patch must be in place before the first MoE
           layer is built. ``apply_general_patches()`` runs from
           ``register_general`` (the ``vllm.general_plugins`` entry point, loaded
           inside ``create_engine_config``), i.e. well before the model is
           instantiated by the model loader. A patch applied later would leave
           the already-built layers holding the community function, which off
           ROCm raises ``NotImplementedError``. Unit tests must likewise call
           ``register()`` before constructing any MoE layer.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p

_SYMBOL = "select_deepseek_v4_mxfp4_moe_backend"


def register() -> None:
    """Register the Iluvatar MXFP4 backend-selection wrapper.

    Delegates to ``register_once`` (idempotent + announcement) and is called
    automatically by ``apply_general_patches()``; the patch-specific work lives
    in ``_apply()``.
    """
    p.register_once(
        __name__,
        _apply,
        detail=f"{_SYMBOL} -> Mxfp4MoeBackend.NONE on Iluvatar",
    )


def _apply() -> None:
    import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as _m

    from vllm_iluvatar.model_executor.layers.fused_moe.oracle.ilu_mxfp4 import (  # noqa: E501
        make_select_deepseek_v4_mxfp4_moe_backend,
    )

    orig = getattr(_m, _SYMBOL)

    # [ILU override] select_deepseek_v4_mxfp4_moe_backend — delta: wrapper
    # returning (Mxfp4MoeBackend.NONE, None) on Iluvatar, delegating elsewhere;
    # rebind_module_func propagates to the quantization/mxfp4.py from-import
    # consumer that Mxfp4MoEMethod.__init__ actually calls.
    p.rebind_module_func(_m, _SYMBOL, make_select_deepseek_v4_mxfp4_moe_backend(orig))
