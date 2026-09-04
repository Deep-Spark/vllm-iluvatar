# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar MiniMax QK-Norm de-fusion.

Corresponds to: vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementation lives in
``vllm_iluvatar.model_executor.layers.minimax_rms_norm.rms_norm_tp``; this
module only performs register-time wiring.

Target   : vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp
           .MiniMaxText01RMSNormTP.forward_qkv
Baseline : main_4cd213ead42e
Why      : the community ``forward_qkv`` routes TP>1 through
           ``torch.ops.vllm.minimax_qk_norm_fusion`` (Lamport / ROCm AITER /
           Triton). On Iluvatar corex the Lamport op is unregistered, AITER is
           not ROCm, and Triton block kernels are risky, so the fusion tiers are
           either dead or unsafe.
How      : ``setattr(MiniMaxText01RMSNormTP, "forward_qkv",
           staticmethod(operator_backed))``. The method is called by class name
           (``MiniMaxText01RMSNormTP.forward_qkv(...)`` in ``models/minimax_m2``)
           and ``minimax_m2`` imports the same class object, so a class-attribute
           rebind reaches every call site -- no ``rebind_module_func`` needed.
           Runtime path: TP==1 -> ixformer ``rms_norm_qk``; TP>1 ->
           ``_C.minimax_qk_var`` + all-reduce + ``_C.minimax_qk_rms_apply``.
           ``forward_qkv_torch`` is retained as a pure-torch reference for tests.
Scope    : all MiniMax-M2 QK-Norm forwards on Iluvatar.
Remove   : when upstream gains a portable platform hook that covers this path.

Timing   : ``forward_qkv`` is called per attention forward, so any point before
           the first forward is early enough; ``apply_general_patches()`` runs
           from ``register_general`` (inside ``create_engine_config``), well
           before model execution. Unit tests must call ``register()`` first.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p

_METHOD = "forward_qkv"


def register() -> None:
    """Register the Iluvatar operator-backed ``forward_qkv`` override.

    Delegates to ``register_once`` (idempotent + announcement) and is called
    automatically by ``apply_general_patches()``; the patch-specific work lives
    in ``_apply()``.
    """
    p.register_once(
        __name__,
        _apply,
        detail=(
            "MiniMaxText01RMSNormTP.forward_qkv -> ixformer rms_norm_qk "
            "(TP=1) / minimax_qk_var+apply (TP>1)"
        ),
    )


def _apply() -> None:
    import vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp as _m

    from vllm_iluvatar.model_executor.layers.minimax_rms_norm import (
        rms_norm_tp as _ilu,
    )

    cls = _m.MiniMaxText01RMSNormTP

    # Guard the call interface (community params must all appear in the ILU
    # override) before rebinding, so drift surfaces at startup.
    p.check_signatures(cls, _ilu, [_METHOD])

    # [ILU override] MiniMaxText01RMSNormTP.forward_qkv — delta: replace the
    # fusion-op dispatch with ixformer (TP=1) / two-stage plugin ops (TP>1).
    # ``forward_qkv`` is a staticmethod called by class name, so a class-attribute
    # rebind is visible to every caller (idempotent: setattr is safe to repeat).
    setattr(cls, _METHOD, staticmethod(_ilu.forward_qkv))
