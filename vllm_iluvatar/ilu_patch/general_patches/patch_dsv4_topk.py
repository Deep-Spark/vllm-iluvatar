# SPDX-License-Identifier: Apache-2.0
"""Wiring for closing the DeepSeek-V4 Triton top-k fast path on Iluvatar.

Corresponds to: vllm/model_executor/layers/fused_moe/router/dsv4_topk.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementation lives in
``vllm_iluvatar.model_executor.layers.fused_moe.router.ilu_dsv4_topk``; this
module only performs register-time wiring.

Target   : vllm.model_executor.layers.fused_moe.router.dsv4_topk.can_use_dsv4_topk
Baseline : main_9c12a81ac
Why      : with ``sqrtsoftplus`` routing, ``fused_topk_bias``
           (``fused_topk_bias_router.py:190-203``) hands the 40 bias layers of
           DeepSeek-V4 to the Triton kernel ``dsv4_topk`` -- all eight gating
           conditions hold on Iluvatar. That bypasses the ixformer
           ``topk_hash_softplus_sqrt`` op the plugin routes every other MoE
           router to (via the ``vllm._custom_ops`` import-hook splat), so router
           behaviour would silently diverge between the FP8 and W4A8 DeepSeek-V4
           forks. See the implementation module for the full rationale,
           including why ``launch_pdl`` is *not* the reason.
How      : delegation wrapper (no method-body copy, so no fingerprint guard) via
           ``make_can_use_dsv4_topk``, installed with ``rebind_module_func`` on
           the **defining** module ``router/dsv4_topk.py``.

           The rebind -- rather than a plain ``setattr`` -- is what makes it
           work: ``fused_topk_bias_router.py:17-20`` is a
           ``from ...dsv4_topk import can_use_dsv4_topk, dsv4_topk`` consumer, so
           the only call site that matters holds its own copy of the symbol.
           ``rebind_module_func`` writes the defining module first and then
           replaces every identity-matching copy across ``sys.modules``, which
           covers that consumer and any future one.
Scope    : every ``sqrtsoftplus`` MoE router on Iluvatar. Not FP8-specific --
           the W4A8 DeepSeek-V4 fork hits the same branch on any baseline that
           ships ``router/dsv4_topk.py`` (it appeared between the 07-06 and
           07-22 vLLM snapshots), so this patch is a shared prerequisite rather
           than part of the FP8 MoE work alone.
Remove   : when the ixformer top-k op stops being the preferred implementation,
           or upstream gains a platform hook for this fast path.

Timing   : the target is called per forward, so any point before the first MoE
           forward is early enough; ``apply_general_patches()`` runs from
           ``register_general`` (inside ``create_engine_config``), well before
           that. Unit tests must call ``register()`` before exercising the
           router.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p

_SYMBOL = "can_use_dsv4_topk"


def register() -> None:
    """Register the Iluvatar ``can_use_dsv4_topk`` wrapper.

    Delegates to ``register_once`` (idempotent + announcement) and is called
    automatically by ``apply_general_patches()``; the patch-specific work lives
    in ``_apply()``.
    """
    p.register_once(
        __name__,
        _apply,
        detail=f"{_SYMBOL} -> False on Iluvatar (Triton dsv4_topk disabled)",
    )


def _apply() -> None:
    import vllm.model_executor.layers.fused_moe.router.dsv4_topk as _m

    # Import the from-import consumer so it is in ``sys.modules`` when
    # ``rebind_module_func`` walks it; otherwise a later
    # ``import fused_topk_bias_router`` would grab the community symbol again.
    import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router  # noqa: F401,E501

    from vllm_iluvatar.model_executor.layers.fused_moe.router.ilu_dsv4_topk import (  # noqa: E501
        make_can_use_dsv4_topk,
    )

    orig = getattr(_m, _SYMBOL)

    # [ILU override] can_use_dsv4_topk — delta: wrapper returning False on
    # Iluvatar so fused_topk_bias falls through to the ixformer top-k op;
    # rebind_module_func propagates to the fused_topk_bias_router from-import
    # copy that is the only call site.
    p.rebind_module_func(_m, _SYMBOL, make_can_use_dsv4_topk(orig))
