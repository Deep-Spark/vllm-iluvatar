# SPDX-License-Identifier: Apache-2.0
"""Iluvatar override for the DeepSeek-V4 Triton top-k fast path.

Corresponds to: vllm/model_executor/layers/fused_moe/router/dsv4_topk.py

Target   : vllm.model_executor.layers.fused_moe.router.dsv4_topk.can_use_dsv4_topk
Baseline : main_9c12a81ac
Why      : ``fused_topk_bias`` diverts ``sqrtsoftplus`` routing into the Triton
           kernel ``dsv4_topk`` whenever ``can_use_dsv4_topk`` says yes.  On
           Iluvatar all eight conditions hold for DeepSeek-V4's 40 bias layers,
           which would run a CoreX-Triton kernel instead of the ixformer
           ``topk_hash_softplus_sqrt`` op and bypass the plugin's
           ``_custom_ops`` override entirely.
How      : delegation wrapper built by ``make_can_use_dsv4_topk`` and installed
           by ``ilu_patch.general_patches.patch_dsv4_topk`` -- always return
           ``False``.  No community body is copied, so no source-fingerprint
           guard is needed.
Scope    : every ``sqrtsoftplus`` MoE router on Iluvatar (DeepSeek-V4 FP8 and
           W4A8) on any baseline that ships ``router/dsv4_topk.py``.
Remove   : when the ixformer top-k op is no longer preferred over the Triton
           kernel, or upstream gains a platform hook for this fast path.
"""

from __future__ import annotations

import functools
from collections.abc import Callable


def make_can_use_dsv4_topk(
    orig: Callable[..., bool],
) -> Callable[..., bool]:
    """Wrap community ``can_use_dsv4_topk`` so Iluvatar always answers ``False``.

    ``functools.wraps`` keeps the community name / signature / ``__wrapped__``
    visible after the rebind, which is what lets ``check_signatures`` and any
    debug output still line up.

    The body reads no arguments, so it takes ``*args, **kwargs`` rather than
    restating the community parameter list: upstream adding a parameter would
    otherwise turn every call into a ``TypeError`` at runtime.
    """

    @functools.wraps(orig)
    def can_use_dsv4_topk(*args: object, **kwargs: object) -> bool:
        # [ILU override] can_use_dsv4_topk — delta: close the Triton fast path
        # so ``fused_topk_bias`` falls through to ``vllm_topk_softplus_sqrt``
        # -> ``ops.topk_hash_softplus_sqrt``, which the plugin's
        # ``_custom_ops`` splat points at ixformer.
        return False

    return can_use_dsv4_topk
