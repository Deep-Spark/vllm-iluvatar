# SPDX-License-Identifier: Apache-2.0
"""Iluvatar override for the DeepSeek-V4 Triton top-k fast path.

Corresponds to: vllm/model_executor/layers/fused_moe/router/dsv4_topk.py

Target   : vllm.model_executor.layers.fused_moe.router.dsv4_topk.can_use_dsv4_topk
Baseline : main_9c12a81ac
Why      : ``fused_topk_bias`` (``fused_topk_bias_router.py:190-203``) diverts
           ``sqrtsoftplus`` routing into the Triton kernel ``dsv4_topk`` whenever
           ``can_use_dsv4_topk`` says yes, and on Iluvatar every one of its eight
           conditions holds for the 40 bias layers of DeepSeek-V4:
           ``is_cuda()`` is true (``IluPlatform`` subclasses ``CudaPlatform``),
           logits are fp32 ``[T, 256]`` contiguous, ``topk == 6``,
           ``renormalize`` is on and the indices dtype is int32. Taking that
           branch would run a CoreX-Triton kernel instead of the ixformer
           ``topk_hash_softplus_sqrt`` op the rest of the plugin routes to, and
           would bypass the ILU ``vllm._custom_ops`` override entirely -- i.e.
           the router would silently stop being the one W4A8 and FP8 share.
How      : delegation wrapper built by ``make_can_use_dsv4_topk`` and installed
           by ``ilu_patch.general_patches.patch_dsv4_topk`` -- return ``False``
           on Iluvatar without calling the community predicate, delegate
           untouched everywhere else. No community body is copied, so no
           source-fingerprint guard is needed.
Scope    : every ``sqrtsoftplus`` MoE router on Iluvatar, i.e. DeepSeek-V4 (both
           the FP8 and the W4A8 fork) on any baseline that ships
           ``router/dsv4_topk.py``.
Remove   : when the ixformer top-k op is no longer preferred over the Triton
           kernel, or upstream gains a platform hook for this fast path.

Correctness note (independent of which kernel is faster)
--------------------------------------------------------
``dsv4_topk`` does not take ``hash_indices_table``: a hash-routed layer that also
carried ``e_score_correction_bias`` would silently get top-k selection instead of
the table lookup -- wrong output, no error. DeepSeek-V4 happens to be safe
because the two are mutually exclusive (hash layers 0/1/2 get ``tid2eid``, the
other 40 get the bias), but closing the branch removes the trap.

On ``launch_pdl``
----------------
An earlier revision of this override justified itself with the
``KeyError: 'Keyword argument launch_pdl was specified but unrecognised'`` that
CoreX Triton raises. That reasoning does **not** apply here and is not why this
patch exists: ``_dsv4_topk_kernel`` declares ``launch_pdl`` as a
``tl.constexpr`` *kernel parameter* (``dsv4_topk.py:49``), so ``:119`` binds a
formal argument rather than passing a Triton runtime kwarg. The recorded CoreX
failure came from ``_save_partial_states_kernel``
(``models/deepseek_v4/common/ops/save_partial_states.py:49``), which does not
declare it. Whether the Triton path would additionally crash on CoreX is
untested; the reason to close the branch is kernel convergence, above.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

# Imported as a module global on purpose: the wrapper below resolves it at call
# time, so tests can monkeypatch ``ilu_dsv4_topk.is_iluvatar_platform`` on a host
# that is not an Iluvatar device (mirrors
# ``distributed/parallel_state.py::_is_ivcore30_fn``).
from vllm_iluvatar.ilu_util.arch import is_iluvatar_platform

if TYPE_CHECKING:
    import torch


def make_can_use_dsv4_topk(
    orig: Callable[..., bool],
) -> Callable[..., bool]:
    """Wrap community ``can_use_dsv4_topk`` so Iluvatar always answers ``False``.

    ``functools.wraps`` keeps the community name / signature / ``__wrapped__``
    visible after the rebind, which is what lets ``check_signatures`` and any
    debug output still line up.
    """

    @functools.wraps(orig)
    def can_use_dsv4_topk(
        gating_output: torch.Tensor,
        correction_bias: torch.Tensor | None,
        topk: int,
        renormalize: bool,
        indices_dtype: torch.dtype,
    ) -> bool:
        # [ILU override] can_use_dsv4_topk — delta: close the Triton fast path
        # on Iluvatar so ``fused_topk_bias`` falls through to
        # ``vllm_topk_softplus_sqrt`` -> ``ops.topk_hash_softplus_sqrt``, which
        # the plugin's ``_custom_ops`` splat points at ixformer. Other platforms
        # keep community behaviour untouched.
        if is_iluvatar_platform():
            return False
        return orig(
            gating_output,
            correction_bias,
            topk,
            renormalize,
            indices_dtype,
        )

    return can_use_dsv4_topk
