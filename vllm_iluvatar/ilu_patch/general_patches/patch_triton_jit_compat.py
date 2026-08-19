# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim for Triton's jit decorator.

Corresponds to:
    vllm/models/minimax_m3/common/ops/{index_topk,sparse_attn}.py

Target   : ``triton.jit`` as imported through ``vllm.triton_utils``.
Baseline : main_6e448d0
Why      : community MiniMax-M3 kernels pass
           ``do_not_specialize_on_alignment`` to ``triton.jit``. Older Triton
           builds used in the Iluvatar runtime reject that keyword at import
           time before any model patch can run.
How      : wrap only Triton ``jit`` callables whose signature lacks the keyword,
           dropping that unsupported kwarg while preserving all other arguments.
Scope    : process-wide Triton jit decorator compatibility.
Remove   : when the Iluvatar runtime ships a Triton version supporting
           ``do_not_specialize_on_alignment``.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable

from vllm_iluvatar.ilu_patch import general_patch as p

_ALIGNMENT_KWARG = "do_not_specialize_on_alignment"
_ILU_WRAPPED_ATTR = "_ilu_drops_do_not_specialize_on_alignment"


def _jit_accepts_alignment_kwarg(jit: Callable[..., object]) -> bool:
    try:
        return _ALIGNMENT_KWARG in inspect.signature(jit).parameters
    except (TypeError, ValueError):
        # If the callable cannot be inspected, leave it untouched rather than
        # hiding an unknown incompatibility.
        return True


def _wrap_jit_if_needed(jit: Callable[..., object]) -> Callable[..., object]:
    if getattr(jit, _ILU_WRAPPED_ATTR, False):
        return jit
    if _jit_accepts_alignment_kwarg(jit):
        return jit

    @functools.wraps(jit)
    def wrapped_jit(*args, **kwargs):
        kwargs.pop(_ALIGNMENT_KWARG, None)
        return jit(*args, **kwargs)

    setattr(wrapped_jit, _ILU_WRAPPED_ATTR, True)
    return wrapped_jit


def _patch_triton_module(triton_module: object) -> bool:
    jit = getattr(triton_module, "jit", None)
    if not callable(jit):
        return False

    patched = _wrap_jit_if_needed(jit)
    if patched is jit:
        return False

    setattr(triton_module, "jit", patched)
    return True


def _apply() -> None:
    patched = False

    try:
        from vllm.triton_utils import triton as vllm_triton
    except ImportError:
        vllm_triton = None
    if vllm_triton is not None:
        patched = _patch_triton_module(vllm_triton) or patched

    try:
        import triton as triton_module
    except ImportError:
        triton_module = None
    if triton_module is not None:
        patched = _patch_triton_module(triton_module) or patched

    if not patched:
        return


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="drop unsupported Triton jit do_not_specialize_on_alignment kwarg",
    )
