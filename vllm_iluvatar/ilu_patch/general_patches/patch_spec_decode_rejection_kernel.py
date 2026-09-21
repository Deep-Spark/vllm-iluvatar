# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar V2 speculative rejection kernel replacement.

Corresponds to: vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementation lives in
``vllm_iluvatar.v1.worker.gpu.spec_decode.rejection_sampler_utils``; this module
only performs register-time wiring.

Target   : vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils._rejection_kernel
Baseline : main_6e448d0
Why      : ilu Triton rejects ``_rejection_kernel`` three ways -- a ``tl.rand``
           draw that keeps a dim and so widens the loop-carried ``accepted`` to
           ``<[1], int1>``, a three-term ``and`` chain, and a constexpr-off
           branch it compiles anyway, dereferencing the ``None`` pointers
           ``rejection_sample`` passes there. Community has no torch path for
           this, and the kernel is reached by any V2 speculative run --
           ``method="dspark"`` forces that runner, whose ``warmup_kernels``
           JIT-compiles it, so a DSpark server dies during warm-up.
How      : rebind the module-level kernel. ``rejection_sample`` resolves it as a
           module global at call time, so nothing else needs patching; the
           kernel itself has to be copied because ``@triton.jit`` compiles from
           the decorated function's own source text.
Scope    : V2 speculative rejection sampling on Iluvatar. Numerics and launch
           geometry are unchanged -- the draw is summed over one element, the
           split chain is the same predicate, and the aliased pointers sit in a
           branch that is never taken -- so this fixes verification for real
           requests too, not just warm-up.
Remove   : when ilu Triton scalarises ``tl.rand`` on loaded offsets, accepts
           chained boolean operators and short-circuits constexpr conditions.

Timing   : ``load_general_plugins()`` runs in ``create_engine_config``, long
           before the worker warm-up that first compiles the kernel.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the Iluvatar V2 speculative rejection kernel replacement."""
    p.register_once(
        __name__,
        _apply,
        detail="V2 spec-decode rejection kernel ilu Triton can compile",
    )


def _apply() -> None:
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as community

    from vllm_iluvatar.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        _rejection_kernel,
    )

    # [ILU override] _rejection_kernel — delta: scalar ``u``, split ``and``
    # chain, typed pointers for the branch ilu compiles but never takes.
    community._rejection_kernel = _rejection_kernel
