# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar sampler-penalties kernel replacement.

Corresponds to: vllm/v1/worker/gpu/sample/penalties.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementation lives in
``vllm_iluvatar.v1.worker.gpu.penalties``; this module only performs
register-time wiring.

Target   : vllm.v1.worker.gpu.sample.penalties.apply_penalties
Baseline : v0.27.1
Why      : ilu's Triton frontend rejects the chained ``or`` in
           ``_penalties_kernel`` ("chained boolean operators (A or B or C) are
           not supported; use parentheses to split the chain"), so the kernel
           fails to compile. It is reached only via the V2 GPU model runner,
           which ``method="dspark"`` forces: that runner's ``warmup_kernels``
           JIT-compiles every sampler kernel with
           ``SamplingParams.for_sampler_warmup()``, which enables penalties.
           Without this patch a DSpark server dies during warm-up.
How      : replace the module-level ``apply_penalties`` with one that launches a
           copy of the kernel whose only change is the parenthesized chain.
           ``PenaltiesState.apply_penalties`` resolves the module global at call
           time, so rebinding the module attribute is enough; the kernel itself
           has to be copied because ``@triton.jit`` compiles from the decorated
           function's own source text.
Scope    : sampler penalties on Iluvatar. Numerics and launch geometry are
           unchanged, so this also fixes penalties for real requests, not just
           warm-up.
Remove   : when the ilu Triton frontend accepts chained boolean operators, or
           when community parenthesizes the chain itself.

Timing   : ``load_general_plugins()`` runs in ``create_engine_config``, long
           before the worker warm-up that first compiles the kernel.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the Iluvatar sampler-penalties kernel replacement."""
    p.register_once(
        __name__,
        _apply,
        detail="sampler penalties kernel without the chained-or ilu Triton rejects",
    )


def _apply() -> None:
    from vllm.v1.worker.gpu.sample import penalties as community

    from vllm_iluvatar.v1.worker.gpu.penalties import apply_penalties

    # [ILU override] apply_penalties — delta: launches a kernel copy whose
    # ``use_penalty`` chain is parenthesized.
    community.apply_penalties = apply_penalties
