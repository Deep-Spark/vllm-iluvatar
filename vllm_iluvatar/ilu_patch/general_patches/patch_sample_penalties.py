# SPDX-License-Identifier: Apache-2.0
"""Wiring for the GPU sampler penalty-kernel ILU override.

Corresponds to: vllm/v1/worker/gpu/sample/penalties.py

Target   : vllm.v1.worker.gpu.sample.penalties.apply_penalties
Baseline : v0.30.0 (ced6857afa)
Why      : the community ``_penalties_kernel`` uses a three-way chained boolean
           (``A or B or C``) that the Iluvatar Triton build rejects at JIT
           compile time. The spec-decode warmup is the first path to compile the
           kernel, so it aborts engine startup with "chained boolean operators
           are not supported".
How      : rebind the module-level ``apply_penalties`` to the ilu mirror
           (``vllm_iluvatar.v1.worker.gpu.sample.penalties``), which launches a
           byte-for-byte copy of the kernel with the chain parenthesised into two
           binary ``or`` operators. ``PenaltiesState.apply_penalties`` calls the
           function through its module globals, so the rebind reaches it.
Scope    : Iluvatar GPU sampling penalty kernel (spec-decode and normal paths).
Remove   : when the Iluvatar Triton build accepts chained boolean operators.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p

_OVERRIDES = ["apply_penalties"]


def register() -> None:
    """Register the sampler penalty-kernel override."""
    p.register_once(
        __name__,
        _apply,
        detail="GPU sampler penalty kernel: split chained boolean for ilu Triton",
    )


def _apply() -> None:
    import vllm.v1.worker.gpu.sample.penalties as _m

    import vllm_iluvatar.v1.worker.gpu.sample.penalties as impl

    p.check_signatures(_m, impl, _OVERRIDES, strict=True)
    for name in _OVERRIDES:
        p.rebind_module_func(_m, name, getattr(impl, name))
