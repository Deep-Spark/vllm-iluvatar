# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar V2 speculative rejection kernel replacement.

Corresponds to: vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py

Target   : vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils._rejection_kernel
Baseline : v0.30.0
Why      : ilu Triton rejects ``_rejection_kernel`` two ways -- a three-term
           ``and`` chain, and a constexpr-off branch it compiles anyway,
           dereferencing the ``None`` pointers ``rejection_sample`` passes
           there. Community has no torch path for this, and the kernel is
           reached by any V2 speculative run -- ``method="dspark"`` forces that
           runner, whose ``warmup_kernels`` JIT-compiles it.
How      : rebind the module-level kernel. ``rejection_sample`` resolves it as a
           module global at call time.
Scope    : V2 speculative rejection sampling on Iluvatar.
Remove   : when ilu Triton accepts chained boolean operators and short-circuits
           constexpr conditions.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
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

    # [ILU override] _rejection_kernel — delta: split ``and`` chain, typed
    # pointers for the branch ilu compiles but never takes.
    community._rejection_kernel = _rejection_kernel
