# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar thinking-budget marker-cache kernel replacement.

Corresponds to: vllm/v1/worker/gpu/sample/thinking_budget.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementation lives in
``vllm_iluvatar.v1.worker.gpu.thinking_budget``; this module only performs
register-time wiring.

Target   : vllm.v1.worker.gpu.sample.thinking_budget.apply_thinking_budget
Baseline : bytedance_release_ds41:20261101v14rc3 / community V2 runner
Why      : ilu's Triton frontend rejects the chained ``and`` in
           ``_update_committed_marker_cache_kernel`` ("chained boolean
           operators (A or B or C) are not supported; use parentheses to
           split the chain"), so the kernel fails to compile. Reached when a
           request sets ``thinking_token_budget`` on the V2 GPU model runner
           (dspark path).
How      : replace the module-level ``apply_thinking_budget`` with one that
           launches a copy of the kernels whose only change is the
           parenthesized cold-scan ``if`` / ``while`` chains.
           ``ThinkingBudgetState.apply`` resolves the module global at call
           time, so rebinding the module attribute is enough; the kernels
           themselves have to be copied because ``@triton.jit`` compiles from
           the decorated function's own source text.
Scope    : thinking_token_budget on Iluvatar. Numerics and launch geometry
           are unchanged.
Remove   : when the ilu Triton frontend accepts chained boolean operators, or
           when community parenthesizes the chains itself.

Timing   : ``load_general_plugins()`` runs in ``create_engine_config``, long
           before a request with ``thinking_token_budget`` first compiles the
           marker-cache kernel.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the Iluvatar thinking-budget marker-cache kernel replacement."""
    p.register_once(
        __name__,
        _apply,
        detail="thinking_budget marker-cache kernel without chained-and ilu Triton rejects",
    )


def _apply() -> None:
    from vllm.v1.worker.gpu.sample import thinking_budget as community

    from vllm_iluvatar.v1.worker.gpu.thinking_budget import apply_thinking_budget

    # [ILU override] apply_thinking_budget — delta: launches kernel copies whose
    # cold-scan ``and`` chains are parenthesized.
    community.apply_thinking_budget = apply_thinking_budget
