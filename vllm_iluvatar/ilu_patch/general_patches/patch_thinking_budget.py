# SPDX-License-Identifier: Apache-2.0
"""Wire the Iluvatar thinking-budget kernels into Model Runner V2.

The override keeps the upstream committed marker cache and replaces draft and
forced-suffix matching with bounded masked scans. It also avoids chained
boolean expressions rejected by the Iluvatar Triton frontend. General plugin
registration applies the override before the sampler handles its first request.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the Iluvatar thinking-budget marker-cache kernel replacement."""
    p.register_once(
        __name__,
        _apply,
        detail="thinking_budget kernels with bounded draft-prefix marker scans",
    )


def _apply() -> None:
    from vllm.v1.worker.gpu.sample import thinking_budget as community

    from vllm_iluvatar.v1.worker.gpu.thinking_budget import apply_thinking_budget

    # ThinkingBudgetState resolves this module global for each application.
    community.apply_thinking_budget = apply_thinking_budget
