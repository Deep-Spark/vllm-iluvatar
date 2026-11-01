# SPDX-License-Identifier: Apache-2.0
"""Use SDK partial-hit/COW support for compact-indexer MLA producers.

Corresponds to: vllm/v1/core/sched/scheduler.py
Target: Scheduler.__init__, after all cache groups have been resolved
Why: a smaller indexer row expands its uniform physical page's token coverage;
     using that page as the local hit unit adds unnecessary producer forward rows.
Scope: compatible DeepSeek V4.1 split-indexer producers, no speculative model.
Remove when the community coordinator exposes an append-only MLA policy.
"""

from __future__ import annotations

from functools import wraps

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(__name__, _apply, detail="compact-indexer producer subpage hits")


def _apply() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_iluvatar.models.deepseek_v41.indexer_prefix_cache import (
        configure_indexer_prefix_hits,
    )

    original = Scheduler.__init__
    if getattr(original, "_ilu_indexer_prefix_hits", False):
        return

    @wraps(original)
    def configured(self, *args, **kwargs):
        original(self, *args, **kwargs)
        configure_indexer_prefix_hits(self)

    configured._ilu_indexer_prefix_hits = True
    Scheduler.__init__ = configured
