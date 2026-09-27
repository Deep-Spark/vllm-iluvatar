# SPDX-License-Identifier: Apache-2.0
"""Iluvatar SharedExperts override.

Baseline : community ``runner/shared_experts.py`` @ v0.30.1rc0
Delta    : override ``_disable_shared_experts_overlap`` to always allow
           multi-stream overlap (controlled by VLLM_DISABLE_SHARED_EXPERTS_STREAM
           env var instead of the community eplb / flashinfer / shard-SP checks).

The 0.30.1 aux-stream rework (``maybe_forward_async`` + ``wait`` replacing
``maybe_sync_shared_experts_stream`` + ``_run_in_aux_stream``) is inherited
unchanged; only the overlap-eligibility predicate differs on Iluvatar.
"""

from __future__ import annotations

from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)


class IluSharedExperts(SharedExperts):
    """Iluvatar override — always allow shared-experts overlap."""

    # [ILU override] _disable_shared_experts_overlap — delta: community checks
    # eplb backend, flashinfer and shard-sequence-parallel conditions to
    # disable overlap; Iluvatar controls this solely via
    # VLLM_DISABLE_SHARED_EXPERTS_STREAM env var, which already forces
    # ``_stream = None`` and therefore NO_OVERLAP.
    @property
    def _disable_shared_experts_overlap(self) -> bool:
        return False
