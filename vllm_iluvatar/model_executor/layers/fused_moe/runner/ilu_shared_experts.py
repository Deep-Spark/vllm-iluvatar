# SPDX-License-Identifier: Apache-2.0
"""Iluvatar SharedExperts override.

Baseline : community ``runner/shared_experts.py`` @ main_6e448d0
Delta    : override ``_disable_shared_experts_overlap`` to always allow
           multi-stream overlap (controlled by VLLM_DISABLE_SHARED_EXPERTS_STREAM
           env var instead of the community eplb / flashinfer checks).
"""

from __future__ import annotations

from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)


class IluSharedExperts(SharedExperts):
    """Iluvatar override — always allow shared-experts overlap."""

    # [ILU override] _disable_shared_experts_overlap — delta: community checks
    # eplb backend + flashinfer conditions to disable overlap; Iluvatar
    # controls this solely via VLLM_DISABLE_SHARED_EXPERTS_STREAM env var.
    @property
    def _disable_shared_experts_overlap(self) -> bool:
        return False
