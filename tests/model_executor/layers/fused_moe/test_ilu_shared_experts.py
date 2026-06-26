# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for IluSharedExperts.

The only override is ``_disable_shared_experts_overlap`` returning ``False``
unconditionally. We verify this via mock to avoid constructing a full MoE
layer stack (which would require GPU + weight tensors).
"""

from __future__ import annotations

from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)

from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_shared_experts import (
    IluSharedExperts,
)


def test_disable_shared_experts_overlap_always_false():
    """IluSharedExperts._disable_shared_experts_overlap must always
    return False, regardless of the underlying moe_config state."""
    # Bypass __init__ to avoid needing real FusedMoEConfig / vllm_config
    obj = IluSharedExperts.__new__(IluSharedExperts)
    assert obj._disable_shared_experts_overlap is False


def test_community_shared_experts_may_return_true():
    """Sanity check: community SharedExperts can return True under certain
    config combinations (eplb + non-default backend), confirming the
    Iluvatar override is meaningful."""
    from unittest.mock import MagicMock

    obj = SharedExperts.__new__(SharedExperts)
    mock_config = MagicMock()
    mock_config.moe_parallel_config.enable_eplb = True
    mock_config.moe_parallel_config.all2all_backend = "deepep_low_latency"
    mock_config.moe_parallel_config.use_fi_nvl_two_sided_kernels = False
    obj._moe_config = mock_config

    assert obj._disable_shared_experts_overlap is True, (
        "Community SharedExperts should disable overlap with eplb + non-default backend"
    )


def test_ilu_override_ignores_config():
    """Even with a config that would make community return True, the
    Iluvatar override must still return False."""
    from unittest.mock import MagicMock

    obj = IluSharedExperts.__new__(IluSharedExperts)
    mock_config = MagicMock()
    mock_config.moe_parallel_config.enable_eplb = True
    mock_config.moe_parallel_config.all2all_backend = "deepep_low_latency"
    mock_config.moe_parallel_config.use_fi_nvl_two_sided_kernels = True
    obj._moe_config = mock_config

    assert obj._disable_shared_experts_overlap is False, (
        "IluSharedExperts must always allow overlap"
    )
