# SPDX-License-Identifier: Apache-2.0
"""Tests for ilu_elastic_ep helper functions and patch installers.

Most tests are CPU-only (pure Python / mock logic). The
``ilu_eplb_map_and_record`` numerical test uses CPU tensors (the function
is pure-torch and works on any device).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_elastic_ep import (
    ilu_eplb_map_and_record,
    ilu_is_monolithic,
    ilu_kernel_handles_own_dispatch,
    install_ilu_elastic_ep_patches,
)

# ---------------------------------------------------------------------------
# ilu_kernel_handles_own_dispatch
# ---------------------------------------------------------------------------


class TestKernelHandlesOwnDispatch:
    def test_non_modular_method_returns_false(self):
        """A quant_method that is NOT FusedMoEModularMethod -> False."""
        quant = MagicMock(spec=[])
        result = ilu_kernel_handles_own_dispatch(quant)
        assert result is False

    def test_modular_monolithic_returns_false(self):
        """A FusedMoEModularMethod wrapping a monolithic W16A16 -> False."""
        from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
            FusedMoEModularMethod,
        )

        quant = MagicMock(spec=FusedMoEModularMethod)
        quant.__class__ = FusedMoEModularMethod
        # Make ilu_is_monolithic return True
        with patch(
            "vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_elastic_ep"
            ".ilu_is_monolithic",
            return_value=True,
        ):
            result = ilu_kernel_handles_own_dispatch(quant)
        assert result is False

    def test_modular_non_monolithic_returns_true(self):
        """A non-monolithic FusedMoEModularMethod -> True (kernel handles dispatch)."""
        from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
            FusedMoEModularMethod,
        )

        quant = MagicMock(spec=FusedMoEModularMethod)
        quant.__class__ = FusedMoEModularMethod
        with patch(
            "vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_elastic_ep"
            ".ilu_is_monolithic",
            return_value=False,
        ):
            result = ilu_kernel_handles_own_dispatch(quant)
        assert result is True


# ---------------------------------------------------------------------------
# ilu_is_monolithic
# ---------------------------------------------------------------------------


class TestIluIsMonolithic:
    def test_detects_wrapped_unquantized_method(self):
        """FusedMoEModularMethod stores the original method on old_quant_method."""
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )

        class IluUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
            pass

        quant = MagicMock()
        quant._ilu_is_monolithic = None
        quant.old_quant_method = IluUnquantizedFusedMoEMethod.__new__(
            IluUnquantizedFusedMoEMethod
        )

        result = ilu_is_monolithic(quant)

        assert result is True
        assert quant._ilu_is_monolithic is True

    def test_caches_result(self):
        """ilu_is_monolithic must cache its result on the modular_method."""
        quant = MagicMock()
        quant._ilu_is_monolithic = None
        quant.moe_kernel.fused_experts = MagicMock()

        with patch(
            "vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_elastic_ep"
            ".IluMOEW16A16",
            create=True,
        ):
            result = ilu_is_monolithic(quant)

        assert result is False
        assert quant._ilu_is_monolithic is False

    def test_returns_cached(self):
        """When _ilu_is_monolithic is already set, skip recomputation."""
        quant = MagicMock()
        quant._ilu_is_monolithic = True
        assert ilu_is_monolithic(quant) is True


# ---------------------------------------------------------------------------
# install_ilu_elastic_ep_patches (idempotency)
# ---------------------------------------------------------------------------


class TestInstallPatches:
    def test_idempotent(self):
        """Calling install_ilu_elastic_ep_patches multiple times must not raise."""
        import vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_elastic_ep as ep_mod

        # Save and restore the global flag
        saved = ep_mod._PATCHES_INSTALLED
        try:
            ep_mod._PATCHES_INSTALLED = False

            with (
                patch.object(ep_mod, "_install_rebalance_patches"),
                patch.object(ep_mod, "_install_eplb_map_patches"),
                patch.object(ep_mod, "_install_modular_method_patches"),
            ):
                install_ilu_elastic_ep_patches()
                install_ilu_elastic_ep_patches()
                install_ilu_elastic_ep_patches()

            assert ep_mod._PATCHES_INSTALLED is True
        finally:
            ep_mod._PATCHES_INSTALLED = saved


# ---------------------------------------------------------------------------
# ilu_eplb_map_and_record (numerical correctness, CPU tensors)
# ---------------------------------------------------------------------------


class TestEplbMapAndRecord:
    def test_empty_topk(self):
        """Empty topk_ids should return immediately."""
        topk_ids = torch.empty((0, 2), dtype=torch.int32)
        expert_load = torch.zeros(8, dtype=torch.float32)
        logical_map = torch.zeros((4, 2), dtype=torch.int32)
        replica_count = torch.ones(4, dtype=torch.int32)
        record_enabled = torch.tensor([1], dtype=torch.int32)

        result = ilu_eplb_map_and_record(
            topk_ids, expert_load, logical_map, replica_count, record_enabled
        )
        assert result.shape == topk_ids.shape
        assert result.numel() == 0

    def test_identity_mapping(self):
        """With a 1:1 logical-to-physical map, output ids should match input."""
        num_experts = 4
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        expert_load = torch.zeros(num_experts, dtype=torch.float32)
        logical_map = torch.arange(num_experts, dtype=torch.int32).unsqueeze(1)
        replica_count = torch.ones(num_experts, dtype=torch.int32)
        record_enabled = torch.tensor([1], dtype=torch.int32)

        result = ilu_eplb_map_and_record(
            topk_ids, expert_load, logical_map, replica_count, record_enabled
        )
        torch.testing.assert_close(result, topk_ids)

    def test_negative_ids_preserved(self):
        """Negative expert ids (padding) must map to -1 in the output."""
        num_experts = 4
        topk_ids = torch.tensor([[0, -1]], dtype=torch.int32)
        expert_load = torch.zeros(num_experts, dtype=torch.float32)
        logical_map = torch.arange(num_experts, dtype=torch.int32).unsqueeze(1)
        replica_count = torch.ones(num_experts, dtype=torch.int32)
        record_enabled = torch.tensor([1], dtype=torch.int32)

        result = ilu_eplb_map_and_record(
            topk_ids, expert_load, logical_map, replica_count, record_enabled
        )
        assert result[0, 0].item() == 0
        assert result[0, 1].item() == -1

    def test_expert_load_recording(self):
        """With record_enabled=1, expert_load should be incremented for
        valid physical ids."""
        num_experts = 4
        topk_ids = torch.tensor([[0, 1], [0, 2]], dtype=torch.int32)
        expert_load = torch.zeros(num_experts, dtype=torch.float32)
        logical_map = torch.arange(num_experts, dtype=torch.int32).unsqueeze(1)
        replica_count = torch.ones(num_experts, dtype=torch.int32)
        record_enabled = torch.tensor([1], dtype=torch.int32)

        ilu_eplb_map_and_record(
            topk_ids, expert_load, logical_map, replica_count, record_enabled
        )
        # Expert 0 selected twice, expert 1 once, expert 2 once
        assert expert_load[0].item() == 2.0
        assert expert_load[1].item() == 1.0
        assert expert_load[2].item() == 1.0
        assert expert_load[3].item() == 0.0

    def test_record_disabled(self):
        """With record_enabled=0, expert_load should NOT be modified."""
        num_experts = 4
        topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
        expert_load = torch.zeros(num_experts, dtype=torch.float32)
        logical_map = torch.arange(num_experts, dtype=torch.int32).unsqueeze(1)
        replica_count = torch.ones(num_experts, dtype=torch.int32)
        record_enabled = torch.tensor([0], dtype=torch.int32)

        ilu_eplb_map_and_record(
            topk_ids, expert_load, logical_map, replica_count, record_enabled
        )
        torch.testing.assert_close(
            expert_load, torch.zeros(num_experts, dtype=torch.float32)
        )

    def test_replica_mapping(self):
        """With 2 replicas per expert, physical ids should depend on
        the Knuth hash of the token index."""
        num_experts = 2
        topk_ids = torch.tensor([[0], [0], [1]], dtype=torch.int32)
        expert_load = torch.zeros(4, dtype=torch.float32)
        # Expert 0 -> physical {0, 1}, Expert 1 -> physical {2, 3}
        logical_map = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        replica_count = torch.full((num_experts,), 2, dtype=torch.int32)
        record_enabled = torch.tensor([1], dtype=torch.int32)

        result = ilu_eplb_map_and_record(
            topk_ids, expert_load, logical_map, replica_count, record_enabled
        )
        # All outputs must be valid physical ids
        assert (result >= 0).all()
        assert (result < 4).all()
        # Token 0 selects expert 0 -> physical 0 or 1
        assert result[0, 0].item() in (0, 1)
        # Token 2 selects expert 1 -> physical 2 or 3
        assert result[2, 0].item() in (2, 3)
