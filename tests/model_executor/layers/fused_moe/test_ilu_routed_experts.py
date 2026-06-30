# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar RoutedExperts OOT replacement."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.model_executor.custom_op import op_registry_oot

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_routed_experts import (
    IluRoutedExperts,
)


class IluAutoGPTQMoEMethod:
    pass


class OtherMoEMethod:
    pass


def _uninitialized_routed_experts() -> IluRoutedExperts:
    obj = IluRoutedExperts.__new__(IluRoutedExperts)
    obj.__dict__["_parameters"] = {}
    obj.__dict__["_buffers"] = {}
    obj.__dict__["_modules"] = {}
    obj.__dict__["_non_persistent_buffers_set"] = set()
    obj.__dict__["training"] = False
    obj.__dict__["moe_config"] = SimpleNamespace(tp_rank=0)
    obj.__dict__["quant_config"] = None
    obj.__dict__["quant_method"] = OtherMoEMethod()
    obj.__dict__["_map_global_expert_id_to_local_expert_id"] = lambda _: 0
    obj.__dict__["_load_model_weight_or_group_weight_scale"] = MagicMock()
    return obj


def test_routed_experts_oot_registration_uses_upstream_class_name():
    assert op_registry_oot["RoutedExperts"] is IluRoutedExperts


def test_ilu_auto_gptq_requests_full_intermediate_size():
    routed = _uninitialized_routed_experts()

    assert routed._needs_intermediate_size_param(IluAutoGPTQMoEMethod()) is True


def test_weight_loader_uses_parameter_shard_dim_override():
    routed = _uninitialized_routed_experts()
    param = torch.nn.Parameter(torch.empty((1, 2, 2)))
    param.shard_dim = 1

    loaded_weight = torch.ones((2, 2))

    assert (
        routed.weight_loader(
            param=param,
            loaded_weight=loaded_weight,
            weight_name="w13_weight",
            shard_id="w1",
            expert_id=0,
            return_success=True,
        )
        is True
    )
    routed._load_model_weight_or_group_weight_scale.assert_called_once()
    assert (
        routed._load_model_weight_or_group_weight_scale.call_args.kwargs["shard_dim"]
        == 1
    )


def test_weight_loader_keeps_moe_loading_marker():
    assert IluRoutedExperts.weight_loader.supports_moe_loading is True
