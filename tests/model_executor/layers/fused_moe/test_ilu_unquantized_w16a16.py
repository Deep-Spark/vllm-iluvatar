# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar W16A16 unquantized MoE wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_unquantized_fused_moe_method import (  # noqa: E501
    IluUnquantizedFusedMoEMethod,
)


class _Method(IluUnquantizedFusedMoEMethod):
    def __init__(self, output: torch.Tensor):
        self.output = output
        self.moe_quant_config = object()

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return self.output


def _layer(*, routed_scaling_factor: float = 7.0):
    layer = MagicMock()
    layer.routed_scaling_factor = routed_scaling_factor
    layer.w13_weight = torch.ones((1, 2, 2))
    layer.w2_weight = torch.ones((1, 2, 2))
    layer.activation = "silu"
    layer.apply_router_weight_on_input = False
    layer.global_num_experts = 1
    layer.expert_map = None
    return layer


def test_unquantized_method_uses_custom_op_oot_registration():
    assert issubclass(IluUnquantizedFusedMoEMethod, UnquantizedFusedMoEMethod)
    assert op_registry_oot["UnquantizedFusedMoEMethod"] is IluUnquantizedFusedMoEMethod


def test_apply_adds_shared_residual_without_runner_scale():
    """Runner, not this method, owns routed_scaling_factor application."""
    method = _Method(torch.full((2,), 3.0))
    shared = torch.full((2,), 5.0)

    result = method.apply(
        layer=_layer(routed_scaling_factor=11.0),
        x=torch.ones((2, 2)),
        topk_weights=torch.ones((2, 1)),
        topk_ids=torch.zeros((2, 1), dtype=torch.int32),
        shared_experts=None,
        shared_experts_input=shared,
    )

    torch.testing.assert_close(result, torch.full((2,), 8.0))


def test_forward_native_calls_plugin_w16a16_helper():
    method = _Method(torch.ones((2, 2)))
    layer = _layer()
    expected = torch.full((2, 2), 9.0)

    with patch(
        "vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe."
        "ilu_w16a16_fused_experts",
        return_value=expected,
    ) as helper:
        result = method.forward_native(
            layer=layer,
            x=torch.ones((2, 2)),
            topk_weights=torch.ones((2, 1)),
            topk_ids=torch.zeros((2, 1), dtype=torch.int32),
            shared_experts=None,
            shared_experts_input=None,
        )

    assert result is expected
    helper.assert_called_once()
    assert helper.call_args.kwargs["w1"] is layer.w13_weight
    assert helper.call_args.kwargs["quant_config"] is method.moe_quant_config


def test_w16_helper_rejects_quantized_config():
    from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe import (
        ilu_w16a16_fused_experts,
    )

    quant_config = MagicMock()
    quant_config.use_fp8_w8a8 = True

    with pytest.raises(ValueError, match="only supports unquantized MoE"):
        ilu_w16a16_fused_experts(
            hidden_states=torch.ones((1, 2)),
            w1=torch.ones((1, 2, 2)),
            w2=torch.ones((1, 2, 2)),
            topk_weights=torch.ones((1, 1)),
            topk_ids=torch.zeros((1, 1), dtype=torch.int32),
            activation="silu",
            quant_config=quant_config,
        )

