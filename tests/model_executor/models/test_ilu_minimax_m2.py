# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar MiniMax-M2 OOT model wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.minimax_m2 import MiniMaxM2MoE

from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
    IluFusedMoEEP,
)
from vllm_iluvatar.model_executor.models import minimax_m2 as ilu_minimax_m2


def test_minimax_m2_oot_registration_points_at_plugin_subclass():
    from vllm import ModelRegistry

    from vllm_iluvatar.ilu_register import apply_oot_registrations

    apply_oot_registrations()
    entry = ModelRegistry.models["MiniMaxM2ForCausalLM"]
    assert entry.module_name == "vllm_iluvatar.model_executor.models.minimax_m2"
    assert entry.class_name == "MiniMaxM2ForCausalLM"


def test_minimax_m2_plugin_class_subclasses_community():
    from vllm.model_executor.models.minimax_m2 import (
        MiniMaxM2ForCausalLM as _CommunityMiniMaxM2ForCausalLM,
    )

    from vllm_iluvatar.model_executor.models.minimax_m2 import MiniMaxM2ForCausalLM

    assert issubclass(MiniMaxM2ForCausalLM, _CommunityMiniMaxM2ForCausalLM)


def test_minimax_m2_moe_mro_includes_ep_mixin():
    mro_names = [cls.__name__ for cls in ilu_minimax_m2.IluMiniMaxM2MoE.__mro__]

    assert "IluFusedMoEEP" in mro_names
    assert mro_names.index("IluMiniMaxM2MoE") < mro_names.index("IluFusedMoEEP")


def test_minimax_m2_moe_promote_attaches_gate_and_ep(monkeypatch):
    gate = GateLinear.__new__(GateLinear)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(torch.ones((2, 4), dtype=torch.float32))
    gate.prefix = "model.layers.3.mlp.gate"

    moe = MiniMaxM2MoE.__new__(MiniMaxM2MoE)
    torch.nn.Module.__init__(moe)
    moe.gate = gate

    init_ep = MagicMock()
    monkeypatch.setattr(IluFusedMoEEP, "init_ep", init_ep)

    ilu_minimax_m2.IluMiniMaxM2MoE.promote(moe)

    assert type(moe) is ilu_minimax_m2.IluMiniMaxM2MoE
    assert type(moe.gate) is ilu_minimax_m2.IluMiniMaxM2GateLinear
    assert moe.gate.weight.dtype is torch.bfloat16
    assert moe.prefix == "model.layers.3.mlp"
    init_ep.assert_called_once_with()


def test_minimax_m2_gate_uses_replicated_linear_and_returns_fp32(monkeypatch):
    calls = []

    def fake_forward(self, x):
        calls.append(x)
        return x.to(torch.bfloat16), "bias"

    monkeypatch.setattr(ReplicatedLinear, "forward", fake_forward)
    gate = ilu_minimax_m2.IluMiniMaxM2GateLinear.__new__(
        ilu_minimax_m2.IluMiniMaxM2GateLinear
    )
    torch.nn.Module.__init__(gate)
    hidden_states = torch.ones((2, 4), dtype=torch.float32)

    output, output_bias = gate.forward(hidden_states)

    assert calls == [hidden_states]
    assert output.dtype is torch.float32
    assert output_bias == "bias"


def _make_stub_moe() -> ilu_minimax_m2.IluMiniMaxM2MoE:
    moe = ilu_minimax_m2.IluMiniMaxM2MoE.__new__(ilu_minimax_m2.IluMiniMaxM2MoE)
    torch.nn.Module.__init__(moe)
    object.__setattr__(moe, "experts", MagicMock())
    object.__setattr__(moe, "gate", MagicMock())
    return moe


def test_minimax_m2_moe_forward_uses_dp_ep():
    moe = _make_stub_moe()
    hidden_states = torch.zeros((2, 4))
    moe._should_use_forward_dp_ep = MagicMock(return_value=True)
    moe.forward_dp_ep = MagicMock(return_value=hidden_states + 1)

    output = moe.forward(hidden_states)

    moe.forward_dp_ep.assert_called_once()
    torch.testing.assert_close(moe.forward_dp_ep.call_args.args[0], hidden_states)
    moe.gate.assert_not_called()
    moe.experts.assert_not_called()
    torch.testing.assert_close(output, hidden_states + 1)


def test_minimax_m2_moe_forward_uses_community_runner_without_dp_ep():
    moe = _make_stub_moe()
    hidden_states = torch.zeros((2, 4))
    router_logits = torch.ones((2, 3), dtype=torch.float32)
    moe._should_use_forward_dp_ep = MagicMock(return_value=False)
    moe.gate.return_value = (router_logits, None)
    moe.experts.return_value = hidden_states + 2

    output = moe.forward(hidden_states)

    moe.gate.assert_called_once()
    torch.testing.assert_close(moe.gate.call_args.args[0], hidden_states)
    moe.experts.assert_called_once()
    torch.testing.assert_close(
        moe.experts.call_args.kwargs["hidden_states"],
        hidden_states,
    )
    assert moe.experts.call_args.kwargs["router_logits"] is router_logits
    torch.testing.assert_close(output, hidden_states + 2)


def test_minimax_m2_model_override_promotes_only_moe_layers(monkeypatch):
    moe = MiniMaxM2MoE.__new__(MiniMaxM2MoE)
    torch.nn.Module.__init__(moe)
    promote = MagicMock()
    monkeypatch.setattr(ilu_minimax_m2.IluMiniMaxM2MoE, "promote", promote)
    model = SimpleNamespace(
        layers=[
            SimpleNamespace(block_sparse_moe=moe),
            SimpleNamespace(block_sparse_moe=None),
            SimpleNamespace(),
        ]
    )

    ilu_minimax_m2.minimax_m2_moe_overrides(model)

    promote.assert_called_once_with(moe)
