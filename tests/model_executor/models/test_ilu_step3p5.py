# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar Step3.5 OOT model registration."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import torch

from vllm_iluvatar.model_executor.layers.fused_moe import (
    ilu_fused_moe_ep,
)
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
    IluFusedMoEEP,
)
from vllm_iluvatar.model_executor.models import step3p5 as ilu_step3p5
from vllm_iluvatar.model_executor.models import step3p5_mtp as ilu_step3p5_mtp


def test_step3p5_oot_model_constructors_expose_vllm_config():
    for model_cls in (
        ilu_step3p5.IluStep3p5ForCausalLM,
        ilu_step3p5_mtp.IluStep3p5MTP,
    ):
        parameters = inspect.signature(model_cls.__init__).parameters
        assert parameters["vllm_config"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["prefix"].default == ""


def test_step3p5_causal_lm_refreshes_pp_intermediate_factory():
    source = inspect.getsource(ilu_step3p5.IluStep3p5ForCausalLM.__init__)

    assert "self.model.make_empty_intermediate_tensors" in source
    assert source.index("set_max_num_seqs") < source.index("super().__init__")


def test_ep_max_num_seqs_falls_back_to_initialized_value(monkeypatch):
    monkeypatch.setattr(ilu_fused_moe_ep, "_MAX_NUM_SEQS", None)
    monkeypatch.setattr(
        ilu_fused_moe_ep,
        "get_current_vllm_config_or_none",
        lambda: None,
    )

    ilu_fused_moe_ep.set_max_num_seqs(256)

    assert ilu_fused_moe_ep._get_max_num_seqs() == 256


def test_step3p5_register_model(monkeypatch):
    calls = []

    import vllm

    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: calls.append((arch, path)),
    )

    ilu_step3p5.register_model()
    ilu_step3p5_mtp.register_model()

    assert calls == [
        (
            "Step3p5ForCausalLM",
            "vllm_iluvatar.model_executor.models.step3p5:IluStep3p5ForCausalLM",
        ),
        (
            "Step3p5MTP",
            "vllm_iluvatar.model_executor.models.step3p5_mtp:IluStep3p5MTP",
        ),
    ]


def test_fused_moe_block_mro_includes_ep_mixin():
    mro_names = [cls.__name__ for cls in ilu_step3p5.IluFusedMoEBlock.__mro__]
    assert "IluFusedMoEEP" in mro_names
    assert mro_names.index("IluFusedMoEBlock") < mro_names.index("IluFusedMoEEP")


def test_decoder_promote_attaches_moe_ep(monkeypatch):
    class _DummyModule(torch.nn.Module):
        pass

    layer = _DummyModule()
    layer.use_moe = True
    layer.prefix = "model.layers.3"
    layer.moe = _DummyModule()
    layer.moe.gate = _DummyModule()
    layer.moe.gate.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    layer.self_attn = _DummyModule()
    init_ep = MagicMock()
    monkeypatch.setattr(IluFusedMoEEP, "init_ep", init_ep)

    ilu_step3p5.IluStep3p5DecoderLayer.promote(layer)

    assert type(layer.moe) is ilu_step3p5.IluFusedMoEBlock
    assert layer.moe.prefix == "model.layers.3.moe"
    init_ep.assert_called_once_with()


def _make_stub_moe_block() -> ilu_step3p5.IluFusedMoEBlock:
    block = ilu_step3p5.IluFusedMoEBlock.__new__(ilu_step3p5.IluFusedMoEBlock)
    block.experts = MagicMock()
    block.experts.is_internal_router = True
    block.gate = MagicMock()
    return block


def test_moe_forward_uses_dp_ep_for_tensor_input():
    block = _make_stub_moe_block()
    hidden_states = torch.zeros((2, 4))
    block._should_use_forward_dp_ep = MagicMock(return_value=True)
    block.forward_dp_ep = MagicMock(return_value=hidden_states + 1)

    output = block.forward(hidden_states)

    block.forward_dp_ep.assert_called_once_with(hidden_states)
    block.experts.assert_not_called()
    torch.testing.assert_close(output, hidden_states + 1)


def test_moe_forward_keeps_w8a8_tuple_on_runner_path():
    block = _make_stub_moe_block()
    origin = torch.zeros((2, 4))
    hidden_states = (torch.zeros((2, 4), dtype=torch.int8), torch.ones(2), origin)
    block._should_use_forward_dp_ep = MagicMock(return_value=True)
    block.forward_dp_ep = MagicMock()
    block.experts.return_value = origin

    output = block.forward(hidden_states)

    block.forward_dp_ep.assert_not_called()
    block.experts.assert_called_once_with(
        hidden_states=hidden_states,
        router_logits=hidden_states,
    )
    torch.testing.assert_close(output, origin)
