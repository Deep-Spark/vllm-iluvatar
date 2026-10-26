# SPDX-License-Identifier: Apache-2.0
"""Auxiliary indices select each completed state exactly once, on CPU."""

from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.models.minimax_m3.mxfp8 import model as impl


class Decoder(torch.nn.Module):
    def forward(self, positions, hidden, residual):
        preceding = hidden if residual is None else hidden + residual
        return torch.ones_like(preceding), preceding, preceding


class Norm(torch.nn.Module):
    def forward(self, hidden, residual):
        complete = hidden + residual
        return complete, complete


@pytest.mark.parametrize("indices", [(0,), (1,), (3,), (0, 1, 3), ()])
@pytest.mark.parametrize("direct", [False, True])
def test_auxiliary_states_are_unique_and_ordered(monkeypatch, indices, direct):
    monkeypatch.setattr(
        impl,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    model = impl.MiniMaxM3Model.__new__(impl.MiniMaxM3Model)
    torch.nn.Module.__init__(model)
    model.layers = torch.nn.ModuleList([Decoder() for _ in range(3)])
    model.start_layer, model.end_layer = 0, 3
    model.aux_hidden_state_layers = indices
    model.fuse_final_norm_allreduce = False
    model.norm = Norm()
    captured = []
    if direct:
        model._dspark_context_bridge = lambda states: captured.extend(states)
    result = model(None, torch.arange(2), None, inputs_embeds=torch.ones(2, 4))
    if not indices:
        assert isinstance(result, torch.Tensor)
        assert not captured
        return
    hidden, exported = result
    states = captured if direct else exported
    assert len(states) == len(indices)
    for index, state in zip(indices, states):
        torch.testing.assert_close(state, torch.full((2, 4), float(index + 1)))
    torch.testing.assert_close(hidden, torch.full((2, 4), 4.0))
    if direct:
        assert exported == []


@pytest.mark.parametrize(
    "invalid", [None, "bias", "quantized", "dtype", "width", "empty"]
)
def test_context_binding_validates_projection_contract(invalid):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    model = impl.MiniMaxM3Model.__new__(impl.MiniMaxM3Model)
    torch.nn.Module.__init__(model)
    model.aux_hidden_state_layers = (0, 1)
    model.config = SimpleNamespace(hidden_size=4)
    projection = SimpleNamespace(
        input_size=8,
        output_size=4,
        weight=torch.ones(4, 8, dtype=torch.bfloat16),
        bias=None,
        quant_method=UnquantizedLinearMethod(),
    )
    output = torch.empty(3, 4, dtype=torch.bfloat16)
    if invalid == "bias":
        projection.bias = torch.zeros(4)
    elif invalid == "quantized":
        projection.quant_method = object()
    elif invalid == "dtype":
        output = output.float()
    elif invalid == "width":
        projection.input_size = 9
    elif invalid == "empty":
        model.aux_hidden_state_layers = ()
    assert model.bind_draft_context(projection, output) is (invalid is None)
    if invalid is None:
        model._dspark_context_bridge([torch.ones(3, 4, dtype=torch.bfloat16)] * 2)
        torch.testing.assert_close(output, torch.full_like(output, 8))
    else:
        assert not hasattr(model, "_dspark_context_bridge")
