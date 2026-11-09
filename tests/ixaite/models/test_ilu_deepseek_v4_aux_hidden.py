# SPDX-License-Identifier: Apache-2.0
"""Aux-hidden-layer id resolution + forward capture for ilu DeepSeek-V4."""

from __future__ import annotations

import importlib
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_iluvatar.models.deepseek_v4.model import (
    IluDeepseekV4Model,
    default_aux_hidden_state_layers,
)

_MODEL_MODULE = "vllm_iluvatar.models.deepseek_v4.model"
_OPS_ATTR = "ixfops"


# The public auxiliary-state behavior is verified against both independent models.
@pytest.fixture(autouse=True, params=["baseline", "ixaite"])
def _model_backend(monkeypatch, request):
    module_name = (
        "vllm_iluvatar.models.deepseek_v4.model"
        if request.param == "baseline"
        else "vllm_iluvatar.backends.ixaite.models.deepseek_v4.model"
    )
    module = importlib.import_module(module_name)
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", request.param)
    monkeypatch.setitem(globals(), "IluDeepseekV4Model", module.IluDeepseekV4Model)
    monkeypatch.setitem(globals(), "_MODEL_MODULE", module_name)
    monkeypatch.setitem(
        globals(), "_OPS_ATTR", "ixfops" if request.param == "baseline" else "ilu_ops"
    )
    return module


def test_dspark_target_layer_ids_are_shifted_by_one():
    cfg = types.SimpleNamespace(dspark_target_layer_ids=[40, 41, 42])
    assert default_aux_hidden_state_layers(cfg) == (41, 42, 43)


def test_missing_dspark_ids_yields_empty():
    cfg = types.SimpleNamespace()
    assert default_aux_hidden_state_layers(cfg) == ()


class _FakeDecoderLayer:
    """Writes layer_idx into feat 0 and stream index into feat 1."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def __call__(self, hidden_states, positions, input_ids):
        out = hidden_states.clone()
        out[..., 0] = float(self.layer_idx)
        for s in range(out.shape[1]):
            out[:, s, 1] = float(s)
        return out


class _Identity:
    def __call__(self, x):
        return x


def _stub_model(*, num_layers: int, hc_mult: int, hidden_size: int, max_tokens: int):
    model = object.__new__(IluDeepseekV4Model)
    # Bypass nn.Module.__setattr__ (object was not Module-inited).
    d = model.__dict__
    d["hc_mult"] = hc_mult
    d["hc_eps"] = 1e-5
    d["rms_norm_eps"] = 1e-5
    d["config"] = types.SimpleNamespace(hidden_size=hidden_size)
    d["start_layer"] = 0
    d["end_layer"] = num_layers
    d["layers"] = [_FakeDecoderLayer(i) for i in range(num_layers)]
    d["aux_hidden_state_layers"] = ()
    d["hc_pre_fusion"] = None
    d["norm"] = _Identity()
    d["hc_head_fn"] = torch.empty(0)
    d["hc_head_scale"] = torch.empty(0)
    d["hc_head_base"] = torch.empty(0)
    d["_mtp_hidden_buffer"] = torch.empty(max_tokens, hc_mult * hidden_size)
    d["embed_input_ids"] = MagicMock()
    return model


def test_aux_forward_mean_pools_and_uses_shifted_layer_ids():
    T, H, hc = 3, 8, 2
    model = _stub_model(num_layers=3, hc_mult=hc, hidden_size=H, max_tokens=T)
    # Capture after source layers 0 and 2 → runner ids 1 and 3.
    model.aux_hidden_state_layers = (1, 3)
    embeds = torch.zeros(T, H)
    positions = torch.arange(T)
    input_ids = torch.zeros(T, dtype=torch.long)

    pp = MagicMock()
    pp.is_first_rank = True
    pp.is_last_rank = True

    def _fake_hc_head(hs, *args, **kwargs):
        return hs.mean(dim=1)

    with (
        patch(
            f"{_MODEL_MODULE}.get_pp_group",
            return_value=pp,
        ),
        patch(
            f"{_MODEL_MODULE}.{_OPS_ATTR}.ds4_hc_head",
            side_effect=_fake_hc_head,
        ),
    ):
        out = IluDeepseekV4Model.forward(
            model,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=None,
            inputs_embeds=embeds,
        )

    assert isinstance(out, tuple)
    hidden, aux = out
    assert hidden.shape == (T, H)
    assert len(aux) == 2
    for a in aux:
        assert a.shape == (T, H)
    # After layer 0: feat0 = 0; mean of streams on feat1 = 0.5
    assert torch.allclose(aux[0][:, 0], torch.zeros(T))
    assert torch.allclose(aux[0][:, 1], torch.full((T,), 0.5))
    # After layer 2: feat0 = 2
    assert torch.allclose(aux[1][:, 0], torch.full((T,), 2.0))
    assert torch.allclose(aux[1][:, 1], torch.full((T,), 0.5))


def test_aux_forward_without_capture_returns_single_tensor():
    T, H, hc = 2, 4, 2
    model = _stub_model(num_layers=2, hc_mult=hc, hidden_size=H, max_tokens=T)
    model.aux_hidden_state_layers = ()
    embeds = torch.zeros(T, H)

    pp = MagicMock()
    pp.is_first_rank = True
    pp.is_last_rank = True

    with (
        patch(
            f"{_MODEL_MODULE}.get_pp_group",
            return_value=pp,
        ),
        patch(
            f"{_MODEL_MODULE}.{_OPS_ATTR}.ds4_hc_head",
            side_effect=lambda hs, *a, **k: hs.mean(dim=1),
        ),
    ):
        out = IluDeepseekV4Model.forward(
            model,
            input_ids=torch.zeros(T, dtype=torch.long),
            positions=torch.arange(T),
            intermediate_tensors=None,
            inputs_embeds=embeds,
        )

    assert isinstance(out, torch.Tensor)
    assert out.shape == (T, H)


# Baseline owns its fused reduction; ixaite reduces before its MHC epilogue.
def test_mhc_provider_keeps_reduction_and_fusion_policy(monkeypatch, _model_backend):
    module = _model_backend
    own = "backends.ixaite" in module.__name__
    monkeypatch.setattr(module, "_DS4_HC_PRE_NORM_FUSION", True)
    monkeypatch.setattr(module, "_DS4_MHC_PRE_VERSION", "v2")
    assert module._hc_pre_norm_fusion_available() is (not own)
    events = []
    x = torch.ones(2, 4)
    reduced = 2 * x
    output = torch.zeros_like(x)

    # The split epilogue must receive the collective's actual result.
    def reduce(value):
        assert value is x
        events.append("reduce")
        return reduced

    # The baseline fused op already performs its own collective.
    def fused(*args, **kwargs):
        assert args[0] is x
        events.append("fused")
        return output

    # The independent post operation follows the selected communicator.
    def split(value, *args):
        assert value is reduced
        events.append("post")
        return output

    if own:
        monkeypatch.setattr(module, "tensor_model_parallel_all_reduce", reduce)
        monkeypatch.setattr(module.ilu_ops, "ds4_mhc_post", split)
    else:
        monkeypatch.setattr(module.ixfops, "allreduce_mhc_post", fused)
        monkeypatch.setattr(
            module, "get_tp_group", lambda: types.SimpleNamespace(device_group=None)
        )
    result = module.IluDeepseekV4DecoderLayer.hc_post(None, x, x, x, x, reduce_x=True)
    assert result is output
    assert events == (["reduce", "post"] if own else ["fused"])
