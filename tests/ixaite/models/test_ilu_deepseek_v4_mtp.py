# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the DeepSeek-V4 W4A8 MTP model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.backends.ixaite.models.deepseek_v4 import mtp
from vllm_iluvatar.backends.ixaite.models.deepseek_v4.mtp import IluDeepSeekV4MTP
from vllm_iluvatar.models.deepseek_v4.model import IluDeepseekV4ForCausalLM


def _empty_mtp() -> IluDeepSeekV4MTP:
    model = IluDeepSeekV4MTP.__new__(IluDeepSeekV4MTP)
    torch.nn.Module.__init__(model)
    return model


@pytest.mark.parametrize(
    ("checkpoint_name", "model_name"),
    [
        (
            "model.layers.43.e_proj.weight",
            "model.layers.43.e_proj.weight",
        ),
        (
            "model.layers.43.attn.wq_b.weight",
            "model.layers.43.mtp_block.attn.wq_b.weight",
        ),
        (
            "model.layers.43.ffn.experts.0.w1.weight",
            "model.layers.43.mtp_block.ffn.experts.0.w1.weight",
        ),
        (
            "model.layers.43.embed_tokens.weight",
            "model.embed_tokens.weight",
        ),
    ],
)
def test_rewrite_spec_layer_name(checkpoint_name: str, model_name: str) -> None:
    assert (
        IluDeepSeekV4MTP._rewrite_spec_layer_name(None, 43, checkpoint_name)
        == model_name
    )


def test_target_exposes_pre_hc_head_hidden_buffer() -> None:
    target = IluDeepseekV4ForCausalLM.__new__(IluDeepseekV4ForCausalLM)
    torch.nn.Module.__init__(target)
    target.model = torch.nn.Module()
    target.model._mtp_hidden_buffer = torch.empty(4, 8192)

    assert target.get_mtp_target_hidden_states() is target.model._mtp_hidden_buffer


def test_missing_mtp_layer_weights_fail_closed(monkeypatch) -> None:
    model = _empty_mtp()
    model.config = SimpleNamespace(num_attention_heads=64, n_routed_experts=256)
    model.model = torch.nn.Module()
    model.model.mtp_start_layer_idx = 43
    model.model.num_mtp_layers = 1
    model.model.layers = torch.nn.ModuleDict()

    monkeypatch.setattr(mtp, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mtp, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mtp, "fused_moe_make_expert_params_mapping", lambda *a, **k: [])

    with pytest.raises(
        ValueError,
        match="MTP speculative decoding layer 43 weights missing from checkpoint",
    ):
        model.load_weights([])


def test_partially_loaded_mtp_layer_fails_closed(monkeypatch) -> None:
    model = _empty_mtp()
    model.config = SimpleNamespace(
        num_attention_heads=64,
        n_routed_experts=256,
        num_hidden_layers=43,
        num_nextn_predict_layers=1,
    )
    model.model = torch.nn.Module()
    model.model.mtp_start_layer_idx = 43
    model.model.num_mtp_layers = 1

    layer = torch.nn.Module()
    layer.enorm = torch.nn.Module()
    layer.enorm.weight = torch.nn.Parameter(torch.empty(1))
    layer.hnorm = torch.nn.Module()
    layer.hnorm.weight = torch.nn.Parameter(torch.empty(1))
    model.model.layers = torch.nn.ModuleDict({"43": layer})

    monkeypatch.setattr(mtp, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mtp, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mtp, "fused_moe_make_expert_params_mapping", lambda *a, **k: [])

    with pytest.raises(
        ValueError,
        match=r"W4A8 MTP: 1 parameter\(s\) were never filled",
    ):
        model.load_weights([("mtp.0.enorm.weight", torch.ones(1))])
