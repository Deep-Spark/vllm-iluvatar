# SPDX-License-Identifier: Apache-2.0
"""DSpark checkpoint adaptation delegates complete, uniquely mapped weights."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

from vllm_iluvatar.models.minimax_m3.dspark import IluMiniMaxDSparkForCausalLM


def checkpoint():
    names = [
        "fc.weight",
        "hidden_norm.weight",
        "final_norm.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
        "confidence_head.proj.bias",
    ]
    stem = "layers.0.decoder_layer."
    names.extend(
        stem + n + ".weight"
        for n in (
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        )
    )
    names.extend(
        stem + n + suffix
        for n in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
        for suffix in (".weight", ".weight_scale_inv")
    )
    return [
        ("language_model.model.dspark." + n, torch.tensor(i))
        for i, n in enumerate(names)
    ]


def draft():
    model = IluMiniMaxDSparkForCausalLM.__new__(IluMiniMaxDSparkForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=1, dspark_target_layer_ids=[0])
    return model


def test_remap_delegates_every_tensor_once(monkeypatch):
    seen = []

    def load(self, tensors):
        seen.extend(tensors)
        return {name for name, _ in seen}

    monkeypatch.setattr(Qwen3DSparkForCausalLM, "load_weights", load)
    source = checkpoint()
    result = draft().load_weights(iter(source))
    assert len(seen) == len(source) == len(result)
    assert "norm.weight" in result
    assert "layers.0.self_attn.q_proj.weight_scale" in result
    assert not any(
        "decoder_layer" in name or "weight_scale_inv" in name for name in result
    )
    assert all(got is original for (_, got), (_, original) in zip(seen, source))


@pytest.mark.parametrize(
    "error", ["missing", "duplicate", "unexpected", "wrong_prefix"]
)
def test_invalid_checkpoint_fails(monkeypatch, error):
    monkeypatch.setattr(
        Qwen3DSparkForCausalLM, "load_weights", lambda self, tensors: list(tensors)
    )
    source = checkpoint()
    if error == "missing":
        source.pop()
    elif error == "duplicate":
        source.append(source[0])
    else:
        source.append(
            (
                "language_model.model.dspark.unknown.weight"
                if error == "unexpected"
                else "target.weight",
                torch.tensor(0),
            )
        )
    with pytest.raises(ValueError, match="[Mm]issing|[Dd]uplicate|[Uu]nexpected"):
        draft().load_weights(source)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_draft_context_capability_delegates_to_target(wrapped, supported):
    model = draft()
    projection = object()
    model.model = SimpleNamespace(fc=projection)
    calls = []
    target = SimpleNamespace()
    if supported:

        def bind(fc, output):
            calls.append((fc, output))
            return True

        target.bind_draft_context = bind
    language = SimpleNamespace(model=target)
    outer = (
        SimpleNamespace(get_language_model=lambda: language) if wrapped else language
    )
    output = torch.empty(1)
    assert model.try_bind_target_context(outer, output) is supported
    assert calls == ([(projection, output)] if supported else [])
