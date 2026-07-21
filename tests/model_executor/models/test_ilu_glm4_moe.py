# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar GLM4-MoE OOT model wiring."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_iluvatar.model_executor.models import glm4_moe as ilu_glm4_moe


def test_glm4_moe_register_model(monkeypatch):
    calls = []

    import vllm

    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: calls.append((arch, path)),
    )

    ilu_glm4_moe.register_model()

    assert calls == [
        (
            "Glm4MoeForCausalLM",
            "vllm_iluvatar.model_executor.models.glm4_moe:Glm4MoeForCausalLM",
        )
    ]


def test_glm4_moe_attention_contiguous_overrides_swaps_attention():
    attn = object.__new__(ilu_glm4_moe._Glm4MoeAttention)
    layer = SimpleNamespace(self_attn=attn)
    model = SimpleNamespace(layers=[layer])

    ilu_glm4_moe.glm4_moe_attention_contiguous_overrides(model)

    assert attn.__class__ is ilu_glm4_moe.Glm4MoeAttention


def test_glm4_moe_bf16_router_overrides_swaps_mlp_and_converts_gate():
    mlp = object.__new__(ilu_glm4_moe._Glm4MoE)
    torch.nn.Module.__init__(mlp)
    gate = torch.nn.Linear(4, 3, bias=False, dtype=torch.float32)
    gate.e_score_correction_bias = torch.nn.Parameter(
        torch.empty(3, dtype=torch.float32)
    )
    mlp.gate = gate
    model = SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)])

    ilu_glm4_moe.glm4_moe_bf16_router_overrides(model)

    assert mlp.__class__ is ilu_glm4_moe.Glm4MoE
    assert mlp.gate.weight.dtype == torch.bfloat16
    assert mlp.gate.e_score_correction_bias.dtype == torch.bfloat16


def test_glm4_moe_attention_forward_makes_split_qkv_contiguous():
    calls = []
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    qkv = torch.arange(2 * 24, dtype=torch.float32).reshape(2, 24)

    def fake_qkv_proj(x):
        calls.append(("qkv_proj", x))
        return qkv, None

    def fake_rotary_emb(positions, q, k):
        calls.append(("rotary_emb", positions, q, k))
        return q, k

    def fake_attn(q, k, v):
        calls.append(("attn", q, k, v))
        return q + k + v

    def fake_o_proj(x):
        calls.append(("o_proj", x))
        return x, None

    attn = object.__new__(ilu_glm4_moe.Glm4MoeAttention)
    object.__setattr__(attn, "qkv_proj", fake_qkv_proj)
    object.__setattr__(attn, "q_size", 8)
    object.__setattr__(attn, "kv_size", 8)
    object.__setattr__(attn, "use_qk_norm", False)
    object.__setattr__(attn, "rotary_emb", fake_rotary_emb)
    object.__setattr__(attn, "attn", fake_attn)
    object.__setattr__(attn, "o_proj", fake_o_proj)

    output = attn.forward("positions", hidden_states)

    _, q, k, v = calls[2]
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert output.shape == (2, 8)
