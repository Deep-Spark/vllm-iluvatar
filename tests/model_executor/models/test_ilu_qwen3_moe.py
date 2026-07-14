# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar Qwen3-MoE OOT model wiring."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_iluvatar.model_executor.models import qwen3_moe as ilu_qwen3_moe


def test_qwen3_moe_register_model(monkeypatch):
    calls = []

    import vllm

    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: calls.append((arch, path)),
    )

    ilu_qwen3_moe.register_model()

    assert calls == [
        (
            "Qwen3MoeForCausalLM",
            "vllm_iluvatar.model_executor.models.qwen3_moe:Qwen3MoeForCausalLM",
        )
    ]


def test_qwen3_moe_attention_fusion_swaps_attention_and_attaches_qk_norm(
    monkeypatch,
):
    created = []

    class FakeRMSNormQK:
        def __init__(self, hidden_size_q, hidden_size_k, eps):
            created.append((hidden_size_q, hidden_size_k, eps))

    monkeypatch.setattr(ilu_qwen3_moe, "RMSNormQK", FakeRMSNormQK)

    attn = object.__new__(ilu_qwen3_moe._Qwen3MoeAttention)
    object.__setattr__(attn, "head_dim", 16)
    object.__setattr__(attn, "q_norm", SimpleNamespace(variance_epsilon=1e-6))
    layer = SimpleNamespace(self_attn=attn)
    model = SimpleNamespace(layers=[layer])

    ilu_qwen3_moe.qwen3_moe_attention_fusion(model)

    assert attn.__class__ is ilu_qwen3_moe.Qwen3MoeAttention
    assert created == [(16, 16, 1e-6)]
    assert isinstance(attn.qk_norm, FakeRMSNormQK)


def test_qwen3_moe_unquantized_router_gates_rebuilds_sparse_gate(
    monkeypatch,
):
    created = []

    class FakeReplicatedLinear:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.input_size = args[0]
            self.output_size = args[1]
            self.bias = object() if kwargs["bias"] else None
            self.skip_bias_add = kwargs["skip_bias_add"]
            self.params_dtype = kwargs["params_dtype"]
            self.quant_config = kwargs["quant_config"]
            self.prefix = kwargs["prefix"]
            self.return_bias = kwargs["return_bias"]
            created.append(self)

    monkeypatch.setattr(ilu_qwen3_moe, "ReplicatedLinear", FakeReplicatedLinear)

    old_gate = SimpleNamespace(
        input_size=32,
        output_size=8,
        bias=None,
        skip_bias_add=False,
        params_dtype=torch.float16,
        prefix="model.layers.0.mlp.gate",
        return_bias=True,
        disable_tp=False,
    )
    experts = SimpleNamespace(
        gate=old_gate,
        shared_expert_gate=None,
        _combined_gate_weight="cached",
        _fse_fuse_gate=True,
    )
    mlp = object.__new__(ilu_qwen3_moe._Qwen3MoeSparseMoeBlock)
    torch.nn.Module.__init__(mlp)
    object.__setattr__(mlp, "gate", old_gate)
    object.__setattr__(mlp, "experts", experts)
    model = SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)])

    ilu_qwen3_moe.qwen3_moe_router_gate_overrides(model)

    assert len(created) == 1
    new_gate = created[0]
    assert new_gate.args == (32, 8)
    assert new_gate.kwargs["quant_config"] is None
    assert new_gate.kwargs["prefix"] == "model.layers.0.mlp.gate"
    assert new_gate.kwargs["params_dtype"] is torch.float16
    assert mlp.gate is new_gate
    assert experts.gate is new_gate
    assert experts._combined_gate_weight is None
    assert experts._fse_fuse_gate is False


def test_qwen3_moe_attention_forward_uses_fused_qk_norm():
    calls = []
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    qkv = torch.arange(2 * 24, dtype=torch.float32).reshape(2, 24)
    q_size = 8
    kv_size = 8
    head_dim = 4

    def fake_qkv_proj(x):
        calls.append(("qkv_proj", x))
        return qkv, None

    def fake_qk_norm(q, k, q_weight, k_weight):
        calls.append(("qk_norm", q, k, q_weight, k_weight))
        return q + 1, k + 2

    def fake_rotary_emb(positions, q, k):
        calls.append(("rotary_emb", positions, q, k))
        return q + 3, k + 4

    def fake_attn(q, k, v):
        calls.append(("attn", q, k, v))
        return q + k + v

    def fake_o_proj(x):
        calls.append(("o_proj", x))
        return x, None

    attn = object.__new__(ilu_qwen3_moe.Qwen3MoeAttention)
    object.__setattr__(attn, "qkv_proj", fake_qkv_proj)
    object.__setattr__(attn, "q_size", q_size)
    object.__setattr__(attn, "kv_size", kv_size)
    object.__setattr__(attn, "head_dim", head_dim)
    object.__setattr__(
        attn, "q_norm", SimpleNamespace(weight=SimpleNamespace(data="q_weight"))
    )
    object.__setattr__(
        attn, "k_norm", SimpleNamespace(weight=SimpleNamespace(data="k_weight"))
    )
    object.__setattr__(attn, "qk_norm", fake_qk_norm)
    object.__setattr__(attn, "rotary_emb", fake_rotary_emb)
    object.__setattr__(attn, "attn", fake_attn)
    object.__setattr__(attn, "o_proj", fake_o_proj)

    output = attn.forward("positions", hidden_states)

    _, q_by_head, k_by_head, q_weight, k_weight = calls[1]
    assert q_by_head.shape == (2, 2, head_dim)
    assert k_by_head.shape == (2, 2, head_dim)
    assert q_weight == "q_weight"
    assert k_weight == "k_weight"
    assert calls[2][0] == "rotary_emb"
    assert calls[3][0] == "attn"
    assert calls[4][0] == "o_proj"
    assert output.shape == (2, q_size)
