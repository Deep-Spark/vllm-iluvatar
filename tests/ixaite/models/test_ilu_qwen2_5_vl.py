# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar Qwen2.5-VL OOT model wiring."""

from __future__ import annotations

from types import SimpleNamespace

import einops
import pytest
import torch

from vllm_iluvatar.backends.ixaite.attention import vit as ilu_vit_attn_wrappers
from vllm_iluvatar.backends.ixaite.attention.vit import (
    vit_attn_output_layout,
    vit_fused_qkv_preproc,
)
from vllm_iluvatar.backends.ixaite.models import qwen2_5_vl as ilu_qwen2_5_vl
from vllm_iluvatar.v1.attention.ops import ilu_vit_attn_wrappers as baseline_vit


def test_qwen2_5_vl_register_model(monkeypatch):
    calls = []

    import vllm

    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: calls.append((arch, path)),
    )

    ilu_qwen2_5_vl.register_model()

    assert calls == [
        (
            "Qwen2_5_VLForConditionalGeneration",
            (
                "vllm_iluvatar.backends.ixaite.models.qwen2_5_vl:"
                "Qwen2_5_VLForConditionalGeneration"
            ),
        )
    ]


def test_qwen2_5_vl_vision_overrides_swaps_block_and_attention_classes():
    block = object.__new__(ilu_qwen2_5_vl._Qwen2_5_VisionBlock)
    attn = object.__new__(ilu_qwen2_5_vl._Qwen2_5_VisionAttention)
    object.__setattr__(block, "attn", attn)
    visual = SimpleNamespace(blocks=[block])
    model = SimpleNamespace(visual=visual)

    ilu_qwen2_5_vl.qwen2_5_vl_vision_overrides(model)

    assert block.__class__ is ilu_qwen2_5_vl.Qwen2_5_VisionBlock
    assert block.attn.__class__ is ilu_qwen2_5_vl.Qwen2_5_VisionAttention


def test_qwen2_5_vision_block_converts_tensor_max_seqlen(monkeypatch):
    captured = {}

    def fake_base_forward(self, **kwargs):
        captured.update(kwargs)
        return "block-output"

    monkeypatch.setattr(
        ilu_qwen2_5_vl._Qwen2_5_VisionBlock,
        "forward",
        fake_base_forward,
    )
    block = object.__new__(ilu_qwen2_5_vl.Qwen2_5_VisionBlock)

    assert (
        block.forward(
            x="x",
            cu_seqlens="cu",
            rotary_pos_emb_cos="cos",
            rotary_pos_emb_sin="sin",
            max_seqlen=torch.tensor(16),
            sequence_lengths="lengths",
        )
        == "block-output"
    )
    assert captured["max_seqlen"] == 16


def test_qwen2_5_vision_attention_uses_fused_qkv_preproc(monkeypatch):
    calls = []
    seq_len = 2
    batch_size = 1
    num_heads = 2
    head_dim = 4
    padded_head_dim = 8
    qkv = torch.arange(
        seq_len * batch_size * 3 * num_heads * head_dim,
        dtype=torch.float32,
    ).reshape(seq_len, batch_size, 3 * num_heads * head_dim)

    def fake_qkv(x):
        calls.append(("qkv", x))
        return qkv, None

    def fake_preproc(**kwargs):
        calls.append(("preproc", kwargs))
        shape = (batch_size, seq_len, num_heads, padded_head_dim)
        return (
            torch.full(shape, 1.0),
            torch.full(shape, 2.0),
            torch.full(shape, 3.0),
        )

    def fake_attn(**kwargs):
        calls.append(("attn", kwargs))
        return kwargs["query"] + kwargs["key"] + kwargs["value"]

    def fake_proj(x):
        calls.append(("proj", x))
        return x, None

    class FakeAttentionOp:
        def __init__(self, padded_head_dim: int) -> None:
            self.padded_head_dim = padded_head_dim

        def __call__(self, **kwargs):
            return fake_attn(**kwargs)

    monkeypatch.setattr(ilu_qwen2_5_vl, "vit_fused_qkv_preproc", fake_preproc)

    attn = object.__new__(ilu_qwen2_5_vl.Qwen2_5_VisionAttention)
    attn.qkv = fake_qkv
    attn.attn = FakeAttentionOp(padded_head_dim)
    attn.proj = fake_proj
    attn.num_attention_heads_per_partition = num_heads
    attn.hidden_size_per_attention_head = head_dim

    output = attn.forward(
        x="hidden",
        cu_seqlens="cu",
        rotary_pos_emb_cos="cos",
        rotary_pos_emb_sin="sin",
        max_seqlen="max",
        sequence_lengths="lengths",
    )

    preproc_kwargs = calls[1][1]
    assert preproc_kwargs == {
        "qkv_packed": qkv,
        "rotary_pos_emb_cos": "cos",
        "rotary_pos_emb_sin": "sin",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "padded_head_dim": padded_head_dim,
    }
    attn_kwargs = calls[2][1]
    assert attn_kwargs["cu_seqlens"] == "cu"
    assert attn_kwargs["max_seqlen"] == "max"
    assert attn_kwargs["sequence_lengths"] == "lengths"
    assert output.shape == (seq_len, batch_size, num_heads * padded_head_dim)


# Check split/padding semantics eagerly so this CPU test needs no C++ compiler.
def test_vit_fused_qkv_preproc_splits_and_pads_without_rotary():
    seq_len = 2
    batch_size = 1
    num_heads = 2
    head_dim = 4
    padded_head_dim = 6
    qkv_packed = torch.arange(
        seq_len * batch_size * 3 * num_heads * head_dim,
        dtype=torch.float32,
    ).reshape(seq_len, batch_size, 3 * num_heads * head_dim)

    q, k, v = baseline_vit.vit_fused_qkv_preproc.__wrapped__(
        qkv_packed=qkv_packed,
        rotary_pos_emb_cos=None,
        rotary_pos_emb_sin=None,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        padded_head_dim=padded_head_dim,
    )

    expected_qkv = qkv_packed.view(seq_len, batch_size, 3, num_heads, head_dim).permute(
        1, 0, 2, 3, 4
    )
    expected_q, expected_k, expected_v = expected_qkv.unbind(dim=2)

    assert q.shape == (batch_size, seq_len, num_heads, padded_head_dim)
    assert k.shape == q.shape
    assert v.shape == q.shape
    torch.testing.assert_close(q[..., :head_dim], expected_q)
    torch.testing.assert_close(k[..., :head_dim], expected_k)
    torch.testing.assert_close(v[..., :head_dim], expected_v)
    assert torch.count_nonzero(q[..., head_dim:]) == 0
    assert torch.count_nonzero(k[..., head_dim:]) == 0
    assert torch.count_nonzero(v[..., head_dim:]) == 0


# Check rotary-path Q/K layout eagerly, independently of Inductor compilation.
def test_vit_fused_qkv_preproc_qk_layout_with_rotary(monkeypatch):
    seq_len, batch_size, num_heads, head_dim = 3, 1, 4, 72
    qkv_packed = torch.arange(
        seq_len * batch_size * 3 * num_heads * head_dim,
        dtype=torch.bfloat16,
    ).reshape(seq_len, batch_size, 3 * num_heads * head_dim)
    monkeypatch.setattr(
        baseline_vit.ApplyRotaryEmb,
        "forward_static",
        staticmethod(lambda qk, *_args: qk),
    )

    q, k, v = baseline_vit.vit_fused_qkv_preproc.__wrapped__(
        qkv_packed=qkv_packed,
        rotary_pos_emb_cos=torch.empty(0),
        rotary_pos_emb_sin=torch.empty(0),
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        padded_head_dim=head_dim,
    )

    expected_qkv = einops.rearrange(
        qkv_packed,
        "s b (three head head_dim) -> b s three head head_dim",
        three=3,
        head=num_heads,
    )
    expected_qk = einops.rearrange(
        expected_qkv[:, :, :2],
        "b s two head head_dim -> (two b) s head head_dim",
        two=2,
    ).view(2, batch_size, seq_len, num_heads, head_dim)
    expected_q, expected_k = expected_qk.unbind(dim=0)

    torch.testing.assert_close(q, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(k, expected_k, atol=0, rtol=0)
    torch.testing.assert_close(v, expected_qkv[:, :, 2], atol=0, rtol=0)


def test_vit_fused_qkv_preproc_ixaite_rope_skips_plugin_rotary(monkeypatch):
    seq_len, batch_size, num_heads, head_dim = 3, 1, 4, 72
    qkv_packed = torch.arange(
        seq_len * batch_size * 3 * num_heads * head_dim,
        dtype=torch.bfloat16,
    ).reshape(seq_len, batch_size, 3 * num_heads * head_dim)
    expected = (
        torch.full((batch_size, seq_len, num_heads, head_dim), 1, dtype=torch.bfloat16),
        torch.full((batch_size, seq_len, num_heads, head_dim), 2, dtype=torch.bfloat16),
        torch.full((batch_size, seq_len, num_heads, head_dim), 3, dtype=torch.bfloat16),
    )
    monkeypatch.setattr(
        ilu_vit_attn_wrappers,
        "_VIT_QKV_ROPE",
        lambda packed, cos, sin, heads, dim: expected,
    )

    q, k, v = vit_fused_qkv_preproc(
        qkv_packed=qkv_packed,
        rotary_pos_emb_cos=torch.ones(seq_len, 18),
        rotary_pos_emb_sin=torch.ones(seq_len, 18),
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        padded_head_dim=head_dim,
    )

    torch.testing.assert_close(q, expected[0], atol=0, rtol=0)
    torch.testing.assert_close(k, expected[1], atol=0, rtol=0)
    torch.testing.assert_close(v, expected[2], atol=0, rtol=0)


@pytest.mark.parametrize("num_heads", [4, 8, 16])
def test_vit_attn_output_layout_matches_einops_shape(num_heads: int):
    batch_size, seq_len, head_dim = 1, 7, 72
    context_layer = torch.arange(
        batch_size * seq_len * num_heads * head_dim,
        dtype=torch.bfloat16,
    ).reshape(batch_size, seq_len, num_heads, head_dim)
    expected = einops.rearrange(
        context_layer,
        "b s h d -> s b (h d)",
        b=batch_size,
    ).contiguous()

    actual = vit_attn_output_layout(context_layer)

    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
