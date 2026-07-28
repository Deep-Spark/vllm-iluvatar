# SPDX-License-Identifier: Apache-2.0
"""Accuracy tests for the Iluvatar MMEncoderAttention OOT override."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from vllm_iluvatar.model_executor.layers.attention import (
    mm_encoder_attention as ilu_mm,
)


def _make_attn(
    *,
    num_heads: int = 2,
    head_size: int = 3,
    opt_level: int = 0,
    block_idx: int = 3,
    monkeypatch: pytest.MonkeyPatch,
) -> ilu_mm.IluMMEncoderAttention:
    monkeypatch.setattr(
        ilu_mm.envs,
        "VLLM_ILU_VIT_ATTN_OPT_LEVEL",
        opt_level,
        raising=False,
    )
    monkeypatch.setattr(
        ilu_mm.envs,
        "VLLM_ILU_VIT_ATTN_LAYER",
        "",
        raising=False,
    )

    attn = object.__new__(ilu_mm.IluMMEncoderAttention)
    attn.num_heads = num_heads
    attn.num_kv_heads = num_heads
    attn.head_size = head_size
    attn.scale = head_size**-0.5
    attn.is_flash_attn_backend = True
    attn._vit_block_idx = block_idx
    return attn


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)


def _varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    outputs = []
    for idx in range(cu_q.numel() - 1):
        q_start, q_end = int(cu_q[idx]), int(cu_q[idx + 1])
        kv_start, kv_end = int(cu_kv[idx]), int(cu_kv[idx + 1])
        out = _reference_attention(
            q[q_start:q_end].unsqueeze(0),
            k[kv_start:kv_end].unsqueeze(0),
            v[kv_start:kv_end].unsqueeze(0),
            scale,
        )
        outputs.append(out.squeeze(0))
    return torch.cat(outputs, dim=0)


def _install_fake_ixformer(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"bf16": 0, "all_int8": 0, "qk_int8": 0}

    ixformer = ModuleType("ixformer")
    ixformer.__path__ = []
    contrib = ModuleType("ixformer.contrib")
    contrib.__path__ = []
    vllm_flash_attn = ModuleType("ixformer.contrib.vllm_flash_attn")
    sageattention = ModuleType("ixformer.contrib.sageattention")
    sageattention.__path__ = []
    sageattention_core = ModuleType("ixformer.contrib.sageattention.core")

    def fake_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        max_seqlen_q,
        max_seqlen_k,
        *,
        softmax_scale,
        causal,
    ):
        del max_seqlen_q, max_seqlen_k
        assert causal is False
        calls["bf16"] += 1
        return _varlen_attention(q, k, v, cu_q, cu_kv, softmax_scale)

    def fake_sageattn_allint8(
        q,
        k,
        v,
        *,
        is_causal,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
    ):
        del max_seqlen_q, max_seqlen_k
        assert is_causal is False
        calls["all_int8"] += 1
        total_out = _varlen_attention(
            q.reshape(-1, q.size(-2), q.size(-1)),
            k.reshape(-1, k.size(-2), k.size(-1)),
            v.reshape(-1, v.size(-2), v.size(-1)),
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale,
        )
        return total_out, None

    def fake_quant_q(x):
        return x, torch.ones((), dtype=torch.float32, device=x.device)

    def fake_ixinfer_flash_attn_unpad_int8_vf16(
        q,
        k,
        v,
        q_scale,
        k_scale,
        *,
        is_causal,
        atten_scale,
    ):
        del q_scale, k_scale
        assert is_causal is False
        calls["qk_int8"] += 1
        return _reference_attention(q, k, v, atten_scale), None

    vllm_flash_attn.flash_attn_varlen_func = fake_flash_attn_varlen_func
    sageattention.sageattn_allint8 = fake_sageattn_allint8
    sageattention.ixinfer_flash_attn_unpad_int8_vf16 = (
        fake_ixinfer_flash_attn_unpad_int8_vf16
    )
    sageattention_core.quant_q = fake_quant_q

    monkeypatch.setitem(sys.modules, "ixformer", ixformer)
    monkeypatch.setitem(sys.modules, "ixformer.contrib", contrib)
    monkeypatch.setitem(
        sys.modules,
        "ixformer.contrib.vllm_flash_attn",
        vllm_flash_attn,
    )
    monkeypatch.setitem(sys.modules, "ixformer.contrib.sageattention", sageattention)
    monkeypatch.setitem(
        sys.modules,
        "ixformer.contrib.sageattention.core",
        sageattention_core,
    )
    return calls


@pytest.mark.parametrize("opt_level", [0, 1, 2])
def test_forward_fa_matches_torch_attention_with_head_padding(
    monkeypatch: pytest.MonkeyPatch,
    opt_level: int,
):
    calls = _install_fake_ixformer(monkeypatch)
    attn = _make_attn(opt_level=opt_level, monkeypatch=monkeypatch)
    torch.manual_seed(0)
    query = torch.randn(2, 3, attn.num_heads, attn.head_size)
    key = torch.randn(2, 4, attn.num_kv_heads, attn.head_size)
    value = torch.randn(2, 4, attn.num_kv_heads, attn.head_size)

    output = attn._forward_fa(query, key, value)

    expected = _reference_attention(query, key, value, attn.scale)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    assert output.shape == expected.shape
    assert calls == {
        "bf16": int(opt_level == 0),
        "all_int8": int(opt_level == 1),
        "qk_int8": int(opt_level == 2),
    }


def test_forward_fa_3d_input_matches_torch_attention(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_ixformer(monkeypatch)
    attn = _make_attn(opt_level=0, monkeypatch=monkeypatch)
    torch.manual_seed(1)
    query_4d = torch.randn(1, 3, attn.num_heads, attn.head_size)
    key_4d = torch.randn(1, 3, attn.num_kv_heads, attn.head_size)
    value_4d = torch.randn(1, 3, attn.num_kv_heads, attn.head_size)

    output = attn._forward_fa(
        query_4d.reshape(1, 3, -1),
        key_4d.reshape(1, 3, -1),
        value_4d.reshape(1, 3, -1),
    )

    expected = _reference_attention(query_4d, key_4d, value_4d, attn.scale).reshape(
        1, 3, -1
    )
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_forward_fa_layer_filter_falls_back_to_bf16(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = _install_fake_ixformer(monkeypatch)
    attn = _make_attn(opt_level=2, block_idx=3, monkeypatch=monkeypatch)
    monkeypatch.setattr(
        ilu_mm.envs,
        "VLLM_ILU_VIT_ATTN_LAYER",
        "4,5",
        raising=False,
    )
    query = torch.randn(1, 2, attn.num_heads, attn.head_size)
    key = torch.randn(1, 2, attn.num_kv_heads, attn.head_size)
    value = torch.randn(1, 2, attn.num_kv_heads, attn.head_size)

    output = attn._forward_fa(query, key, value)

    expected = _reference_attention(query, key, value, attn.scale)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    assert calls == {"bf16": 1, "all_int8": 0, "qk_int8": 0}
