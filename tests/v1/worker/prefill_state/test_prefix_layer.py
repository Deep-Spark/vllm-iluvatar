"""Numerical CPU check of explicit padded-prefix execution (no CUDA kernels)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_iluvatar.models.deepseek_v41 import model as module


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.rows = []

    def forward(self, positions, x, scaling):
        self.rows.append(len(x))
        return x + 3


class Engram(nn.Module):
    def __init__(self, n):
        super().__init__()
        self._rows = torch.ones(n, 2) if n else None
        self._hash_ids = torch.arange(n)
        self._token_mask = None
        self.rows = []

    def forward(self, x, hashes, mask=None):
        self.rows.append(len(x))
        self._rows = None
        return x * 2


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x, ids):
        self.input = x.clone()
        return x + 1


def layer(n):
    result = module.IluDeepseekV41DecoderLayer.__new__(
        module.IluDeepseekV41DecoderLayer
    )
    nn.Module.__init__(result)
    result.attn = Attention()
    result.engram = Engram(n)
    result.ffn = MoE()
    for name in (
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "attn_norm",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "ffn_norm",
    ):
        setattr(result, name, None)
    result._hc_pre_mixes = lambda residual, tag, fn, scale, base, pre, squares, norm: (
        None,
        None,
        residual.mean(1),
        pre,
    )
    return result


@pytest.mark.parametrize("valid", [0, 1, 2, 3])
def test_only_live_rows_enter_prefix_attention_but_all_rows_enter_moe(
    monkeypatch, valid
):
    monkeypatch.setattr(
        module.mhc,
        "hc_post_sqrsum",
        lambda x, residual, post, mix: (residual + x.unsqueeze(1), None),
    )
    monkeypatch.setattr(
        module, "get_forward_context", lambda: SimpleNamespace(additional_kwargs={})
    )
    x = torch.zeros(4, 4, 2)
    x[:valid] = torch.arange(valid * 8).reshape(valid, 4, 2)
    ids = positions = torch.arange(4)
    pre = torch.ones(4, 4)
    if valid:
        reference = layer(valid)
        expected = reference(
            x[:valid].clone(), positions[:valid], ids[:valid], pre[:valid]
        )[0]
    test = layer(valid)
    # Original methods must remain attached: no transient method replacement.
    original_attention = test.attn.forward.__func__
    original_engram = test.engram.forward.__func__
    monkeypatch.setattr(
        module,
        "get_forward_context",
        lambda: SimpleNamespace(additional_kwargs={"prefix_decode_tokens": valid}),
    )
    actual = test(x.clone(), positions, ids, pre)[0]
    assert test.attn.rows == ([valid] if valid else [])
    assert test.engram.rows == ([valid] if valid else [])
    assert len(test.ffn.input) == 4
    assert torch.equal(test.ffn.input[valid:], torch.zeros_like(test.ffn.input[valid:]))
    if valid:
        assert torch.equal(actual[:valid], expected)
    assert test.attn.forward.__func__ is original_attention
    assert test.engram.forward.__func__ is original_engram
