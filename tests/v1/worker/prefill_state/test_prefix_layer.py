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


@pytest.mark.parametrize("rows", [None, (), (0, 2, 5), tuple(range(6))])
def test_tail_ffn_preserves_inputs_and_scopes_selected_padding(rows):
    from vllm.forward_context import ForwardContext, override_forward_context

    context = ForwardContext(
        no_compile_layers={}, attn_metadata={}, slot_mapping={},
        is_padding=torch.tensor([False, True, False, True, False, True]),
    )
    seen = []

    class InspectMoE(nn.Module):
        def forward(self, x, ids):
            current = module.get_forward_context()
            current.moe_layer_index += 1
            seen.append((x.clone(), ids.clone(), current.is_padding.clone()))
            return x + ids[:, None]

    block = layer(0)
    block.ffn = InspectMoE()
    h = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    original = h.clone()
    ids = torch.arange(6)
    kept = tuple(range(6)) if rows is None else rows
    with override_forward_context(context):
        result = block._forward_tail_ffn(h, ids, rows)
        assert module.get_forward_context() is context
    expected = torch.zeros_like(h)
    if kept:
        index = torch.tensor(kept)
        expected[index] = h[index] + ids[index, None]
        assert len(seen) == 1
        torch.testing.assert_close(seen[0][0], h[index])
        torch.testing.assert_close(seen[0][1], ids[index])
        torch.testing.assert_close(seen[0][2], context.is_padding[index])
    else:
        assert not seen
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(h, original)
    assert context.moe_layer_index == bool(kept)


def test_tail_ffn_restores_context_and_inputs_on_exception():
    from vllm.forward_context import ForwardContext, override_forward_context

    class FailingMoE(nn.Module):
        def forward(self, x, ids):
            assert len(module.get_forward_context().is_padding) == 2
            raise ValueError("synthetic FFN failure")

    block = layer(0)
    block.ffn = FailingMoE()
    context = ForwardContext(
        no_compile_layers={}, attn_metadata={}, slot_mapping={},
        is_padding=torch.zeros(6, dtype=torch.bool),
    )
    h = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    original = h.clone()
    with override_forward_context(context):
        with pytest.raises(ValueError, match="synthetic FFN failure"):
            block._forward_tail_ffn(h, torch.arange(6), (1, 5))
        assert module.get_forward_context() is context
    torch.testing.assert_close(h, original)
