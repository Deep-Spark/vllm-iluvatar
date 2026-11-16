from inspect import signature

import pytest
import torch
from torch import nn

from vllm_iluvatar.models.deepseek_v41 import engram as engram_mod
from vllm_iluvatar.models.deepseek_v41 import mhc
from vllm_iluvatar.models.deepseek_v41.model import IluDeepseekV41DecoderLayer


@pytest.mark.parametrize("abi", ["keyword", "legacy", "opaque"])
def test_mhc_pre_resolves_fuse_reductions_once(abi, monkeypatch):
    inspected = []

    def legacy(x):
        return x, None

    def modern(x, *, fuse_reductions):
        return x, fuse_reductions

    def op_signature(op):
        inspected.append(op)
        if abi == "opaque":
            raise ValueError("signature unavailable")
        return signature(op)

    monkeypatch.setattr(mhc, "signature", op_signature)
    op = modern if abi == "keyword" else legacy
    call = mhc._bind_mhc_pre(op)
    assert (call is op) == (abi == "keyword")
    for enabled in (True, False):
        expected = enabled if abi == "keyword" else None
        assert call("input", fuse_reductions=enabled) == ("input", expected)
    assert inspected == [op]


@pytest.mark.parametrize("layer_id", [1, 14])
def test_engram_apply_sqrsum_restarts_streaming_pre(layer_id, monkeypatch):
    layer = object.__new__(IluDeepseekV41DecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_id = layer_id
    layer.hc_mult = 4
    layer.rms_norm_eps = layer.hc_eps = 1e-6
    layer.hc_sinkhorn_iters = 20
    layer.hc_post_alpha = 1.0
    layer._hc_fn_io_cache = {}
    layer._fuse_attn_allreduce_post = False
    for stage in ("attn", "ffn"):
        for suffix in ("fn", "scale", "base"):
            setattr(layer, f"hc_{stage}_{suffix}", torch.ones(1))

    class Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(8))
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x

    layer.attn_norm = Norm()
    layer.ffn_norm = Norm()

    class Engram(nn.Module):
        def __init__(self):
            super().__init__()
            self._rows = torch.ones(2, 24, 256)
            self._hash_ids = torch.zeros(2, 24, dtype=torch.int32)
            self._token_mask = torch.ones(2, dtype=torch.bool)

        def forward(self, residual, hashes, mask):
            residual = residual + 3
            return residual, residual.float().square().sum((-1, -2), keepdim=False)[
                :, None
            ]

    layer.engram = Engram()
    layer.attn = lambda positions, h, unused: h
    layer.attn.fused_wqa_wkv = nn.Linear(8, 8, bias=False)
    layer.ffn = lambda h, input_ids: h
    calls = []

    def post(x, residual, post_mix, res_mix):
        updated = residual + x[:, None, :]
        return updated, updated.float().square().sum((-1, -2))

    def pre(x, *args, **kwargs):
        calls.append((x.clone(), kwargs["sqrsum"], kwargs["layer_norm_weight"]))
        return None, None, x.mean(1), torch.ones(2, 4), None, None

    monkeypatch.setattr(mhc, "hc_post_sqrsum", post)
    monkeypatch.setattr(mhc, "hc_pre_mixes", pre)
    layer(
        torch.ones(2, 8),
        torch.arange(2),
        None,
        torch.ones(2, 4),
        None,
        None,
        torch.ones(2, 4, 8),
    )
    assert torch.equal(calls[0][0], torch.full((2, 4, 8), 5.0))
    torch.testing.assert_close(calls[0][1], torch.full((2, 1), 800.0))
    assert calls[0][2] is layer.attn_norm.weight
    assert layer.attn_norm.calls == 0
    assert calls[1][1] is not None and calls[1][2] is layer.ffn_norm.weight
    assert layer.ffn_norm.calls == 0


@pytest.mark.parametrize(("tokens", "expect_sqrsum"), [(128, True), (512, False)])
def test_engram_sqrsum_apply_dispatch(tokens, expect_sqrsum, monkeypatch):
    embedding = object.__new__(engram_mod.IluEngram)
    nn.Module.__init__(embedding)
    embedding._qk_weight = torch.ones(4, 8)
    embedding._shard_keys = True
    embedding.key_offset = 0
    embedding.eps = 1e-6
    embedding.clamp_value = 1e-8

    x = torch.zeros(tokens, 4, 8)
    kv = torch.zeros(tokens, 4 * 4 + 8)
    partials = torch.ones(tokens, 4, 2)
    calls = []
    monkeypatch.setattr(engram_mod.ixfops, "engram_gate_reduce", lambda *args: partials)
    monkeypatch.setattr(
        engram_mod, "tensor_model_parallel_all_reduce", lambda value: value
    )
    monkeypatch.setattr(
        engram_mod.ixfops,
        "engram_gate_apply_with_sqrsum",
        lambda *args: (x + 1, torch.ones(tokens, 1)),
    )
    monkeypatch.setattr(
        engram_mod.ixfops,
        "engram_gate_apply",
        lambda *args: calls.append("fallback") or x + 2,
    )

    out, sqrsum = embedding._gate(x, kv)
    assert (sqrsum is not None) is expect_sqrsum
    assert ("fallback" not in calls) is expect_sqrsum
    assert torch.all(out == (1 if expect_sqrsum else 2))
