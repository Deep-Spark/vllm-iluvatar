"""Guard draft weight ownership and non-causal SWA replay semantics."""

from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.models.deepseek_v41.attention import IluDeepseekV41Attention
from vllm_iluvatar.models.deepseek_v41.dspark import remap_dspark_name


@pytest.mark.parametrize(
    "name,expected",
    [
        ("layers.0.attn.wkv.weight", None),
        ("mtp.3.attn.wkv.weight", None),
        ("mtp.2.confidence_head.weight", None),
        ("mtp.0.ffn.gate.bias_vl", None),
        ("mtp.0.main_proj.weight", "model.main_proj.weight"),
        ("mtp.2.norm.weight", "model.norm.weight"),
        ("mtp.2.markov_head.embed.weight", "model.markov_head.markov_w1.weight"),
        ("mtp.2.markov_head.head.weight", "model.markov_head.markov_w2.weight"),
        ("mtp.1.attn.wkv.scale", "model.layers.1.attn.wkv.scale"),
    ],
)
def test_draft_weight_ownership(name, expected):
    assert remap_dspark_name(name) == expected


def test_noncausal_block_includes_future_queries_and_replays_live_lengths():
    a = object.__new__(IluDeepseekV41Attention)
    a.window_size = 128
    a._window_idx = torch.full((10, 256), -1, dtype=torch.int32, device="cuda")
    a._window_lens = torch.zeros(10, dtype=torch.int32, device="cuda")
    a._window_offsets = torch.arange(128, device="cuda")
    a._draft_window_offsets = torch.arange(256, device="cuda")
    pos = torch.arange(10, device="cuda") + 65536
    table = torch.arange(1, 2053, device="cuda", dtype=torch.int32).view(2, 1026)
    valid = torch.ones(10, device="cuda", dtype=torch.bool)
    valid[-1] = False
    md = SimpleNamespace(
        causal=False,
        seq_lens=torch.tensor([65541, 65541], device="cuda", dtype=torch.int32),
        query_start_loc=torch.tensor([0, 5, 10], device="cuda", dtype=torch.int32),
        token_to_req_indices=torch.arange(10, device="cuda", dtype=torch.int32) // 5,
        is_valid_token=valid,
        block_table=table,
        block_size=64,
    )

    def run():
        return a._window_indices(md, 0, 10, pos)

    out, lens = run()
    assert torch.equal(out[0], out[4])
    assert lens[0] == 133
    assert lens[-1] == 0
    logical = torch.arange(65536 - 128, 65541, device="cuda")
    expected = table[0, logical // 64] * 64 + logical % 64
    assert torch.equal(out[0, :133], expected)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out, lens = run()
    md.seq_lens.add_(64)
    g.replay()
    torch.cuda.synchronize()
    logical.add_(64)
    expected = table[0, logical // 64] * 64 + logical % 64
    assert torch.equal(out[0, :133], expected)
    assert torch.all(out[-1] == -1)


def test_dp_padding_insert_slots_are_invalid_and_replay_live_values():
    a = object.__new__(IluDeepseekV41Attention)
    a._insert_slots = torch.full((10,), -1, dtype=torch.int64, device="cuda")
    slots = torch.arange(64, 69, dtype=torch.int64, device="cuda")
    out = a._padded_insert_slots(slots, 10)
    assert torch.equal(out[:5], slots)
    assert torch.all(out[5:] == -1)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = a._padded_insert_slots(slots, 10)
    slots.add_(64)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out[:5], slots)
    assert torch.all(out[5:] == -1)


def test_draft_decode_dispatches_noncausal_indices(monkeypatch):
    a = object.__new__(IluDeepseekV41Attention)
    indices, lens = object(), object()
    seen = []
    monkeypatch.setattr(
        IluDeepseekV41Attention, "_window_indices", lambda *args: (indices, lens)
    )
    monkeypatch.setattr(
        IluDeepseekV41Attention,
        "_sparse_mla",
        lambda self, q, idx, ln, *args: seen.append((idx, ln)),
    )
    md = SimpleNamespace(causal=False, num_decodes=1, num_decode_tokens=5)
    a._forward_decode(None, None, None, md, None, True, None)
    assert seen == [(indices, lens)]
