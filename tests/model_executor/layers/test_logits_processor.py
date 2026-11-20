# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage for CQ vocabulary-parallel greedy selection."""

from types import SimpleNamespace

import pytest
import torch
from conftest import requires_corex30
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_iluvatar.model_executor.layers.logits_processor import IluLogitsProcessor

pytestmark = requires_corex30


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (2, 1023),
        (3, 1024),
        (4, 1025),
        (8, 77440),
        (12, 77440),
        (16, 77440),
        (32, 154880),
    ],
)
@pytest.mark.parametrize("kind", ["random", "ties", "masked", "all_inf"])
def test_local_max(default_vllm_config, dtype, shape, kind):
    processor = LogitsProcessor(shape[1])
    assert isinstance(processor, IluLogitsProcessor)
    x = torch.randn(shape[0], shape[1] + 7, device="cuda", dtype=dtype)[:, : shape[1]]
    if kind == "ties":
        x.zero_()
        x[:, 0] = x[:, -1] = 5
    elif kind == "masked":
        x[:, ::3] = -torch.inf
        x[:, -1] = 1
    elif kind == "all_inf":
        x.fill_(-torch.inf)
    before = x.clone()
    rng = torch.cuda.get_rng_state()
    values, indices = processor._local_max(x)
    reference = x.max(-1)
    assert torch.equal(indices, reference.indices)
    assert torch.equal(values, reference.values)
    assert torch.equal(x, before)
    assert torch.equal(torch.cuda.get_rng_state(), rng)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("warm_metadata", [False, True])
def test_graph_replay(default_vllm_config, dtype, warm_metadata):
    x = torch.randn(12, 77440, device="cuda", dtype=dtype)
    warmup = LogitsProcessor(x.shape[1])
    warmup._local_max(x)
    processor = warmup if warm_metadata else LogitsProcessor(x.shape[1])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, indices = processor._local_max(x)
    for winner in [0, 1024, 77439]:
        x.zero_()
        x[:, winner] = 5
        graph.replay()
        expected = x.max(-1)
        assert torch.equal(indices, expected.indices)
        assert torch.equal(values, expected.values)


class PrecomputedLogits(IluLogitsProcessor):
    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        return hidden_states.clone()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("scale,soft_cap", [(1.0, None), (0.5, None), (2.0, 3.0)])
def test_top_tokens_padding_and_transform(default_vllm_config, dtype, scale, soft_cap):
    x = torch.randn(8, 1031, device="cuda", dtype=dtype)
    x[:, -7:] = 100
    processor = PrecomputedLogits(1024, scale=scale, soft_cap=soft_cap)
    head = SimpleNamespace(
        tp_size=1,
        shard_indices=SimpleNamespace(
            num_org_vocab_padding=7, org_vocab_start_index=77440
        ),
    )
    expected = x[:, :-7]
    if soft_cap is not None:
        expected = torch.tanh(expected / soft_cap) * soft_cap
    expected = (expected * scale).argmax(-1) + 77440
    assert torch.equal(processor.get_top_tokens(head, x), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_strided_vocabulary(default_vllm_config, dtype):
    x = torch.randn(4, 2050, device="cuda", dtype=dtype)[:, ::2]
    processor = LogitsProcessor(x.shape[1])
    values, indices = processor._local_max(x)
    expected = x.max(-1)
    assert torch.equal(indices, expected.indices)
    assert torch.equal(values, expected.values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1), (4, 1025), (16, 77440), (96, 77440)])
@pytest.mark.parametrize("kind", ["random", "ties", "all_inf"])
def test_packed_shard_candidate(default_vllm_config, dtype, shape, kind):
    from vllm_iluvatar.model_executor.layers.greedy_tp import greedy_local_pair

    processor = LogitsProcessor(shape[1])
    x = torch.randn(shape[0], shape[1] + 7, device="cuda", dtype=dtype)[:, : shape[1]]
    if kind == "ties":
        x.zero_()
        x[:, 0] = x[:, -1] = 5
    elif kind == "all_inf":
        x.fill_(-torch.inf)
    reference = x.cpu().float().max(-1)
    actual = greedy_local_pair(x, processor._get_greedy_metadata(x), 77440)
    expected = torch.stack((reference.values, (reference.indices + 77440).float()), -1)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("tp", [2, 4, 8, 16])
@pytest.mark.parametrize("kind", ["random", "ties", "all_inf", "nan"])
def test_gathered_candidate_winner(tp, kind):
    from vllm_iluvatar.model_executor.layers.greedy_tp import greedy_rank_winner

    pairs = torch.randn(16, tp, 2)
    pairs[:, :, 1] = torch.randint(0, 154880, (16, tp))
    if kind == "ties":
        pairs[:, :, 0] = 1
    elif kind == "all_inf":
        pairs[:, :, 0] = -torch.inf
    elif kind == "nan":
        pairs[:, -1, 0] = float("nan")
    winner = pairs[:, :, 0].argmax(-1, keepdim=True)
    expected = pairs[:, :, 1].gather(-1, winner).squeeze(-1).long()
    actual = greedy_rank_winner(pairs.cuda())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_packed_candidate_graph_updates(default_vllm_config):
    from vllm_iluvatar.model_executor.layers.greedy_tp import (
        greedy_local_pair,
        greedy_rank_winner,
    )

    x = torch.randn(16, 1025, device="cuda")
    processor = LogitsProcessor(x.shape[1])
    metadata = processor._get_greedy_metadata(x)
    pair = greedy_local_pair(x, metadata, 77440)
    greedy_rank_winner(pair[:, None, :].expand(-1, 2, -1).contiguous())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pair = greedy_local_pair(x, metadata, 77440)
        actual = greedy_rank_winner(torch.stack((pair, pair), dim=1))
    for index in [0, 1024, 513]:
        x.zero_()
        x[:, index] = 5
        graph.replay()
        torch.testing.assert_close(
            actual.cpu(), torch.full((16,), 77440 + index), rtol=0, atol=0
        )
