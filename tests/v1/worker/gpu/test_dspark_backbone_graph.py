# SPDX-License-Identifier: Apache-2.0
"""Backbone replay must update hidden states while sampling stays eager."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_iluvatar import envs as ilu_envs
from vllm_iluvatar.v1.worker.gpu.spec_decode.dspark import (
    make_init_cudagraph_manager,
)
from vllm_iluvatar.v1.worker.gpu.spec_decode.dspark_backbone_graph import (
    DSparkBackboneGraph,
)


def test_registered_default_keeps_independent_draft_eager(monkeypatch):
    ilu_envs.apply()
    monkeypatch.delenv("VLLM_ILU_DSPARK_BACKBONE_GRAPH", raising=False)
    modes = []
    initialize = make_init_cudagraph_manager(lambda spec, mode: modes.append(mode))
    spec = SimpleNamespace(speculative_config=SimpleNamespace(enforce_eager=True))
    initialize(spec, CUDAGraphMode.FULL_DECODE_ONLY)
    assert modes == [CUDAGraphMode.NONE]


def _speculator(device):
    inputs = torch.arange(60, dtype=torch.float32, device=device).view(15, 4)
    cache = torch.zeros_like(inputs)
    sampled = []
    spec = SimpleNamespace(num_query_per_req=5)

    def model(tokens, *args):
        result = inputs[:tokens] * 2 + 1
        cache[:tokens].copy_(result)
        return result

    def sample(requests, hidden):
        if device == "cuda":
            assert not torch.cuda.is_current_stream_capturing()
        sampled.append((requests, hidden[: requests * 5].clone()))

    def generate(requests, tokens, *args, **kwargs):
        sample(requests, model(tokens))

    manager = SimpleNamespace(run_fullgraph=lambda desc: None)
    spec._run_model = model
    spec._sample_sequential = sample
    spec._generate_draft = generate
    spec.capture_impl = lambda: spec._generate_draft(3, 15, None, None)
    spec.capture = lambda: spec.capture_impl()
    spec.propose = lambda input_batch, desc: manager.run_fullgraph(desc)
    return spec, manager, inputs, cache, sampled


def test_capture_and_failed_proposal_do_not_leave_stale_scope():
    spec, manager, _, _, sampled = _speculator("cpu")
    state = DSparkBackboneGraph(spec)
    state.bind_manager(manager)
    spec.capture()
    assert not sampled
    assert not state.capturing
    with pytest.raises(KeyError):
        spec.propose(SimpleNamespace(num_reqs=1), SimpleNamespace(num_tokens=5))
    assert state.num_reqs is None
    spec._generate_draft(1, 5, None, None)
    assert sampled[-1][0] == 1


def test_failed_capture_and_manager_recreation_discard_stale_outputs():
    spec, manager, _, _, _ = _speculator("cpu")
    state = DSparkBackboneGraph(spec)
    state.bind_manager(manager)
    spec.capture()
    fresh_manager = SimpleNamespace(run_fullgraph=lambda desc: None)
    state.bind_manager(fresh_manager)
    assert not state.outputs

    def fail():
        raise ValueError("capture failed")

    spec.capture_impl = fail
    with pytest.raises(ValueError, match="capture failed"):
        spec.capture()
    assert not state.capturing


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_replay_uses_new_inputs_and_samples_only_local_rows():
    spec, manager, inputs, cache, sampled = _speculator("cuda")
    state = DSparkBackboneGraph(spec)
    graphs = {}

    def capture():
        for tokens in (5, 15):
            for _ in range(3):
                spec._generate_draft(tokens // 5, tokens, None, None)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                spec._generate_draft(tokens // 5, tokens, None, None)
            graphs[tokens] = graph

    spec.capture_impl = capture
    manager.run_fullgraph = lambda desc: graphs[desc.num_tokens].replay()
    state.bind_manager(manager)
    spec.capture()
    assert not sampled
    for turn, (tokens, local_reqs) in enumerate(((15, 1), (5, 1), (15, 3))):
        inputs.add_(turn + 1)
        cache.fill_(-1)
        expected = inputs[:tokens] * 2 + 1
        spec.propose(
            SimpleNamespace(num_reqs=local_reqs), SimpleNamespace(num_tokens=tokens)
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(cache[:tokens], expected, rtol=0, atol=0)
        requests, hidden = sampled[-1]
        assert requests == local_reqs
        torch.testing.assert_close(hidden, expected[: local_reqs * 5], rtol=0, atol=0)
        assert state.num_reqs is None
