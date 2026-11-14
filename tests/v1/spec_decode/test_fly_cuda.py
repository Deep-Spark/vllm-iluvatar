# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_iluvatar.v1.spec_decode.fly import compute_fly_entropy

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for entropy kernel tests", allow_module_level=True)


def _reference(values, top_k, from_logits):
    values = values.float()
    selected = values.topk(min(top_k, values.shape[-1]), dim=-1).values
    if from_logits:
        log_probabilities = selected - values.logsumexp(-1, keepdim=True)
        probabilities = log_probabilities.exp()
    else:
        probabilities = selected
        log_probabilities = selected.log()
    terms = torch.where(probabilities > 0, probabilities * log_probabilities, 0.0)
    return -terms.sum(-1)


@pytest.mark.parametrize("vocab_size", [4097, 8193])
def test_entropy_odd_tail_does_not_duplicate_the_last_candidate(vocab_size):
    probabilities = torch.zeros(2, vocab_size, device="cuda")
    probabilities[:, 0] = 0.4
    probabilities[:, -1] = 0.6

    actual = compute_fly_entropy(probabilities, 3)

    torch.testing.assert_close(actual, _reference(probabilities, 3, False))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("from_logits", [False, True])
def test_entropy_strided_input_matches_torch_without_mutation(dtype, from_logits):
    generator = torch.Generator().manual_seed(17)
    owner = torch.randn(4, 16400, generator=generator).to(dtype).cuda()
    values = owner[1:4, 3:16389:2]
    if not from_logits:
        values.copy_(values.float().softmax(-1))
    before = owner.clone()

    actual = compute_fly_entropy(values, 3, from_logits=from_logits)
    expected = _reference(values, 3, from_logits)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(owner, before, rtol=0, atol=0)
    if dtype == torch.float64:
        # The unsupported kernel dtype must retain exact FP32 Torch fallback.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_entropy_graph_replay_uses_changed_inputs():
    values = torch.zeros(3, 8193, device="cuda")
    compute_fly_entropy(values, 3, from_logits=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = compute_fly_entropy(values, 3, from_logits=True)
    torch.cuda.current_stream().wait_stream(stream)

    for finite_columns in (1, 3, 17):
        values.fill_(-float("inf"))
        values[:, :finite_columns] = 0
        graph.replay()
        expected = _reference(values, 3, True)
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=1e-5)
