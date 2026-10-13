# SPDX-License-Identifier: Apache-2.0
"""Causal decode spans: compressed floor, padding and current-stream replay."""
import pytest
import torch
import vllm_iluvatar._C  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def reference(lengths, query_bounds, real_tokens, padded_tokens, ratio):
    starts, ends = [0] * padded_tokens, [0] * padded_tokens
    prefix = 0
    for request, length in enumerate(lengths):
        begin, end = query_bounds[request:request + 2]
        for token in range(begin, min(end, real_tokens)):
            starts[token] = prefix
            ends[token] = prefix + (length - (end - begin) + token - begin + 1) // ratio
        prefix += length // ratio
    return starts, ends


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("lengths,bounds,real,padded", [
    ([1], [0, 6], 6, 6),
    ([0, 1], [0, 6, 12], 12, 12),
    ([511, 512], [0, 6, 12], 12, 12),
    ([1048582, 1048569], [0, 6, 12], 12, 12),
    ([32769, 0], [0, 6, 6], 6, 12),
    ([17, 513], [0, 0, 6], 6, 12),
    ([0, 0], [0, 0, 0], 0, 12),
])
def test_decode_spans_exact_floor_and_padded_canaries(ratio, lengths, bounds, real, padded):
    sequence = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    query = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    raw = torch.full((2, padded + 8), -12345, dtype=torch.int32, device="cuda")
    starts, ends = raw[0, 4:-4], raw[1, 4:-4]
    torch.ops._C.v41_indexer_decode_spans(
        sequence, query, starts, ends, len(lengths), real, padded, ratio)
    expected = reference(lengths, bounds, real, padded, ratio)
    assert starts.tolist() == expected[0]
    assert ends.tolist() == expected[1]
    assert bool((raw[:, :4] == -12345).all())
    assert bool((raw[:, -4:] == -12345).all())


@pytest.mark.parametrize("ratio", [1, 2])
def test_decode_spans_changing_inputs_on_nondefault_graph_stream(ratio):
    sequence = torch.zeros(2, dtype=torch.int32, device="cuda")
    query = torch.tensor([0, 6, 12], dtype=torch.int32, device="cuda")
    starts = torch.empty(12, dtype=torch.int32, device="cuda")
    ends, valid = torch.empty_like(starts), torch.empty_like(starts)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.ops._C.v41_indexer_decode_spans(
            sequence, query, starts, ends, 2, 12, 12, ratio)
        torch.sub(ends, starts, out=valid)
    for lengths, bounds in (
        ([1048582, 200001], [0, 6, 12]),
        ([1, 17], [0, 6, 12]),
        ([17, 513], [0, 0, 6]),
        ([0, 0], [0, 0, 0]),
        ([1048582, 200001], [0, 6, 12]),
    ):
        with torch.cuda.stream(stream):
            sequence.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
            query.copy_(torch.tensor(bounds, dtype=torch.int32, device="cuda"))
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        expected = reference(lengths, bounds, 12, 12, ratio)
        assert starts.tolist() == expected[0]
        assert ends.tolist() == expected[1]
        assert valid.tolist() == [end - start for start, end in zip(*expected)]


def test_decode_spans_rejects_overlapping_outputs():
    sequence = torch.tensor([10], dtype=torch.int32, device="cuda")
    query = torch.tensor([0, 6], dtype=torch.int32, device="cuda")
    raw = torch.empty(8, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="must not overlap"):
        torch.ops._C.v41_indexer_decode_spans(sequence, query, raw[:6], raw[2:8], 1, 6, 6, 1)
