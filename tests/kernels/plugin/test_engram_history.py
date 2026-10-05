# SPDX-License-Identifier: Apache-2.0
"""Engram history must survive eviction of SWA pages before prefix replay."""

import pytest
import torch
import vllm_iluvatar._C  # noqa: F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def test_original_history_with_evicted_prefix_and_graph_replay():
    # Three requests begin at different positions, including the sequence start.
    # The second request replays a prefix whose preceding SWA page was evicted.
    prompts = [[3, 5, 7, 9], [1, 4, 6, 8, 10, 12, 14], [2, 11, 13, 15]]
    starts = [0, 4, 2]
    lengths = [3, 3, 2]
    token_map = [19 - i for i in range(20)]
    multipliers = [3, 5, 7, 11]
    primes = [[13, 17], [19, 23], [29, 31]]
    offsets = [0, 20, 40, 70, 100, 140]
    pad_id = 2

    def reference(sequences):
        expected = []
        for seq, start, length in zip(sequences, starts, lengths):
            for pos in range(start, start + length):
                # Build each n-gram from the complete original sequence, without
                # consulting request/chunk mappings or any page-cache contents.
                row = []
                for size, head_primes in enumerate(primes, start=2):
                    value = 0
                    blocked = False
                    for shift in range(size):
                        raw = seq[pos - shift] if pos >= shift else -1
                        blocked |= raw < 0
                        compressed = pad_id if blocked else token_map[raw]
                        value ^= compressed * multipliers[shift]
                    row.extend(
                        value % prime + offsets[(size - 2) * 2 + head]
                        for head, prime in enumerate(head_primes)
                    )
                expected.append(row)
        return torch.tensor(expected, dtype=torch.int64, device="cuda")

    def gpu(values, dtype=torch.int64):
        return torch.tensor(values, dtype=dtype, device="cuda")

    ids = gpu([seq[p] for seq, start, length in zip(prompts, starts, lengths)
               for p in range(start, start + length)])
    positions = gpu([p for start, length in zip(starts, lengths)
                     for p in range(start, start + length)])
    cache = torch.full((65,), -1, dtype=torch.int64, device="cuda")
    # Block 0 means not resident. All prefix pages deliberately remain absent.
    block_table = gpu([[0, 0]] * 3, torch.int32)
    reqs = gpu([req for req, length in enumerate(lengths) for _ in range(length)])
    qsl = gpu([0, 3, 6, 8], torch.int32)
    lookback = gpu([[seq[start - 1 - j] if start > j else -1 for j in range(3)]
                    for seq, start in zip(prompts, starts)], torch.int32)
    args = (ids, positions, gpu(token_map), cache, block_table, reqs, qsl,
            gpu(multipliers), gpu(primes), gpu(offsets), pad_id, 64, 4, 3, True)

    def run():
        return torch.ops._C.engram_ngram_hash(*args, lookback)

    expected = reference(prompts)
    torch.testing.assert_close(run(), expected, rtol=0, atol=0)
    # Legacy callers still work, but cannot recover history from evicted pages.
    assert not torch.equal(torch.ops._C.engram_ngram_hash(*args), expected)
    # With resident pages, the optional window agrees with the legacy API.
    block_table[1, 0] = 1
    block_table[2, 0] = 2
    for req in [1, 2]:
        for pos in range(starts[req]):
            cache[(req * 4) + pos] = token_map[prompts[req][pos]]
    torch.testing.assert_close(
        torch.ops._C.engram_ngram_hash(*args), expected, rtol=0, atol=0
    )
    block_table.zero_()

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for value in range(20):
        cache.fill_(value)
        graph.replay()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    # Reuse the captured graph for a changed request history. It must read the
    # live runner buffer rather than values fixed during capture.
    prompts[1][3] = 16
    lookback[1, 0] = 16
    graph.replay()
    torch.testing.assert_close(result, reference(prompts), rtol=0, atol=0)
    # A missing history ID pads that and all older members of its n-gram.
    prompts[1][3] = -1
    lookback[1, 0] = -1
    graph.replay()
    torch.testing.assert_close(result, reference(prompts), rtol=0, atol=0)
