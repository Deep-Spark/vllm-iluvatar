# SPDX-License-Identifier: Apache-2.0
"""Engram history must survive eviction of SWA pages before prefix replay."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import vllm_iluvatar._C  # noqa: F401

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm_iluvatar.models.deepseek_v41.engram import NgramHashState

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
    # V2's missing history is authoritative even when stale cache is resident.
    block_table[1, 0] = 1
    cache[7] = token_map[8]
    graph.replay()
    torch.testing.assert_close(result, reference(prompts), rtol=0, atol=0)


@pytest.mark.parametrize("prompt_only", [True, False], ids=["v1", "v2"])
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
@torch.inference_mode()
def test_runner_history_across_continuous_decode(prompt_only, use_graph):
    # Different prompt lengths cross their generation boundary in the same
    # batch. Later steps cross physical pages and finish with multi-token decode.
    sequences = [list(range(1, 17)), list(range(17, 33))]
    prompt_lengths = np.array([4, 6])
    token_map = [67 - i for i in range(40)]
    multipliers = [3, 5, 7, 11]
    primes = [[13, 17], [19, 23], [29, 31]]
    offsets = [0, 13, 30, 49, 72, 101]
    pad_id = 2
    state = NgramHashState(
        SimpleNamespace(max_ngram_size=4, primes=(primes,)),
        SimpleNamespace(block_size=4, kv_cache=torch.empty(9, 4, 1, 8, device="cuda")),
        token_map, torch.tensor([multipliers]), pad_id, "cuda",
        lookback_is_prompt_only=prompt_only,
    )
    assert state.ensure_cache()
    # Deliberately permute physical pages to detect logical/physical confusion.
    bt_cpu = [[4, 1, 3, 2], [8, 6, 5, 7]]
    bt = torch.tensor(bt_cpu, dtype=torch.int32, device="cuda")
    starts = np.array([3, 5])
    # Execute the actual V1 gather method: CPU generated IDs are placeholders.
    cpu_ids = np.zeros((2, 16), dtype=np.int32)
    for req, count in enumerate(prompt_lengths):
        cpu_ids[req, :count] = sequences[req][:count]
    window_cpu = np.full((2, 3), -1, dtype=np.int32)
    window = torch.empty((2, 3), dtype=torch.int32, device="cuda")
    def copy_window():
        window.copy_(torch.from_numpy(window_cpu))
        return window
    runner = SimpleNamespace(
        lookback_token_ids=SimpleNamespace(np=window_cpu, copy_to_gpu=copy_window),
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu=starts,
            num_prompt_tokens=prompt_lengths,
            token_ids_cpu=cpu_ids,
        ),
    )

    def reference(req, pos):
        row = []
        for size, head_primes in enumerate(primes, start=2):
            value = 0
            for shift in range(size):
                value ^= token_map[sequences[req][pos - shift]] * multipliers[shift]
            row.extend(value % prime + offsets[(size - 2) * 2 + head]
                       for head, prime in enumerate(head_primes))
        return row

    def slot(req, pos):
        return bt_cpu[req][pos // 4] * 4 + pos % 4

    state._cache.fill_(-1)
    # Seed the first chunk's history. Poison prompt entries: runner-provided
    # prompt history must take priority over any stale paged-cache content.
    for req, start in enumerate(starts):
        for pos in range(start):
            state._cache[slot(req, pos)] = 999

    for step in range(9):
        count = 3 if step == 8 else 1
        positions_cpu = [int(start) + j for start in starts for j in range(count)]
        ids = torch.tensor([sequences[req][pos] for req, start in enumerate(starts)
                            for pos in range(int(start), int(start) + count)], device="cuda")
        positions = torch.tensor(positions_cpu, device="cuda")
        reqs = torch.arange(2, device="cuda").repeat_interleave(count)
        qsl = torch.tensor([0, count, count * 2], dtype=torch.int32, device="cuda")
        slots = torch.tensor([slot(req, int(start) + j) for req, start in enumerate(starts)
                              for j in range(count)], device="cuda")
        if prompt_only:
            assert GPUModelRunner._prepare_lookback_token_ids(runner, 2) is window
        else:
            window.copy_(torch.tensor([[sequences[req][int(start) - 1 - j]
                                       for j in range(3)]
                                      for req, start in enumerate(starts)], dtype=torch.int32))
        expected = torch.tensor([reference(req, int(start) + j)
                                 for req, start in enumerate(starts) for j in range(count)],
                                device="cuda")[:, None, :]
        def run(ids=ids, positions=positions, qsl=qsl, slots=slots, reqs=reqs):
            return state(ids, positions, qsl, None, window, None, slots, bt, reqs, 2)
        result = run()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        if use_graph and (step == 0 or count > 1):
            # Hold every external captured tensor, including request metadata.
            # CUDA graphs retain device pointers, not Python tensor references.
            graph_inputs = (ids, positions, qsl, slots, reqs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_result = run()
        if use_graph:
            if step > 0 and count == 1:
                # Captured tensors remain fixed; refill their contents so graph
                # replay sees live input IDs, positions, slots and history.
                graph_inputs[0].copy_(ids)
                graph_inputs[1].copy_(positions)
                graph_inputs[3].copy_(slots)
            graph.replay()
            torch.testing.assert_close(graph_result, expected, rtol=0, atol=0,
                                       msg=f"step={step}")
        starts += count


@pytest.mark.parametrize("resident", [False, True])
def test_v1_unavailable_generated_history_stays_padding(resident):
    # A V1 fallback must still reject an absent page or a dead cache entry.
    def gpu(values, dtype=torch.int64):
        return torch.tensor(values, dtype=dtype, device="cuda")
    args = (
        gpu([13]), gpu([5]), gpu(list(range(20))), gpu([-1] * 17),
        gpu([[1 if resident else 0]], torch.int32), gpu([0]),
        gpu([0, 1], torch.int32), gpu([3, 5, 7, 11]),
        gpu([[13], [19], [29]]), gpu([0, 13, 32]), 2, 16, 8, 1, True,
        gpu([[-1, 9, 7]], torch.int32),
    )
    result = torch.ops._C.engram_ngram_hash(*args, lookback_is_prompt_only=True)
    expected = []
    for size, prime, offset in [(2, 13, 0), (3, 19, 13), (4, 29, 32)]:
        rolling = 13 * 3
        for multiplier in [5, 7, 11][:size - 1]:
            rolling ^= 2 * multiplier
        expected.append(rolling % prime + offset)
    torch.testing.assert_close(result, gpu([expected]), rtol=0, atol=0)
