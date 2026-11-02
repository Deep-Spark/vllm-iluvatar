# SPDX-License-Identifier: Apache-2.0
"""GPU regressions for budget forcing across speculative token boundaries."""

import pytest
import torch

from vllm_iluvatar.v1.worker.gpu.thinking_budget import apply_thinking_budget

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")

START = [10, 11]
NATURAL_END = [20, 21]
FORCED_END = [15, 20, 21]


def _last(sequence, marker):
    return max(
        (i for i in range(len(sequence) - len(marker) + 1)
         if sequence[i:i + len(marker)] == marker),
        default=-1,
    )


def _expected(history, budget):
    start = _last(history, START)
    end = _last(history, NATURAL_END)
    if budget < 0 or start < 0 or start <= end:
        return None
    if len(history) - start - len(START) < budget:
        return None
    prefix = max(
        (n for n in range(1, len(FORCED_END))
         if history[-n:] == FORCED_END[:n]),
        default=0,
    )
    return FORCED_END[prefix]


@pytest.mark.parametrize(
    "history,draft,budget",
    [
        (START + [40], [41, 42, 43, 44, 45], 512),
        (START, [15, 20, 21, 40, 41], 0),
        (START + list(range(40, 48)), [15, 20, 21, 40, 41], 8),
        ([40, 10], [11, 15, 20, 21, 41], 0),
        (START + [40] * 8 + [20], [21, 40, 41, 42, 43], 8),
        (START + [40] * 8 + [15, 20], [21, 40, 41, 42, 43], 8),
        (START + [40] * 8, [15, 20, 21, 40, 41], -1),
        ([40] * 15, [41, 42, 43, 44, 45], 0),
        (START + [40] * 8 + NATURAL_END, [41, 42, 43, 44, 45], 0),
    ],
)
def test_budget_forcing_matches_complete_history(history, draft, budget):
    def ints(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    rows = len(draft) + 1
    tokens = torch.full((4, 128), 99, dtype=torch.int32, device="cuda")
    tokens[3, :len(history)] = ints(history)
    lengths = ints([0, 0, 0, len(history)])
    budgets = ints([-1, -1, -1, budget])
    cached_start = ints([-1] * 4)
    cached_end = ints([-1] * 4)
    cached_scan = ints([0] * 4)
    logits = torch.full((rows, 64), -3.0, device="cuda")
    apply_thinking_budget(
        logits, ints([3]), ints([3] * rows), budgets, tokens, lengths,
        ints([history[-1]] + draft), ints(list(range(rows))),
        cached_start, cached_end, cached_scan,
        ints(START), ints(NATURAL_END), ints(FORCED_END),
    )
    expected = torch.full_like(logits, -3.0)
    for row in range(rows):
        token = _expected(history + draft[:row], budget)
        if token is not None:
            expected[row, token] = 1.0e9
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


def test_budget_forcing_keeps_interleaved_request_slots_independent():
    def ints(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    histories = {3: START + [40] * 8, 0: START + [40]}
    drafts = {3: [15, 20], 0: [41, 42]}
    tokens = torch.full((4, 128), 99, dtype=torch.int32, device="cuda")
    length = [0] * 4
    for slot, history in histories.items():
        tokens[slot, :len(history)] = ints(history)
        length[slot] = len(history)
    mapping = [3, 3, 3, 0, 0, 0]
    local = [0, 1, 2, 0, 1, 2]
    inputs = [histories[3][-1], *drafts[3], histories[0][-1], *drafts[0]]
    logits = torch.full((6, 64), -3.0, device="cuda")
    apply_thinking_budget(
        logits, ints([3, 0]), ints(mapping), ints([512, -1, -1, 8]),
        tokens, ints(length), ints(inputs), ints(local),
        ints([-1] * 4), ints([-1] * 4), ints([0] * 4),
        ints(START), ints(NATURAL_END), ints(FORCED_END),
    )
    expected = torch.full_like(logits, -3.0)
    for row, (slot, pos) in enumerate(zip(mapping, local)):
        token = _expected(histories[slot] + drafts[slot][:pos], 8 if slot == 3 else 512)
        if token is not None:
            expected[row, token] = 1.0e9
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "start,natural,end",
    [([10], [20], [20]), ([10], [20], [15, 20]), (START, NATURAL_END, FORCED_END)],
)
def test_budget_cache_handles_incremental_and_unchanged_history(start, natural, end):
    def ints(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    tokens = torch.full((1, 128), 99, dtype=torch.int32, device="cuda")
    cache_start, cache_end, cache_scan = ints([-1]), ints([-1]), ints([0])
    lengths = ints([len(start) + 1])
    tokens[0, :len(start)] = ints(start)
    tokens[0, len(start)] = 40
    logits = torch.full((1, 64), -3.0, device="cuda")

    def apply():
        apply_thinking_budget(
            logits, ints([0]), ints([0]), ints([8]), tokens, lengths,
            ints([40]), ints([0]), cache_start, cache_end, cache_scan,
            ints(start), ints(natural), ints(end),
        )

    apply()
    torch.testing.assert_close(logits, torch.full_like(logits, -3.0))
    tokens[0, len(start):len(start) + 8] = 40
    lengths.fill_(len(start) + 8)
    apply()
    expected = torch.full_like(logits, -3.0)
    expected[0, end[0]] = 1.0e9
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    logits.fill_(-3.0)
    apply()
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
