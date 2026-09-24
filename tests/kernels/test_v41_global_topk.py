# SPDX-License-Identifier: Apache-2.0
"""The V4.1 local-to-global top-k slot translation, at compress ratio 1 and 2.

``ds4_compute_global_topk_indices_and_lens`` is the one piece of the sparse
path that was written for V4.0's 1/4/128 ratios and is now driven with V4.1's
0/1/2, on both the decode (``attention.py`` ``_forward_decode``) and the
prefill (``_forward_prefill``) side. It turns the indexer's per-token selection
of compressed rows into the global paged slots the sparse MLA kernel reads.

The check that matters is not that the kernel matches some transcription of
itself, but that the rows it hands the reader are the rows the *writer* put
them in. So the fixture builds the slots with ``get_compressed_slot_mapping``
-- the same helper the metadata builders use to tell the compressor where to
publish -- and asserts the translation lands on exactly those.
"""

from __future__ import annotations

import pytest
import torch

ixfops = pytest.importorskip("ixformer.inference.functions")
from vllm.v1.attention.backends.mla.compressor_utils import (  # noqa: E402
    get_compressed_slot_mapping,
)

DEVICE = "cuda"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _build_batch(seq_lens: list[int], rows_per_page: int, compress_ratio: int):
    """A full-prefill batch plus the compressed slots its compressor writes.

    Returns the per-token positions and request ids, the block table, and
    ``written[req][group]``: the global compressed slot holding that request's
    group, as ``get_compressed_slot_mapping`` assigned it.
    """
    num_reqs = len(seq_lens)
    max_groups = max(
        (length + compress_ratio - 1) // compress_ratio for length in seq_lens
    )
    max_cols = (max_groups + rows_per_page - 1) // rows_per_page + 1
    # Page 0 is the null page everywhere in vLLM, so hand out pages from 1.
    block_table = (
        torch.arange(1, num_reqs * max_cols + 1, dtype=torch.int32)
        .reshape(num_reqs, max_cols)
        .to(DEVICE)
    )

    positions = torch.cat(
        [torch.arange(length, dtype=torch.int64) for length in seq_lens]
    ).to(DEVICE)
    token_to_req = torch.cat(
        [
            torch.full((length,), r, dtype=torch.int32)
            for r, length in enumerate(seq_lens)
        ]
    ).to(DEVICE)

    lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    query_start_loc[1:] = torch.cumsum(lens, 0)
    num_tokens = int(query_start_loc[-1])

    # A full prefill: seq_len == query_len, so every request starts at 0.
    slots = get_compressed_slot_mapping(
        num_tokens,
        query_start_loc,
        lens,
        block_table,
        rows_per_page,
        compress_ratio,
    )

    written: dict[int, dict[int, int]] = {r: {} for r in range(num_reqs)}
    for t in range(num_tokens):
        slot = int(slots[t])
        if slot < 0:
            continue
        req = int(token_to_req[t])
        written[req][int(positions[t]) // compress_ratio] = slot
    return positions, token_to_req, block_table, written, num_tokens


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_global_topk_lands_on_the_rows_the_compressor_wrote(compress_ratio: int):
    """Identity selection must resolve to every group the token can see.

    Feeding ``[0, 1, 2, ...]`` is what the short-context bypass
    (``v41_fill_short_context_topk``) produces, so this is a real input shape,
    and it makes the expected answer exactly "every causally visible group".
    """
    seq_lens = [9, 4, 17]
    rows_per_page, topk = 4, 16
    positions, token_to_req, block_table, written, num_tokens = _build_batch(
        seq_lens, rows_per_page, compress_ratio
    )

    topk_indices = (
        torch.arange(topk, dtype=torch.int32, device=DEVICE)
        .expand(num_tokens, topk)
        .contiguous()
    )
    is_valid = torch.ones(num_tokens, dtype=torch.bool, device=DEVICE)

    global_indices, lens = ixfops.ds4_compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req,
        block_table,
        rows_per_page,
        is_valid,
        positions,
        compress_ratio,
    )

    for t in range(num_tokens):
        req = int(token_to_req[t])
        # Causality in the compressed domain: a token at position p sees the
        # groups that closed at or before it.
        visible = min((int(positions[t]) + 1) // compress_ratio, topk)
        expected = [written[req][g] for g in range(visible)]
        assert int(lens[t]) == len(expected), (
            f"token {t} (req {req}, pos {int(positions[t])}) kept "
            f"{int(lens[t])} slots, expected {len(expected)}"
        )
        got = global_indices[t, : len(expected)].tolist()
        assert got == expected, (
            f"token {t} (req {req}, pos {int(positions[t])}) resolved to "
            f"{got}, but the compressor wrote those groups at {expected}"
        )
        assert torch.all(global_indices[t, len(expected) :] == -1)


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_global_topk_honours_selection_and_padding(compress_ratio: int):
    """A sparse, out-of-order selection, plus CUDA-graph padding rows."""
    seq_lens = [12, 7]
    rows_per_page, topk = 4, 8
    positions, token_to_req, block_table, written, num_tokens = _build_batch(
        seq_lens, rows_per_page, compress_ratio
    )

    generator = torch.Generator(device="cpu").manual_seed(0)
    # -1 is the producer's "no selection" sentinel and must be dropped, not
    # translated; the kept slots compact to the front.
    choices = torch.randint(
        -1, 6, (num_tokens, topk), generator=generator, dtype=torch.int32
    )
    topk_indices = choices.to(DEVICE)
    is_valid = torch.ones(num_tokens, dtype=torch.bool, device=DEVICE)
    is_valid[::5] = False

    global_indices, lens = ixfops.ds4_compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req,
        block_table,
        rows_per_page,
        is_valid,
        positions,
        compress_ratio,
    )

    for t in range(num_tokens):
        if not bool(is_valid[t]):
            assert int(lens[t]) == 0
            assert torch.all(global_indices[t] == -1)
            continue
        req = int(token_to_req[t])
        visible = min((int(positions[t]) + 1) // compress_ratio, topk)
        expected = []
        for j in range(visible):
            local = int(topk_indices[t, j])
            if local < 0:
                continue
            # A local slot past what the request has written is not a row the
            # compressor ever filled; the fixture's block table still has a
            # page for it, so the translation is defined and we follow it.
            page = int(block_table[req, local // rows_per_page])
            expected.append(page * rows_per_page + local % rows_per_page)
        assert int(lens[t]) == len(expected)
        assert global_indices[t, : len(expected)].tolist() == expected
        assert torch.all(global_indices[t, len(expected) :] == -1)


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_global_topk_ignores_columns_past_the_causal_length(compress_ratio: int):
    """Stale columns of a long-lived buffer must not leak into the selection.

    ``topk_indices_buffer`` is reused across steps, so a row whose causal
    length shrank still holds last step's slots past that length. Passing
    ``positions`` is what bounds the read; this pins that it does.
    """
    seq_lens = [6]
    rows_per_page, topk = 4, 8
    positions, token_to_req, block_table, _, num_tokens = _build_batch(
        seq_lens, rows_per_page, compress_ratio
    )
    # Every column selects a real slot, so anything read past the causal
    # length would show up as extra kept entries.
    topk_indices = torch.zeros(num_tokens, topk, dtype=torch.int32, device=DEVICE)
    is_valid = torch.ones(num_tokens, dtype=torch.bool, device=DEVICE)

    _, lens = ixfops.ds4_compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req,
        block_table,
        rows_per_page,
        is_valid,
        positions,
        compress_ratio,
    )
    expected = [
        min((int(positions[t]) + 1) // compress_ratio, topk) for t in range(num_tokens)
    ]
    assert lens.tolist() == expected
