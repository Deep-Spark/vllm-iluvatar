# SPDX-License-Identifier: Apache-2.0
"""Bit-preserving, compact all-gather for strongly uneven row counts."""

from collections.abc import Callable, Sequence
import math

import torch


def compact_gather_beneficial(sizes: Sequence[int], row_bytes: int) -> bool:
    """Compare ring traffic, allowing 1 MiB for packing/reduction overhead.

    A padded gather moves ``world * max_rows`` rows; a compact all-reduce
    moves twice the actual total. Their common ``(world - 1) / world``
    factor cancels. Whole int32 words must belong to a single producer.
    """
    if not sizes or row_bytes % 4 or any(size < 0 for size in sizes):
        return False
    total = sum(sizes)
    saved_bytes = (len(sizes) * max(sizes) - 2 * total) * row_bytes
    return total > 0 and saved_bytes >= 1024 * 1024


def compact_all_gather(
    input_: torch.Tensor,
    sizes: Sequence[int],
    rank: int,
    all_reduce: Callable[[torch.Tensor], torch.Tensor | None],
) -> torch.Tensor:
    """Gather contiguous rows through an existing in-place int32 SUM.

    Each rank writes only its exclusive interval; all other words are zero.
    Integer addition therefore preserves every payload bit, including NaN
    payloads and negative zero. Row boundaries must not split int32 words.
    The input remains unchanged and the returned tensor owns its storage.
    Callers keep the original collective while capturing a CUDA graph.
    """
    if not 0 <= rank < len(sizes) or any(size < 0 for size in sizes):
        raise ValueError("Invalid ragged gather sizes or rank")
    if input_.shape[0] != sizes[rank]:
        raise ValueError("Local tensor rows do not match ragged gather sizes")
    row_bytes = math.prod(input_.shape[1:]) * input_.element_size()
    if row_bytes % 4:
        raise ValueError("Ragged gather rows must contain whole int32 words")

    output = input_.new_zeros((sum(sizes),) + tuple(input_.shape[1:]))
    if sizes[rank]:
        output.narrow(0, sum(sizes[:rank]), sizes[rank]).copy_(input_)
    all_reduce(output.view(torch.int32))
    return output
