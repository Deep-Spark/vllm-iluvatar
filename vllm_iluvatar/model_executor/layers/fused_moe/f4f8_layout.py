# SPDX-License-Identifier: Apache-2.0
"""Build [begin, end, weight expert] tasks for the two CQ F4F8 projections."""

import torch
import triton
import triton.language as tl


@triton.jit
def _split_expert_ranges(
    start, end, slot, E: tl.constexpr, B: tl.constexpr, LOG_B: tl.constexpr
):
    # Each nonempty expert owns a slot. Lend empty slots to earlier experts
    # with additional 256-row tiles, keeping the launch grid at E slots.
    tile_rows: tl.constexpr = 256
    rows = end - start
    tiles = tl.cdiv(rows.to(tl.uint32), tile_rows).to(tl.int32)
    wanted = tl.maximum(tiles - 1, 0)
    spare = E - tl.sum((rows > 0).to(tl.int32), axis=0)
    extra = tl.minimum(wanted, tl.maximum(spare - (tl.cumsum(wanted, 0) - wanted), 0))
    pieces = (rows > 0).to(tl.int32) + extra
    ends = tl.cumsum(pieces, 0)
    used = tl.sum(pieces, 0)

    # The owner is the first expert whose cumulative piece count exceeds slot.
    owner = tl.full((B,), 0, tl.int32)
    for i in tl.static_range(LOG_B):
        step = 1 << (LOG_B - 1 - i)
        probe = owner + step
        prev = tl.gather(ends, tl.minimum(probe - 1, B - 1), 0)
        owner = tl.where((probe < B) & (prev <= slot), probe, owner)

    count = tl.maximum(tl.gather(pieces, owner, 0), 1)
    index = slot - (tl.gather(ends, owner, 0) - count)
    num_tiles = tl.gather(tiles, owner, 0)
    quotient = num_tiles // count
    remainder = num_tiles % count
    first = index * quotient + tl.minimum(index, remainder)
    last = first + quotient + (index < remainder).to(tl.int32)
    begin = tl.gather(start, owner, 0)
    # The exclusive end is representable even when the next tile boundary is not.
    limit = tl.where(
        last == num_tiles, tl.gather(end, owner, 0), begin + last * tile_rows
    )
    active = slot < used
    return (
        tl.where(active, begin + first * tile_rows, 0),
        tl.where(active, limit, 0),
        tl.where(active, owner, 0),
    )


@triton.jit
def _sort_layout(
    src, dst, E: tl.constexpr, B: tl.constexpr, SPLIT: tl.constexpr, LOG_B: tl.constexpr
):
    pos = tl.arange(0, B)
    start = tl.load(src + pos * 2, pos < E, 0)
    end = tl.load(src + pos * 2 + 1, pos < E, 0)
    if SPLIT:
        start, end, expert = _split_expert_ranges(start, end, pos, E, B, LOG_B)

    key = tl.where(pos < E, (end - start).to(tl.int64) * B + (B - 1 - pos), -1)
    key = tl.sort(key, descending=True)
    order = B - 1 - (key & (B - 1)).to(tl.int32)
    if SPLIT:
        start = tl.gather(start, order, 0)
        end = tl.gather(end, order, 0)
        expert = tl.gather(expert, order, 0)
    else:
        expert = order
        start = tl.load(src + expert * 2, pos < E, 0)
        end = tl.load(src + expert * 2 + 1, pos < E, 0)
    tl.store(dst + pos * 3, start, pos < E)
    tl.store(dst + pos * 3 + 1, end, pos < E)
    tl.store(dst + pos * 3 + 2, expert, pos < E)


def sort_group_layout(
    layout: torch.Tensor, *, split_large: bool = False
) -> torch.Tensor:
    experts = layout.shape[0]
    block = triton.next_power_of_2(experts)
    output = torch.empty((experts, 3), device=layout.device, dtype=torch.int32)
    _sort_layout[(1,)](
        layout, output, experts, block, split_large, block.bit_length() - 1, num_warps=4
    )
    return output
