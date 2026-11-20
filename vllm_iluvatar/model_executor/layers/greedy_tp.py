# SPDX-License-Identifier: Apache-2.0
"""Pack vocabulary-shard maxima and select the winning TP candidate."""

import torch
import triton
import triton.language as tl
from vllm.v1.worker.gpu.sample.gumbel import _gumbel_sample_kernel


@triton.jit
def _pack_winner(
    maxima,
    indices,
    pair,
    NUM_BLOCKS: tl.constexpr,
    OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    values = tl.load(maxima + row * NUM_BLOCKS + col, col < NUM_BLOCKS, float("-inf"))
    is_nan = values != values
    has_nan = tl.max(is_nan.to(tl.int32), 0) > 0
    maximum = tl.max(values, 0)
    winners = (col < NUM_BLOCKS) & tl.where(has_nan, is_nan, values == maximum)
    winner = tl.min(tl.where(winners, col, 2147483647), 0)
    value = tl.load(maxima + row * NUM_BLOCKS + winner)
    token = tl.load(indices + row * NUM_BLOCKS + winner) + OFFSET
    tl.store(pair + row * 2, value)
    tl.store(pair + row * 2 + 1, token.to(tl.float32))


@triton.jit
def _rank_winner(
    pairs, output, TP: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    rank = tl.arange(0, BLOCK)
    values = tl.load(pairs + row * STRIDE + rank * 2, rank < TP, float("-inf"))
    is_nan = values != values
    has_nan = tl.max(is_nan.to(tl.int32), 0) > 0
    maximum = tl.max(values, 0)
    winners = (rank < TP) & tl.where(has_nan, is_nan, values == maximum)
    winner = tl.min(tl.where(winners, rank, 2147483647), 0)
    token = tl.load(pairs + row * STRIDE + winner * 2 + 1)
    tl.store(output + row, token.to(tl.int64))


def greedy_local_pair(
    logits: torch.Tensor, metadata: tuple, offset: int
) -> torch.Tensor:
    """Return FP32 [score, global token ID], preserving the existing TP format."""
    batch, vocab = logits.shape
    blocks = triton.cdiv(vocab, 1024)
    indices = torch.empty((batch, blocks), device=logits.device, dtype=torch.int64)
    values = torch.empty((batch, blocks), device=logits.device, dtype=torch.float32)
    pair = torch.empty((batch, 2), device=logits.device, dtype=torch.float32)
    mapping, temperature, seeds, positions = metadata
    _gumbel_sample_kernel[(batch, blocks)](
        indices,
        blocks,
        values,
        blocks,
        None,
        0,
        0,
        None,
        logits,
        logits.stride(0),
        mapping,
        seeds,
        positions,
        temperature,
        vocab,
        BLOCK_SIZE=1024,
        IS_DRAFTING=False,
        APPLY_TEMPERATURE=False,
        USE_FP64=False,
        PER_TOKEN_COL=False,
    )
    _pack_winner[(batch,)](
        values,
        indices,
        pair,
        blocks,
        offset,
        triton.next_power_of_2(blocks),
        num_warps=4,
    )
    return pair


def greedy_rank_winner(pairs: torch.Tensor) -> torch.Tensor:
    """Select the first winning rank from [batch, TP, 2] gathered candidates."""
    batch, tp, _ = pairs.shape
    output = torch.empty(batch, device=pairs.device, dtype=torch.int64)
    _rank_winner[(batch,)](
        pairs,
        output,
        tp,
        pairs.stride(0),
        triton.next_power_of_2(tp),
        num_warps=4,
    )
    return output
