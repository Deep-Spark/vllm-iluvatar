# SPDX-License-Identifier: Apache-2.0
"""Iluvatar two-level indexer candidate selection and masking.

Corresponds to: vllm/model_executor/kernels/attention/dsa/candidate_blocks.py

Target   : ``select_candidate_blocks`` / ``apply_candidate_mask``
Baseline : main_6e448d0
Why      : community spells both in Triton, which does not run on corex, and
           the module is not even shipped in this vLLM tree. Without them the
           V4.1 index sources above ``candidate_source_layer_id`` take their
           top-k over the whole context instead of over the shortlist the
           candidate source published.
How      : same two entry points, same argument order and contract; the two
           reductions move to ``torch.ops._C.v41_candidate_block_scores`` /
           ``v41_apply_candidate_mask`` (``csrc/plugin/v41_candidate.cu``) and
           decode top-k scans the GPU row bounds with the native radix kernel;
           other shapes retain the bounded Torch implementation.
Scope    : DeepSeek-V4.1 indexers with a candidate source.
Remove   : when community grows a vendor hook for these two.
"""

from __future__ import annotations

import torch
import ixformer.inference.functions as ixfops

# Rows per ``topk`` call in ``select_candidate_blocks``. Above 128 rows corex
# torch's topk takes scratch memory outside the caching allocator -- tens of
# MiB at 256 rows, over 256 MiB at a 4096-row prefill chunk -- and once the
# pool has grown into the utilization headroom that allocation fails
# (``ixErrorMemoryAllocation``). Up to 128 rows it needs none. Rows are
# independent, so slicing them changes nothing but the scratch.
_TOPK_ROWS = 128


def select_candidate_blocks(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
) -> None:
    """Shortlist block ids by best score, pinning each row's newest block.

    Row bounds are in packed column space; an absent start means zero. Decode
    rows share bounds in groups of ``row_repeat``. ``out`` is ``-1`` padded.
    """
    rows, width = logits.shape
    if not rows:
        return
    if not width:
        out.fill_(-1)
        return

    nblocks = -(-width // block_size)
    k = min(topk_blocks, nblocks)
    if (
        rows in (6, 12, 18, 24)
        and logits.dtype == torch.bfloat16
        and row_ks is None
        and row_repeat == 1
        and block_size == 8
        and topk_blocks == 2048
        and nblocks >= topk_blocks
        and width % 16 == 0
        and out.is_contiguous()
    ):
        scores = torch.ops._C.v41_candidate_block_scores(
            logits, None, row_ke, block_size, row_repeat
        )
        ends = torch.div(
            row_ke.clamp(min=0, max=width) + block_size - 1,
            block_size,
            rounding_mode="floor",
        )
        starts = torch.zeros_like(ends)
        ixfops.dsa_update_topk_indices(scores, starts, ends, topk_blocks, out)
        # The score producer initializes all columns. The radix pads with -1;
        # an in-span block with no finite column is padding as well.
        selected = scores.gather(1, out.clamp(min=0).long())
        out.masked_fill_((out < 0) | (selected == -float("inf")), -1)
        return
    # Slice on group boundaries so each slice's bounds start at a group.
    step = max(1, _TOPK_ROWS // row_repeat) * row_repeat
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        g0, g1 = r0 // row_repeat, -(-r1 // row_repeat)
        scores = torch.ops._C.v41_candidate_block_scores(
            logits[r0:r1],
            None if row_ks is None else row_ks[g0:g1],
            row_ke[g0:g1],
            block_size,
            row_repeat,
        )
        top = scores.topk(k, dim=-1)
        # One kernel writes the slice's whole ``out`` row: an index in the
        # first ``k`` columns (``-1`` where the slot scored ``-inf``, a block
        # with no live column, which is padding not a candidate; NaN is left as
        # a kept index, the reference's tie behaviour) and ``-1`` in the tail.
        # This replaces a select, an int32 cast, a row copy and a separate
        # ``out[:, k:]`` fill with a single launch per slice.
        torch.ops._C.v41_write_candidate_topk(out[r0:r1], top.values, top.indices, k)


def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
) -> None:
    """Mask packed logits outside the causal bounds and the shortlist."""
    rows, width = logits.shape
    if not rows or not width:
        return
    torch.ops._C.v41_apply_candidate_mask(
        logits, row_ks, row_ke, candidate_blocks, block_size, row_repeat
    )


def apply_initialized_decode_candidate_mask(
    logits: torch.Tensor,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
) -> None:
    """Mask block8 decode logits whose exterior the scorer already initialized."""
    if (logits.dtype == torch.bfloat16 and logits.shape[0] in (6, 12, 18, 24)
            and logits.stride(1) == 1):
        if logits.shape[1]:
            torch.ops._C.v41_apply_initialized_decode_mask(
                logits, row_ke, candidate_blocks
            )
        return
    apply_candidate_mask(logits, None, row_ke, candidate_blocks, 8)
