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
           reductions move to ``ixfops.candidate_block_scores`` /
           ``apply_candidate_mask`` (ixformer deepseekv41/candidate).
           decode top-k for 6/12/18/24 rows scans GPU row bounds with the
           native radix kernel; other shapes retain the bounded Torch
           implementation. writing the shortlist uses
           ``write_candidate_topk``.
Scope    : DeepSeek-V4.1 indexers with a candidate source.
Remove   : when community grows a vendor hook for these two.
"""

from __future__ import annotations

import contextlib
import torch
from ixformer.cuda import cuda_arch
from ixformer.inference.functions import dsa_update_topk_indices
from ixformer.inference.functions.models import deepseek_v41 as ixfops
from vllm import envs


# Rows per ``topk`` call in ``select_candidate_blocks``. Above 128 rows corex
# torch's topk takes scratch memory outside the caching allocator -- tens of
# MiB at 256 rows, over 256 MiB at a 4096-row prefill chunk -- and once the
# pool has grown into the utilization headroom that allocation fails
# (``ixErrorMemoryAllocation``). Up to 128 rows it needs none. Rows are
# independent, so slicing them changes nothing but the scratch.
_TOPK_ROWS = 128

# The same scratch also grows with ``nblocks`` (the top-k width): at a long
# prefill chunk nblocks reaches ~1e4 and even 128 rows overruns the headroom
# left once a 1024-token chunk's activations are resident. Cap ``rows *
# nblocks`` per call so the scratch stays bounded as context grows, and bisect
# the row span on OOM as a safety net.
_TOPK_COL_BUDGET = 128 * 2048


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
        logits.dtype == torch.bfloat16
        and row_ks is None
        and row_repeat == 1
        and block_size == 8
        and topk_blocks == 2048
        and nblocks >= topk_blocks
        and out.is_contiguous()
    ):
        # The native selector accepts arbitrary row counts. Bound temporary
        # storage per call and preserve its packed-load row alignment even
        # when the last candidate block is partial.
        stride = (nblocks + 31) // 32 * 32
        for r0 in range(0, rows, _TOPK_ROWS):
            r1 = min(rows, r0 + _TOPK_ROWS)
            scores = logits.new_empty((r1 - r0, stride))[:, :nblocks]
            bounds = torch.empty((2, r1 - r0), dtype=torch.int32, device=logits.device)
            ixfops.candidate_block_scores_bounded(
                logits[r0:r1],
                None,
                row_ke[r0:r1],
                block_size,
                1,
                scores,
                bounds,
            )
            dsa_update_topk_indices(
                scores, bounds[0], bounds[1], topk_blocks, out[r0:r1]
            )
            # Empty rows may leave scores wholly uninitialized. In particular,
            # a padding index must never be clamped to zero and gathered.
            ixfops.finalize_bounded_candidates(scores, bounds, out[r0:r1])
        return
    if k >= nblocks:
        ke = row_ke if row_repeat == 1 else row_ke.repeat_interleave(row_repeat)
        ke = ke[:rows].to(torch.int64)
        nb_row = (ke + (block_size - 1)) // block_size
        cols = torch.arange(nblocks, device=out.device, dtype=torch.int32)
        out[:rows, :nblocks] = torch.where(
            cols.unsqueeze(0) < nb_row.unsqueeze(1),
            cols.unsqueeze(0),
            out.new_full((), -1),
        )
        if nblocks < out.shape[1]:
            out[:rows, nblocks:] = -1
        return
    # Slice on group boundaries so each slice's bounds start at a group. The row
    # cap shrinks as nblocks grows so ``rows * nblocks`` stays within budget.
    # This bounded BF16 decode case fits in one top-k call. Its consumers
    # build membership flags, so ordering the selected blocks is unnecessary.
    # Keep the conservative scratch budget for larger/prefill workloads.
    compact_decode = (
        cuda_arch == "ivcore30"
        and 48 <= rows <= 96
        and width == 278528
        and logits.dtype == torch.bfloat16
        and row_ks is None
        and row_repeat == 1
        and topk_blocks == 2048
        and block_size == 8
    )
    fast_prefill = (
        envs.VLLM_ILU_V41_PREFILL_UNSORTED
        and cuda_arch == "ivcore30"
        and 128 < rows <= 16384
        and 16384 < width <= 32768
        and logits.dtype == torch.bfloat16
        and row_ks is None
        and row_repeat == 1
        and topk_blocks == 2048
        and block_size == 8
    )
    max_rows = (
        rows
        if compact_decode
        else max(1, min(_TOPK_ROWS, _TOPK_COL_BUDGET // max(1, nblocks)))
    )
    step = max(1, max_rows // row_repeat) * row_repeat
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        g0, g1 = r0 // row_repeat, -(-r1 // row_repeat)
        scores = ixfops.candidate_block_scores(
            logits[r0:r1],
            None if row_ks is None else row_ks[g0:g1],
            row_ke[g0:g1],
            block_size,
            row_repeat,
        )
        # topk is per-row, so its scratch can be shrunk further by halving the
        # row span and retrying -- the safety net when the proactive cap above
        # still leaves the scratch too big for the current headroom.
        n = scores.shape[0]
        sub = n
        s = 0
        while s < n:
            e = min(n, s + sub)
            try:
                top = scores[s:e].topk(
                    k, dim=-1, sorted=not (compact_decode or fast_prefill)
                )
            except RuntimeError as err:
                if "out of memory" in str(err).lower() and sub > 1:
                    with contextlib.suppress(Exception):
                        torch.cuda.empty_cache()
                    sub = max(1, sub // 2)
                    continue
                raise
            # One launch writes the kept indices and the ``-1`` tail. A block
            # that scored ``-inf`` is padding; a NaN is kept, matching
            # community.
            ixfops.write_candidate_topk(
                out[r0 + s : r0 + e], top.values, top.indices, k
            )
            s = e


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
    ixfops.apply_candidate_mask(
        logits, row_ks, row_ke, candidate_blocks, block_size, row_repeat
    )


def apply_initialized_decode_candidate_mask(
    logits: torch.Tensor,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
) -> None:
    """Mask block8 decode logits whose exterior the scorer already initialized."""
    if (
        logits.dtype == torch.bfloat16
        and logits.shape[0] in (6, 12, 18, 24, 30)
        and logits.stride(1) == 1
    ):
        if logits.shape[1]:
            ixfops.apply_initialized_decode_mask(logits, row_ke, candidate_blocks)
        return
    apply_candidate_mask(logits, None, row_ke, candidate_blocks, 8)
