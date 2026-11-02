# SPDX-License-Identifier: Apache-2.0
"""Thinking-budget kernels adapted to the Iluvatar Triton backend.

The committed marker cache retains the upstream cold/incremental scan. Draft
prefix marker matching uses bounded masked vectors: the expanded logits batch
bounds every request's draft prefix, so no dynamic zero-trip loop is needed.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

_COLD_SCAN_BLOCK = 1024

@triton.jit
def _update_committed_marker_cache_kernel(
    req_ids_ptr,
    thinking_token_budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    cached_scan_pos_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    MAX_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req_state_idx = tl.load(req_ids_ptr + tl.program_id(0))
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    scan_pos = tl.load(cached_scan_pos_ptr + req_state_idx)
    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    if scan_pos > total_len:
        scan_pos = 0
        last_start = -1
        last_end = -1

    # [ILU delta] parenthesized; ilu Triton rejects chained ``and``.
    if (scan_pos == 0 and last_start < 0) and last_end < 0:
        # Cold scan: walk backward in vectorized blocks, stopping at the first
        # block with a marker; only the relative order of the two positions
        # found matters below.
        block_hi = total_len
        # [ILU delta] parenthesized; same chained-bool restriction.
        while (block_hi > 0 and last_start < 0) and last_end < 0:
            block_lo = block_hi - BLOCK
            if block_lo < 0:
                block_lo = 0
            offs = block_lo + tl.arange(0, BLOCK)

            start_match = (offs < block_hi) & (offs + START_LEN <= total_len)
            for j in tl.static_range(0, START_LEN):
                expected = tl.load(reasoning_start_token_ids_ptr + j)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs + j,
                    mask=offs + j < total_len,
                    other=-1,
                )
                start_match = start_match & (actual == expected)

            end_match = (offs < block_hi) & (offs + NATURAL_END_LEN <= total_len)
            for j in tl.static_range(0, NATURAL_END_LEN):
                expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs + j,
                    mask=offs + j < total_len,
                    other=-1,
                )
                end_match = end_match & (actual == expected)

            last_start = tl.max(tl.where(start_match, offs, -1), axis=0)
            last_end = tl.max(tl.where(end_match, offs, -1), axis=0)
            block_hi = block_lo
    elif scan_pos < total_len:
        for i in tl.range(scan_pos, total_len):
            if i + START_LEN <= total_len:
                start_match = True
                for j in tl.static_range(0, START_LEN):
                    expected = tl.load(reasoning_start_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr + req_state_idx * all_token_ids_stride + i + j
                    )
                    start_match = start_match & (actual == expected)
                if start_match:
                    last_start = i

            if i + NATURAL_END_LEN <= total_len:
                end_match = True
                for j in tl.static_range(0, NATURAL_END_LEN):
                    expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr + req_state_idx * all_token_ids_stride + i + j
                    )
                    end_match = end_match & (actual == expected)
                if end_match:
                    last_end = i

    tl.store(cached_last_start_ptr + req_state_idx, last_start)
    tl.store(cached_last_end_ptr + req_state_idx, last_end)
    new_scan_pos = total_len - (MAX_LEN - 1)
    if new_scan_pos < 0:
        new_scan_pos = 0
    tl.store(cached_scan_pos_ptr + req_state_idx, new_scan_pos)


@triton.jit
def _last_effective_marker(
    all_token_ids_ptr,
    all_token_ids_stride,
    input_ids_ptr,
    cur_req_first_pos,
    req_state_idx,
    total_len,
    effective_len,
    marker_ptr,
    MARKER_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    scan_lo = tl.maximum(total_len - MARKER_LEN + 1, 0)
    positions = scan_lo + tl.arange(0, BLOCK)
    matches = positions + MARKER_LEN <= effective_len
    for j in tl.static_range(0, MARKER_LEN):
        pos = positions + j
        committed = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos,
            mask=matches & (pos < total_len),
            other=-1,
        )
        draft = tl.load(
            input_ids_ptr + cur_req_first_pos + pos - total_len + 1,
            mask=matches & (pos >= total_len),
            other=-1,
        )
        actual = tl.where(pos < total_len, committed, draft)
        matches = matches & (actual == tl.load(marker_ptr + j))
    return tl.max(tl.where(matches, positions, -1), axis=0)


@triton.jit
def _thinking_budget_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    thinking_token_budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
    SCAN_BLOCK: tl.constexpr,
    PREFIX_BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return

    local_pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - local_pos
    total_len = tl.load(total_len_ptr + req_state_idx)
    effective_len = total_len + local_pos

    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    last_start = tl.maximum(
        last_start,
        _last_effective_marker(
            all_token_ids_ptr, all_token_ids_stride, input_ids_ptr,
            cur_req_first_pos, req_state_idx, total_len, effective_len,
            reasoning_start_token_ids_ptr, MARKER_LEN=START_LEN, BLOCK=SCAN_BLOCK,
        ),
    )
    last_end = tl.maximum(
        last_end,
        _last_effective_marker(
            all_token_ids_ptr, all_token_ids_stride, input_ids_ptr,
            cur_req_first_pos, req_state_idx, total_len, effective_len,
            natural_reasoning_end_token_ids_ptr,
            MARKER_LEN=NATURAL_END_LEN, BLOCK=SCAN_BLOCK,
        ),
    )

    if last_start < 0 or last_start <= last_end:
        return

    reasoning_start = last_start + START_LEN
    # If the request resumes from a prompt that already contains generated
    # reasoning content, count it against the remaining budget.
    num_reasoning_tokens = effective_len - reasoning_start
    if num_reasoning_tokens < budget:
        return

    end_prefix_len = 0
    if END_LEN > 1:
        prefix_lengths = tl.arange(0, PREFIX_BLOCK)
        prefix_match = ((prefix_lengths > 0) & (prefix_lengths < END_LEN)
                        & (prefix_lengths <= effective_len))
        suffix_start = effective_len - prefix_lengths
        for j in tl.static_range(0, END_LEN - 1):
            pos = suffix_start + j
            active = prefix_match & (j < prefix_lengths)
            committed = tl.load(
                all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos,
                mask=active & (pos < total_len), other=-1,
            )
            draft = tl.load(
                input_ids_ptr + cur_req_first_pos + pos - total_len + 1,
                mask=active & (pos >= total_len), other=-1,
            )
            actual = tl.where(pos < total_len, committed, draft)
            expected = tl.load(reasoning_end_token_ids_ptr + j)
            prefix_match = prefix_match & ((j >= prefix_lengths)
                                           | (actual == expected))
        end_prefix_len = tl.max(tl.where(prefix_match, prefix_lengths, 0), axis=0)

    force_token_id = tl.load(reasoning_end_token_ids_ptr + end_prefix_len)
    tl.store(logits_ptr + token_idx * logits_stride + force_token_id, 1.0e9)


def apply_thinking_budget(
    logits: torch.Tensor,
    req_ids: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    thinking_token_budget: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    cached_last_start: torch.Tensor,
    cached_last_end: torch.Tensor,
    cached_scan_pos: torch.Tensor,
    reasoning_start_token_ids: torch.Tensor,
    natural_reasoning_end_token_ids: torch.Tensor,
    reasoning_end_token_ids: torch.Tensor,
) -> None:
    num_tokens = logits.shape[0]
    if num_tokens == 0:
        return
    start_len = reasoning_start_token_ids.shape[0]
    natural_end_len = natural_reasoning_end_token_ids.shape[0]
    end_len = reasoning_end_token_ids.shape[0]

    _update_committed_marker_cache_kernel[(req_ids.shape[0],)](
        req_ids,
        thinking_token_budget,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        cached_last_start,
        cached_last_end,
        cached_scan_pos,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        MAX_LEN=max(start_len, natural_end_len),
        BLOCK=_COLD_SCAN_BLOCK,
    )

    _thinking_budget_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        thinking_token_budget,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        input_ids,
        expanded_local_pos,
        cached_last_start,
        cached_last_end,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        reasoning_end_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        END_LEN=end_len,
        # Every request draft prefix fits in the expanded logits batch.
        SCAN_BLOCK=triton.next_power_of_2(num_tokens),
        PREFIX_BLOCK=triton.next_power_of_2(end_len),
    )
