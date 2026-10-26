"""Length-aware candidate consumer scoring without host synchronization."""

import ixformer.inference.functions as ops
import torch
from ixformer import cuda
from ixformer.inference.functions.deepseek.candidate_indexer import (
    candidate_logits_split,
    prepare_candidate_metadata,
    remap_candidate_indices,
)

from .ilu_candidate_blocks import (
    apply_candidate_mask,
    apply_initialized_decode_candidate_mask,
)


def maybe_topk(
    q,
    cq,
    ck,
    values,
    scales,
    table,
    weights,
    visible,
    candidates,
    output,
    *,
    max_q_len,
    max_kv_len,
    group,
    topk,
    decode,
):
    """Publish hybrid dense/compact TopK; unsupported layouts retain the caller path."""
    if cuda.cuda_arch != "ivcore11":
        return False
    if scales is None or candidates is None or q.ndim != 3 or candidates.ndim != 2:
        return False
    rows = q.shape[0]
    cap = candidates.shape[1]
    width = (4 * cap * group + 31) // 32 * 32
    if width < topk:
        return False
    if not (
        0 < rows <= 65535
        and q.shape[1] in (32, 64)
        and q.shape[2] == 128
        and q.dtype == torch.int8
        and q.is_contiguous()
        and values.dtype == torch.uint8
        and values.ndim == 3
        and values.shape[1] in (64, 256)
        and values.shape[2] == 128
        and values.stride(1) == 128
        and values.stride(2) == 1
        and values.stride(0) >= values.shape[1] * 128
        and values.stride(0) % 64 == 0
        and scales.dtype == torch.float32
        and scales.shape == values.shape[:2]
        and scales.stride(1) == 1
        and scales.stride(0) >= values.shape[1]
        and weights.dtype == torch.float32
        and weights.shape == (rows, q.shape[1])
        and weights.is_contiguous()
        and 4 <= group <= values.shape[1]
        and group & (group - 1) == 0
        and cap > 0
        and topk == 512
        and output.shape == (rows, topk)
        and output.dtype == torch.int32
        and output.is_contiguous()
        and candidates.shape[0] == rows
        and candidates.dtype == torch.int32
        and candidates.stride(1) == 1
        and table.ndim == 2
        and table.dtype == torch.int32
        and table.is_contiguous()
        and cq.ndim == ck.ndim == visible.ndim == 1
        and cq.numel() == ck.numel() == table.shape[0] + 1
        and visible.numel() == rows
        and cq.dtype == ck.dtype == visible.dtype == torch.int32
        and cq.is_contiguous()
        and ck.is_contiguous()
        and visible.is_contiguous()
        and max_q_len > 0
        and max_kv_len > 0
    ):
        return False
    # Eager prefill already supplies a CPU upper bound for this query chunk.
    # If it proves every row short, preserve the original dense chain rather
    # than launching both hybrid branches. Decode uses GPU row lengths because
    # its captured maximum need not match the lengths at replay.
    if not decode and max_kv_len <= width:
        return False
    tensors = (q, cq, ck, values, scales, table, weights, visible, candidates, output)
    if not all(t.is_cuda and t.device == q.device for t in tensors):
        return False
    if q.device.index != torch.cuda.current_device():
        return False
    if q.data_ptr() % 64 or values.data_ptr() % 64:
        return False
    if rows * (width * 2 + topk * 4 + 24) + 4 * ck.numel() > 112 * 1024**2:
        return False
    # Standard decoder table capacity covers1M. Eager chunks can have a smaller
    # table; the dense kernel's bound cannot exceed its supplied columns.
    dense_max = min(width, max_kv_len, table.shape[1] * values.shape[1])
    if dense_max <= 0:
        return False
    metadata = torch.empty((6, rows), device=q.device, dtype=torch.int32)
    requests, sparse_ends, dense_ends, select_ends, _flags, starts = metadata.unbind(0)
    dense_ck = torch.empty_like(ck)
    logits = torch.empty((rows, width), device=q.device, dtype=torch.bfloat16)
    selected = torch.empty_like(output)
    prepare_candidate_metadata(
        candidates, cq, ck, visible, group, width, metadata, dense_ck
    )
    ops.dsa_indexer_mqa_logits_split(
        q,
        cq,
        dense_ck,
        values,
        scales,
        table,
        weights,
        max_q_len=max_q_len,
        max_kv_len=dense_max,
        valid_lens=dense_ends,
        logits=logits,
    )
    if decode and group == 8:
        apply_initialized_decode_candidate_mask(logits, visible, candidates)
    else:
        apply_candidate_mask(logits, None, visible, candidates, group)
    candidate_logits_split(
        q,
        values,
        scales,
        cq,
        ck,
        table,
        weights,
        candidates,
        requests,
        visible,
        sparse_ends,
        group,
        max_q_len,
        max_kv_len,
        logits,
    )
    ops.dsa_update_topk_indices(logits, starts, select_ends, topk, selected)
    remap_candidate_indices(selected, candidates, visible, metadata, group, output)
    return True
