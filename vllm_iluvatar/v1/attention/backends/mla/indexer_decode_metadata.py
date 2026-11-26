# SPDX-License-Identifier: Apache-2.0
"""Build paged indexer decode bounds directly in graph-stable buffers."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _request_bounds(
    Q,
    L,
    CQ,
    CK,
    Lengths,
    Compressed,
    Legacy,
    B,
    Ratio: tl.constexpr,
    Flat: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    b = tl.arange(0, BLOCK_B)
    base = tl.load(Q)
    starts = tl.load(Q + b, b < B, 0) - base
    ends = tl.load(Q + b + 1, b < B, 0) - base
    seq = tl.load(L + b, b < B, 0)
    compressed = seq // Ratio
    prefix = tl.cumsum(compressed)
    tl.store(CQ + b, starts, b < B)
    tl.store(CQ + B, tl.load(Q + B) - base)
    tl.store(CK, 0)
    tl.store(CK + b + 1, prefix, b < B)
    tl.store(Lengths + b, ends - starts, b < B)
    tl.store(Compressed + b, compressed, b < B)
    if not Flat:
        tl.store(Legacy + b, ends - starts, b < B)
    return prefix - compressed, ends, seq


@triton.jit
def _query_requests(
    Q,
    RequestIds,
    rows,
    live,
    base,
    B,
    HAS_REQUEST_IDS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    if HAS_REQUEST_IDS:
        req = tl.load(RequestIds + rows, live, 0)
    else:
        lo = tl.full(rows.shape, 0, tl.int32)
        hi = lo + B
        # First request whose exclusive query end exceeds this row. Repeated
        # boundaries (including empty requests in the middle) are skipped.
        for _ in range(SEARCH_STEPS):
            mid = (lo + hi) // 2
            end = tl.load(Q + mid + 1, live & (mid < B), 2147483647) - base
            right = rows >= end
            lo = tl.where(right, mid + 1, lo)
            hi = tl.where(right, hi, mid)
        req = lo
    return tl.where(live, tl.minimum(req, tl.maximum(B - 1, 0)), 0)


@triton.jit(do_not_specialize=["B", "P"])
def _fused_decode_metadata(
    Q,
    L,
    RequestIds,
    CQ,
    CK,
    Lengths,
    Compressed,
    Legacy,
    KS,
    KE,
    Valid,
    B,
    P,
    Ratio: tl.constexpr,
    Flat: tl.constexpr,
    HAS_REQUEST_IDS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    starts, ends, seq = _request_bounds(
        Q, L, CQ, CK, Lengths, Compressed, Legacy, B, Ratio, Flat, BLOCK_B
    )
    rows = tl.arange(0, BLOCK_T)
    base = tl.load(Q)
    real = tl.load(Q + B) - base
    live = (rows < P) & (rows < real)
    req = _query_requests(
        Q, RequestIds, rows, live, base, B, HAS_REQUEST_IDS, SEARCH_STEPS
    )
    start = tl.gather(starts, req, 0)
    end = tl.gather(ends, req, 0)
    length = tl.gather(seq, req, 0)
    visible = (length - end + rows + 1) // Ratio
    tl.store(KS + rows, tl.where(live, start, 0), rows < P)
    tl.store(KE + rows, tl.where(live, start + visible, 0), rows < P)
    tl.store(Valid + rows, tl.where(live, visible, 0), rows < P)
    if Flat:
        tl.store(Legacy + rows, 1, rows < P)


@triton.jit(do_not_specialize=["B"])
def _decode_request_metadata(
    Q,
    L,
    CQ,
    CK,
    Lengths,
    Compressed,
    Legacy,
    B,
    Ratio: tl.constexpr,
    Flat: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    _request_bounds(Q, L, CQ, CK, Lengths, Compressed, Legacy, B, Ratio, Flat, BLOCK_B)


@triton.jit(do_not_specialize=["B", "P"])
def _decode_query_metadata(
    Q,
    L,
    RequestIds,
    CK,
    Legacy,
    KS,
    KE,
    Valid,
    B,
    P,
    Ratio: tl.constexpr,
    Flat: tl.constexpr,
    HAS_REQUEST_IDS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    base = tl.load(Q)
    real = tl.load(Q + B) - base
    live = (rows < P) & (rows < real)
    req = _query_requests(
        Q, RequestIds, rows, live, base, B, HAS_REQUEST_IDS, SEARCH_STEPS
    )
    start = tl.load(CK + req, live, 0)
    end = tl.load(Q + req + 1, live, base) - base
    length = tl.load(L + req, live, 0)
    visible = (length - end + rows + 1) // Ratio
    tl.store(KS + rows, tl.where(live, start, 0), rows < P)
    tl.store(KE + rows, tl.where(live, start + visible, 0), rows < P)
    tl.store(Valid + rows, tl.where(live, visible, 0), rows < P)
    if Flat:
        tl.store(Legacy + rows, 1, rows < P)


def build_indexer_decode_metadata(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    compress_ratio: int,
    flatten_decode_lens: bool,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    per_req_decode_lens: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    decode_lens: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    valid_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor | None = None,
) -> None:
    """Write request and query bounds without temporary tensors or host copies.

    ``seq_lens`` counts uncompressed tokens including this round's queries.
    Each request's queries are its last ``diff(query_start_loc)`` positions.
    Outputs use the compressed domain, except query offsets/lengths. All are
    contiguous int32 tensors, preallocated by the metadata builder.

    ``valid_lens`` sets the padded output row count. Unmapped rows receive
    ks=ke=valid_lens=0 on every invocation. Optional request IDs must describe
    the same query rows; only the real prefix is read. ``decode_lens`` keeps
    the legacy flattened-ones representation when requested; cache writers
    use ``per_req_decode_lens`` instead.
    """
    b = seq_lens.numel()
    p = valid_lens.numel()
    assert compress_ratio > 0
    assert query_start_loc.numel() == b + 1
    assert cu_seqlens_q.numel() == cu_seqlens_kv.numel() == b + 1
    assert per_req_decode_lens.numel() == compressed_seq_lens.numel() == b
    assert cu_seqlen_ks.numel() == cu_seqlen_ke.numel() == p
    assert decode_lens.numel() == (p if flatten_decode_lens else b)
    has_ids = token_to_req_indices is not None
    ids = token_to_req_indices if has_ids else query_start_loc
    if has_ids:
        assert ids.numel() >= p
    block_b = triton.next_power_of_2(max(b, 1))
    # Derive the search bound from the tile size to share compiled variants.
    search_steps = block_b.bit_length()
    common = (
        query_start_loc,
        seq_lens,
        ids,
        cu_seqlens_q,
        cu_seqlens_kv,
        per_req_decode_lens,
        compressed_seq_lens,
        decode_lens,
        cu_seqlen_ks,
        cu_seqlen_ke,
        valid_lens,
    )
    if b <= 128 and p <= 512:
        _fused_decode_metadata[(1,)](
            *common,
            b,
            p,
            compress_ratio,
            flatten_decode_lens,
            has_ids,
            search_steps,
            block_b,
            triton.next_power_of_2(max(p, 1)),
            num_warps=4,
        )
    else:
        # Larger batches avoid replicating the request scan in every token tile.
        _decode_request_metadata[(1,)](
            query_start_loc,
            seq_lens,
            cu_seqlens_q,
            cu_seqlens_kv,
            per_req_decode_lens,
            compressed_seq_lens,
            decode_lens,
            b,
            compress_ratio,
            flatten_decode_lens,
            block_b,
            num_warps=4,
        )
        if p:
            _decode_query_metadata[(triton.cdiv(p, 256),)](
                query_start_loc,
                seq_lens,
                ids,
                cu_seqlens_kv,
                decode_lens,
                cu_seqlen_ks,
                cu_seqlen_ke,
                valid_lens,
                b,
                p,
                compress_ratio,
                flatten_decode_lens,
                has_ids,
                search_steps,
                256,
                num_warps=4,
            )
