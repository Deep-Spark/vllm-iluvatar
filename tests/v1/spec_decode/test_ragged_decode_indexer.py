# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for ragged (non-uniform query length) decode in the
DSA lightning-indexer metadata path.

Context
-------
With ``num_speculative_tokens >= 2`` (``next_n >= 3``) the indexer metadata
builder sets ``use_flattening=True`` and ``split_decodes_and_prefills`` no longer
forces uniform decode query lengths, so a decode batch can be *ragged* (e.g. one
request verifying 4 MTP tokens while another only decodes 1). The indexer ops
used to ``raise NotImplementedError`` on such batches
(``requires_padding``). That guard was removed because the ilu decode path drives
``dsa_indexer_mqa_logits_with_blocks`` -- the *same* variable-q kernel prefill
uses -- via per-request ``cu_seqlens_q`` plus one per-token causal span in
``cu_seqlen_ks/ke``. Each decode token is therefore an independent logits/topk
row and ragged batches need no pack/unpack.

These tests lock the correctness of the per-token causal span computation
(``kv_spans_from_batches``) that makes that claim true. It is a pure, device
agnostic tensor function, so no GPU / ixformer kernels are required.
"""

from __future__ import annotations

import torch

from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import kv_spans_from_batches

CPU = torch.device("cpu")


def _spans(start_seq_loc, seq_len_per_batch, compress_ratio=1):
    ks, ke = kv_spans_from_batches(
        torch.tensor(start_seq_loc, dtype=torch.int32),
        torch.tensor(seq_len_per_batch, dtype=torch.int32),
        device=CPU,
        compress_ratio=compress_ratio,
    )
    return ks.tolist(), ke.tolist()


def test_ragged_decode_spans_uncompressed():
    """decode_lens=[4,4,1] (ragged), seq_lens=[10,12,7], compress_ratio=1.

    Selected tokens are the *last* ``m`` positions of each request, so token j
    (0-based) attends to ``L - m + j + 1`` causal KV positions.
    """
    # cu_seqlens_q for decode_lens [4,4,1]
    ks, ke = _spans([0, 4, 8, 9], [10, 12, 7], compress_ratio=1)

    # per-request KV start offsets in the concatenated cache: [0, 10, 22]
    assert ks == [0, 0, 0, 0, 10, 10, 10, 10, 22]
    assert ke == [7, 8, 9, 10, 19, 20, 21, 22, 29]

    kv_lens = [e - s for s, e in zip(ks, ke)]
    # req0 (L=10, last 4 -> valid counts 7..10), req1 (L=12 -> 9..12), req2 (L=7 -> 7)
    assert kv_lens == [7, 8, 9, 10, 9, 10, 11, 12, 7]


def test_ragged_decode_spans_compressed_ratio4():
    """Same ragged batch on the V4 compressed cache (compress_ratio=4).

    end = start + (L - m + pos_within) // compress_ratio.
    """
    ks, ke = _spans([0, 4, 8, 9], [40, 48, 28], compress_ratio=4)

    assert ks == [0, 0, 0, 0, 10, 10, 10, 10, 22]
    assert ke == [9, 9, 9, 10, 21, 21, 21, 22, 29]

    kv_lens = [e - s for s, e in zip(ks, ke)]
    assert kv_lens == [9, 9, 9, 10, 11, 11, 11, 12, 7]
    # valid_lens == ke - ks; MTP next_n=4 first request pos 36..39 → (pos+1)//4
    assert kv_lens[:4] == [(p + 1) // 4 for p in range(36, 40)]


def test_ragged_matches_per_request_independent():
    """The per-token causal length (ke - ks) of a request inside a ragged batch
    must equal what it would get if computed in isolation.

    This is the invariant that justifies removing the ``requires_padding`` guard:
    a neighbouring request with a different query length cannot corrupt another
    request's spans, because each token carries its own span.
    """
    decode_lens = [4, 4, 1]
    seq_lens = [10, 12, 7]
    cu = [0]
    for m in decode_lens:
        cu.append(cu[-1] + m)

    _, ke_batched = _spans(cu, seq_lens)
    ks_batched, _ = _spans(cu, seq_lens)
    kv_lens_batched = [e - s for s, e in zip(ks_batched, ke_batched)]

    kv_lens_independent: list[int] = []
    for m, seq in zip(decode_lens, seq_lens):
        ks_i, ke_i = _spans([0, m], [seq])
        kv_lens_independent.extend(e - s for s, e in zip(ks_i, ke_i))

    assert kv_lens_batched == kv_lens_independent


def test_uniform_decode_is_special_case():
    """Uniform MTP (all requests verify next_n tokens) is just the ragged path
    with equal lengths -- it must still produce correct per-token spans.
    """
    # 3 requests, each verifying next_n=2 tokens; seq_lens [5, 8, 3].
    ks, ke = _spans([0, 2, 4, 6], [5, 8, 3])

    kv_lens = [e - s for s, e in zip(ks, ke)]
    # req0 (L=5, last 2 -> 4,5), req1 (L=8 -> 7,8), req2 (L=3 -> 2,3)
    assert kv_lens == [4, 5, 7, 8, 2, 3]


def test_empty_decode_returns_empty():
    ks, ke = _spans([0], [])
    assert ks == []
    assert ke == []
