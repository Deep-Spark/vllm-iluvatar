"""Compressor state/pool/norm and compressed-cache insert.

Mirrors ``vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py``:
same function names, same arguments, same contract, so the callers read exactly
like the official ones. Official drives both entry points with Triton, which
does not run on this platform, so each body dispatches to the kernel covering
the same work.

The ratio-2 pooling is a plugin kernel rather than ixformer's
``ds4_v41_compress_prepare_out``. That one gathers a group's rows from the state
cache by absolute position, ``block_table[position / capacity]``, so it wants a
state cache paged over the whole context -- which is what the 0.27.1
``IluCompressorStateSpec`` handed it. V4.1's ``CircularBufferSpec`` reserves one
block per request instead, and every group past the first ``capacity`` positions
then indexes off the end of the block-table row. ``v41_compress_pool_norm``
follows official's division of labour, where only a chunk's first token reads
the ring and every other group is pooled from the batch, which is what makes one
block per request enough.

The cache record is whatever ``kv_cache``'s dtype says: the packed fp8_ds_mla
row (448 E4M3 NoPE bytes, 64 BF16 RoPE values, 7 UE8M0 scales), or a plain
[448 NoPE | 64 RoPE] bf16 / e4m3 row.
"""

from __future__ import annotations

import torch


def fused_save_compress_norm(
    kv_score: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    token_to_req_indices: torch.Tensor | None,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    compress_ratio: int,
    latent_out: torch.Tensor,
) -> None:
    """Pool each closed group into a normalized BF16 latent; save FP32 states.

    The latent feeds the main-cache insert and the indexer K path, which the
    attention layer schedules on separate streams.

    Ratio 2 keeps one ring block per request holding the open group's rows:
    position ``p`` lives in row ``p % capacity`` and ``slot_mapping`` encodes
    ``block * capacity + p % capacity``. A request's program handles the group
    that the chunk's first token closes with its predecessor's ring row, then
    stores the chunk's last ``capacity`` rows; every group that ends inside the
    batch is pooled from ``kv_score`` and never touches the ring. Ratio 1 has
    no ring, and ``slot_mapping`` then only marks valid tokens.

    Args:
        kv_score: FP32 [tokens, 512] for CR1, [tokens, 1024] for CR2.
        positions: Absolute positions of the packed request tokens.
        state_cache: Ring FP32 [blocks, capacity, 1024] KV/score states (CR2).
        slot_mapping: Ring slots (CR2) or main-cache slots (CR1). Negative
            marks a token that publishes nothing.
        query_start_loc: [num_reqs + 1] token offsets of each request's chunk.
        token_to_req_indices: Request indices for the packed token rows.
        rms_norm_weight: BF16 [512] normalization weight.
        rms_norm_eps: RMSNorm epsilon.
        compress_ratio: Group size, either 1 or 2.
        latent_out: BF16 [tokens, 512], written only at valid group boundaries.

    """
    assert compress_ratio in (1, 2)
    assert kv_score.dtype == torch.float32
    assert kv_score.shape[1] == 512 * compress_ratio and kv_score.stride(1) == 1
    assert latent_out.shape == (kv_score.shape[0], 512)
    assert latent_out.is_contiguous() and latent_out.dtype == torch.bfloat16
    assert positions.is_contiguous() and slot_mapping.is_contiguous()

    num_tokens = slot_mapping.numel()
    if num_tokens == 0:
        return

    if compress_ratio == 1:
        # Ratio 1 pools nothing and carries no state, so the latent is just the
        # normalized row.
        torch.ops._C.v41_compress_norm_cr1(
            kv_score,
            slot_mapping,
            rms_norm_weight,
            rms_norm_eps,
            latent_out,
        )
        return

    assert state_cache is not None and token_to_req_indices is not None
    assert query_start_loc is not None
    torch.ops._C.v41_compress_pool_norm(
        kv_score,
        positions,
        state_cache,
        slot_mapping,
        query_start_loc,
        token_to_req_indices,
        rms_norm_weight,
        rms_norm_eps,
        latent_out,
    )


def rope_quant_insert(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    row_out: torch.Tensor | None = None,
) -> None:
    """Apply GPT-J RoPE and publish a latent into the compressed KV cache.

    ``slot_mapping`` is already compressed-domain: the metadata builder runs
    ``get_compressed_slot_mapping``, whose slot is the global compressed row
    index ``page * rows_per_page + row``. The kernel splits it the way the
    reference does, so no block table is involved.

    ``kv_cache``'s dtype selects the record: ``uint8`` with a 584-byte row is
    the packed fp8_ds_mla one, ``bfloat16`` the plain [448 NoPE | 64 RoPE] row.
    Official also offers a per-tensor-fp8 plain row and takes an ``fp8_scale``
    for it; nothing on this platform reads that record, so neither the record
    nor the argument is carried here.

    ``row_out``, when a pipeline cut leaves this layer's consumers on a later
    rank, receives the same record in token order so it can cross the boundary
    and be replayed there.
    """
    if slot_mapping.numel() == 0:
        return
    torch.ops._C.v41_rope_insert(
        latent,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        compress_ratio,
        row_out,
    )


def move_rows_to_paged_cache(
    rows: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    compress_ratio: int,
) -> None:
    """Replay records received from an earlier pipeline rank into a local cache.

    ``rows`` is the dense ``[num_tokens, record_width]`` mirror the producing
    rank filled through ``row_out``; ``slot_mapping`` is this rank's. Nothing
    is recomputed, so the replica holds the producer's bytes exactly.
    """
    if slot_mapping.numel() == 0:
        return
    torch.ops._C.v41_move_rows_to_paged_cache(
        rows, kv_cache, slot_mapping, positions, compress_ratio
    )
