"""Decode-side paged MLA: global slots + ``ds4_flash_mla_with_kvcache`` v3.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_iluvatar.models.deepseek_v41.paged_caches import V41AttentionMetadata

_TRUE = ("1", "true", "yes", "on")


def decode_wkv_enabled() -> bool:
    return os.environ.get("V41_DECODE_WKV", "1").strip().lower() in _TRUE


def prefill_as_decode_enabled() -> bool:
    """Route prefill through this file too, instead of building a dense plane.

    ``ds4_compute_swa_decode_indices_and_lens`` derives each token's position
    from ``query_start_loc`` / ``seq_lens``, so it covers a multi-token query
    unchanged; ``b = num_tokens, s_q = 1`` then makes ``topk_length``'s
    per-batch semantics per-token.
    """
    return os.environ.get("V41_PREFILL_AS_DECODE", "0").strip().lower() in _TRUE


def as_wkv_cache(cache: torch.Tensor) -> torch.Tensor:
    if cache.dim() == 4:
        if cache.size(-2) != 1:
            raise ValueError(
                f"paged decode cache expected h_kv=1 at dim -2, got {tuple(cache.shape)}"
            )
        return cache
    if cache.dim() == 3:
        return cache.unsqueeze(-2)
    raise ValueError(
        f"paged decode cache must be 3D or 4D, got {tuple(cache.shape)}"
    )


def mask_null_page_slots(
    slots: torch.Tensor, storage_block_size: int
) -> torch.Tensor:
    """Map slots that land in vLLM's null block (page 0) back to ``-1``.

    Off the production path: both index builders now require ``phys_block > 0``
    before emitting a slot, and a slot from a real block is always
    ``>= storage_block_size``, so this can no longer match anything they
    produce. Kept for the XID-24 repro, which replays raw kernel output.
    """
    if storage_block_size <= 0:
        return slots
    null_page = (slots >= 0) & (slots < storage_block_size)
    return torch.where(null_page, slots.new_full((), -1), slots)


def pack_valid_slots(
    slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack nonnegative slots to the front of each row.

    ``ds4_flash_mla_with_kvcache`` v3 sets ``need_gather_indices=false`` and
    walks the first ``extra_len`` columns. CUDA extra-index translation counts
    null page 0 as valid; we then mask those to ``-1``. Leaving the holes in
    place (or keeping the pre-mask length) makes the kernel load slot ``-1``.
    Prefix reuse punches more 0s into the compress ``block_table``, which is
    why extra+prefix detonates and extra-off / prefix-off do not.
    """
    if slots.numel() == 0:
        return slots, slots.new_zeros(slots.size(0) if slots.ndim else 0, dtype=torch.int32)
    valid = slots >= 0
    counts = valid.sum(-1, dtype=torch.int32)
    order = valid.to(torch.int32).argsort(dim=-1, descending=True, stable=True)
    packed = slots.gather(-1, order)
    keep = torch.arange(slots.size(-1), device=slots.device) < counts.unsqueeze(-1)
    packed = torch.where(keep, packed, packed.new_full((), -1))
    return packed, counts


def fill_walk_padding(slots: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Replace ``-1`` tail with the last valid slot (or 0).

    Even after pack, v3 ``fetchKFast`` (gather=false) DMA-loads a whole
    ``K_ROWS`` tile. Columns past ``extra_len`` stay ``-1`` and become
    ``(size_t)-1 * stride``. Duplicating the last in-range slot keeps those
    loads inside the cache; ``extra_len`` still truncates the softmax.
    """
    if slots.numel() == 0:
        return slots
    last = (lens.to(torch.int64) - 1).clamp(min=0)
    rows = torch.arange(slots.size(0), device=slots.device)
    filler = slots[rows, last].clamp(min=0)
    return torch.where(slots >= 0, slots, filler.unsqueeze(-1))


def cache_slot_limit(cache: torch.Tensor) -> int:
    """Rows a paged cache exposes to v3's dense slot walk.

    Cache is ``[num_blocks, page_block_size, 1, D]``. A slot ``>= num_blocks *
    page_block_size`` (or negative) is an OOB load in ``cuInferDSV4Attention``
    when ``need_gather_indices=false``.
    """
    return int(cache.size(0) * cache.size(1))


def clamp_slots_to_cache(slots: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    """Drop slots that ixinfer v3 would use as dense extra/SWA row ids.

    Off the CUDA path: the index builders take ``slot_limit`` and apply this
    while they emit, before they compact. Still used by the CPU fallback and by
    the XID-24 repro, which replays raw kernel output.
    """
    limit = cache_slot_limit(cache)
    ok = (slots >= 0) & (slots < limit)
    return torch.where(ok, slots, slots.new_full((), -1))


def finalize_rows_torch(
    slots: torch.Tensor, lens: torch.Tensor, slot_limit: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Python mirror of what the CUDA builders now do in place.

    Only the CPU fallback needs it: drop slots the cache cannot hold, compact
    what is left, then repeat the last kept slot across the tail that v3's
    unchecked tile load still reads.
    """
    del lens
    if slot_limit > 0:
        ok = (slots >= 0) & (slots < slot_limit)
        slots = torch.where(ok, slots, slots.new_full((), -1))
    slots, counts = pack_valid_slots(slots)
    return fill_walk_padding(slots, counts), counts


def build_swa_decode_indices(
    block_table: torch.Tensor,
    block_size: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal SWA global slots from ``positions``. Block 0 → ``-1``.
    """
    nd = int(positions.size(0))
    device = positions.device
    pos = positions.to(torch.int64)
    n_win = (pos + 1).clamp(max=window_size)
    lo = pos + 1 - n_win
    offs = torch.arange(window_size, device=device, dtype=torch.int64)
    raw = lo.unsqueeze(1) + offs
    in_win = offs.unsqueeze(0) < n_win.unsqueeze(1)

    reqs = token_to_req_indices[:nd].to(torch.int64)
    n_reqs, max_blocks = int(block_table.size(0)), int(block_table.size(1))
    if n_reqs == 0 or max_blocks == 0:
        empty = pos.new_full((nd, window_size), -1, dtype=torch.int32)
        return empty, n_win.to(torch.int32)

    blk_idx = torch.div(raw, block_size, rounding_mode="floor")
    in_range = in_win & (blk_idx >= 0) & (blk_idx < max_blocks)
    safe_req = reqs.clamp(0, n_reqs - 1)
    safe_blk = blk_idx.clamp(0, max_blocks - 1)
    blocks = (
        block_table.to(torch.int64)
        .index_select(0, safe_req.reshape(-1))
        .gather(1, safe_blk)
    )
    slots = blocks * block_size + raw % block_size
    ok = in_range & (blocks > 0)
    indices = torch.where(ok, slots, slots.new_full((), -1)).to(torch.int32)
    return pack_valid_slots(indices)


def build_extra_decode_indices(
    block_table: torch.Tensor,
    storage_block_size: int,
    token_to_req_indices: torch.Tensor,
    local_idxs: torch.Tensor,
    positions: torch.Tensor | None = None,
    compress_ratio: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local compressed-group ids to global slots. Block 0 → ``-1``.

    ``storage_block_size`` is the compressed-domain page
    (``cache_md.block_size // ratio``), matching ``gather_group_rows``.

    Packed like the CUDA builder: v3's dense walk reads the first
    ``topk_lens`` columns, so a correct count with a mid-row hole still loads
    slot ``-1``.
    """
    nd, topk = local_idxs.shape
    local = local_idxs.to(torch.int64)
    valid = local >= 0
    if positions is not None and compress_ratio > 0:
        eff = ((positions.to(torch.int64) + 1) // compress_ratio).clamp(0, topk)
        col = torch.arange(topk, device=local.device)
        valid = valid & (col.unsqueeze(0) < eff.unsqueeze(1))

    reqs = token_to_req_indices[:nd].to(torch.int64)
    n_reqs, max_blocks = int(block_table.size(0)), int(block_table.size(1))
    if n_reqs == 0 or max_blocks == 0 or storage_block_size <= 0:
        empty = local.new_full(local.shape, -1, dtype=torch.int32)
        return empty, local.new_zeros(nd, dtype=torch.int32)

    blk_idx = torch.div(local, storage_block_size, rounding_mode="floor")
    blk_off = local % storage_block_size
    in_range = valid & (blk_idx >= 0) & (blk_idx < max_blocks)
    safe_req = reqs.clamp(0, n_reqs - 1)
    safe_blk = blk_idx.clamp(0, max_blocks - 1)
    blocks = (
        block_table.to(torch.int64)
        .index_select(0, safe_req.reshape(-1))
        .gather(1, safe_blk)
    )
    slots = blocks * storage_block_size + blk_off
    ok = in_range & (blocks > 0)
    indices = torch.where(ok, slots, slots.new_full((), -1)).to(torch.int32)
    return pack_valid_slots(indices)


def _swa_indices_cuda(
    md: V41AttentionMetadata,
    nd: int,
    window_size: int,
    device: torch.device,
    ixfops,
    slot_limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # pad_walk_tail makes the kernel write every row in [0, nd), including
    # padding tokens, so neither buffer needs a sentinel prefill.
    swa_indices = torch.empty((nd, window_size), dtype=torch.int32, device=device)
    swa_lens = torch.empty(nd, dtype=torch.int32, device=device)
    valid = md.slot_mapping[:nd] >= 0 if md.slot_mapping.numel() >= nd else torch.ones(
        nd, dtype=torch.bool, device=device
    )
    ixfops.ds4_compute_swa_decode_indices_and_lens(
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        window_size=window_size,
        query_start_loc=md.query_start_loc,
        seq_lens=md.seq_lens,
        token_to_req_indices=md.token_to_req_indices,
        is_valid_token=valid,
        block_table=md.block_table,
        block_size=md.block_size,
        num_decode_tokens=nd,
        slot_limit=slot_limit,
        pad_walk_tail=True,
    )
    # The kernel drops page 0 and out-of-cache slots, compacts the row, writes
    # the packed count and pads the walked tail, so the row is already in the
    # shape v3 wants.
    return swa_indices, swa_lens


def _extra_indices_cuda(
    cache_md: V41AttentionMetadata,
    local_idxs: torch.Tensor,
    positions: torch.Tensor,
    ratio: int,
    token_to_req_indices: torch.Tensor,
    is_valid: torch.Tensor,
    ixfops,
    slot_limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    storage_bs = cache_md.block_size // ratio
    # Same contract as the SWA builder: page 0 and out-of-cache slots dropped,
    # row compacted, extra_lens holding the packed count, walked tail padded.
    return ixfops.ds4_compute_global_topk_indices_and_lens(
        local_idxs.contiguous(),
        token_to_req_indices.contiguous(),
        cache_md.block_table.contiguous(),
        storage_bs,
        is_valid,
        positions,
        ratio,
        slot_limit,
        True,
    )


def decode_swa_indices(
    md: V41AttentionMetadata,
    positions: torch.Tensor,
    window_size: int,
    ixfops=None,
    slot_limit: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    nd = int(positions.size(0))
    use_cuda = (
        positions.is_cuda
        and ixfops is not None
        and hasattr(ixfops, "ds4_compute_swa_decode_indices_and_lens")
    )
    if use_cuda:
        return _swa_indices_cuda(
            md, nd, window_size, positions.device, ixfops, slot_limit
        )
    return finalize_rows_torch(
        *build_swa_decode_indices(
            md.block_table,
            md.block_size,
            md.token_to_req_indices,
            positions,
            window_size,
        ),
        slot_limit,
    )


def decode_extra_indices(
    cache_md: V41AttentionMetadata,
    local_idxs: torch.Tensor,
    positions: torch.Tensor,
    ratio: int,
    token_to_req_indices: torch.Tensor,
    is_valid: torch.Tensor | None = None,
    ixfops=None,
    slot_limit: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    nd = local_idxs.size(0)
    t2r = token_to_req_indices[:nd]
    use_cuda = (
        local_idxs.is_cuda
        and ixfops is not None
        and hasattr(ixfops, "ds4_compute_global_topk_indices_and_lens")
    )
    if use_cuda:
        valid = (
            is_valid
            if is_valid is not None
            else torch.ones(nd, dtype=torch.bool, device=local_idxs.device)
        )
        return _extra_indices_cuda(
            cache_md, local_idxs, positions, ratio, t2r, valid, ixfops, slot_limit
        )
    return finalize_rows_torch(
        *build_extra_decode_indices(
            cache_md.block_table,
            cache_md.block_size // ratio,
            t2r,
            local_idxs,
            positions,
            ratio,
        ),
        slot_limit,
    )


def decode_attn_wkv(
    q: torch.Tensor,
    window_cache: torch.Tensor,
    window_md: V41AttentionMetadata,
    positions: torch.Tensor,
    window_size: int,
    attn_sink: torch.Tensor,
    scale: float,
    compress_ratio: int = 0,
    compress_cache: torch.Tensor | None = None,
    compress_md: V41AttentionMetadata | None = None,
    group_idxs: torch.Tensor | None = None,
    group_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged decode attention. ``q`` is ``(nd, H, 512)``; returns the same shape.

    Inverse RoPE stays outside this call (``forward`` still does
    ``apply_rope(..., inverse=True)``).
    """
    del group_lens
    import ixformer.inference.functions as ixfops

    window_cache = as_wkv_cache(window_cache)
    # The builders take the cache size and hand back rows already in v3's
    # shape; clamping, packing and walk-padding here cost three more passes
    # over every row, per layer, for a result the kernel can produce as it
    # emits.
    swa_idx, swa_len = decode_swa_indices(
        window_md,
        positions,
        window_size,
        ixfops=ixfops,
        slot_limit=cache_slot_limit(window_cache),
    )

    extra_cache = extra_idx = extra_len = None
    if (
        compress_ratio > 0
        and compress_cache is not None
        and compress_md is not None
        and group_idxs is not None
        and group_idxs.size(-1) > 0
    ):
        extra_cache = as_wkv_cache(compress_cache)
        extra_idx, extra_len = decode_extra_indices(
            compress_md,
            group_idxs,
            positions,
            compress_ratio,
            compress_md.token_to_req_indices,
            is_valid=(
                compress_md.slot_mapping[: q.size(0)] >= 0
                if compress_md.slot_mapping.numel() >= q.size(0)
                else None
            ),
            ixfops=ixfops,
            slot_limit=cache_slot_limit(extra_cache),
        )
        extra_idx = extra_idx.unsqueeze(1).contiguous()

    out = ixfops.ds4_flash_mla_with_kvcache(
        q=q.unsqueeze(1).contiguous(),
        k_cache=as_wkv_cache(window_cache),
        indices=swa_idx.unsqueeze(1).contiguous(),
        topk_length=swa_len.contiguous(),
        sm_scale=scale,
        attn_sink=attn_sink,
        extra_k_cache=extra_cache,
        extra_indices=extra_idx,
        extra_topk_length=extra_len,
        version=3,
    )
    return out.squeeze(1)
