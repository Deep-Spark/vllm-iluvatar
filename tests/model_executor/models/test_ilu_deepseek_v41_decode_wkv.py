# SPDX-License-Identifier: Apache-2.0
"""CPU tests for V4.1 decode paged-slot builders (no kernel launch)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.models.deepseek_v41.decode_wkv import (
    as_wkv_cache,
    build_extra_decode_indices,
    build_swa_decode_indices,
    decode_wkv_enabled,
    mask_null_page_slots,
    pack_valid_slots,
)
from vllm_iluvatar.models.deepseek_v41.paged_caches import (
    gather_rows,
    group_slots,
    slots_for_positions,
    window_slots,
    write_cache,
)


def test_decode_wkv_enabled_defaults_on(monkeypatch):
    monkeypatch.delenv("V41_DECODE_WKV", raising=False)
    assert decode_wkv_enabled() is True
    monkeypatch.setenv("V41_DECODE_WKV", "0")
    assert decode_wkv_enabled() is False
    monkeypatch.setenv("V41_DECODE_WKV", "off")
    assert decode_wkv_enabled() is False
    monkeypatch.setenv("V41_DECODE_WKV", "yes")
    assert decode_wkv_enabled() is True


def test_as_wkv_cache_accepts_3d_and_4d():
    cache3 = torch.zeros(4, 16, 512)
    viewed = as_wkv_cache(cache3)
    assert viewed.shape == (4, 16, 1, 512)
    cache4 = torch.zeros(4, 16, 1, 512)
    assert as_wkv_cache(cache4) is cache4
    with pytest.raises(ValueError, match="3D or 4D"):
        as_wkv_cache(torch.zeros(4, 512))


def _manual_swa(block_table, block_size, reqs, positions, window_size):
    nd = positions.size(0)
    out = torch.full((nd, window_size), -1, dtype=torch.int32)
    lens = torch.zeros(nd, dtype=torch.int32)
    for t in range(nd):
        pos = int(positions[t])
        n_win = min(pos + 1, window_size)
        lo = pos + 1 - n_win
        win_pos = torch.arange(lo, pos + 1)
        slots = slots_for_positions(block_table[int(reqs[t])], win_pos, block_size)
        packed, counts = pack_valid_slots(slots.unsqueeze(0))
        out[t, :n_win] = packed[0, :n_win]
        out[t, n_win:] = -1
        lens[t] = counts[0]
    return out, lens


def test_swa_decode_indices_match_slots_for_positions():
    # req0: pos 19 (window 8 → [12, 19]); page 0 is null, page 1 holds 16..31.
    # req1: pos 4 (window → [0, 4]); first page is null so those slots are -1.
    block_size = 16
    window_size = 8
    block_table = torch.tensor(
        [
            [0, 3, 0, 0],
            [0, 5, 0, 0],
        ],
        dtype=torch.int32,
    )
    reqs = torch.tensor([0, 1], dtype=torch.int32)
    positions = torch.tensor([19, 4], dtype=torch.int64)

    got_idx, got_len = build_swa_decode_indices(
        block_table, block_size, reqs, positions, window_size
    )
    exp_idx, exp_len = _manual_swa(block_table, block_size, reqs, positions, window_size)
    torch.testing.assert_close(got_idx, exp_idx)
    torch.testing.assert_close(got_len, exp_len)
    # Null-page holes packed to the tail; lens is the valid-slot count.
    assert got_len.tolist() == [4, 0]
    assert got_idx[0, :4].tolist() == [48, 49, 50, 51]
    assert got_idx[0, 4:].tolist() == [-1, -1, -1, -1]
    assert got_idx[1].tolist() == [-1, -1, -1, -1, -1, -1, -1, -1]


def test_swa_decode_indices_short_seq_is_prefix_of_window():
    block_table = torch.tensor([[7, 0]], dtype=torch.int32)
    reqs = torch.tensor([0], dtype=torch.int32)
    positions = torch.tensor([2], dtype=torch.int64)
    idx, lens = build_swa_decode_indices(block_table, 16, reqs, positions, 8)
    assert lens.tolist() == [3]
    assert idx[0, :3].tolist() == [7 * 16 + 0, 7 * 16 + 1, 7 * 16 + 2]
    assert idx[0, 3:].tolist() == [-1, -1, -1, -1, -1]


def test_extra_decode_indices_match_group_slots():
    ratio = 2
    block_size = 16
    storage_bs = block_size // ratio
    block_table = torch.tensor(
        [
            [2, 4, 0],
            [0, 6, 0],
        ],
        dtype=torch.int32,
    )
    reqs = torch.tensor([0, 1], dtype=torch.int32)
    local = torch.tensor(
        [
            [3, 1, -1, 0],
            [8, 0, 2, -1],
        ],
        dtype=torch.int32,
    )
    positions = torch.tensor([20, 16], dtype=torch.int64)
    md0 = SimpleNamespace(block_table=block_table, block_size=block_size)

    got_idx, got_len = build_extra_decode_indices(
        block_table, storage_bs, reqs, local, positions, ratio
    )
    for t in range(2):
        valid = local[t] >= 0
        eff = min((int(positions[t]) + 1) // ratio, local.size(-1))
        keep = valid & (torch.arange(local.size(-1)) < eff)
        slots = group_slots(md0, int(reqs[t]), local[t].clamp(min=0), ratio)
        expected = torch.where(keep, slots, slots.new_full((), -1))
        torch.testing.assert_close(got_idx[t], expected)
        assert int(got_len[t]) == int((expected >= 0).sum())


def test_extra_positions_drop_stale_columns():
    # pos=3, ratio=2 → only first (3+1)//2 = 2 columns may contribute.
    block_table = torch.tensor([[3, 0]], dtype=torch.int32)
    reqs = torch.tensor([0], dtype=torch.int32)
    local = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    idx, lens = build_extra_decode_indices(
        block_table, 8, reqs, local, torch.tensor([3]), compress_ratio=2
    )
    assert lens.tolist() == [2]
    assert idx[0, 2:].tolist() == [-1, -1]


def test_mask_null_page_slots_clears_block_zero_only():
    slots = torch.tensor([0, 15, 16, 31, -1], dtype=torch.int32)
    out = mask_null_page_slots(slots, 16)
    assert out.tolist() == [-1, -1, 16, 31, -1]


def test_pack_valid_slots_moves_holes_to_the_tail():
    # Prefix-reuse compress block_table often has 0-holes in the middle.
    # CUDA extra_len still counts those; v3 then walks -1 and smashes the heap.
    slots = torch.tensor([[5, -1, 12, -1, 8], [-1, -1, 3, -1, -1]], dtype=torch.int32)
    packed, counts = pack_valid_slots(slots)
    assert counts.tolist() == [3, 1]
    assert packed[0].tolist() == [5, 12, 8, -1, -1]
    assert packed[1].tolist() == [3, -1, -1, -1, -1]


def test_pack_valid_slots_repairs_cuda_null_page_len():
    # CUDA extra-index kernel counts phys_block==0; slot = 0*bs+off lands in
    # [0, storage_bs) and is later masked to -1. extra_len stays inflated.
    storage_bs = 8
    cuda_slots = torch.tensor([[16, 0, 32, -1]], dtype=torch.int32)
    cuda_len = 3
    masked = mask_null_page_slots(cuda_slots, storage_bs)
    packed, counts = pack_valid_slots(masked)
    assert int(cuda_len) == 3
    assert counts.tolist() == [2]
    assert packed[0].tolist() == [16, 32, -1, -1]


def test_swa_builder_matches_ixformer_ref_after_null_mask():
    from ixformer.inference.functions.deepseek.decode_attn import (
        ref_ds4_compute_swa_decode_indices_and_lens,
    )

    block_size = 16
    window_size = 8
    block_table = torch.tensor([[0, 3, 0], [5, 0, 0]], dtype=torch.int32)
    reqs = torch.tensor([0, 1], dtype=torch.int32)
    positions = torch.tensor([19, 4], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    seq_lens = torch.tensor([20, 5], dtype=torch.int32)
    valid = torch.ones(2, dtype=torch.bool)

    ref_idx = torch.full((2, window_size), -1, dtype=torch.int32)
    ref_len = torch.zeros(2, dtype=torch.int32)
    ref_ds4_compute_swa_decode_indices_and_lens(
        ref_idx,
        ref_len,
        window_size,
        query_start_loc,
        seq_lens,
        reqs,
        valid,
        block_table,
        block_size,
        2,
    )
    ref_idx, ref_len = pack_valid_slots(mask_null_page_slots(ref_idx, block_size))

    got_idx, got_len = build_swa_decode_indices(
        block_table, block_size, reqs, positions, window_size
    )
    torch.testing.assert_close(got_idx, ref_idx)
    torch.testing.assert_close(got_len, ref_len)


def test_extra_builder_matches_ixformer_ref_after_null_mask():
    from ixformer.inference.functions.deepseek.decode_attn import (
        ref_ds4_compute_global_topk_indices_and_lens,
    )

    ratio = 2
    storage_bs = 8
    block_table = torch.tensor([[2, 0], [0, 6]], dtype=torch.int32)
    reqs = torch.tensor([0, 1], dtype=torch.int32)
    local = torch.tensor([[1, 0, -1], [8, 2, -1]], dtype=torch.int32)
    positions = torch.tensor([20, 16], dtype=torch.int64)
    valid = torch.ones(2, dtype=torch.bool)

    ref_idx, ref_len = ref_ds4_compute_global_topk_indices_and_lens(
        local, reqs, block_table, storage_bs, valid, positions, ratio
    )
    ref_idx = mask_null_page_slots(ref_idx, storage_bs)
    # Ref counts phys==0 as a hit; after the mask those slots are -1.
    ref_len = (ref_idx >= 0).sum(-1, dtype=torch.int32)

    got_idx, got_len = build_extra_decode_indices(
        block_table, storage_bs, reqs, local, positions, ratio
    )
    torch.testing.assert_close(got_idx, ref_idx)
    torch.testing.assert_close(got_len, ref_len)


_HEAD = 512
_SCALE = _HEAD**-0.5


def _md(block_table: torch.Tensor, block_size: int) -> SimpleNamespace:
    return SimpleNamespace(block_table=block_table, block_size=block_size)


def _fill_window_cache(
    cache: torch.Tensor,
    md,
    req: int,
    n_tokens: int,
    seed: int,
) -> torch.Tensor:
    """Write ``n_tokens`` unique rows into the window page of ``req``."""
    g = torch.Generator().manual_seed(seed)
    rows = torch.randn(n_tokens, _HEAD, generator=g)
    pos = torch.arange(n_tokens)
    write_cache(cache, rows, window_slots(md, req, pos))
    return rows


def _fill_compress_cache(
    cache: torch.Tensor,
    md,
    req: int,
    n_groups: int,
    ratio: int,
    seed: int,
) -> None:
    g = torch.Generator().manual_seed(seed)
    rows = torch.randn(n_groups, _HEAD, generator=g)
    groups = torch.arange(n_groups)
    write_cache(cache, rows, group_slots(md, req, groups, ratio))


def _old_gather_decode(
    q: torch.Tensor,
    window_kv: torch.Tensor,
    window_cache: torch.Tensor,
    window_md,
    positions: torch.Tensor,
    seq_lens: list[int],
    window_size: int,
    attn_sink: torch.Tensor,
    compress_cache: torch.Tensor | None = None,
    compress_md=None,
    ratio: int = 1,
    group_idxs: torch.Tensor | None = None,
    group_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """CPU copy of ``prefill_attn``'s decode gather + plane indices."""
    from ixformer.inference.functions.deepseek.prefill_attn import (
        ref_ds4_flash_mla_sparse_fwd,
    )

    nd = q.size(0)
    spans = [(i, i, i + 1, seq_lens[i]) for i in range(nd)]
    plane_parts: list[torch.Tensor] = []
    base = 0
    win_bases: list[int] = []
    grp_bases: list[int] = []
    n_groups: list[int] = []
    win_origins: list[int] = []
    n_win_max = min(window_size, max(seq_lens))

    for req, t0, t1, seq_len in spans:
        n_g = seq_len // ratio if compress_cache is not None else 0
        if n_g:
            slots = group_slots(
                compress_md, req, torch.arange(n_g, device=q.device), ratio
            )
            plane_parts.append(gather_rows(compress_cache, slots))
        grp_bases.append(base)
        base += n_g
        n_groups.append(n_g)

        start_pos = seq_len - (t1 - t0)
        if start_pos > 0:
            win_lo = max(0, start_pos + 1 - window_size)
            hist = start_pos - win_lo
            parts = []
            if hist:
                win_pos = torch.arange(win_lo, start_pos)
                parts.append(gather_rows(window_cache, window_slots(window_md, req, win_pos)))
            parts.append(window_kv[t0:t1])
            plane_parts.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=0))
            win_origins.append(win_lo)
            win_bases.append(base)
            base += hist + (t1 - t0)
        else:
            plane_parts.append(window_kv[t0:t1])
            win_origins.append(start_pos)
            win_bases.append(base)
            base += t1 - t0

    plane = torch.cat(plane_parts, dim=0)
    topk_w = 0 if group_idxs is None else group_idxs.size(-1)
    width = n_win_max + topk_w
    idxs = torch.full((nd, width), -1, dtype=torch.int32)
    lens = torch.zeros(nd, dtype=torch.int32)
    col = torch.arange(width)
    for (_req, t0, t1, _sl), wb, gb, n_g, wlo in zip(
        spans, win_bases, grp_bases, n_groups, win_origins
    ):
        pos = positions[t0:t1].to(torch.int64)
        n_win = (pos + 1).clamp(max=window_size).unsqueeze(1)
        first = (pos + 1).unsqueeze(1) - n_win
        c = col.unsqueeze(0)
        win_val = wb + (first - wlo) + c
        if n_g and group_idxs is not None:
            g = group_idxs[t0:t1].to(torch.int64)
            gl = group_lens[t0:t1].unsqueeze(1)
            gi = (c - n_win).clamp(0, g.size(-1) - 1)
            grp_val = gb + g.gather(1, gi.expand(g.size(0), -1))
            row = torch.where(c < n_win, win_val, torch.where(c < n_win + gl, grp_val, -1))
            lens[t0:t1] = (n_win + gl).squeeze(1).to(torch.int32)
        else:
            row = torch.where(c < n_win, win_val, -1)
            lens[t0:t1] = n_win.squeeze(1).to(torch.int32)
        idxs[t0:t1] = row.to(torch.int32)

    return ref_ds4_flash_mla_sparse_fwd(
        q,
        plane.unsqueeze(1),
        idxs.unsqueeze(1).contiguous(),
        _SCALE,
        d_v=_HEAD,
        attn_sink=attn_sink,
        topk_length=lens,
    )


def _new_wkv_decode(
    q: torch.Tensor,
    window_cache: torch.Tensor,
    window_md,
    positions: torch.Tensor,
    window_size: int,
    attn_sink: torch.Tensor,
    compress_cache: torch.Tensor | None = None,
    compress_md=None,
    ratio: int = 0,
    group_idxs: torch.Tensor | None = None,
) -> torch.Tensor:
    from ixformer.inference.functions.deepseek.decode_attn import (
        ref_ds4_flash_mla_with_kvcache,
    )

    reqs = torch.arange(q.size(0), dtype=torch.int32)
    swa_idx, swa_len = build_swa_decode_indices(
        window_md.block_table, window_md.block_size, reqs, positions, window_size
    )
    extra = extra_idx = extra_len = None
    if ratio > 0 and compress_cache is not None and group_idxs is not None:
        extra_idx, extra_len = build_extra_decode_indices(
            compress_md.block_table,
            compress_md.block_size // ratio,
            reqs,
            group_idxs,
            positions,
            ratio,
        )
        extra = as_wkv_cache(compress_cache)
        extra_idx = extra_idx.unsqueeze(1)
    out = ref_ds4_flash_mla_with_kvcache(
        q.unsqueeze(1),
        as_wkv_cache(window_cache),
        swa_idx.unsqueeze(1),
        swa_len,
        _SCALE,
        attn_sink=attn_sink,
        extra_k_cache=extra,
        extra_indices=extra_idx,
        extra_topk_length=extra_len,
    )
    return out.squeeze(1)


def test_cpu_ref_decode_swa_only_matches_gather_path():
    torch.manual_seed(0)
    block_size = 16
    window_size = 8
    # req0 pos=19 uses pages 2 (0..15) and 3 (16..19); req1 pos=10 uses page 4.
    block_table = torch.tensor([[2, 3, 0], [4, 0, 0]], dtype=torch.int32)
    md = _md(block_table, block_size)
    cache = torch.zeros(8, block_size, _HEAD)
    rows0 = _fill_window_cache(cache, md, 0, 20, seed=1)
    rows1 = _fill_window_cache(cache, md, 1, 11, seed=2)

    positions = torch.tensor([19, 10], dtype=torch.int64)
    seq_lens = [20, 11]
    window_kv = torch.stack([rows0[19], rows1[10]])
    q = torch.randn(2, 4, _HEAD)
    sink = torch.randn(4)

    old = _old_gather_decode(
        q, window_kv, cache, md, positions, seq_lens, window_size, sink
    )
    new = _new_wkv_decode(q, cache, md, positions, window_size, sink)
    torch.testing.assert_close(new, old, atol=1e-5, rtol=1e-5)


def test_cpu_ref_decode_ratio2_extra_matches_gather_path():
    torch.manual_seed(1)
    block_size = 16
    window_size = 8
    ratio = 2
    win_table = torch.tensor([[2, 3, 0], [4, 0, 0]], dtype=torch.int32)
    cmp_table = torch.tensor([[5, 0, 0], [6, 0, 0]], dtype=torch.int32)
    win_md = _md(win_table, block_size)
    cmp_md = _md(cmp_table, block_size)
    window_cache = torch.zeros(8, block_size, _HEAD)
    compress_cache = torch.zeros(8, block_size, _HEAD)
    rows0 = _fill_window_cache(window_cache, win_md, 0, 20, seed=3)
    rows1 = _fill_window_cache(window_cache, win_md, 1, 11, seed=4)
    _fill_compress_cache(compress_cache, cmp_md, 0, 10, ratio, seed=5)
    _fill_compress_cache(compress_cache, cmp_md, 1, 5, ratio, seed=6)

    positions = torch.tensor([19, 10], dtype=torch.int64)
    seq_lens = [20, 11]
    window_kv = torch.stack([rows0[19], rows1[10]])
    # Compact local groups, same contract as ``topk_groups``.
    group_idxs = torch.tensor([[3, 0, -1], [4, 1, -1]], dtype=torch.int32)
    group_lens = torch.tensor([2, 2], dtype=torch.int32)
    q = torch.randn(2, 4, _HEAD)
    sink = torch.randn(4)

    old = _old_gather_decode(
        q,
        window_kv,
        window_cache,
        win_md,
        positions,
        seq_lens,
        window_size,
        sink,
        compress_cache=compress_cache,
        compress_md=cmp_md,
        ratio=ratio,
        group_idxs=group_idxs,
        group_lens=group_lens,
    )
    new = _new_wkv_decode(
        q,
        window_cache,
        win_md,
        positions,
        window_size,
        sink,
        compress_cache=compress_cache,
        compress_md=cmp_md,
        ratio=ratio,
        group_idxs=group_idxs,
    )
    torch.testing.assert_close(new, old, atol=1e-5, rtol=1e-5)
