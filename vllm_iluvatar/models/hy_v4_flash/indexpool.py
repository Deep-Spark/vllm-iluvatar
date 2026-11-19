# SPDX-License-Identifier: Apache-2.0
"""Gated four-token IndexPool with paged keys and causal group selection."""

import torch
import triton
import triton.language as tl
from torch import nn
from vllm.model_executor.layers.layernorm import RMSNorm

from .attention import _paged_indices, _store_rows


@triton.jit
def _group_inputs(
    NEW,
    REQ,
    START,
    SEQS,
    TABLE,
    SLOT,
    KEYS,
    OUT,
    CLOSE,
    POS,
    LIVE,
    PAGE: tl.constexpr,
    STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    t = tl.program_id(0)
    req = tl.load(REQ + t, t < LIVE, other=0)
    first = tl.load(START + req)
    end = tl.load(START + req + 1)
    p = tl.load(SEQS + req) - end + t
    live = (t < LIVE) & (t < end)
    block = tl.load(TABLE + req.to(tl.int64) * STRIDE + p // PAGE, live, other=0)
    slot = block.to(tl.int64) * PAGE + (p % PAGE) // 4 * 4
    j = tl.arange(0, 4)
    dim = tl.arange(0, 128)
    closed = live & (p % 4 == 3)
    query_row = t - 3 + j
    current = closed & (query_row >= first)
    raw = tl.load(
        KEYS + (slot + j[:, None]) * 128 + dim[None, :],
        closed & ~current[:, None],
        other=0.0,
    )
    fresh = tl.load(
        NEW + query_row[:, None].to(tl.int64) * 128 + dim[None, :],
        current[:, None],
        other=0.0,
    )
    raw = tl.where(current[:, None], fresh, raw)
    tl.store(OUT + (t * 4 + j[:, None]) * 128 + dim[None, :], raw)
    physical = tl.load(SLOT + t, live, other=-1).to(tl.int64)
    store = (physical >= 0) & (physical < CAPACITY)
    value = tl.load(NEW + t.to(tl.int64) * 128 + dim, store, other=0.0)
    tl.store(KEYS + physical * 128 + dim, value, store)
    tl.store(CLOSE + t, tl.where(closed, slot // 4, -1))
    tl.store(POS + t, tl.where(live, p - p % 4, 0))


@triton.jit
def _pool4(VALUES, LOGITS, APE, NORM, OUT, EPS: tl.constexpr):
    token = tl.program_id(0)
    j = tl.arange(0, 4)
    d = tl.arange(0, 128)
    offset = j[:, None] * 128 + d[None, :]
    scores = tl.load(LOGITS + token * 512 + offset) + tl.load(APE + offset)
    exps = tl.exp(scores - tl.max(scores, 0)[None, :])
    probs = exps / tl.sum(exps, 0)[None, :]
    values = tl.load(VALUES + token * 512 + offset)
    pooled = tl.sum(values * probs, 0)
    inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / 128 + EPS)
    tl.store(OUT + token * 128 + d, pooled * inv * tl.load(NORM + d))


@triton.jit
def _scores(
    Q,
    WEIGHTS,
    KEYS,
    REQ,
    START,
    SEQS,
    TABLE,
    LOGITS,
    LIVE,
    GROUPS,
    PAGE: tl.constexpr,
    STRIDE: tl.constexpr,
    HEAD_SCALE: tl.constexpr = 1.0,
    WRITE_INVALID: tl.constexpr = True,
):
    g = tl.program_id(0)
    t = tl.program_id(1)
    r = tl.load(REQ + t, t < LIVE, other=0)
    end = tl.load(START + r + 1)
    grp = (tl.load(SEQS + r) - end + t) // 4
    valid = (t < LIVE) & (t < end) & (grp > 511) & (g < grp)
    d = tl.arange(0, 128)
    h = tl.arange(0, 16)
    value = -float("inf")
    if valid:
        block = tl.load(TABLE + r.to(tl.int64) * STRIDE + (4 * g) // PAGE)
        slot = block.to(tl.int64) * (PAGE // 4) + g % (PAGE // 4)
        k = tl.load(KEYS + slot * 128 + d).to(tl.float32)
        q = tl.load(Q + (t * 16 + h[:, None]) * 128 + d[None, :]).to(tl.float32)
        prob = tl.load(WEIGHTS + t * 16 + h).to(tl.float32) * HEAD_SCALE
        value = tl.sum(tl.maximum(tl.sum(q * k[None, :], 1), 0.0) * prob, 0)
    tl.store(LOGITS + t.to(tl.int64) * GROUPS + g, value, valid | WRITE_INVALID)


@triton.jit
def _expand(
    TOP,
    REQ,
    START,
    SEQS,
    TABLE,
    IDX,
    LEN,
    LIVE,
    PAGE: tl.constexpr,
    STRIDE: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    GROUPS: tl.constexpr = 0,
    PAD_GROUPS: tl.constexpr = 0,
):
    t = tl.program_id(0)
    group = tl.arange(0, 512)
    inner = tl.arange(0, 4)
    r = tl.load(REQ + t, t < LIVE, other=0)
    end = tl.load(START + r + 1)
    p = tl.load(SEQS + r) - end + t
    cur = p // 4
    kept = tl.minimum(cur, 511)
    live = (t < LIVE) & (t < end)
    # For short prefixes the complete groups all survive, so emit in order.
    g = group
    if live & (cur > 511):
        if GROUPS:
            i = tl.arange(0, PAD_GROUPS)
            score = tl.load(
                TOP + t.to(tl.int64) * GROUPS + i,
                (i < cur) & (i < GROUPS),
                other=-float("inf"),
            )
            # Transform IEEE scores into unsigned, ordered high bits and use
            # descending group ID priority for deterministic ties.
            bits = tl.where(score != score, float("inf"), score).to(
                tl.uint32, bitcast=True
            )
            key = tl.where(bits >> 31 != 0, ~bits, bits ^ 0x80000000)
            packed = (key.to(tl.uint64) << 32) | (0xFFFFFFFF - i.to(tl.uint32))
            ranked = tl.sort(packed, descending=True)
            best = tl.gather(ranked, group, 0).to(tl.uint32)
            g = (0xFFFFFFFF - best).to(tl.int32)
        else:
            g = tl.load(TOP + t.to(tl.int64) * 511 + group, group < kept, other=0).to(
                tl.int32
            )
    g = tl.where(group == kept, cur, g)
    logical = g[:, None] * 4 + inner[None, :]
    j = group[:, None] * 4 + inner[None, :]
    count = tl.where(live, 4 * kept + p % 4 + 1, 0)
    page = tl.load(
        TABLE + r.to(tl.int64) * STRIDE + logical // PAGE, j < count, other=0
    )
    tl.store(
        IDX + t.to(tl.int64) * ROW_WIDTH + j,
        tl.where(j < count, page * PAGE + logical % PAGE, -1),
    )
    tl.store(LEN + t, count)


class FlashIndexPool(nn.Module):
    def __init__(self, c, max_len):
        super().__init__()
        self.max_groups = triton.cdiv(max_len, 4)
        self.fuse_topk = self.max_groups <= 1024
        self.query_projection = nn.Linear(c.q_lora_rank, 2048, bias=False)
        self.head_weight_projection = nn.Linear(c.hidden_size, 16, bias=False)
        self.token_key_projection = nn.Linear(c.hidden_size, 128, bias=False)
        self.token_key_norm = RMSNorm(128, c.rms_norm_eps)
        self.key_gate_projection = nn.Linear(128, 128, bias=False, dtype=torch.float32)
        self.key_value_projection = nn.Linear(128, 128, bias=False, dtype=torch.float32)
        self.key_norm = nn.RMSNorm(128, eps=c.rms_norm_eps, dtype=torch.float32)
        self.intra_group_ape = nn.Parameter(torch.empty(4, 128, dtype=torch.float32))

    def forward(self, x, latent, positions, rotary, cache, meta, out=None):
        t = x.shape[0]
        page = meta.block_size
        row = meta.block_table.stride(0)
        if out is None:
            out = (
                torch.empty((t, 1, 2048), device=x.device, dtype=torch.int32),
                torch.empty(t, device=x.device, dtype=torch.int32),
            )
        idx, lens = out
        keys = self.token_key_norm(self.token_key_projection(x))
        slots = torch.empty(t, device=x.device, dtype=torch.int64)
        starts = torch.empty_like(slots)
        raw = x.new_empty((t, 4, 128), dtype=torch.float32)
        _group_inputs[(t,)](
            keys,
            meta.req_id_per_token,
            meta.query_start_loc,
            meta.seq_lens,
            meta.block_table,
            meta.slot_mapping,
            cache.raw_index_keys,
            raw,
            slots,
            starts,
            meta.num_actual_tokens,
            page,
            row,
            cache.raw_index_keys.shape[0],
        )
        val = self.key_value_projection(raw)
        scores = self.key_gate_projection(raw)
        pooled = x.new_empty((t, 128))
        _pool4[(t,)](
            val,
            scores,
            self.intra_group_ape,
            self.key_norm.weight,
            pooled,
            self.key_norm.eps,
            enable_fp_fusion=False,
        )
        # Text positions coincide on all three M-RoPE planes.
        p = positions[0] if positions.ndim == 2 else positions
        kpe, _ = rotary(starts, pooled[:, 64:].contiguous(), None)
        k = torch.cat((pooled[:, :64], kpe.reshape(t, 64)), dim=-1)
        _store_rows[(t,)](k, cache.pooled_index_keys, slots, t, 128, 128)
        # Keep extending the pooled cache even when every causal key fits.
        # V2 supplies the configured maximum at graph capture, so long
        # contexts still use the scoring path on replay.
        if meta.max_seq_len <= 2048:
            _paged_indices[(t,)](
                meta.req_id_per_token,
                meta.query_start_loc,
                meta.seq_lens,
                meta.block_table,
                idx,
                lens,
                meta.num_actual_tokens,
                row,
                page,
                2048,
            )
            return idx, lens
        q = self.query_projection(latent).view(t, 16, 128)
        qpe, _ = rotary(p, q[:, :, 64:].contiguous(), None)
        q = torch.cat((q[:, :, :64], qpe.reshape(t, 16, 64)), dim=-1)
        head_weights = self.head_weight_projection(x)
        return self._select_groups(
            q, head_weights, cache.pooled_index_keys, meta, idx, lens
        )

    def _select_groups(self, q, head_weights, pooled_keys, meta, idx, lens):
        t = q.size(0)
        page = meta.block_size
        row = meta.block_table.stride(0)
        # V2 metadata supplies a CPU upper bound for this batch in eager mode
        # and the configured context limit during full CUDA graph capture.
        groups = min(self.max_groups, triton.cdiv(meta.max_seq_len, page) * (page // 4))
        logits = torch.empty((t, groups), device=q.device, dtype=torch.float32)
        _scores[(groups, t)](
            q,
            head_weights,
            pooled_keys,
            meta.req_id_per_token,
            meta.query_start_loc,
            meta.seq_lens,
            meta.block_table,
            logits,
            meta.num_actual_tokens,
            groups,
            page,
            row,
            2048**-0.5,
            not self.fuse_topk,
        )
        top = logits if self.fuse_topk else logits.topk(511, sorted=False).indices
        _expand[(t,)](
            top,
            meta.req_id_per_token,
            meta.query_start_loc,
            meta.seq_lens,
            meta.block_table,
            idx,
            lens,
            meta.num_actual_tokens,
            page,
            row,
            idx.shape[-1],
            groups if self.fuse_topk else 0,
            triton.next_power_of_2(groups) if self.fuse_topk else 0,
        )
        return idx, lens
