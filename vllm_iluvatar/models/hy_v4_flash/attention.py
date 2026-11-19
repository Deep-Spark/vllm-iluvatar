# SPDX-License-Identifier: Apache-2.0
"""Paged absorbed MLA with learned sinks and device-side indices."""

import ixformer.inference.functions as ops
import torch
import triton
import triton.language as tl
from torch import nn
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)

from vllm_iluvatar.ilu_util.arch import is_ivcore30
from vllm_iluvatar.models.deepseek_v32.attention import IluDeepseekV32IndexerCache
from vllm_iluvatar.models.hy_v4.attention import apply_qk_rope, build_hy4_ropes
from vllm_iluvatar.models.hy_v4.flashmla_sparse import HYV4FlashMLASparseBackend
from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
    IluFlashMLASparseMetadataBuilder,
)

from .gates import gate_product

BF16_KV_WIDTH = 576
FP8_KV_WIDTH = 640


@triton.jit
def _store_rows(X, CACHE, SLOT, LIVE, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    slot = tl.load(SLOT + row, row < LIVE, other=-1).to(tl.int64)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, other=0)
    tl.store(CACHE + slot * WIDTH + col, x, (slot >= 0) & (col < WIDTH))


@triton.jit
def _pack_absorbed_qk(
    QUERY,
    QPE,
    KEY,
    KPE,
    QOUT,
    KOUT,
    SLOTS,
    LIVE,
    HEADS: tl.constexpr,
    CACHE_SLOTS: tl.constexpr,
    PAGED: tl.constexpr,
    QUERY_ROW: tl.constexpr,
    QUERY_HEAD: tl.constexpr,
    QPE_ROW: tl.constexpr,
    QPE_HEAD: tl.constexpr,
    KEY_ROW: tl.constexpr,
    KPE_ROW: tl.constexpr,
    ACTIVE=None,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    cols = tl.arange(0, 512)
    rope = tl.arange(0, 64)
    if head < HEADS:
        query = tl.load(QUERY + row * QUERY_ROW + head * QUERY_HEAD + cols)
        pos = tl.load(QPE + row * QPE_ROW + head * QPE_HEAD + rope)
        dest = QOUT + (row * HEADS + head) * 576
        tl.store(dest + cols, query)
        tl.store(dest + 512 + rope, pos)
    else:
        slot = row
        if PAGED:
            slot = tl.load(SLOTS + row, row < LIVE, other=-1).to(tl.int64)
        valid = (slot >= 0) & (slot < CACHE_SLOTS)
        if ACTIVE is not None:
            valid = valid & tl.load(ACTIVE + row // 6)
        key = tl.load(KEY + row * KEY_ROW + cols, valid, other=0)
        pos = tl.load(KPE + row * KPE_ROW + rope, valid, other=0)
        dest = KOUT + slot * 576
        tl.store(dest + cols, key, valid)
        tl.store(dest + 512 + rope, pos, valid)


@triton.jit
def _paged_indices(
    REQ,
    START,
    SEQLEN,
    TABLE,
    OUT,
    LENGTHS,
    LIVE,
    STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, WINDOW)
    req = tl.load(REQ + row, row < LIVE, other=0)
    end = tl.load(START + req + 1)
    pos = tl.load(SEQLEN + req) - end + row
    length = tl.where((row < LIVE) & (row < end), tl.minimum(pos + 1, WINDOW), 0)
    logical = tl.maximum(0, pos + 1 - WINDOW) + j
    block = tl.load(
        TABLE + req.to(tl.int64) * STRIDE + logical // PAGE, j < length, other=0
    )
    tl.store(
        OUT + row * WINDOW + j, tl.where(j < length, block * PAGE + logical % PAGE, -1)
    )
    tl.store(LENGTHS + row, length)


@triton.jit
def _store_tokens_and_history(
    IDS,
    REQ,
    START,
    SEQLEN,
    TABLE,
    SLOT,
    CACHE,
    OUT,
    LIVE,
    STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    PAD: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, 4)
    req = tl.load(REQ + row, row < LIVE, other=0)
    first = tl.load(START + req)
    end = tl.load(START + req + 1)
    pos = tl.load(SEQLEN + req) - end + row - j
    valid = (row < LIVE) & (row < end) & (pos >= 0) & (j < 3)
    # Tokens within this query chunk come from IDS. Older tokens are already
    # in the cache, so concurrent CTAs never read each other's writes.
    current = valid & (row - j >= first)
    older = valid & ~current
    fresh = tl.load(IDS + row - j, current, other=PAD)
    blk = tl.load(TABLE + req.to(tl.int64) * STRIDE + pos // PAGE, older, other=0)
    cached = tl.load(CACHE + blk.to(tl.int64) * PAGE + pos % PAGE, older, other=PAD)
    token = tl.where(current, fresh, cached)
    slot = tl.load(SLOT + row, row < LIVE, other=-1).to(tl.int64)
    store = (slot >= 0) & (slot < CAPACITY)
    own = tl.load(IDS + row, store, other=PAD)
    tl.store(CACHE + slot, own, store)
    tl.store(OUT + row * 3 + j, token, j < 3)


@triton.jit
def _request_ids(
    START, OUT, REQS: tl.constexpr, N: tl.constexpr, B: tl.constexpr, LOG: tl.constexpr
):
    rows = tl.program_id(0) * B + tl.arange(0, B)
    lo = tl.full((B,), 0, tl.int32)
    hi = tl.full((B,), REQS, tl.int32)
    for _ in tl.static_range(LOG):
        mid = (lo + hi) // 2
        advance = (lo < hi) & (
            tl.load(START + mid + 1, mid < REQS, other=2147483647) <= rows
        )
        hi = tl.where(advance, hi, mid)
        lo = tl.where(advance, mid + 1, lo)
    tl.store(OUT + rows, tl.where(lo < REQS, lo, 0), rows < N)


class FlashMetadataBuilder(IluFlashMLASparseMetadataBuilder):
    def _stage_req_id_per_token(self, common_attn_metadata):
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        out = self.req_id_per_token_buffer[:num_tokens]
        if num_tokens:
            _request_ids[(triton.cdiv(num_tokens, 128),)](
                query_start_loc,
                out,
                query_start_loc.shape[0] - 1,
                num_tokens,
                128,
                (query_start_loc.shape[0] - 1).bit_length(),
            )
        return out


class FlashBackend(HYV4FlashMLASparseBackend):
    @staticmethod
    def get_builder_cls():
        return FlashMetadataBuilder


class FlashPageStorage(IluDeepseekV32IndexerCache):
    """All page-aligned Flash tensors use the same full-context block table."""

    def get_attn_backend(self):
        return FlashBackend


class FlashCache(FlashPageStorage):
    def __init__(self, prefix, config, with_history, with_index, fp8):
        width = FP8_KV_WIDTH if fp8 else BF16_KV_WIDTH
        dtype = torch.uint8 if fp8 else torch.bfloat16
        super().__init__(width, dtype, prefix, config)
        if with_index:
            self.raw_keys = FlashPageStorage(
                128, torch.bfloat16, f"{prefix}.raw_keys", config
            )
            # One pooled 128-wide vector for each group of four physical slots.
            self.pooled_keys = FlashPageStorage(
                32, torch.bfloat16, f"{prefix}.pooled_keys", config
            )
        if with_history:
            self.token_ids = FlashPageStorage(
                1, torch.int32, f"{prefix}.token_ids", config
            )

    @property
    def raw_index_keys(self):
        return self.raw_keys.kv_cache.view(-1, 128)

    @property
    def pooled_index_keys(self):
        return self.pooled_keys.kv_cache.view(-1, 128)

    @property
    def history(self):
        return self.token_ids.kv_cache.view(-1)

    def metadata(self):
        meta = get_forward_context().attn_metadata
        return None if meta is None else meta[self.prefix]

    def triplets(self, ids, pad):
        meta = self.metadata()
        out = ids.new_empty((ids.numel(), 3), dtype=torch.int32)
        if meta is None:
            return out.fill_(pad)
        _store_tokens_and_history[(ids.numel(),)](
            ids,
            meta.req_id_per_token,
            meta.query_start_loc,
            meta.seq_lens,
            meta.block_table,
            meta.slot_mapping,
            self.history,
            out,
            meta.num_actual_tokens,
            meta.block_table.stride(0),
            meta.block_size,
            pad,
            self.history.numel(),
        )
        return out


class FlashAttention(nn.Module):
    TP_WEIGHT_DIMS = {
        "q_b_proj.weight": 0,
        "kv_b_proj.weight": 0,
        "linear_gate.weight": 0,
        "learnable_sink_param": 0,
        "o_proj.weight": 1,
    }

    def __init__(
        self,
        config,
        layer,
        vllm_config,
        prefix,
        *,
        quant_config,
        with_history=False,
        draft=False,
    ):
        super().__init__()
        swa = config.layer_types[layer] == "sliding_attention"
        heads = config.swa_num_attention_heads if swa else config.num_attention_heads
        self.tp = get_tp_group()
        if heads % self.tp.world_size:
            raise ValueError("Flash attention heads must divide evenly across TP")
        self.heads = heads // self.tp.world_size
        cache_dtype = vllm_config.cache_config.cache_dtype
        self.fp8 = cache_dtype in ("fp8", "fp8_e4m3")
        if cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise ValueError(f"Unsupported Flash KV cache dtype: {cache_dtype}")
        if self.fp8:
            if not is_ivcore30() or self.heads not in (16, 32, 64):
                raise ValueError(
                    "Flash FP8 KV requires TG300 and 16, 32 or 64 local heads"
                )
            self.register_buffer(
                "kv_scale", torch.ones(1, dtype=torch.float32), persistent=False
            )
        self.window = config.sliding_window if swa else config.index_topk
        from .indexpool import FlashIndexPool

        self.compressed_indexer = (
            FlashIndexPool(config, vllm_config.model_config.max_model_len)
            if not swa
            and not draft
            and vllm_config.model_config.max_model_len > config.index_topk
            else None
        )
        h = config.hidden_size

        self.qkv_a_sizes = (config.q_lora_rank, BF16_KV_WIDTH)
        self.fused_qkv_a_proj = MergedColumnParallelLinear(
            h,
            list(self.qkv_a_sizes),
            bias=False,
            disable_tp=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        total_width = heads * 256
        self.q_b_proj = ColumnParallelLinear(
            config.q_lora_rank,
            total_width,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(512, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(512, self.heads * 448, bias=False)
        self.o_proj = RowParallelLinear(
            total_width,
            h,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.linear_gate = ColumnParallelLinear(
            h,
            total_width,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_gate",
        )
        self.post_proj_norm = (
            RMSNorm(h, config.rms_norm_eps)
            if config.sandwich_norm and (swa or not config.dsa_skip_attn_post_norm)
            else nn.Identity()
        )
        self.learnable_sink_param = nn.Parameter(
            torch.empty(self.heads, dtype=torch.float32)
        )
        self.rotary, self.mrope = build_hy4_ropes(
            64, vllm_config.model_config.max_model_len, config.rope_for_layer(layer)
        )
        self.cache = (
            None
            if draft
            else FlashCache(
                f"{prefix}.attn",
                vllm_config.cache_config,
                with_history,
                self.compressed_indexer is not None,
                self.fp8,
            )
        )

        self.snapshot = None
        self.snapshot_len = None
        hf = vllm_config.model_config.hf_config
        if (
            not draft
            and vllm_config.speculative_config
            and layer in getattr(hf, "dfly_target_index_layer", ())
        ):
            n = vllm_config.scheduler_config.max_num_batched_tokens
            self.snapshot = torch.full((n, self.window), -1, dtype=torch.int32)
            self.snapshot_len = torch.zeros(n, dtype=torch.int32)

    def projections(self, positions, hidden_states):
        t = hidden_states.shape[0]
        q_latent, kv = self.fused_qkv_a_proj(hidden_states).split(
            self.qkv_a_sizes, dim=-1
        )
        latent = self.q_a_layernorm(q_latent)
        q = self.q_b_proj(latent).view(t, self.heads, 256)
        k = self.kv_a_layernorm(kv[:, :512])
        qpe, kpe = apply_qk_rope(
            self.rotary, positions, q[:, :, 192:], kv[:, None, 512:], self.mrope
        )
        w = self.kv_b_proj.weight.view(self.heads, 448, 512)
        q_abs = torch.bmm(q[:, :, :192].transpose(0, 1), w[:, :192]).transpose(0, 1)
        return latent, q_abs, qpe, k, kpe

    @staticmethod
    def pack_bf16(q_abs, qpe, k, kpe, cache=None, slots=None, live=None, active=None):
        t, heads = q_abs.shape[:2]
        if live is None:
            live = t
        packed_q = q_abs.new_empty((t, heads, 576))
        keys = k.new_empty((t, 576)) if cache is None else cache
        _pack_absorbed_qk[(t, heads + 1)](
            q_abs,
            qpe,
            k,
            kpe,
            packed_q,
            keys,
            slots,
            live,
            heads,
            keys.shape[0],
            slots is not None,
            q_abs.stride(0),
            q_abs.stride(1),
            qpe.stride(0),
            qpe.stride(1),
            k.stride(0),
            kpe.stride(0),
            active,
        )
        return packed_q, keys

    def store_fp8_kv(self, k, kpe, cache, slots):
        ops.glm_dsa_fp8_kv_insert(k, kpe, cache, slots, self.kv_scale, cache.shape[1])

    def forward(self, positions, hidden_states):
        t = hidden_states.shape[0]
        meta = self.cache.metadata()
        if meta is None:
            return torch.zeros_like(hidden_states)
        latent, q_abs, qpe, k, kpe = self.projections(positions, hidden_states)
        cache = self.cache.kv_cache
        if self.fp8:
            self.store_fp8_kv(
                k, kpe, cache, meta.slot_mapping[: meta.num_actual_tokens]
            )
        else:
            q_abs, _ = self.pack_bf16(
                q_abs,
                qpe,
                k,
                kpe,
                cache.view(-1, 1, BF16_KV_WIDTH),
                meta.slot_mapping,
                meta.num_actual_tokens,
            )
            qpe = None
        if self.snapshot is None:
            idx = torch.empty(
                (t, 1, self.window), dtype=torch.int32, device=q_abs.device
            )
            lens = torch.empty(t, dtype=torch.int32, device=q_abs.device)
        else:
            idx = self.snapshot[:t].view(t, 1, self.window)
            lens = self.snapshot_len[:t]
        if self.compressed_indexer is not None:
            self.compressed_indexer(
                hidden_states,
                latent,
                positions,
                self.rotary,
                self.cache,
                meta,
                out=(idx, lens),
            )
        else:
            _paged_indices[(t,)](
                meta.req_id_per_token,
                meta.query_start_loc,
                meta.seq_lens,
                meta.block_table,
                idx,
                lens,
                meta.num_actual_tokens,
                meta.block_table.stride(0),
                meta.block_size,
                self.window,
            )
        return self.attend(hidden_states, q_abs, qpe, cache, idx, lens)

    def attend(self, hidden_states, q_abs, qpe, cache, idx, lens):
        t = hidden_states.shape[0]
        w = self.kv_b_proj.weight.view(self.heads, 448, 512)
        if self.fp8:
            q = q_abs.new_empty((t, self.heads, FP8_KV_WIDTH), dtype=torch.uint8)
            q_scale = q_abs.new_empty((t, self.heads), dtype=torch.float32)
            ops.glm_dsa_fp8_q_pack(q_abs, qpe, q, q_scale)
            y = ops.dsa_flash_mla_sparse_with_fp8_kvcache_diff_kv(
                q.unsqueeze(1),
                cache.unsqueeze(2),
                idx,
                lens,
                q_scale,
                self.kv_scale,
                1 / 16,
                attn_sink=self.learnable_sink_param,
            )
        else:
            y = ops.sparse_flash_attn(
                q_abs if qpe is None else torch.cat((q_abs, qpe), dim=-1),
                cache.view(-1, 1, BF16_KV_WIDTH),
                idx,
                1 / 16,
                v_head_dim=512,
                topk_length=lens,
                attn_sink=self.learnable_sink_param,
                backend="ixinfer",
            )
        y = torch.bmm(
            y.reshape(t, self.heads, 512).transpose(0, 1), w[:, 192:].transpose(1, 2)
        )
        gated = gate_product(self.linear_gate(hidden_states), y, head_dim=256)
        out = self.o_proj(gated)
        return self.post_proj_norm(out)
