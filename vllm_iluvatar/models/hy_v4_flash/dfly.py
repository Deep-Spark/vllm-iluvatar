# SPDX-License-Identifier: Apache-2.0
"""Five parallel DFly mask slots with slot-0 fusion and learned hidden correction."""

import torch
import triton
import triton.language as tl
from safetensors import safe_open
from torch import nn
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from .attention import BF16_KV_WIDTH, FP8_KV_WIDTH, FlashAttention
from .norm import sum_rms_norm
from .quantization import PACKED_LINEAR_PROJECTIONS


@triton.jit
def _pack_kv(
    CACHE,
    SNAP,
    LENGTH,
    ROW,
    LIVE,
    NEW,
    PACKED,
    WIDTH: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    FP8: tl.constexpr,
    TILE_ROWS: tl.constexpr,
):
    first = tl.program_id(0) * TILE_ROWS
    col = first + tl.arange(0, TILE_ROWS)
    b = tl.program_id(1)
    d = tl.arange(0, 1024)
    live = tl.load(LIVE + b)
    src = tl.load(ROW + b).to(tl.int64)
    length = tl.load(LENGTH + src, live, other=0)
    if live & (first < length):
        kvslot = tl.load(SNAP + src * 2048 + col, col < length, other=-1).to(tl.int64)
        val = tl.load(
            CACHE + kvslot[:, None] * ROW_WIDTH + d[None, :],
            (d[None, :] < ROW_WIDTH) & (kvslot[:, None] >= 0),
            other=0,
        )
        tl.store(
            PACKED + (b.to(tl.int64) * WIDTH + col[:, None]) * ROW_WIDTH + d[None, :],
            val,
            (col[:, None] < length) & (d[None, :] < ROW_WIDTH),
        )
    elif not FP8 and live & (first < length + 6):
        val = tl.load(
            NEW + (b.to(tl.int64) * 6 + first - length) * ROW_WIDTH + d,
            d < ROW_WIDTH,
            other=0,
        )
        tl.store(
            PACKED + (b.to(tl.int64) * WIDTH + first) * ROW_WIDTH + d,
            val,
            d < ROW_WIDTH,
        )


@triton.jit
def _draft_indices(
    LENGTH,
    ROW,
    LIVE,
    IDX,
    LENS,
    SLOTS,
    WIDTH: tl.constexpr,
    PADDED: tl.constexpr,
    FP8: tl.constexpr,
    SNAP=None,
):
    t = tl.program_id(0)
    b = t // 6
    active = tl.load(LIVE + b)
    src = tl.load(ROW + b)
    history = tl.load(LENGTH + src, active, other=0)
    length = history + t % 6 + 1
    length = tl.where(active, length, 0)
    j = tl.arange(0, 4096)
    slot = b * WIDTH + j
    if SNAP is not None:
        old = tl.load(
            SNAP + src.to(tl.int64) * 2048 + j, active & (j < history), other=-1
        )
        slot = tl.where(j < history, old, b * 6 + j - history)
    tl.store(IDX + t * PADDED + j, tl.where(j < length, slot, -1), j < PADDED)
    tl.store(LENS + t, length)
    if FP8:
        tl.store(SLOTS + t, tl.where(active, b * WIDTH + length - 1, -1))


@triton.jit
def _norm_concat(
    FIRST,
    SECOND,
    FIRST_W,
    SECOND_W,
    OUT,
    D: tl.constexpr,
    FIRST_STRIDE: tl.constexpr,
    SECOND_STRIDE: tl.constexpr,
    FIRST_EPS: tl.constexpr,
    SECOND_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    half = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    if half == 0:
        v = tl.load(FIRST + row * FIRST_STRIDE + col, col < D, other=0.0).to(tl.float32)
        w = tl.load(FIRST_W + col, col < D, other=0.0).to(tl.float32)
        norm = tl.rsqrt(tl.sum(v * v, 0) / D + FIRST_EPS)
    else:
        v = tl.load(SECOND + row * SECOND_STRIDE + col, col < D, other=0.0).to(
            tl.float32
        )
        w = tl.load(SECOND_W + col, col < D, other=0.0).to(tl.float32)
        norm = tl.rsqrt(tl.sum(v * v, 0) / D + SECOND_EPS)
    tl.store(OUT + (row * 2 + half) * D + col, v * norm * w, col < D)


def _normalized_pair(first, second, first_norm, second_norm):
    rows, width = first.shape
    out = first.new_empty((rows, 2 * width))
    _norm_concat[(rows, 2)](
        first,
        second,
        first_norm.weight,
        second_norm.weight,
        out,
        width,
        first.stride(0),
        second.stride(0),
        first_norm.variance_epsilon,
        second_norm.variance_epsilon,
        triton.next_power_of_2(width),
    )
    return out


@triton.jit
def _update_anchor(X, VALUE, GATE, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = col < WIDTH
    offset = row.to(tl.int64) * WIDTH + col
    anchor_offset = row.to(tl.int64) * 6 * WIDTH + col
    value = tl.load(VALUE + offset, valid, other=0).to(tl.float32)
    gate = tl.load(GATE + col, valid, other=0).to(tl.float32)
    anchor = tl.load(X + anchor_offset, valid, other=0).to(tl.float32)
    delta = (gate * value).to(X.dtype.element_ty).to(tl.float32)
    tl.store(X + anchor_offset, anchor + delta, valid)


@triton.jit
def _write_draft_tokens(TOKENS, ACTIVE, OUT, ROWS: tl.constexpr, STRIDE: tl.constexpr):
    row = tl.program_id(0) * 128 + tl.arange(0, 128)
    valid = row < ROWS
    active = tl.load(ACTIVE + row, valid, other=False)
    token = tl.load(TOKENS + row, valid, other=-1)
    tl.store(OUT + row * STRIDE, tl.where(active, token, -1), valid)


class DraftLayer(nn.Module):
    def __init__(self, c, vc, target_layer, index, quant):
        super().__init__()
        prefix = f"model.dfly_layers.{index}"
        self.self_attn = FlashAttention(
            c, target_layer, vc, f"{prefix}.self_attn", draft=True, quant_config=quant
        )
        self.use_target_cache = vc.kv_transfer_config is None
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = MergedColumnParallelLinear(
            3072,
            [9216, 9216],
            bias=False,
            return_bias=False,
            quant_config=quant,
            prefix=f"{prefix}.mlp.gate_up_proj",
        )
        self.mlp.act = SiluAndMul()
        self.mlp.down_proj = ReplicatedLinear(
            9216,
            3072,
            bias=False,
            return_bias=False,
            quant_config=quant,
            prefix=f"{prefix}.mlp.down_proj",
        )

    def forward(self, x, residual, positions, target, source, active):
        batch = source.numel()
        if residual is None:
            inp, residual = self.input_layernorm(x), x
        else:
            inp, residual = self.input_layernorm(x, residual)
        _, q, qpe, k, kpe = self.self_attn.projections(positions, inp)
        fp8 = self.self_attn.fp8
        direct = (
            not fp8
            and self.use_target_cache
            and batch * 6 <= target.cache.kv_cache.shape[1]
        )
        if fp8:
            page = target.cache.kv_cache.shape[1]
            width = triton.cdiv(2054, page) * page
            row_width = FP8_KV_WIDTH
            kv = x.new_empty(
                (batch * width // page, page, row_width), dtype=torch.uint8
            )
            slots = torch.empty(batch * 6, dtype=torch.int64, device=x.device)
            keys = k
        else:
            width = 2054
            row_width = BF16_KV_WIDTH
            kv = (
                target.cache.kv_cache.view(-1, 1, row_width)
                if direct
                else x.new_empty((batch * width, 1, row_width))
            )
            slots = None
            q, keys = self.self_attn.pack_bf16(
                q, qpe, k, kpe, kv if direct else None, active=active
            )
            qpe = None
        ids = torch.empty((batch * 6, 1, 2112), dtype=torch.int32, device=x.device)
        lens = torch.empty((batch * 6,), dtype=torch.int32, device=x.device)
        tile_rows = 4 if fp8 else 1
        staged_rows = 2048 if fp8 else 2054
        if not direct:
            _pack_kv[(triton.cdiv(staged_rows, tile_rows), batch)](
                target.cache.kv_cache,
                target.snapshot,
                target.snapshot_len,
                source,
                active,
                keys,
                kv,
                width,
                row_width,
                fp8,
                tile_rows,
            )
        _draft_indices[(6 * batch,)](
            target.snapshot_len,
            source,
            active,
            ids,
            lens,
            slots,
            width,
            2112,
            fp8,
            target.snapshot if direct else None,
        )
        if fp8:
            self.self_attn.store_fp8_kv(k, kpe, kv, slots)
        attn = self.self_attn.attend(inp, q, qpe, kv, ids, lens)
        inp, residual = sum_rms_norm(
            attn, residual, self.post_attention_layernorm, keep_sum=True
        )
        return self.mlp.down_proj(self.mlp.act(self.mlp.gate_up_proj(inp))), residual


class DFly(nn.Module):
    def __init__(self, vc, target):
        super().__init__()
        hf = vc.model_config.hf_config
        c = vc.model_config.hf_text_config
        self.target_layers = tuple(hf.dfly_target_kv_layers)
        if (
            c.hidden_size != 3072
            or self.target_layers != tuple(hf.dfly_target_index_layer)
            or any(c.layer_types[i] == "sliding_attention" for i in self.target_layers)
            or len(self.target_layers) != 5
            or hf.dfly_mtp_fusion != "slot0"
            or hf.dfly_context_local_window is not None
            or not hf.dfly_enable_hidden_correction
        ):
            raise ValueError("Unexpected HY4 Flash DFly weight contract")
        self.layers = nn.ModuleList(
            [
                DraftLayer(c, vc, layer, i, target.quant_config)
                for i, layer in enumerate(self.target_layers)
            ]
        )
        self.final_norm = RMSNorm(3072, c.rms_norm_eps)
        self.mtp_fusion = nn.Module()
        self.mtp_fusion.enorm = RMSNorm(3072, c.rms_norm_eps)
        self.mtp_fusion.hnorm = RMSNorm(3072, c.rms_norm_eps)
        self.mtp_fusion.eh_proj = nn.Linear(6144, 3072, bias=False)
        self.mtp_fusion.gate = nn.Parameter(torch.empty(3072))
        self.hidden_correction = nn.Module()
        self.hidden_correction.hidden_norm = RMSNorm(3072, c.rms_norm_eps)
        self.hidden_correction.embed_norm = RMSNorm(3072, c.rms_norm_eps)
        self.hidden_correction.gate_up_proj = MergedColumnParallelLinear(
            6144,
            [3072, 3072],
            bias=False,
            return_bias=False,
            quant_config=target.quant_config,
            prefix="model.dfly_layers.hidden_correction.gate_up_proj",
        )
        self.hidden_correction.act = SiluAndMul()
        self.hidden_correction.down_proj = nn.Linear(3072, 3072, bias=False)
        self.confidence_head = nn.Module()
        self.confidence_head.proj = nn.Linear(3072, 1)
        self.load_checkpoint(target.directory, target.weight_map)
        process_weights_after_loading(
            self, vc.model_config, self.final_norm.weight.device
        )
        # The runner owns these modules and storage. Keep direct references
        # without adding target parameters to the draft module's state dict.
        object.__setattr__(self, "target", target)

    @torch.no_grad()
    def load_checkpoint(self, path, index):
        for name, par in self.named_parameters():
            suffix = (
                name.removeprefix("layers.") if name.startswith("layers.") else name
            )
            key = "model.dfly_layers." + suffix
            parent, projection, role = key.rsplit(".", 2)
            if projection in PACKED_LINEAR_PROJECTIONS:
                for shard, leaf in enumerate(PACKED_LINEAR_PROJECTIONS[projection]):
                    source = f"{parent}.{leaf}.{role}"
                    with safe_open(path / index[source], framework="pt") as f:
                        par.weight_loader(par, f.get_tensor(source), shard)
                continue
            with safe_open(path / index[key], framework="pt") as f:
                value = f.get_tensor(key)
                if hasattr(par, "weight_loader"):
                    par.weight_loader(par, value)
                else:
                    par.copy_(value.reshape_as(par))

    def forward(self, ids, positions, source, active, hidden, out):
        x = self.target.embed_input_ids(ids)
        anchor = x[::6]
        fusion = self.mtp_fusion
        delta = fusion.eh_proj(
            _normalized_pair(anchor, hidden, fusion.enorm, fusion.hnorm)
        )
        rows, width = delta.shape
        _update_anchor[(triton.cdiv(width, 256), rows)](
            x, delta, fusion.gate, width, 256, enable_fp_fusion=False
        )
        residual = None
        for layer, idx in zip(self.layers, self.target_layers):
            x, residual = layer(
                x,
                residual,
                positions,
                self.target.layers[idx].self_attn,
                source,
                active,
            )
        x, _ = self.final_norm(x, residual)
        h = x.view(-1, 6, 3072)
        prev = ids[::6].long()
        corr = self.hidden_correction
        for i in range(5):
            row = h[:, i + 1]
            combined = _normalized_pair(
                row,
                self.target.embed_input_ids(prev),
                corr.hidden_norm,
                corr.embed_norm,
            )
            row = row + corr.down_proj(corr.act(corr.gate_up_proj(combined)))
            prev = self.target.compute_logits(row).argmax(-1)
            _write_draft_tokens[(triton.cdiv(rows, 128),)](
                prev, active, out[:, i], rows, out.stride(0)
            )
