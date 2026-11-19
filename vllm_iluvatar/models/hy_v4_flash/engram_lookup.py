# SPDX-License-Identifier: Apache-2.0
"""Flash BF16 checkpoint row lookup on V4.1 shared pinned mmap storage."""

from pathlib import Path

import torch
import triton
import triton.language as tl

from vllm_iluvatar.models.deepseek_v41.engram import _safetensors_entry
from vllm_iluvatar.models.deepseek_v41.engram_storage import map_shared_checkpoint_rows


def map_flash_table(directory, weight_map, layer_idx, group):
    name = (
        f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.embedding.weight"
    )
    path = Path(directory) / weight_map[name]
    _, shape, dtype = _safetensors_entry(path, name)
    if len(shape) != 2 or shape[1] != 96 or dtype != torch.bfloat16:
        raise ValueError(f"Unexpected Flash Engram table {name}: {shape}, {dtype}")
    return map_shared_checkpoint_rows(
        path, name, 0, shape[0], group, _safetensors_entry
    )


@triton.jit
def _lookup(
    table,
    hashes,
    gamma,
    out,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUND_FP8: tl.constexpr,
):
    token_head = tl.program_id(0)
    row = tl.load(hashes + token_head).to(tl.int64)
    col = tl.arange(0, BLOCK)
    values = tl.load(
        table + row * WIDTH + col, (col < WIDTH) & (row >= 0) & (row < ROWS), other=0.0
    ).to(tl.float32)
    if ROUND_FP8:
        # Match the BF16 intermediates of the checkpoint reference. E4M3's
        # subnormal step is 2^-9. Use IEEE nearest-even without native FP8.
        scale = (
            (tl.maximum(tl.max(tl.abs(values), 0), 1.0e-12) / 448.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        scaled = tl.minimum(
            (tl.abs(values) / scale).to(tl.bfloat16).to(tl.float32), 448.0
        )
        exponent = ((scaled.to(tl.int32, bitcast=True) >> 23) & 255) - 127
        step = ((tl.maximum(exponent - 3, -9) + 127) << 23).to(tl.float32, bitcast=True)
        units = scaled / step
        floor = tl.floor(units)
        frac = units - floor
        rounded = floor + (
            (frac > 0.5) | ((frac == 0.5) & ((floor.to(tl.int32) & 1) != 0))
        ).to(tl.float32)
        dequant = (tl.where(values < 0, -rounded, rounded) * step * scale).to(
            tl.bfloat16
        )
        values = dequant.to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(values * values, 0) / WIDTH + 1.1920928955078125e-7)
    weight = tl.load(gamma + (token_head % HEADS) * WIDTH + col, col < WIDTH, other=0.0)
    tl.store(out + token_head * WIDTH + col, values * inverse_rms * weight, col < WIDTH)


def lookup_normalized_rows(table, hashes, gamma, out, *, round_fp8):
    """Read flattened global row IDs. Negative IDs denote padding."""
    tokens, heads = hashes.shape
    rows, width = table.shape
    if (
        width != 96
        or gamma.shape != (heads, width)
        or out.shape != (tokens, heads, width)
    ):
        raise ValueError("Flash Engram lookup shape mismatch")
    if any(not x.is_contiguous() for x in (table, hashes, gamma, out)):
        raise ValueError("Flash Engram lookup requires contiguous row-major tensors")
    if tokens:
        _lookup[(tokens * heads,)](
            table,
            hashes,
            gamma,
            out,
            rows,
            heads,
            width,
            triton.next_power_of_2(width),
            round_fp8,
        )
    return out


@triton.jit
def _hash_rows(
    TRIPLETS, MAP, MULT, SIZES, OFFSETS, OUT, MAP_SIZE: tl.constexpr, PAD: tl.constexpr
):
    t = tl.program_id(0)
    head = tl.arange(0, 16)
    ids = tl.load(TRIPLETS + t * 3 + 0)
    one = tl.load(MAP + tl.where((ids >= 0) & (ids < MAP_SIZE), ids, PAD)).to(
        tl.int64
    ) * tl.load(MULT)
    ids = tl.load(TRIPLETS + t * 3 + 1)
    two = one ^ (
        tl.load(MAP + tl.where((ids >= 0) & (ids < MAP_SIZE), ids, PAD)).to(tl.int64)
        * tl.load(MULT + 1)
    )
    ids = tl.load(TRIPLETS + t * 3 + 2)
    three = two ^ (
        tl.load(MAP + tl.where((ids >= 0) & (ids < MAP_SIZE), ids, PAD)).to(tl.int64)
        * tl.load(MULT + 2)
    )
    tl.store(
        OUT + t * 16 + head,
        tl.where(head < 8, two, three) % tl.load(SIZES + head)
        + tl.load(OFFSETS + head),
    )


@triton.jit
def _write_residual(
    HIDDEN, KV, KEY_NORM, QUERY_NORM, GAMMA, SCALES, IDS, OUT, PAD: tl.constexpr
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, 4096)
    mask = d < 3072
    key = tl.load(KV + (t * 5 + h) * 3072 + d, mask, other=0).to(tl.float32)
    query = tl.load(HIDDEN + (t * 4 + h) * 3072 + d, mask, other=0).to(tl.float32)
    val = tl.load(KV + (t * 5 + 4) * 3072 + d, mask, other=0).to(tl.float32)
    eps = 1.1920928955078125e-7
    kn = tl.load(KEY_NORM + h * 3072 + d, mask, other=0).to(tl.float32)
    qn = tl.load(QUERY_NORM + h * 3072 + d, mask, other=0).to(tl.float32)
    kg = (
        (key * tl.rsqrt(tl.sum(key * key, 0) / 3072 + eps) * kn)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    qg = (
        (query * tl.rsqrt(tl.sum(query * query, 0) / 3072 + eps) * qn)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    dot = (
        tl.sum((kg * qg).to(tl.bfloat16).to(tl.float32), 0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    dot = (dot * 0.018042195912175808).to(tl.bfloat16).to(tl.float32)
    gate = (
        tl.sigmoid(
            tl.where(dot < 0, -1.0, tl.where(dot > 0, 1.0, 0.0))
            * tl.sqrt(tl.maximum(tl.abs(dot), 1.0e-6))
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    signal = (
        (val * tl.rsqrt(tl.sum(val * val, 0) / 3072 + eps))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    signal = (signal * tl.load(GAMMA + d, mask, other=0)).to(tl.bfloat16).to(tl.float32)
    scale = tl.load(SCALES + h * 3072 + d, mask, other=0)
    branch = (
        (
            (signal * gate).to(tl.bfloat16).to(tl.float32)
            + (signal * scale).to(tl.bfloat16).to(tl.float32)
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    result = query + tl.where(tl.load(IDS + t * 3) == PAD, 0.0, branch)
    tl.store(OUT + (t * 4 + h) * 3072 + d, result, mask)
