"""Compressor state/pool/norm and compressed-cache insert.

Mirrors ``vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py``:
same function names, same positional arguments, same contract, so the callers
read exactly like the official ones. Official drives both entry points with
Triton, which does not run on this platform, so each body dispatches to the
ixformer kernel covering the same work.

Two arguments are ours and keyword-only, ``block_table`` and
``token_to_req_indices``. Official derives the state ring slot in a separate
Triton kernel in the metadata builder and finds a chunk's predecessor row with a
dedicated per-request program; our kernels absorb both, so they need the block
table instead of a pre-encoded slot.

The cache records here are plain BF16 rows with a qdq round-trip, where official
packs FP8/FP4 with per-record scales. Layout is the one thing not shared, which
is why the quantization format lives in this layer and not in the callers.
"""
from __future__ import annotations

import ixformer.inference.functions as ixfops
import torch

# The compressed KV record is BF16 with an FP4 qdq round-trip over 16-dim
# blocks, so head_dim 512 is 32 blocks. Official quantizes into the record.
COMPRESS_KV_QDQ_BLOCK = 16
COMPRESS_KV_QDQ_FMT = "fp4_e4m3"


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
    *,
    block_table: torch.Tensor | None = None,
) -> None:
    """Pool each closed group into a normalized BF16 latent; save FP32 states.

    The latent feeds the main-cache insert and the indexer K path, which the
    attention layer schedules on separate streams.

    Ratio 2 keeps one ring block per request holding the open group's rows. A
    group that a chunk's first token closes with its predecessor needs that
    predecessor's ring row, which is why the block table comes in.

    Args:
        kv_score: FP32 [tokens, 512] for CR1, [tokens, 1024] for CR2.
        positions: Absolute positions of the packed request tokens.
        state_cache: Ring FP32 [blocks, capacity, 1024] KV/score states (CR2).
        slot_mapping: Ring slots (CR2) or main-cache slots (CR1).
        query_start_loc: Unused here; our kernel takes request boundaries from
            ``token_to_req_indices`` and the block table.
        token_to_req_indices: Request indices for the packed token rows.
        rms_norm_weight: BF16 [512] normalization weight.
        rms_norm_eps: RMSNorm epsilon.
        compress_ratio: Group size, either 1 or 2.
        latent_out: BF16 [tokens, 512], written only at valid group boundaries.
        block_table: [num_reqs, max_blocks] pages backing the state ring.

    """
    del query_start_loc
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
        # No group to pool and no state to carry, so the latent is just the
        # normalized row. TODO(ops): ds4_v41_compress_prepare_out needs a state
        # cache, which a ratio-1 layer never allocates, so this stays at torch
        # level until the kernel takes a null state.
        row = kv_score[:num_tokens]
        rms = torch.rsqrt(row.pow(2).mean(-1, keepdim=True) + rms_norm_eps)
        latent_out[:num_tokens] = (row * rms).to(latent_out.dtype) * rms_norm_weight
        return

    assert state_cache is not None and token_to_req_indices is not None
    assert block_table is not None
    # kv and score are the two halves of the merged wkv/wgate GEMM. The kernel
    # takes a row stride, so these views go in without a copy.
    ixfops.ds4_v41_compress_prepare_out(
        kv_score[:num_tokens, :512],
        kv_score[:num_tokens, 512:],
        state_cache,
        token_to_req_indices[:num_tokens],
        positions[:num_tokens],
        slot_mapping,
        block_table,
        rms_norm_weight,
        latent_out[:num_tokens],
        compress_ratio,
        rms_norm_eps,
    )


def rope_quant_insert(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    *,
    fp8_scale: torch.Tensor | None = None,
    block_size: int,
    block_table: torch.Tensor | None = None,
    token_to_req_indices: torch.Tensor | None = None,
) -> None:
    """Apply GPT-J RoPE and publish a latent to the compressed KV cache.

    One kernel for what was four calls -- derive the slots, rotate, quantize,
    scatter -- of which the slot derivation alone cost a cumsum and two scatters
    over the whole batch.

    Args:
        latent: BF16 [tokens, 512] pre-RoPE rows, one per token.
        positions: Absolute positions of the packed request tokens.
        cos_sin_cache: The layer's rotary table.
        kv_cache: The compressed cache to write.
        slot_mapping: Main-cache slots of the packed tokens.
        compress_ratio: Group size, either 1 or 2.
        fp8_scale: Unused; BF16 rows carry no per-tensor scale.
        block_size: Page size in *tokens*. Ours, and not derivable from the
            cache: these kernels take a token-domain ``slot_mapping`` and
            convert it to a row in closed form, where official hands its Triton
            kernels a slot already in the group domain. So the cache holds
            ``block_size // compress_ratio`` rows per page and passing its dim 1
            here silently halves every page number on a ratio-2 layer.
        block_table: Pages of the compressed cache (ratio > 1).
        token_to_req_indices: Request indices for the token rows (ratio > 1).

    """
    assert fp8_scale is None, "BF16 compressed rows carry no per-tensor scale"
    assert kv_cache.shape[1] * compress_ratio == block_size, (
        f"{kv_cache.shape[1]} rows per page x ratio {compress_ratio} is not the "
        f"{block_size}-token page the slot arithmetic assumes"
    )
    # Request rows only, never the padding a captured batch carries: positions
    # is the one metadata field the builder already cuts to num_actual_tokens.
    num_tokens = positions.numel()
    if num_tokens == 0:
        return

    if compress_ratio == 1:
        # Every token closes its own group, so the per-token rows are already
        # the packed per-group rows this kernel wants.
        ixfops.v41_compress_rope_qdq_insert(
            kv_cache,
            latent,
            slot_mapping[:num_tokens],
            positions[:num_tokens],
            cos_sin_cache,
            block_size,
            compress_ratio,
            COMPRESS_KV_QDQ_BLOCK,
            COMPRESS_KV_QDQ_FMT,
        )
        return

    assert block_table is not None and token_to_req_indices is not None
    ixfops.v41_rope_qdq_insert_by_token(
        kv_cache,
        latent,
        positions[:num_tokens],
        slot_mapping[:num_tokens],
        block_table,
        token_to_req_indices[:num_tokens],
        cos_sin_cache,
        block_size,
        compress_ratio,
        COMPRESS_KV_QDQ_BLOCK,
        COMPRESS_KV_QDQ_FMT,
        None,
        1e-6,
    )
