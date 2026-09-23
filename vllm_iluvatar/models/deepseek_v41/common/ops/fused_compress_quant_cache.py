"""Compressor state/pool/norm and compressed-cache insert.

Mirrors ``vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py``:
same function names, same positional arguments, same contract, so the callers
read exactly like the official ones. Official drives both entry points with
Triton, which does not run on this platform, so each body dispatches to the
ixformer kernel covering the same work.

Three arguments are ours and keyword-only: ``block_table``,
``token_to_req_indices`` and the token-domain ``block_size``. Official derives
the state ring slot in a separate Triton kernel in the metadata builder and
finds a chunk's predecessor row with a dedicated per-request program; our
kernels absorb both, so they need the block table instead of a pre-encoded slot.

The cache record is the packed fp8_ds_mla row: 448 E4M3 NoPE bytes, 64 BF16
RoPE values and 7 UE8M0 scales.
"""
from __future__ import annotations

import ixformer.inference.functions as ixfops
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
    block_table: torch.Tensor | None = None,
    token_to_req_indices: torch.Tensor | None = None,
    block_size: int | None = None,
) -> None:
    """Apply GPT-J RoPE and publish a latent as a packed fp8 compressed row.

    ``block_size`` is the token page size. The cache's second dimension is the
    compressed row count, which is ``block_size // compress_ratio``.
    """
    assert fp8_scale is None
    assert block_table is not None and token_to_req_indices is not None
    assert block_size is not None
    num_tokens = positions.numel()
    if num_tokens == 0:
        return
    if kv_cache.dim() == 3:
        assert kv_cache.shape[1] * compress_ratio == block_size
    ixfops.v41_rope_fp8_insert_by_token(
        kv_cache,
        latent[:num_tokens],
        positions[:num_tokens],
        slot_mapping[:num_tokens],
        block_table,
        token_to_req_indices[:num_tokens],
        cos_sin_cache,
        block_size,
        compress_ratio,
    )
