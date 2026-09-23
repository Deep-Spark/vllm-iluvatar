"""Indexer key production: k_norm, RoPE, quantize, paged store.

Mirrors ``vllm/models/deepseek_v41/common/ops/indexer_k_store.py``: same
function name, same arguments, same contract, so the indexer's ``_produce_k``
reads like the official one.

Our index-key records are BF16 rows with an FP4 qdq round-trip over 32-dim
blocks and power-of-two scales, where official packs MXFP4 nibbles (or per-token
FP8) into a uint8 page. ``use_fp4_cache`` therefore selects the qdq format
rather than the record width.
"""
from __future__ import annotations

import ixformer.inference.functions as ixfops
import torch

# The indexer quantizes over 32 dims with E8M0 (power-of-two) scales, where the
# compressed KV cache uses 16 dims with E4M3.
MXFP4_BLOCK_SIZE = 32


def indexer_k_norm_rope_store(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    compress_ratio: int,
    use_fp4_cache: bool,
    *,
    block_table: torch.Tensor | None = None,
    token_to_req_indices: torch.Tensor | None = None,
    block_size: int | None = None,
) -> None:
    """k_norm -> RoPE -> quant -> paged store for indexer keys.

    Args:
        k_pre: [num_tokens, 128] bf16, the ``wk(latent)`` projection. Only
            group-boundary rows are read.
        positions: [num_tokens] int64 token positions.
        cos_sin_cache: [max_pos, rope_head_dim] GPT-J layout (cos half, then
            sin half), from the layer's compress-RoPE instance.
        rms_norm_weight: [128] k_norm weight.
        rms_norm_eps: Epsilon of the k_norm RMSNorm.
        k_cache: The paged indexer cache.
        kv_slot_mapping: [num_tokens] slots in the indexer cache (-1 = skip).
        compress_ratio: group size; keys are emitted at group boundaries.
        use_fp4_cache: FP4 with power-of-two scales when True.
        block_table: Pages of the indexer cache (ratio > 1).
        token_to_req_indices: Request indices for the token rows (ratio > 1).
        block_size: Raw tokens per page. The cache's second dimension is the
            compressed row count, ``block_size // compress_ratio``.

    """
    num_tokens = kv_slot_mapping.numel()
    page_tokens = block_size if block_size is not None else int(k_cache.shape[1])
    assert k_pre.ndim == 2 and k_pre.shape[1] == 128
    assert k_pre.dtype == torch.bfloat16 and k_pre.stride(1) == 1
    assert num_tokens <= k_pre.shape[0] and num_tokens <= positions.numel()
    assert compress_ratio in (1, 2)
    assert use_fp4_cache, "the BF16 index-key record is always FP4 qdq here"
    if num_tokens == 0:
        return

    if compress_ratio == 1:
        # The packed-row insert has no norm to fold into, so it runs separately.
        # TODO(ops): fold once the packed kernel takes a norm weight, as the
        # by-token one does.
        k = torch.empty_like(k_pre[:num_tokens])
        ixfops.rms_norm(k_pre[:num_tokens], rms_norm_weight, rms_norm_eps, k)
        ixfops.v41_compress_rope_qdq_insert(
            k_cache,
            k,
            kv_slot_mapping,
            positions[:num_tokens],
            cos_sin_cache,
            page_tokens,
            compress_ratio,
            MXFP4_BLOCK_SIZE,
            "fp4_pow2",
        )
        return

    # Folding the norm in matters here: normalising in torch first would
    # normalise every token's row and then keep only the ones closing a group.
    assert block_table is not None and token_to_req_indices is not None
    if k_cache.dim() == 3:
        assert k_cache.shape[1] * compress_ratio == page_tokens
    ixfops.v41_rope_qdq_insert_by_token(
        k_cache,
        k_pre[:num_tokens],
        positions[:num_tokens],
        kv_slot_mapping,
        block_table,
        token_to_req_indices[:num_tokens],
        cos_sin_cache,
        page_tokens,
        compress_ratio,
        MXFP4_BLOCK_SIZE,
        "fp4_pow2",
        rms_norm_weight,
        rms_norm_eps,
    )
