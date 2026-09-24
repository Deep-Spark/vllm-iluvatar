"""Indexer key production: k_norm, RoPE, quantize, paged store.

Mirrors ``vllm/models/deepseek_v41/common/ops/indexer_k_store.py``: same
function name, same arguments, same contract, so the indexer's ``_produce_k``
reads like the official one.

One argument differs, and deliberately. Official's last parameter is
``use_fp4_cache``, selecting between an MXFP4 page and a per-token FP8 one.
This platform carries no MXFP4 record -- the scoring kernels here read either
plain BF16 rows or the V3.2 per-token FP8 layout -- so the parameter is
``store_fp8`` and names what it actually selects:

* ``False``: BF16 [num_blocks, block_size, 128] rows.
* ``True``:  uint8 [num_blocks, block_size, 132] pages holding 128 E4M3 bytes
  per row followed by one FP32 scale per row, which is official's
  ``use_fp4_cache=False`` branch and what
  ``dsa_indexer_mqa_logits_with_blocks_fp8`` reads.

Neither path quantize-dequantizes. What lands in the cache is what the scoring
kernel reads.
"""

from __future__ import annotations

import torch


def indexer_k_norm_rope_store(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    compress_ratio: int,
    store_fp8: bool,
    row_out: torch.Tensor | None = None,
) -> None:
    """k_norm -> RoPE -> store for indexer keys.

    ``kv_slot_mapping`` is already compressed-domain (the indexer metadata
    builder runs ``get_compressed_slot_mapping``), so the kernel splits the
    slot into page and row itself and needs no block table.

    Args:
        k_pre: [num_tokens, 128] bf16, the ``wk(latent)`` projection. Only
            group-boundary rows are read.
        positions: [num_tokens] int32/int64 token positions.
        cos_sin_cache: [max_pos, rope_head_dim] fp32 GPT-J layout (cos half,
            then sin half), from the layer's compress-RoPE instance.
        rms_norm_weight: [128] k_norm weight.
        rms_norm_eps: Epsilon of the k_norm RMSNorm.
        k_cache: The paged indexer cache, bf16 or uint8 per ``store_fp8``.
        kv_slot_mapping: [num_tokens] slots in the indexer cache (-1 = skip).
        compress_ratio: group size; keys are emitted at group boundaries.
        store_fp8: per-token E4M3 with one FP32 scale when True, else BF16.
        row_out: When a pipeline cut leaves this layer's consumers on a later
            rank, receives the same record in token order so it can cross the
            boundary and be replayed there.

    """
    assert k_pre.ndim == 2 and k_pre.shape[1] == 128
    assert k_pre.dtype == torch.bfloat16 and k_pre.stride(1) == 1
    assert compress_ratio in (1, 2)
    num_tokens = kv_slot_mapping.numel()
    assert num_tokens <= k_pre.shape[0] and num_tokens <= positions.numel()
    if num_tokens == 0:
        return

    torch.ops._C.v41_indexer_k_norm_rope_store(
        k_pre,
        positions,
        cos_sin_cache,
        rms_norm_weight,
        rms_norm_eps,
        k_cache,
        kv_slot_mapping,
        compress_ratio,
        store_fp8,
        row_out,
    )
