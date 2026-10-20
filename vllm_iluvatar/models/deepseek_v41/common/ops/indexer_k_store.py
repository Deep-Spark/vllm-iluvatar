"""Indexer key production: k_norm, RoPE, quantize, paged store.

Mirrors ``vllm/models/deepseek_v41/common/ops/indexer_k_store.py``: same
function name, same arguments, same contract, so the indexer's ``_produce_k``
reads like the official one.

One argument differs, and deliberately. Official's last parameter is
``use_fp4_cache``, selecting between an MXFP4 page and a per-token FP8 one.
This platform carries no MXFP4 record, so the parameter is ``record`` and names
the layout the scoring kernels here actually read:

* ``"bf16"``: BF16 [num_blocks, block_size, 128] rows.
* ``"fp8"``: uint8 [num_blocks, block_size, 132] pages holding 128 E4M3 bytes
  per row followed by one FP32 scale per row, which is official's
  ``use_fp4_cache=False`` branch and what
  ``dsa_indexer_mqa_logits_with_blocks_fp8`` reads.
* ``"int8"``: the same 132-byte page, quantized with ixformer's scheme-A
  ``amax / 127`` instead of a power-of-two E4M3 scale, which is what
  ``dsa_indexer_mqa_logits`` reads on the ixinfer backend. Unlike ``"fp8"``
  this needs no FP8 hardware.

No path quantize-dequantizes. What lands in the cache is what the scoring
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
    record: str,
    row_out: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
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
        k_cache: The paged indexer cache, bf16 or uint8 per ``record``.
        kv_slot_mapping: [num_tokens] slots in the indexer cache (-1 = skip).
        compress_ratio: group size; keys are emitted at group boundaries.
        record: ``"bf16"``, ``"fp8"`` or ``"int8"``, as described above.
        row_out: When a pipeline cut leaves this layer's consumers on a later
            rank, receives the same record in token order so it can cross the
            boundary and be replayed there.

    """
    assert k_pre.ndim == 2 and k_pre.shape[1] == 128
    assert k_pre.dtype == torch.bfloat16 and k_pre.stride(1) == 1
    assert compress_ratio in (1, 2)
    assert record in ("bf16", "fp8", "int8")
    num_tokens = kv_slot_mapping.numel()
    assert num_tokens <= k_pre.shape[0] and num_tokens <= positions.numel()
    if num_tokens == 0:
        return

    if scales is not None:
        if record != "int8":
            raise ValueError("Separate indexer scales require INT8 keys")
        from ixformer.inference.functions.deepseek.int8_split_cache import (
            indexer_norm_rope_int8_store_split,
        )

        indexer_norm_rope_int8_store_split(
            k_pre, positions, cos_sin_cache, rms_norm_weight, rms_norm_eps,
            k_cache, scales, kv_slot_mapping, compress_ratio, row_out,
        )
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
        record,
        row_out,
    )
