# SPDX-License-Identifier: Apache-2.0
"""Decode-only metadata helpers for Iluvatar MoE paths.

The decode GEMV paths must avoid ``expert_sizes_gpu.cpu()`` during CUDA graph
capture. These helpers keep the attention-metadata shape checks shared by W16A16
and quantized MoE implementations without importing their kernel modules.
"""

from __future__ import annotations

import torch


def _is_supported_decode_metadata(value) -> bool:
    """Accept community / Iluvatar metadata used for decode-vs-general dispatch."""
    name = value.__class__.__name__
    return name in {
        "IluFlashAttentionMetadata",
        "IluDeepseekV32IndexerMetadata",
        "MLACommonMetadata",
        "DeepseekSparseSWAMetadata",
        "GDNAttentionMetadata",
        "IluGDNAttentionMetadata",
    } or (hasattr(value, "num_prefills") and hasattr(value, "num_decodes"))


def _metadata_values(attn_metadata) -> list[object]:
    if attn_metadata is None:
        return []
    if isinstance(attn_metadata, dict):
        return list(attn_metadata.values())
    return [attn_metadata]


def _is_only_decode(attn_metadata) -> bool:
    """Decide whether to take the CG-safe decode GEMV path."""
    # FULL_DECODE_ONLY CUDA graph capture must never hit expert_sizes_gpu.cpu().
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return True
    except Exception:
        pass

    values = _metadata_values(attn_metadata)
    if not values:
        return False

    decode_meta = None
    for value in values:
        if (
            _is_supported_decode_metadata(value)
            and hasattr(value, "num_prefills")
            and hasattr(value, "num_decodes")
        ):
            decode_meta = value
            break

    value_types = {value.__class__.__name__ for value in values}
    if len(value_types) == 1:
        return decode_meta is not None and all(
            value.num_decodes > 0 and value.num_prefills == 0 for value in values
        )

    if decode_meta is None:
        return False
    return decode_meta.num_decodes > 0 and decode_meta.num_prefills == 0
