# Source: general_patches/patch_mrope.py at 47fb804; bind only the owned fused path.
# SPDX-License-Identifier: Apache-2.0
"""Bind owned MRoPE for contiguous 2D NeoX inputs; reject other layouts."""

from __future__ import annotations

from vllm_iluvatar.backends.ixaite.layers.mrope import (
    ilu_fused_mrope,
    ilu_fused_mrope_supported,
)
from vllm_iluvatar.ilu_patch import general_patch as p


# Install the override only in the selected process-wide backend.
def register() -> None:
    from vllm_iluvatar.backends import provider_mode

    if provider_mode() == "baseline":
        return
    p.register_once(
        __name__,
        _apply,
        detail="MRotaryEmbedding.forward_cuda (fused cos/sin gather mrope)",
    )


# Repeated startup assigns the same implementation without retaining a fallback.
def _apply() -> None:
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    MRotaryEmbedding.forward_cuda = _ilu_mrope_forward_cuda


# Reject unsupported inputs before the in-place kernel changes query or key.
def _ilu_mrope_forward_cuda(self, positions, query, key=None, offsets=None):
    if key is None or positions.ndim != 2 or not self.is_neox_style:
        raise RuntimeError("ixaite MRoPE requires a key and 2D NeoX positions")
    cos_sin_cache = self._match_cos_sin_cache_dtype(query)
    if not ilu_fused_mrope_supported(positions, query, key, cos_sin_cache):
        raise RuntimeError("ixaite MRoPE tensor layout is unsupported")
    query_shape, key_shape = query.shape, key.shape
    ilu_fused_mrope(
        positions,
        query,
        key,
        cos_sin_cache,
        self.mrope_section,
        self.head_size,
        self.rotary_dim,
        self.mrope_interleaved,
    )
    return query.reshape(query_shape), key.reshape(key_shape)
