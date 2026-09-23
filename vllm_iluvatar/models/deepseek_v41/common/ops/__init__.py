"""Fused ops behind the official DeepSeek-V4.1 entry points.

Mirrors ``vllm/models/deepseek_v41/common/ops/``: every function keeps the
official name, argument order and contract, so the model code above this layer
reads like the official one. Official implements these in Triton, which does not
run on this platform, so each body dispatches to the ixformer kernel covering
the same work. Where a kernel does not line up yet the difference is confined to
this layer and marked ``TODO(ops)``.
"""

from vllm_iluvatar.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
    fused_save_compress_norm,
    rope_quant_insert,
)
from vllm_iluvatar.models.deepseek_v41.common.ops.indexer_k_store import (
    MXFP4_BLOCK_SIZE,
    indexer_k_norm_rope_store,
)

__all__ = [
    "MXFP4_BLOCK_SIZE",
    "fused_save_compress_norm",
    "indexer_k_norm_rope_store",
    "rope_quant_insert",
]
