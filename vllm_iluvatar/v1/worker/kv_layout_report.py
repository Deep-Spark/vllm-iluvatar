# SPDX-License-Identifier: Apache-2.0
"""Startup report of the resolved KV cache layout, with a 32-bit span guard.

v0.30.0 allocates the whole KV pool as one backing tensor and gives every layer
an ``as_strided`` view into it, so a layer's blocks sit ``bytes_per_block``
apart even when that layer's own page is much smaller. The distance from a
layer's first byte to its last is therefore the size of the entire pool, not
``num_blocks * page_size_bytes`` as it was under the 0.27 Iluvatar layout.

That distance is what the attention kernels have to address. The DeepSeek-V4
gather kernels that still compute a per-lane ``voff`` with
``__ivcorex_ml_mem_load_i32`` wrap silently once it exceeds 4 GiB -- see
``kernels/踩坑记录/大kv-cache越界问题.md`` in llm-adapt-notes. The packed FP8 decode path moved to ``__ivcorex_ml_mem_a64_load_i32`` in
ixformer e3d1c525. The plain per-tensor E4M3 path uses native 64-bit pointer
arithmetic and has dedicated >4 GiB gather/attention tests. BF16 / W4A8 / int8
DeepSeek-V4 kernels remain unsafe, so for those we refuse to start rather than
return wrong tokens.
"""

from __future__ import annotations

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

logger = init_logger(__name__)

_SPAN_32BIT_LIMIT = 2**32 - 1

# Both FP8 layouts have >4 GiB coverage: packed uses the ixformer e3d1c525
# a64 load fix; plain E4M3 uses int64_t page-stride pointer arithmetic.
_A64_SAFE_CACHE_DTYPES = frozenset({"fp8_ds_mla", "fp8_e4m3"})


def _layer_specs(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
    specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            specs.update(spec.kv_cache_specs)
        else:
            specs.update({name: spec for name in group.layer_names})
    return specs


def report_kv_cache_layout(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """Log the resolved layout and abort if a layer span overflows 32 bits."""
    tensors = kv_cache_config.kv_cache_tensors
    if not tensors:
        return

    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    num_blocks = kv_cache_config.num_blocks
    pool_bytes = tensors[0].size
    specs = _layer_specs(kv_cache_config)

    lines = [
        f"ILU KV cache layout: {layout.name}, num_blocks={num_blocks}, "
        f"pool={pool_bytes} B ({pool_bytes / 2**30:.2f} GiB)"
    ]

    for gi, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        pages = sorted({specs[name].page_size_bytes for name in group.layer_names})
        lines.append(
            f"  group[{gi}] layers={len(group.layer_names)} "
            f"block_size={spec.block_size} "
            f"page_size_bytes={','.join(str(p) for p in pages)} "
            f"spec={type(spec).__name__}"
        )

    max_span = 0
    for tensor in tensors:
        page_size_bytes = specs[tensor.layers[0]].page_size_bytes
        span = (num_blocks - 1) * tensor.block_stride + page_size_bytes
        max_span = max(max_span, span)
        lines.append(
            f"  tensor layers={len(tensor.layers)} offset={tensor.offset} "
            f"layer_stride={tensor.layer_stride} "
            f"block_stride={tensor.block_stride} "
            f"page_size_bytes={page_size_bytes} layer_span={span} B"
        )

    lines.append(
        f"  max layer span = {max_span} B ({max_span / 2**30:.2f} GiB), "
        f"over 32-bit limit = {max_span > _SPAN_32BIT_LIMIT}"
    )
    logger.info("\n".join(lines))

    if max_span <= _SPAN_32BIT_LIMIT:
        return

    cache_dtype = vllm_config.cache_config.cache_dtype
    archs = vllm_config.model_config.architectures or []
    if cache_dtype in _A64_SAFE_CACHE_DTYPES or not any(
        "eepseek" in arch and "V4" in arch for arch in archs
    ):
        return

    bytes_per_block = max(tensor.block_stride for tensor in tensors)
    max_blocks = _SPAN_32BIT_LIMIT // bytes_per_block
    max_pool_bytes = max_blocks * bytes_per_block
    raise RuntimeError(
        f"DeepSeek-V4 KV cache layer span is {max_span} B, over the "
        f"{_SPAN_32BIT_LIMIT} B that the {cache_dtype} gather kernels can "
        f"address (their per-lane voffset is 32-bit; the packed FP8 and plain "
        f"E4M3 paths are the a64-safe exceptions). Results would be silently wrong. "
        f"Shrink the pool to at most {max_pool_bytes} B "
        f"({max_pool_bytes / 2**30:.2f} GiB, {max_blocks} blocks of "
        f"{bytes_per_block} B) with --kv-cache-memory, or lower "
        f"--gpu-memory-utilization."
    )
