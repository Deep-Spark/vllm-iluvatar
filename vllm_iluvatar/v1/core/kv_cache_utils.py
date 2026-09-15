# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/core/kv_cache_utils.py

ILU override implementations for attention quantization (Int8 KV cache
scale) support.

Target   : vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups
           vllm.v1.core.kv_cache_utils._pool_bytes_per_block
Baseline : vllm 0.29.0 (b53bdfecb11d)
Why      : Iluvatar attention quantization Int8 support -- for allocating
           KV cache scales and adjusting the num gpu blocks.
How      : B2 "shared backing buffer": the K-scale storage shares the SAME
           int8 backing allocation as the KV cache. Scale is PER-LAYER --
           each layer in a packed ``KVCacheTensor`` owns its K-scale page
           (and static V-scale, if any), matching 0.29.0's ``layers`` packing
           where every layer has its own KV page. (In 0.27.1 ``shared_by``
           meant true byte-sharing, so one scale per tensor sufficed; 0.29.0
           ``layers`` packs same-spec layers each-own-page, so scale must be
           ``* len(layer_names)``.)

           * Block-outermost layouts (BLHNC / BLNHC, the mixed-page-size and
             P2P-transfer case): scale is CO-LOCATED inside every block, right
             after that spec-region's KV pages -- block_stride =
             kv_bytes_per_block + scale_per_block_total. Within a block the
             layout is ``[KV region (L pages) | scale region (L scale pages)]``;
             layer ``l``'s scale sits at ``scale_offset + l * scale_page``.
             Transferring block b carries its KV and its scale together.
             allocate_kv_cache_scale carves a strided float32
             ``[num_blocks, scale_page/4]`` view per layer out of the per-block
             padding.
           * Layer-outermost layouts (LBNHC, uniform page sizes): scale is
             appended as a contiguous tail after the KV region instead, and
             block_stride stays KV-only. allocate_kv_cache_scale carves
             contiguous float32 views per layer from the tail.

           In both cases the fixed-size region (e.g. KVINT8 static V scale) is
           appended once after the per-block / KV region, since it does not
           scale with num_blocks; it is carved PER-LAYER too.

           _pool_bytes_per_block is overridden to return
           kv_bytes_per_block + scale_per_block_total so the community
           shrink loop in get_kv_cache_configs re-plans each worker with a
           scale-aware budget. This keeps the override pure
           (num_blocks = available // _pool_bytes_per_block), so all ranks
           still converge to the same num_blocks after the single re-plan.

           get_kv_cache_configs is not overridden: its shrink loop re-plans
           each worker by calling this override, so scale_size is recomputed
           consistently.
Scope    : only active when attention quantization is enabled
           (VLLM_ILU_FLASH_ATTN_QUANT_TYPE / VLLM_ILU_MLA_ATTN_QUANT_TYPE);
           see ilu_patch/general_patches/patch_kv_cache_utils.py.
Remove   : When upstream supports int8 kv cache quantization.
"""

from collections import defaultdict

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import (
    compute_layout_strides,
    may_override_num_blocks,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

logger = init_logger(__name__)

origin_get_kv_cache_config_from_groups = kv_cache_utils.get_kv_cache_config_from_groups
origin_get_kv_cache_bytes_per_block = kv_cache_utils._get_kv_cache_bytes_per_block
origin_pool_bytes_per_block = kv_cache_utils._pool_bytes_per_block

_SCALE_KV_SPEC_TYPES = (
    FullAttentionSpec,
    SlidingWindowSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


def _layers_by_spec(group: KVCacheGroupSpec) -> defaultdict[KVCacheSpec, list[str]]:
    """Group a cache group's layer names by their concrete KVCacheSpec."""
    group_spec = group.kv_cache_spec
    layers_by_spec: defaultdict[KVCacheSpec, list[str]] = defaultdict(list)
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        for layer_name, spec in group_spec.kv_cache_specs.items():
            layers_by_spec[spec].append(layer_name)
    elif group.layer_names:
        layers_by_spec[group_spec].extend(group.layer_names)
    return layers_by_spec


def _scale_bytes_for_spec(spec: KVCacheSpec) -> int:
    """Return scale_page_size_bytes for a spec, or 0 if it is not scale-aware."""
    if not isinstance(spec, _SCALE_KV_SPEC_TYPES):
        return 0
    return getattr(spec, "scale_page_size_bytes", 0) or 0


def _ilu_scale_per_block_total(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """Total per-block scale bytes across all scale-bearing spec-regions.

    Scale is PER-LAYER: each layer in a packed tensor owns its K-scale page,
    so a spec-region's contribution is ``scale_page * len(layer_names)``.
    """
    total = 0
    for group in kv_cache_groups:
        for spec, layer_names in _layers_by_spec(group).items():
            scale_page = _scale_bytes_for_spec(spec)
            total += scale_page * len(layer_names)
    return total


def _ilu_v_per_block_total(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """Total per-block V bytes for KINT8 (flash) specs, where V is allocated
    as a SEPARATE dense bf16 buffer per layer (not in the V2 backing buffer,
    which holds K only). Each such layer contributes
    ``num_heads * block_size * head_size_v * 2`` (bf16) per block, so a
    spec-region contributes that times ``len(layer_names)``. This must be
    included in the num_blocks divisor so num_blocks reflects the real
    total memory (K in V2 + V separate + scale separate), otherwise num_blocks
    is over-sized (K-only is small) and the per-layer V buffers OOM.
    """
    total = 0
    for group in kv_cache_groups:
        for spec, layer_names in _layers_by_spec(group).items():
            if not (
                isinstance(spec, (FullAttentionSpec, SlidingWindowSpec))
                and envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE in ("KINT8", "KVINT8")
            ):
                continue
            v_head = getattr(spec, "head_size_v", 0) or 0
            total += spec.num_heads * spec.block_size * v_head * 2 * len(layer_names)
    return total


def _ilu_v2_scale_per_block_total(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Per-block scale bytes that live in the V2 backing buffer.

    KINT8 (flash) scale is allocated as a SEPARATE dense buffer per layer (see
    allocate_kv_cache_scale), so it is excluded from the V2 buffer size /
    scale_tail; only KVINT8 / MLA-INT8 scale (carved out of the shared buffer)
    counts here.
    """
    total = 0
    for group in kv_cache_groups:
        for spec, layer_names in _layers_by_spec(group).items():
            if isinstance(
                spec, (FullAttentionSpec, SlidingWindowSpec)
            ) and envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE in ("KINT8", "KVINT8"):
                continue
            scale_page = _scale_bytes_for_spec(spec)
            total += scale_page * len(layer_names)
    return total


def _ilu_pool_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """Scale-aware bytes-per-block used by the community shrink loop.

    available_memory // _ilu_pool_bytes_per_block must equal the num_blocks
    produced by get_kv_cache_config_from_groups so the single re-plan in
    get_kv_cache_configs keeps every rank aligned.
    """
    return (
        origin_get_kv_cache_bytes_per_block(kv_cache_groups)
        + _ilu_v_per_block_total(kv_cache_groups)
        + _ilu_scale_per_block_total(kv_cache_groups)
    )


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """ILU override: plan KV cache + K-scale sharing one backing buffer.

    Scale and fixed are PER-LAYER (each layer in a packed tensor owns its
    K-scale page / static V-scale), so a spec-region's contribution is
    ``scale_page * len(layer_names)`` per block.

    Block-outermost: scale is co-located inside every block (after each
    spec-region's KV pages), so block_stride includes the scale pages and
    transferring a block carries its scale. Within a block the region is
    ``[KV region (L pages) | scale region (L scale pages)]``; ``scale_offset``
    is the byte offset of the scale region's start in block 0, and per-layer
    offsets within it (``+ l * scale_page``) are resolved by
    ``allocate_kv_cache_scale``. Layer-outermost: scale is appended as a
    contiguous tail (block_stride stays KV-only). The fixed-size region is
    always appended once after the per-block / KV region.
    """
    if len(kv_cache_groups) == 0:
        return origin_get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, available_memory
        )

    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    assert layout.name in ("LBNHC", "LBHNC"), (
        "Only LBNHC or LBHNC layout is supported for int8 quantization"
    )
    kv_bytes_per_block = origin_get_kv_cache_bytes_per_block(kv_cache_groups)
    scale_per_block_total = _ilu_scale_per_block_total(kv_cache_groups)
    v_per_block_total = _ilu_v_per_block_total(kv_cache_groups)

    # K (V2 backing buffer) + V (separate, KINT8) + scale (shared/tail) per
    # block, so num_blocks reflects the real total memory footprint.
    bytes_per_block = kv_bytes_per_block + v_per_block_total + scale_per_block_total
    num_blocks = available_memory // bytes_per_block
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    # Scale (K-scale, and static V-scale for KVINT8) lives in a TAIL appended
    # after the KV region -- NOT co-located inside every block. The ILU int8
    # read kernels (vllm_paged_attention_mla_int8 / vllm_paged_attention_int8
    # / cuInferPageAttentionV7) require kv_cache_scale.is_contiguous(); a
    # co-located (block-outermost) scale would be a strided as_strided view
    # whose block_stride is generally != 4*block_size, failing that check.
    # Tail placement carves scale as a contiguous slice, which always passes.
    # block_stride stays KV-only (page_size), matching community, so
    # compute_layout_strides derives block_stride = page_size (per-layer) and
    # layer_stride = num_blocks * page. KINT8/KVINT8 flash stores K alone in
    # the V2 buffer (V + scale are separate dense buffers), so its block_stride
    # is also K-only -- handled per-tensor below.
    v2_scale_per_block_total = _ilu_v2_scale_per_block_total(kv_cache_groups)
    scale_tail = v2_scale_per_block_total * num_blocks
    size = kv_bytes_per_block * num_blocks + scale_tail

    kv_cache_tensors: list[KVCacheTensor] = []
    for group in kv_cache_groups:
        byte_offset = 0
        for spec, layer_names in _layers_by_spec(group).items():
            num_layers = len(layer_names)
            is_flash_int8 = isinstance(
                spec, (FullAttentionSpec, SlidingWindowSpec)
            ) and envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE in ("KINT8", "KVINT8")
            # block_stride is always KV-only (None -> page_size): scale lives in
            # the tail, not inside every block. KINT8/KVINT8 flash is also K-only.
            blk_stride = None
            layer_stride, block_stride, _, _, _ = compute_layout_strides(
                spec,
                num_blocks,
                num_layers,
                layout,
                fixed_strides=(None, blk_stride, None, None, None),
            )
            offset = (
                byte_offset
                * max(layer_stride, spec.page_size_bytes)
                // spec.page_size_bytes
            )
            scale_page = _scale_bytes_for_spec(spec)
            # KINT8 (flash) stores K alone in the V2 buffer (dense,
            # C = head_size); V is a separate dense bf16 buffer allocated
            # per layer by allocate_kv_cache_scale. Flag the tensor so the
            # allocator knows to allocate fresh V + K-scale instead of carving
            # scale out of the shared backing buffer, and carry v_head_size for
            # sizing the V buffer.
            is_flash_int8 = isinstance(
                spec, (FullAttentionSpec, SlidingWindowSpec)
            ) and envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE in ("KINT8", "KVINT8")
            tensor = KVCacheTensor(
                size=size,
                layers=list(layer_names),
                layer_stride=layer_stride,
                block_stride=block_stride,
                offset=offset,
            )
            # ILU-added fields (injected by patch_kv_cache_interface.py).
            # Scale and fixed are PER-LAYER: each layer in the packed tensor
            # owns its K-scale page (and static V-scale, if any).
            tensor.is_flash_int8 = is_flash_int8
            tensor.v_head_size = spec.head_size_v if is_flash_int8 else 0
            if is_flash_int8:
                # K + scale are allocated as separate dense buffers per layer
                # (see allocate_kv_cache_scale); they do NOT live in the V2
                # backing buffer, so the V2 size / carve logic must ignore them.
                tensor.scale_size = 0
            else:
                tensor.scale_size = scale_page * num_layers * num_blocks
            # Scale lives in the tail (not co-located inside blocks); flash int8
            # also has no in-buffer scale (V + scale are separate buffers). Mark
            # scale_offset = -1 (no in-block carve) for all tensors; the per-tensor
            # tail offset is resolved later by allocate_kv_cache_scale.
            tensor.scale_offset = -1
            byte_offset += num_layers * spec.page_size_bytes
            kv_cache_tensors.append(tensor)

    logger.info(
        "ilu kv+scale plan: layout=%s kv_bytes_per_block=%s "
        "scale_per_block=%s num_blocks=%s size=%s",
        layout.name,
        kv_bytes_per_block,
        scale_per_block_total,
        num_blocks,
        size,
    )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=(
            vllm_config.cache_config.prefix_cache_retention_interval
        ),
    )
