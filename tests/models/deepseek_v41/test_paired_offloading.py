"""Exercise shared planning and real strided payload/scale offload views."""

from types import SimpleNamespace

import pytest
import torch
from ixformer.inference.functions.deepseek.quantized_kv_format import I4G32, I8G64
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
)

from vllm_iluvatar.ilu_patch.general_patches.patch_pp_cpu_offloading import (
    _worker_bytes_per_block,
)
from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    CompressedKVState,
    plan_scale_layout,
)
from vllm_iluvatar.models.deepseek_v41.indexer_kv import (
    SPLIT_INDEXER_FORMAT,
    IndexerScaleState,
    plan_indexer_scales,
)
from vllm_iluvatar.models.deepseek_v41.offloading import (
    canonical_paired_pages,
    plan_offload_layout,
)


def paired_cache(fmt, block_outer):
    count, page = 3, 65536
    main = MLAAttentionSpec(
        block_size=fmt.rows_per_page(512) * 2,
        tokens_per_state=2,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=fmt.payload_bytes(512),
        cache_dtype_str=fmt.name,
    )
    index = MLAAttentionSpec(
        block_size=512,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        cache_dtype_str=SPLIT_INDEXER_FORMAT,
    )
    swa = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
    )
    groups = [
        KVCacheGroupSpec(["main", "main_alias"], main),
        KVCacheGroupSpec(["index"], index),
        KVCacheGroupSpec(["swa"], swa),
    ]
    stride = 3 * page if block_outer else page
    layer = page if block_outer else count * page
    tensors = [
        KVCacheTensor(
            size=count * page * 3,
            layers=names,
            offset=offset,
            block_stride=stride,
            layer_stride=layer,
        )
        for names, offset in [
            (["main"], 0),
            (["main_alias"], 0),
            (["index"], layer),
            (["swa"], layer * 2),
        ]
    ]
    config = KVCacheConfig(count, tensors, groups)
    raw = torch.randint(0, 256, (count * page * 3,), dtype=torch.uint8)
    context = {
        name: SimpleNamespace(
            compressed_kv_format=fmt if name.startswith("main") else None,
            kv_cache_scales=None,
        )
        for name in ("main", "main_alias", "index", "swa")
    }
    states = (
        CompressedKVState(plan_scale_layout(config, fmt), fmt, "cpu", context),
        IndexerScaleState(plan_indexer_scales(config), "cpu", context),
    )
    caches = {}
    for names, offset in [(t.layers, t.offset) for t in tensors]:
        for name in names:
            caches[name] = raw.as_strided((count, page), (stride, 1), offset)
    return config, caches, context, states


@pytest.mark.parametrize("fmt", [I4G32, I8G64])
@pytest.mark.parametrize("block_outer", [False, True])
def test_offload_geometry_and_roundtrip_include_only_real_paired_pages(
    fmt, block_outer
):
    config, caches, context, states = paired_cache(fmt, block_outer)
    try:
        layout = plan_offload_layout(config)
        canonical = canonical_paired_pages(config, caches, context)
        assert len(layout.regions) == 5  # three payloads, two scale planes
        assert len(layout.group_regions[0]) == 2  # aliases are not transferred twice
        assert layout.bytes_per_block == _worker_bytes_per_block(config)
        expected = 3 * 65536 + fmt.scale_page_bytes(512) + 512 * 4
        assert layout.bytes_per_block == expected
        assert sum(t.page_size_bytes for t in canonical.tensors) == expected
        # The main sideplane mirrors the whole pool; unused holes do not count
        # toward CPU capacity or get assigned to a different manager block.
        assert states[0].layout.scale_bytes // config.num_blocks > fmt.scale_page_bytes(
            512
        )
        for tensor in canonical.tensors:
            tensor.tensor[1].random_(-128, 127)
        snapshots = [t.tensor.clone() for t in canonical.tensors]
        for refs in canonical.group_data_refs:
            saved = [
                (r.tensor_idx, canonical.tensors[r.tensor_idx].tensor[1].clone())
                for r in refs
            ]
            for index, _ in saved:
                canonical.tensors[index].tensor[1].zero_()
            for index, data in saved:
                canonical.tensors[index].tensor[1].copy_(data)
        assert all(
            torch.equal(t.tensor, before)
            for t, before in zip(canonical.tensors, snapshots)
        )
    finally:
        for state in reversed(states):
            state.release()


def test_reject_scale_view_bound_to_the_wrong_manager_offset():
    config, caches, context, states = paired_cache(I4G32, False)
    try:
        context["main"].kv_cache_scales = context["main"].kv_cache_scales[1:]
        with pytest.raises((AssertionError, ValueError), match="planned|page|^$"):
            canonical_paired_pages(config, caches, context)
    finally:
        for state in reversed(states):
            state.release()
