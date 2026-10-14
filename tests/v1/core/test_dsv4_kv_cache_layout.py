# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for the community DeepSeek-V4 KV layout."""

from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
from vllm.v1.core.kv_cache_utils import (
    _get_packed_kv_cache_groups,
    get_kv_cache_config_from_groups,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout


def _attention_spec(cls, *, block_size: int, page_size: int, **kwargs):
    return cls(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        head_size_v=0,
        dtype=torch.uint8,
        state_content_bytes=1,
        page_size_padded=page_size,
        **kwargs,
    )


def _flash_fp8_specs():
    specs = {}

    # 43 per-layer SWA pages become groups of 22 and 21 because the widest
    # mixed group below can hold 26 such pages in one physical block.
    for i in range(43):
        specs[f"swa.{i}"] = _attention_spec(
            SlidingWindowMLASpec,
            block_size=64,
            page_size=37376,
            sliding_window=4096,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
        )

    # The unbalanced 21:21:20 mixed-page bucket is kept whole and determines
    # the shared physical block stride:
    # 21 * 8448 + 21 * 37376 + 20 * 1216 = 986624.
    for i in range(21):
        specs[f"mla.{i}"] = _attention_spec(
            MLAAttentionSpec,
            block_size=256,
            page_size=8448,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
        )
    for i in range(21):
        specs[f"local.{i}"] = _attention_spec(
            MLAAttentionSpec,
            block_size=256,
            page_size=37376,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
        )
    for i in range(20):
        specs[f"index.{i}"] = _attention_spec(
            MLAAttentionSpec,
            block_size=256,
            page_size=1216,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
        )

    state_specs = (
        (8, 21, "state_a"),
        (2, 21, "state_b"),
        (4, 20, "state_c"),
    )
    for block_size, count, prefix in state_specs:
        for i in range(count):
            specs[f"{prefix}.{i}"] = _attention_spec(
                CircularBufferSpec,
                block_size=block_size,
                page_size=16384,
            )
    return specs


def _config():
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        speculative_config=None,
        model_config=None,
    )


def test_flash_fp8_group_table_uses_community_block_outermost_layout():
    config = _config()
    groups = _get_packed_kv_cache_groups(config, _flash_fp8_specs())
    assert groups is not None

    table = []
    for group in groups:
        spec = group.kv_cache_spec
        layer_specs = (
            spec.kv_cache_specs.values()
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else (spec,)
        )
        table.append(
            (
                len(group.layer_names),
                spec.block_size,
                sorted({layer.page_size_bytes for layer in layer_specs}),
            )
        )
    assert table == [
        (22, 64, [37376]),
        (21, 64, [37376]),
        (62, 256, [1216, 8448, 37376]),
        (21, 8, [16384]),
        (21, 2, [16384]),
        (20, 4, [16384]),
    ]

    bytes_per_block = 21 * 8448 + 21 * 37376 + 20 * 1216
    cache = get_kv_cache_config_from_groups(
        config, groups, available_memory=bytes_per_block * 7
    )
    assert cache.num_blocks == 7
    assert {tensor.size for tensor in cache.kv_cache_tensors} == {
        bytes_per_block * 7
    }
    assert {tensor.block_stride for tensor in cache.kv_cache_tensors} == {
        bytes_per_block
    }

    mixed = {
        tensor.layers[0].split(".", 1)[0]: (
            tensor.offset,
            tensor.layer_stride,
        )
        for tensor in cache.kv_cache_tensors
        if tensor.layers
        and tensor.layers[0].split(".", 1)[0] in {"mla", "local", "index"}
    }
    assert mixed == {
        "mla": (0, 8448),
        "local": (21 * 8448, 37376),
        "index": (21 * 8448 + 21 * 37376, 1216),
    }


def test_fp8_ds_mla_state_width_stays_584_bytes():
    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        head_size_v=0,
        dtype=torch.uint8,
        state_content_bytes=584,
        tokens_per_state=1,
        cache_dtype_str="fp8_ds_mla",
        alignment=64,
        model_version="deepseek_v4",
        storage_block_size=64,
    )

    assert spec.state_content_size_bytes == 584
    assert spec.unpadded_page_size_bytes == 64 * 584
    assert spec.page_size_bytes == 64 * 584
    assert spec.page_size_padded is None
    assert spec.storage_block_size == 64
    assert spec.page_size_bytes // spec.num_states == 584


def test_flat_kv_row_view_reports_physical_block_stride():
    block_size, head_dim, block_stride_rows = 64, 512, 1927
    cache = torch.empty_strided(
        (3, block_size, head_dim),
        (block_stride_rows * head_dim, head_dim, 1),
        dtype=torch.bfloat16,
    )

    rows, actual_block_stride_rows = flat_kv_row_view(cache, block_size)

    assert actual_block_stride_rows == cache.stride(0) // head_dim
    assert actual_block_stride_rows == block_stride_rows
    assert rows.shape == (
        (cache.shape[0] - 1) * block_stride_rows + block_size,
        head_dim,
    )
    assert rows.stride() == (head_dim, 1)
