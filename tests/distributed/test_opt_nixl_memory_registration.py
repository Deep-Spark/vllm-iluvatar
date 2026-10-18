from __future__ import annotations

import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheTensor,
    MLAAttentionSpec,
)

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.memory_registration import (
    MemoryRegistrationPlanner,
)


def test_packed_layers_use_per_layer_offsets() -> None:
    num_blocks = 2
    block_size = 64
    row_width = 256
    layer_stride = block_size * row_width
    block_stride = 2 * layer_stride
    backing = torch.empty(num_blocks * block_stride, dtype=torch.uint8)
    layers = [
        "model.layers.5.attn.indexer.k_cache",
        "model.layers.8.attn.indexer.k_cache",
    ]
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=backing.numel(),
                layers=layers,
                layer_stride=layer_stride,
                block_stride=block_stride,
            )
        ],
        kv_cache_groups=[],
    )
    spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=row_width,
        dtype=torch.uint8,
    )
    views = [
        torch.as_strided(
            backing,
            size=(num_blocks, block_size, row_width),
            stride=(block_stride, row_width, 1),
            storage_offset=layer_index * layer_stride,
        )
        for layer_index in range(len(layers))
    ]

    planner = MemoryRegistrationPlanner(config)
    registrations = [
        planner.describe(
            layer_name=layer_name,
            tensor=view,
            spec=spec,
            num_blocks=num_blocks,
        )
        for layer_name, view in zip(layers, views, strict=True)
    ]

    assert [registration.payload_segments for registration in registrations] == [
        ((0, layer_stride),),
        ((layer_stride, layer_stride),),
    ]
