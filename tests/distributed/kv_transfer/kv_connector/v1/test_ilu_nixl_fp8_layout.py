# SPDX-License-Identifier: Apache-2.0

import re
from unittest.mock import MagicMock

import torch
from vllm.v1.core.kv_cache_utils import (
    _project_kv_cache_groups_to_worker,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.fp8_layout import (
    build_fp8_ds_mla_wire_config,
    get_fp8_ds_mla_wire_groups,
    use_fp8_ds_mla_wire_staging,
)


def _mock_vllm_config(staging: bool = False, connector: str = "IluNixlConnector"):
    config = MagicMock()
    config.attention_config.hisparse_config = None
    config.cache_config.num_gpu_blocks_override = None
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.speculative_config = None
    config.kv_transfer_config = None
    if staging:
        transfer_config = config.kv_transfer_config = MagicMock()
        transfer_config.kv_role = "kv_producer"
        transfer_config.kv_connector = connector
        transfer_config.kv_connector_extra_config = {
            "enable_fp8_ds_mla_wire_staging": "True"
        }
        transfer_config.get_from_extra_config.side_effect = lambda key, default: (
            transfer_config.kv_connector_extra_config.get(key, default)
        )
    return config


def _deepseek_v4_compute_specs():
    specs = {}
    ratios = [1, 1] + [4 if i % 2 == 0 else 128 for i in range(41)]
    for layer_idx, ratio in enumerate(ratios):
        prefix = f"layers.{layer_idx}"
        if ratio == 4:
            specs[f"{prefix}.indexer.k_cache"] = MLAAttentionSpec(
                block_size=256,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.bfloat16,
                tokens_per_state=4,
            )
            specs[f"{prefix}.indexer.state"] = SlidingWindowMLASpec(
                block_size=4,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.float32,
                sliding_window=8,
                alignment=512,
            )

        specs[f"{prefix}.swa"] = SlidingWindowMLASpec(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=128,
            head_size_v=0,
            cache_dtype_str="bfloat16",
            model_version=None,
        )
        if ratio > 1:
            specs[f"{prefix}.main"] = MLAAttentionSpec(
                block_size=256,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.bfloat16,
                tokens_per_state=ratio,
                cache_dtype_str="bfloat16",
                alignment=512,
                model_version=None,
            )
            specs[f"{prefix}.state"] = SlidingWindowMLASpec(
                block_size=4 if ratio == 4 else 8,
                num_kv_heads=1,
                head_size=2048 if ratio == 4 else 1024,
                dtype=torch.float32,
                sliding_window=8 if ratio == 4 else 128,
                alignment=512,
            )
    return specs


def test_wire_staging_gate_is_ilu_connector_only():
    assert use_fp8_ds_mla_wire_staging(_mock_vllm_config(staging=True))
    assert use_fp8_ds_mla_wire_staging(
        _mock_vllm_config(staging=True, connector="IluNixlPullConnector")
    )
    assert not use_fp8_ds_mla_wire_staging(
        _mock_vllm_config(staging=True, connector="NixlConnector")
    )


def _layer_index(layer_name: str) -> int:
    match = re.search(r"layers\.(\d+)", layer_name)
    assert match is not None
    return int(match.group(1))


def test_wire_group_builder_preserves_registration_order_and_compute_groups():
    config = _mock_vllm_config()
    first = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        tokens_per_state=4,
        cache_dtype_str="bfloat16",
    )
    second = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        tokens_per_state=128,
        cache_dtype_str="bfloat16",
    )
    groups = [
        KVCacheGroupSpec(
            ["registered.second", "registered.first"],
            UniformTypeKVCacheSpecs(
                block_size=256,
                kv_cache_specs={
                    "registered.second": second,
                    "registered.first": first,
                },
            ),
        ),
        KVCacheGroupSpec([], first),
    ]
    ordered_specs = {
        "registered.first": first,
        "registered.second": second,
    }

    wire_groups = get_fp8_ds_mla_wire_groups(config, groups, ordered_specs)

    assert len(wire_groups) == 2
    assert wire_groups[0].layer_names == ["registered.first", "registered.second"]
    assert isinstance(wire_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs)
    assert list(wire_groups[0].kv_cache_spec.kv_cache_specs) == [
        "registered.first",
        "registered.second",
    ]
    assert wire_groups[1].layer_names == []


def test_wire_layout_has_golden_stride_and_no_compute_group_aliases():
    config = _mock_vllm_config()
    specs = _deepseek_v4_compute_specs()
    compute_groups = get_kv_cache_groups(config, specs)
    wire_groups = get_fp8_ds_mla_wire_groups(config, compute_groups, specs)
    wire = build_fp8_ds_mla_wire_config(compute_groups, wire_groups, num_blocks=8)

    wire_specs = {}
    for group in wire.kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        wire_specs.update(group.kv_cache_spec.kv_cache_specs)

    wire_indexer = wire_specs["layers.2.indexer.k_cache"]
    assert wire_indexer.dtype == torch.uint8
    assert wire_indexer.head_size == 132
    assert wire_indexer.real_page_size_bytes == 8448
    assert wire_indexer.page_size_bytes == 8640
    assert wire_specs["layers.2.indexer.state"].page_size_bytes == 8640
    assert wire_specs["layers.2.state"].page_size_bytes == 32832
    assert wire_specs["layers.3.state"].page_size_bytes == 32832

    # Current vLLM packs each cache group densely and overlays groups because
    # they have independent block-id namespaces. The backing stride is the
    # largest group's dense size rather than the sum of every group's slots.
    expected_stride = max(
        sum(wire_specs[name].page_size_bytes for name in group.layer_names)
        for group in wire_groups
    )
    assert expected_stride == 893952
    assert {tensor.block_stride for tensor in wire.kv_cache_tensors} == {
        expected_stride
    }

    owner = {
        name: idx
        for idx, group in enumerate(compute_groups)
        for name in group.layer_names
    }
    slots: dict[tuple[int, int], list[str]] = {}
    for tensor in wire.kv_cache_tensors:
        for name in tensor.layers:
            slots.setdefault((owner[name], tensor.offset), []).append(name)
    assert not [names for names in slots.values() if len(names) > 1]


def test_int8_indexer_compute_layout_maps_to_same_padded_wire_format():
    config = _mock_vllm_config()
    layer_name = "layers.2.indexer.k_cache"
    compute_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=4,
    )
    specs = {layer_name: compute_spec}
    compute_groups = [
        KVCacheGroupSpec(
            [layer_name],
            UniformTypeKVCacheSpecs(
                block_size=compute_spec.block_size,
                kv_cache_specs=specs,
            ),
        )
    ]

    wire_groups = get_fp8_ds_mla_wire_groups(config, compute_groups, specs)
    wire_spec = wire_groups[0].kv_cache_spec.kv_cache_specs[layer_name]

    assert compute_spec.real_page_size_bytes == wire_spec.real_page_size_bytes == 8448
    assert compute_spec.page_size_bytes == 8448
    assert wire_spec.dtype == torch.uint8
    assert wire_spec.head_size == 132
    assert wire_spec.page_size_bytes == 8640


def test_pp8_projected_wire_layout_has_no_compute_group_aliases():
    config = _mock_vllm_config()
    specs = _deepseek_v4_compute_specs()
    compute_groups = get_kv_cache_groups(config, specs)
    partition = [5, 5, 5, 5, 5, 6, 6, 6]
    start = 0
    for n_layers in partition:
        end = start + n_layers
        worker_spec = {
            name: spec
            for name, spec in specs.items()
            if start <= _layer_index(name) < end
        }
        projected = _project_kv_cache_groups_to_worker(compute_groups, worker_spec)
        wire_groups = get_fp8_ds_mla_wire_groups(config, projected, worker_spec)
        assert [len(group.layer_names) for group in wire_groups] == [
            len(group.layer_names) for group in projected
        ]
        build_fp8_ds_mla_wire_config(projected, wire_groups, num_blocks=8)
        start = end
