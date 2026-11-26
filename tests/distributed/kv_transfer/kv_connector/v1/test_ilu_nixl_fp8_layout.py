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
    get_fp8_wire_scratch_blocks,
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


def _mock_multi_config(
    *,
    pull_connector: str = "IluNixlPullConnector",
    child_staging: bool = True,
    parent_staging: bool = False,
    scratch_blocks=None,
):
    child_extra = {}
    if child_staging:
        child_extra["enable_fp8_ds_mla_wire_staging"] = "True"
    if scratch_blocks is not None:
        child_extra["fp8_wire_scratch_blocks"] = scratch_blocks
    extra = {
        "connectors": [
            {
                "kv_connector": pull_connector,
                "kv_role": "kv_producer",
                "kv_connector_extra_config": child_extra,
            },
            {
                "kv_connector": "OffloadingConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {"cpu_bytes_to_use": 1},
            },
        ]
    }
    if parent_staging:
        extra["enable_fp8_ds_mla_wire_staging"] = "True"
    config = MagicMock()
    transfer_config = config.kv_transfer_config = MagicMock()
    transfer_config.kv_role = "kv_producer"
    transfer_config.kv_connector = "MultiConnector"
    transfer_config.kv_connector_extra_config = extra
    transfer_config.get_from_extra_config.side_effect = lambda key, default: extra.get(
        key, default
    )
    return config


def test_wire_staging_gate_is_ilu_connector_only():
    assert use_fp8_ds_mla_wire_staging(_mock_vllm_config(staging=True))
    assert use_fp8_ds_mla_wire_staging(
        _mock_vllm_config(staging=True, connector="IluNixlPullConnector")
    )
    assert not use_fp8_ds_mla_wire_staging(
        _mock_vllm_config(staging=True, connector="NixlConnector")
    )


def test_wire_staging_gate_reads_multiconnector_ilu_child():
    assert use_fp8_ds_mla_wire_staging(_mock_multi_config())
    assert use_fp8_ds_mla_wire_staging(
        _mock_multi_config(child_staging=False, parent_staging=True)
    )
    assert not use_fp8_ds_mla_wire_staging(_mock_multi_config(child_staging=False))
    assert not use_fp8_ds_mla_wire_staging(
        _mock_multi_config(pull_connector="NixlConnector", parent_staging=True)
    )
    assert get_fp8_wire_scratch_blocks(_mock_multi_config(scratch_blocks="3")) == 3
    assert get_fp8_wire_scratch_blocks(_mock_multi_config()) == 4


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


def test_dsv41_i8g64_and_split_indexer_map_to_fp8_wire():
    config = _mock_vllm_config()
    main_name = "layers.2.main"
    idx_name = "layers.2.indexer.k_cache"
    main = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=512,
        tokens_per_state=2,
        cache_dtype_str="i8g64",
        storage_block_size=256,
    )
    indexer = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        tokens_per_state=2,
        cache_dtype_str="int8_indexer_split",
    )
    specs = {main_name: main, idx_name: indexer}
    groups = [
        KVCacheGroupSpec(
            [main_name, idx_name],
            UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=specs),
        )
    ]

    wire_groups = get_fp8_ds_mla_wire_groups(config, groups, specs)
    wire_specs = wire_groups[0].kv_cache_spec.kv_cache_specs

    wire_main = wire_specs[main_name]
    assert wire_main.dtype == torch.uint8
    assert wire_main.cache_dtype_str == "fp8_ds_mla"
    assert wire_main.model_version == "deepseek_v4"
    assert wire_main.state_content_bytes == 584
    assert wire_main.alignment == 64
    assert getattr(wire_main, "storage_block_size", None) is None

    wire_idx = wire_specs[idx_name]
    assert wire_idx.dtype == torch.uint8
    assert wire_idx.head_size == 132
    assert wire_idx.cache_dtype_str is None
    # One compute block stores block_size / tokens_per_state = 128 states at
    # 132 B each (128 INT8 values + one FP32 scale): 128 * 132 = 16896. Without
    # an ``fp8_wire_block_size`` override the wire page keeps the legacy 576 B
    # alignment, so 16896 rounds up to 17280.
    assert wire_idx.real_page_size_bytes == 128 * 132
    assert wire_idx.page_size_bytes == 17280


def test_dsv41_i4g32_maps_to_fp8_wire():
    config = _mock_vllm_config()
    name = "layers.8.main"
    spec = MLAAttentionSpec(
        block_size=512,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=256,
        tokens_per_state=2,
        cache_dtype_str="i4g32",
        storage_block_size=512,
    )
    specs = {name: spec}
    groups = [
        KVCacheGroupSpec(
            [name],
            UniformTypeKVCacheSpecs(block_size=512, kv_cache_specs=specs),
        )
    ]
    wire = get_fp8_ds_mla_wire_groups(config, groups, specs)[0].kv_cache_spec
    wire_spec = wire.kv_cache_specs[name]
    assert wire_spec.cache_dtype_str == "fp8_ds_mla"
    assert wire_spec.model_version == "deepseek_v4"
    assert wire_spec.state_content_bytes == 584


def test_fp8_wire_block_size_subblocks_match_d_payloads():
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.fp8_layout import (
        get_fp8_wire_block_size,
        packed_bytes_per_block,
        wire_sub_blocks_per_compute_block,
    )

    config = _mock_multi_config()
    config.kv_transfer_config.kv_connector_extra_config["connectors"][0][
        "kv_connector_extra_config"
    ]["fp8_wire_block_size"] = 64
    config.model_config.hf_config.model_type = "deepseek_v41"
    assert get_fp8_wire_block_size(config) == 64

    main_cr2 = MLAAttentionSpec(
        block_size=512,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=256,
        tokens_per_state=2,
        cache_dtype_str="i4g32",
        storage_block_size=512,
    )
    idx_cr2 = MLAAttentionSpec(
        block_size=1024,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        tokens_per_state=2,
        cache_dtype_str="int8_indexer_split",
    )
    main_cr1 = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=256,
        tokens_per_state=1,
        cache_dtype_str="i4g32",
        storage_block_size=256,
    )
    idx_cr1 = MLAAttentionSpec(
        block_size=512,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        tokens_per_state=1,
        cache_dtype_str="int8_indexer_split",
    )
    specs = {
        "layers.2.main": main_cr2,
        "layers.2.indexer.k_cache": idx_cr2,
        "layers.20.main": main_cr1,
        "layers.20.indexer.k_cache": idx_cr1,
    }
    compute_groups = [
        KVCacheGroupSpec(["layers.2.main"], main_cr2),
        KVCacheGroupSpec(["layers.2.indexer.k_cache"], idx_cr2),
        KVCacheGroupSpec(["layers.20.main"], main_cr1),
        KVCacheGroupSpec(["layers.20.indexer.k_cache"], idx_cr1),
    ]
    wire_groups = get_fp8_ds_mla_wire_groups(config, compute_groups, specs)
    w_main_cr2 = wire_groups[0].kv_cache_spec.kv_cache_specs["layers.2.main"]
    w_idx_cr2 = wire_groups[1].kv_cache_spec.kv_cache_specs["layers.2.indexer.k_cache"]
    w_main_cr1 = wire_groups[2].kv_cache_spec.kv_cache_specs["layers.20.main"]
    w_idx_cr1 = wire_groups[3].kv_cache_spec.kv_cache_specs["layers.20.indexer.k_cache"]

    assert w_main_cr2.block_size == 64
    assert w_main_cr2.page_size_bytes == 18688
    assert wire_sub_blocks_per_compute_block(main_cr2, w_main_cr2) == 8

    assert w_idx_cr2.block_size == 64
    assert w_idx_cr2.page_size_bytes == 4224
    assert wire_sub_blocks_per_compute_block(idx_cr2, w_idx_cr2) == 16

    assert w_main_cr1.block_size == 64
    assert w_main_cr1.page_size_bytes == 37376
    assert wire_sub_blocks_per_compute_block(main_cr1, w_main_cr1) == 4

    assert w_idx_cr1.block_size == 64
    assert w_idx_cr1.page_size_bytes == 8448
    assert wire_sub_blocks_per_compute_block(idx_cr1, w_idx_cr1) == 8

    wire_cfg = build_fp8_ds_mla_wire_config(compute_groups, wire_groups, num_blocks=4)
    assert [t.block_stride for t in wire_cfg.kv_cache_tensors] == [
        18688,
        4224,
        37376,
        8448,
    ]
    assert [t.layer_stride for t in wire_cfg.kv_cache_tensors] == [
        4 * 8 * 18688,
        4 * 16 * 4224,
        4 * 4 * 37376,
        4 * 8 * 8448,
    ]
    assert (
        packed_bytes_per_block(wire_groups, compute_groups=compute_groups) == 4 * 37376
    )
