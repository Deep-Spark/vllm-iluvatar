# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Ilu FP8 wire-cache planning wrappers."""

from unittest.mock import MagicMock

import pytest
import torch
import vllm.v1.core.kv_cache_utils as community
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

import vllm_iluvatar.v1.core.kv_cache_utils as ilu_planning
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.fp8_layout import (
    FP8_WIRE_SCRATCH_BUFFER_COUNT,
)
from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.ilu_nixl.fp8_staging import (
    FP8DSMLAWireStaging,
)


def _config(connector: str = "IluNixlConnector"):
    config = MagicMock()
    config.attention_config.hisparse_config = None
    config.cache_config.num_gpu_blocks_override = None
    config.cache_config.prefix_cache_retention_interval = None
    config.cache_config.get_resolved_kv_cache_layout.return_value = (
        KVCacheLayout.BLHNC
    )
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.speculative_config = None
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


def _single_uniform_group():
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        tokens_per_state=4,
        cache_dtype_str="bfloat16",
        alignment=512,
    )
    specs = {"layers.0.main": spec}
    uniform = UniformTypeKVCacheSpecs(
        block_size=spec.block_size,
        kv_cache_specs=specs,
    )
    return specs, [KVCacheGroupSpec(list(specs), uniform)]


def test_planning_activation_reserves_memory_and_scheduler_strips_wire_config():
    config = _config()
    specs, groups = _single_uniform_group()
    available_memory = 64 * 1024**2
    patched = ilu_planning.make_get_kv_cache_config_from_groups(
        community.get_kv_cache_config_from_groups
    )

    baseline = community.get_kv_cache_config_from_groups(
        config, groups, available_memory
    )
    planned = patched(config, groups, available_memory, specs)

    wire = planned.wire_kv_cache_config
    assert wire is not None
    wire_bytes_per_block = wire.kv_cache_tensors[0].block_stride
    scratch_bytes = FP8_WIRE_SCRATCH_BUFFER_COUNT * 4 * wire_bytes_per_block
    expected = community.get_kv_cache_config_from_groups(
        config,
        groups,
        available_memory - scratch_bytes,
    )
    assert wire.num_blocks == planned.num_blocks
    assert planned.num_blocks == expected.num_blocks
    assert planned.num_blocks < baseline.num_blocks
    assert (
        sum(tensor.size for tensor in planned.kv_cache_tensors)
        <= available_memory - scratch_bytes
    )

    scheduler_planner = ilu_planning.make_generate_scheduler_kv_cache_config(
        community.generate_scheduler_kv_cache_config
    )
    scheduler_config = scheduler_planner([planned])
    assert getattr(scheduler_config, "wire_kv_cache_config", None) is None
    assert planned.wire_kv_cache_config is wire


def test_planning_preserves_zero_memory_profiling_and_connector_gate():
    specs, groups = _single_uniform_group()
    patched = ilu_planning.make_get_kv_cache_config_from_groups(
        community.get_kv_cache_config_from_groups
    )

    profiling = patched(_config(), groups, available_memory=0)
    assert profiling.num_blocks == 0
    assert profiling.wire_kv_cache_config is None

    upstream_connector = patched(
        _config(connector="NixlConnector"),
        groups,
        64 * 1024**2,
        specs,
    )
    assert not hasattr(upstream_connector, "wire_kv_cache_config")


def test_staging_reads_dynamic_wire_config_from_baseline_dataclass():
    baseline_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )

    with pytest.raises(ValueError, match="was not planned"):
        FP8DSMLAWireStaging(
            _config(),
            baseline_config,
            compute_caches={},
        )


def test_multiworker_wrapper_supplies_order_and_resizes_wire_config():
    config = _config()
    specs, groups = _single_uniform_group()
    one_worker_planner = ilu_planning.make_get_kv_cache_config_from_groups(
        community.get_kv_cache_config_from_groups
    )

    def baseline_multiworker(_config, worker_specs, memories):
        configs = [one_worker_planner(_config, groups, memory) for memory in memories]
        min_blocks = min(item.num_blocks for item in configs)
        for item in configs:
            old_blocks = item.num_blocks
            item.num_blocks = min_blocks
            for tensor in item.kv_cache_tensors:
                assert tensor.size % old_blocks == 0
                tensor.size = tensor.size // old_blocks * min_blocks
        return configs

    patched_multiworker = ilu_planning.make_get_kv_cache_configs(baseline_multiworker)
    configs = patched_multiworker(
        config,
        [specs, dict(specs)],
        [96 * 1024**2, 64 * 1024**2],
    )

    assert configs[0].num_blocks == configs[1].num_blocks
    for item in configs:
        wire = item.wire_kv_cache_config
        assert wire.num_blocks == item.num_blocks
        assert all(
            tensor.size == tensor.block_stride * item.num_blocks
            for tensor in wire.kv_cache_tensors
        )


def test_planning_wrapper_preserves_current_packed_layout():
    config = _config(connector="NixlConnector")
    config.kv_transfer_config.kv_connector_extra_config[
        "enable_cross_layers_blocks"
    ] = "True"
    specs, groups = _single_uniform_group()
    spec = next(iter(specs.values()))
    second_specs = {"layers.1.main": spec}
    groups.append(
        KVCacheGroupSpec(
            list(second_specs),
            UniformTypeKVCacheSpecs(
                block_size=spec.block_size,
                kv_cache_specs=second_specs,
            ),
        )
    )
    baseline = community.get_kv_cache_config_from_groups(
        config, groups, available_memory=8 * 1024**2
    )
    patched = ilu_planning.make_get_kv_cache_config_from_groups(
        community.get_kv_cache_config_from_groups
    )
    packed_config = patched(config, groups, available_memory=8 * 1024**2)

    assert packed_config.num_blocks == baseline.num_blocks
    assert packed_config.kv_cache_tensors == baseline.kv_cache_tensors
    assert all(tensor.block_stride > 0 for tensor in packed_config.kv_cache_tensors)
