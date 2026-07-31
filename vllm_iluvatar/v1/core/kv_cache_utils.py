# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/v1/core/kv_cache_utils.py

ILU override implementation for Attention quantization support.

Target   : vllm.v1.core.kv_cache_utils.get_kv_cache_configs
           vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups
Baseline : main_61ac3680210
Why      : Iluvatar attention quantization Int8 support.
           For allocating kv cache scales and adjust the num gpu blocks.
How      : get_kv_cache_configs: full-method copy with one delta.
           get_kv_cache_config_from_groups: super method call with deltas.
Scope    : always enabled if attention quantization is enabled.
Remove   : When upstream support int8 kv cache quantization.
"""

from typing import Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

logger = init_logger("vllm." + __name__)

origin_get_kv_cache_config_from_groups = kv_cache_utils.get_kv_cache_config_from_groups

_SCALE_KV_SPEC_TYPES = (FullAttentionSpec, MLAAttentionSpec)


def _get_shared_scale_kv_spec(
    shared_by: list[str],
    kv_cache_specs: dict[str, KVCacheSpec],
) -> Any | None:
    """Return the scale-aware spec after validating a shared KV tensor.

    Scale-unaware specs such as Mamba may share the main allocation with an
    attention spec, but only the attention layer owns and consumes the
    separately allocated scale cache.
    """
    scale_specs: set[Any] = set()
    scale_layers: list[str] = []
    scale_unaware_layers: list[str] = []

    for layer_name in shared_by:
        layer_kv_spec = kv_cache_specs[layer_name]
        spec_name = layer_kv_spec.__class__.__name__
        if isinstance(layer_kv_spec, _SCALE_KV_SPEC_TYPES):
            scale_specs.add(layer_kv_spec)
            scale_layers.append(f"{layer_name} ({spec_name})")
        else:
            scale_unaware_layers.append(f"{layer_name} ({spec_name})")

    if len(scale_specs) > 1:
        raise ValueError(
            "Scale-aware layers sharing one KV cache tensor must have the "
            f"same KV spec, but got: {scale_layers}."
        )

    if not scale_specs:
        logger.debug(
            "KV cache tensor does not need a scale cache; shared_by=%s",
            scale_unaware_layers,
        )
        return None

    if scale_unaware_layers:
        logger.debug(
            "Allocating a scale cache only for scale-aware layers %s; "
            "scale-unaware sharing layers are %s",
            scale_layers,
            scale_unaware_layers,
        )
    return next(iter(scale_specs))


def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    from functools import partial

    from vllm.v1.core.kv_cache_utils import (
        _auto_fit_max_model_len,
        _check_enough_kv_cache_memory,
        _estimate_max_model_len_from_groups,
        _max_memory_usage_bytes_from_groups,
        _pool_bytes_per_block,
        _project_kv_cache_groups_to_worker,
        get_kv_cache_capacity,
        get_kv_cache_groups,
    )
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    # Merge the KV cache specs of all workers. Different PP stages may have
    # different layer names, and different TP ranks of the same PP stage should
    # have the same KV cache spec.
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    # Check if the KV cache specs are registered correctly.
    # This is to prevent that some layers are initialized with unregistered specs.
    KVCacheSpecRegistry.check_kv_cache_spec_registry(merged_kv_cache_specs)
    # Get global KV cache groups. This also handles spec unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_kv_cache_specs may be modified in-place.
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use per-worker projected groups to account for PP sharding.
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
        for worker_spec in kv_cache_specs
    ]

    # If `num_gpu_blocks_override` is set, the cache size that will actually
    # be allocated is decoupled from the profiled `available_memory`:
    # `may_override_num_blocks` in `get_kv_cache_config_from_groups` clamps
    # `num_blocks` to the override. Reflect that in `available_memory` here so
    # auto-fit, the admission check, and the per-worker config builder all
    # plan against the same effective capacity.
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        adjusted_memory: list[int] = []
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
            if not groups:
                adjusted_memory.append(avail_mem)
                continue
            bytes_per_block = _pool_bytes_per_block(vllm_config, groups)
            logger.info(
                "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
                avail_mem // bytes_per_block,
                override,
            )
            adjusted_memory.append(override * bytes_per_block)
        available_memory = adjusted_memory

    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory
        )

    # Check if the available memory is enough per worker.
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )

    kv_cache_configs: list[KVCacheConfig] = []
    for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        assert sum(len(group.layer_names) for group in projected_groups) == len(
            kv_cache_spec_one_worker
        ), "Some layers are not assigned to any group."
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, projected_groups, available_memory_one_worker
            )
        )

    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        # Shrink tensor size proportionally
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks

            # NOTE >>> iluvatar specific logic
            if tensor.scale_size:
                assert tensor.scale_size % num_blocks_old == 0
                tensor.scale_size = tensor.scale_size // num_blocks_old * min_num_blocks
            # <<< iluvatar specific logic

        if len(kv_cache_config.kv_cache_groups) > 0:
            max_model_len = vllm_config.model_config.max_model_len
            # GPU KV cache size in tokens = max_concurrency * max_model_len:
            # the total tokens of context the pool can hold at peak
            # utilization. Sourcing this from the concurrency calculation
            # handles hybrid layouts correctly.
            num_tokens, max_concurrency = get_kv_cache_capacity(
                vllm_config, kv_cache_config
            )

            logger.info_once("GPU KV cache size: %s tokens", f"{num_tokens:,}")
            logger.info_once(
                "Maximum concurrency for %s tokens per request: %.2fx",
                f"{max_model_len:,}",
                max_concurrency,
            )

    return kv_cache_configs


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    # first, use original function to get the config
    config = origin_get_kv_cache_config_from_groups(
        vllm_config, kv_cache_groups, available_memory
    )
    # graph capture mode
    num_blocks = config.num_blocks

    kv_cache_specs = {}
    all_quant_layers: set[str] = set()
    for kv_cache_group in kv_cache_groups:
        if isinstance(kv_cache_group.kv_cache_spec, UniformTypeKVCacheSpecs):
            kv_cache_specs.update(kv_cache_group.kv_cache_spec.kv_cache_specs)
            all_quant_layers.update(kv_cache_group.kv_cache_spec.kv_cache_specs.keys())
        else:
            kv_cache_specs.update(
                {
                    layer_name: kv_cache_group.kv_cache_spec
                    for layer_name in kv_cache_group.layer_names
                }
            )
            all_quant_layers.update(kv_cache_group.layer_names)

    # calculate the total size of the kv cache and the total size of the kv cache with scale
    total_size = 0
    total_size_with_scale = 0
    scale_specs_by_tensor: list[Any | None] = []
    for kv_cache_tensor in config.kv_cache_tensors:
        total_size += kv_cache_tensor.size
        total_size_with_scale += kv_cache_tensor.size
        layer_kv_spec = _get_shared_scale_kv_spec(
            kv_cache_tensor.shared_by, kv_cache_specs
        )
        scale_specs_by_tensor.append(layer_kv_spec)

        if layer_kv_spec is not None:
            total_size_with_scale += (
                layer_kv_spec.scale_page_size_bytes
            ) * num_blocks + layer_kv_spec.fixed_size_bytes

    # shrink the num blocks
    total_used_memory = total_size
    assert total_used_memory <= available_memory or available_memory == 0
    if available_memory != 0:
        new_blocks = int(total_used_memory / total_size_with_scale * num_blocks)
    else:
        # on graph capturing, the num_blocks is the smallest block that forward needed, so, do not shrink it.
        new_blocks = num_blocks
    logger.info(
        "available_memory: %s, total_used_memory: %s, total_size_with_scale: %s, init_blocks: %s, new_blocks: %s",
        available_memory,
        total_used_memory,
        total_size_with_scale,
        num_blocks,
        new_blocks,
    )

    # adjust the size of the kv cache tensors and add kv scale and fixed size to kv_cache_tensors
    # scale_size and fixed_size are only used on gpu model runner and will not influence the others.
    for kv_cache_tensor, layer_kv_spec in zip(
        config.kv_cache_tensors, scale_specs_by_tensor
    ):
        assert kv_cache_tensor.size % num_blocks == 0
        kv_cache_tensor.size = kv_cache_tensor.size // num_blocks * new_blocks

        if layer_kv_spec is not None:
            kv_cache_tensor.scale_size = (
                layer_kv_spec.scale_page_size_bytes * new_blocks
            )
            kv_cache_tensor.fixed_size = layer_kv_spec.fixed_size_bytes

    config.num_blocks = new_blocks
    return config
