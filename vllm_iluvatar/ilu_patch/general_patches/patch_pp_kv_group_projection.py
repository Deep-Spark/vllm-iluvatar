# SPDX-License-Identifier: Apache-2.0
"""Keep empty PP KV-cache groups, but do not allocate their tensors.

Target   : vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups
Baseline : v0.31.0
Why      : _project_kv_cache_groups_to_worker keeps a group whose
           layer_names is empty and whose UniformTypeKVCacheSpecs still lists
           every layer. get_kv_cache_config_from_groups walks that spec, so
           the last PP stage allocates compressor state for kv sources it does
           not own, and allocate_kv_cache raises StopIteration.
           Dropping the group instead makes this rank's group count smaller
           than the scheduler, which copies rank 0's config. Block ids are
           then ``7 != 6`` and the rank send never starts.
How      : build tensors from the layers this rank owns, then put the
           original groups back so the group count and specs stay aligned
           with the scheduler.
"""

from __future__ import annotations

from dataclasses import replace

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="allocate only local layers of a PP KV-cache group",
    )


def _apply() -> None:
    from vllm.v1.core import kv_cache_utils as kv_utils
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    origin = kv_utils.get_kv_cache_config_from_groups

    def _config_local_layers_only(vllm_config, kv_cache_groups, available_memory):
        narrowed = []
        for group in kv_cache_groups:
            spec = group.kv_cache_spec
            if not isinstance(spec, UniformTypeKVCacheSpecs):
                narrowed.append(group)
                continue
            owned = {
                name: spec.kv_cache_specs[name]
                for name in group.layer_names
                if name in spec.kv_cache_specs
            }
            if len(owned) == len(spec.kv_cache_specs):
                narrowed.append(group)
                continue
            narrowed.append(
                replace(
                    group,
                    kv_cache_spec=UniformTypeKVCacheSpecs(
                        block_size=spec.block_size,
                        kv_cache_specs=owned,
                    ),
                )
            )
        config = origin(vllm_config, narrowed, available_memory)
        config.kv_cache_groups = kv_cache_groups
        return config

    kv_utils.get_kv_cache_config_from_groups = _config_local_layers_only
