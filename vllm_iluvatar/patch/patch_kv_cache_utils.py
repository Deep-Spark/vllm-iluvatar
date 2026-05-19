"""Patch ``vllm.v1.core.kv_cache_utils`` for DeepSeek-V4 KV-cache scale tensors.

Why
---
The iluvatar W4A8 V4 KV cache needs per-layer ``scale_page_size_bytes``
buffers in addition to the main KV blocks. Upstream
``_get_kv_cache_config_deepseek_v4`` returns a 2-tuple
``(num_blocks, kv_cache_tensors)`` so the model runner has nowhere to
read the scale tensor sizes from. The support_dsv4 branch evolves the
function to a 3-tuple
``(num_blocks, kv_cache_tensors, kv_cache_scale_tensors)`` and updates
the sole caller (``get_kv_cache_config_from_groups``) to unpack it.

How
---
We replace both functions with iluvatar-aware versions via ``setattr`` on
the imported ``vllm.v1.core.kv_cache_utils`` module. The replacement code
uses only names already exported by the upstream module so we re-import
them inside ``apply()`` to avoid hard-coding paths.
"""

from __future__ import annotations


def apply() -> None:
    from collections import defaultdict

    import vllm.envs as envs
    from vllm.config import VllmConfig
    from vllm.logger import init_logger
    from vllm.v1.core import kv_cache_utils as _kvcu
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        UniformTypeKVCacheSpecs,
    )

    logger = init_logger(__name__)

    if getattr(_kvcu, "_ilu_patched", False):
        return

    may_override_num_blocks = _kvcu.may_override_num_blocks
    get_uniform_page_size = _kvcu.get_uniform_page_size
    get_num_blocks = _kvcu.get_num_blocks

    def _get_kv_cache_config_deepseek_v4(
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> tuple[int, list[KVCacheTensor], list[KVCacheTensor]]:
        full_mla_spec = kv_cache_groups[0].kv_cache_spec
        assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
        page_sizes = sorted(full_mla_spec.get_page_sizes())
        layer_tuple_page_bytes = sum(page_sizes)

        bucketed: list[dict[int, list[str]]] = []
        for group in kv_cache_groups:
            assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            specs = group.kv_cache_spec.kv_cache_specs
            b: dict[int, list[str]] = defaultdict(list)
            for name in group.layer_names:
                b[specs[name].page_size_bytes].append(name)
            bucketed.append(b)

        num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values())
        num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)

        kv_cache_tensors: list[KVCacheTensor] = []
        kv_cache_scale_tensors: list[KVCacheTensor] = []
        for tuple_idx in range(num_layer_tuples):
            for ps in page_sizes:
                shared_by: list[str] = []
                scale_ps: int = 0
                for b, group in zip(bucketed, kv_cache_groups):
                    bucket = b.get(ps)
                    if bucket is not None and tuple_idx < len(bucket):
                        layer_name = bucket[tuple_idx]
                        shared_by.append(layer_name)
                        if scale_ps == 0:
                            assert isinstance(
                                group.kv_cache_spec, UniformTypeKVCacheSpecs
                            )
                            scale_ps = group.kv_cache_spec.kv_cache_specs[
                                layer_name
                            ].scale_page_size_bytes
                kv_cache_tensors.append(
                    KVCacheTensor(size=ps * num_blocks, shared_by=shared_by)
                )
                kv_cache_scale_tensors.append(
                    KVCacheTensor(size=scale_ps * num_blocks, shared_by=shared_by)
                )

        return num_blocks, kv_cache_tensors, kv_cache_scale_tensors

    def get_kv_cache_config_from_groups(
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> KVCacheConfig:
        if len(kv_cache_groups) == 0:
            return KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_groups,
            )

        if len(kv_cache_groups) == 1 and isinstance(
            kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
        ):
            num_blocks = (
                available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
            )
            num_blocks = may_override_num_blocks(vllm_config, num_blocks)
            per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
            kv_cache_tensors = [
                KVCacheTensor(
                    size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                    shared_by=[layer_name],
                )
                for layer_name in kv_cache_groups[0].layer_names
            ]
            kv_cache_scale_tensors = [
                KVCacheTensor(
                    size=per_layer_specs[layer_name].scale_page_size_bytes * num_blocks,
                    shared_by=[layer_name],
                )
                for layer_name in kv_cache_groups[0].layer_names
            ]
        elif all(
            isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            for group in kv_cache_groups
        ):
            (
                num_blocks,
                kv_cache_tensors,
                kv_cache_scale_tensors,
            ) = _get_kv_cache_config_deepseek_v4(
                vllm_config, kv_cache_groups, available_memory
            )
        else:
            group_size = max(len(group.layer_names) for group in kv_cache_groups)

            if envs.VLLM_ATTN_OPT_LEVEL == 2:
                page_size, scale_page_size, v_cache_scale_size = get_uniform_page_size(
                    [group.kv_cache_spec for group in kv_cache_groups]
                )
            else:
                page_size, scale_page_size = get_uniform_page_size(
                    [group.kv_cache_spec for group in kv_cache_groups]
                )
                v_cache_scale_size = 0
            assert group_size > 0, "group_size must be greater than 0"
            num_blocks = get_num_blocks(
                vllm_config,
                group_size,
                available_memory,
                page_size,
                scale_page_size,
            )
            kv_cache_tensors = []
            kv_cache_scale_tensors = []
            for i in range(group_size):
                shared_by = []
                for j in range(len(kv_cache_groups)):
                    if i < len(kv_cache_groups[j].layer_names):
                        shared_by.append(kv_cache_groups[j].layer_names[i])
                kv_cache_tensors.append(
                    KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
                )
                kv_cache_scale_tensors.append(
                    KVCacheTensor(
                        size=scale_page_size * num_blocks,
                        shared_by=shared_by,
                        size_scale=v_cache_scale_size,
                    )
                )

        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_scale_tensors=kv_cache_scale_tensors,
            kv_cache_groups=kv_cache_groups,
        )

    _kvcu._get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_deepseek_v4
    _kvcu.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups
    _kvcu._ilu_patched = True  # type: ignore[attr-defined]

    logger.info(
        "vllm-iluvatar: patched vllm.v1.core.kv_cache_utils "
        "(_get_kv_cache_config_deepseek_v4 + get_kv_cache_config_from_groups)"
    )
