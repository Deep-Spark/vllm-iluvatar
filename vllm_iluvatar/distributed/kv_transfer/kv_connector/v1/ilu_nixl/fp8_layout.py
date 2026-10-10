# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-side planning for the DeepSeek-V4 packed FP8 wire cache."""

from dataclasses import replace

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

_ILU_WIRE_CONNECTORS = ("IluNixlConnector", "IluNixlPullConnector")
FP8_WIRE_SCRATCH_BUFFER_COUNT = 2


def use_fp8_ds_mla_wire_staging(vllm_config: VllmConfig) -> bool:
    """Whether this Ilu producer publishes a D-compatible FP8 staging cache."""
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        return False
    if str(kv_transfer_config.kv_role).lower() != "kv_producer":
        return False
    if kv_transfer_config.kv_connector not in _ILU_WIRE_CONNECTORS:
        return False
    enabled = kv_transfer_config.get_from_extra_config(
        "enable_fp8_ds_mla_wire_staging", False
    )
    return str(enabled).lower() == "true"


def get_fp8_wire_scratch_blocks(vllm_config: VllmConfig) -> int:
    """Return and validate blocks per GPU wire scratch buffer."""
    kv_transfer_config = vllm_config.kv_transfer_config
    raw_value = (
        4
        if kv_transfer_config is None
        else kv_transfer_config.get_from_extra_config("fp8_wire_scratch_blocks", 4)
    )
    if isinstance(raw_value, bool):
        raise ValueError("fp8_wire_scratch_blocks must be a positive integer")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"fp8_wire_scratch_blocks must be a positive integer, got {raw_value!r}"
        ) from exc
    if value <= 0 or str(raw_value).strip() != str(value):
        raise ValueError(
            f"fp8_wire_scratch_blocks must be a positive integer, got {raw_value!r}"
        )
    return value


def _to_fp8_ds_mla_wire_spec(
    spec: KVCacheSpec, layer_name: str | None = None
) -> KVCacheSpec:
    """Return the D-side wire-layout counterpart of a producer cache spec."""
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return replace(
            spec,
            kv_cache_specs={
                name: _to_fp8_ds_mla_wire_spec(layer_spec, name)
                for name, layer_spec in spec.kv_cache_specs.items()
            },
        )
    if isinstance(spec, MLAAttentionSpec):
        is_indexer_cache = layer_name is not None and ".indexer." in layer_name
        if is_indexer_cache:
            # D stores 128 FP8 bytes and one FP32 UE8M0 scale per token.
            return replace(
                spec,
                head_size=132,
                dtype=torch.uint8,
                cache_dtype_str=None,
                model_version=None,
                alignment=576,
                page_size_padded=None,
            )

        is_main_ds_v4_cache = spec.model_version == "deepseek_v4" or (
            spec.head_size == 512
            and spec.cache_dtype_str in ("auto", "bfloat16", "fp8_ds_mla")
        )
        if is_main_ds_v4_cache:
            return replace(
                spec,
                dtype=torch.uint8,
                cache_dtype_str="fp8_ds_mla",
                model_version="deepseek_v4",
                alignment=576,
                page_size_padded=None,
            )
        raise ValueError(
            "Unsupported MLA cache for FP8 DS MLA wire staging: "
            f"layer={layer_name}, spec={spec}"
        )

    if isinstance(spec, SlidingWindowMLASpec):
        is_main_ds_v4_cache = spec.model_version == "deepseek_v4" or (
            spec.head_size == 512
            and spec.cache_dtype_str in ("auto", "bfloat16", "fp8_ds_mla")
            and (layer_name is None or ".state_cache" not in layer_name)
        )
        if is_main_ds_v4_cache:
            return replace(
                spec,
                dtype=torch.uint8,
                cache_dtype_str="fp8_ds_mla",
                model_version="deepseek_v4",
                alignment=576,
                page_size_padded=None,
            )

        if spec.sliding_window == 8:
            block_size = 4
        elif spec.sliding_window == 128:
            block_size = 8
        else:
            raise ValueError(
                "Unsupported compressor-state sliding window for FP8 DS MLA "
                f"wire staging: layer={layer_name}, sliding_window="
                f"{spec.sliding_window}"
            )
        return replace(
            spec,
            block_size=block_size,
            alignment=576,
            page_size_padded=None,
        )

    raise ValueError(
        "FP8 DS MLA wire staging only supports MLA, indexer, and compressor "
        f"state cache specs, got {type(spec).__name__}"
    )


def get_fp8_ds_mla_wire_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Convert each compute group to D-compatible wire specs without regrouping.

    Pipeline-parallel workers only see a projected layer subset. Re-running
    ``get_kv_cache_groups`` on that subset can split one compute group and
    alias packed wire slots. Keep compute-group membership (including empty
    projected groups) and only rewrite specs. ``vllm_config`` is unused but
    kept so callers can pass the same arguments as before.
    """
    del vllm_config

    flattened: dict[str, KVCacheSpec] = {}
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        for layer_name in group.layer_names:
            flattened[layer_name] = (
                group_spec.kv_cache_specs[layer_name]
                if isinstance(group_spec, UniformTypeKVCacheSpecs)
                else group_spec
            )

    order = [name for name in kv_cache_spec if name in flattened]
    if len(order) != len(flattened):
        missing = sorted(set(flattened) - set(order))
        raise ValueError(
            "FP8 DS MLA wire staging needs the registration order of every "
            "grouped layer, but these are absent from the KV cache spec: "
            + ", ".join(missing[:8])
        )

    order_index = {name: idx for idx, name in enumerate(order)}
    wire_groups: list[KVCacheGroupSpec] = []
    for group in kv_cache_groups:
        if not group.layer_names:
            wire_groups.append(group)
            continue
        ordered_names = sorted(group.layer_names, key=order_index.__getitem__)
        converted = {
            name: _to_fp8_ds_mla_wire_spec(flattened[name], name)
            for name in ordered_names
        }
        uniform = UniformTypeKVCacheSpecs.from_specs(converted)
        if uniform is None:
            raise ValueError(
                "FP8 DS MLA wire staging requires each compute group to remain "
                "a uniform wire spec after conversion, but these layers do "
                "not: " + ", ".join(ordered_names[:8])
            )
        wire_groups.append(
            replace(group, layer_names=ordered_names, kv_cache_spec=uniform)
        )
    return wire_groups


def packed_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """Return the 0.31 block-outermost shared-pool stride."""
    group_sizes = [
        sum(
            (
                group.kv_cache_spec.kv_cache_specs[layer_name]
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else group.kv_cache_spec
            ).page_size_bytes
            for layer_name in group.layer_names
        )
        for group in kv_cache_groups
        if group.layer_names
    ]
    if not group_sizes:
        raise ValueError("FP8 DS MLA wire staging needs at least one cache layer")
    return max(group_sizes)


def _assert_no_aliased_wire_slots(
    kv_cache_groups: list[KVCacheGroupSpec],
    tensors: list[KVCacheTensor],
) -> None:
    """Reject two layers of one compute group sharing a packed byte slot."""
    owner = {
        layer_name: group_idx
        for group_idx, group in enumerate(kv_cache_groups)
        for layer_name in group.layer_names
    }
    slots: dict[tuple[int, int], list[str]] = {}
    for tensor in tensors:
        for layer_name in tensor.layers:
            slots.setdefault((owner[layer_name], tensor.offset), []).append(layer_name)
    aliased = {key: names for key, names in slots.items() if len(names) > 1}
    if not aliased:
        return

    detail = "; ".join(
        f"group={group_idx} offset={offset} layers={','.join(sorted(names))}"
        for (group_idx, offset), names in sorted(aliased.items())[:4]
    )
    raise ValueError(
        f"FP8 DS MLA wire staging assigned {len(aliased)} packed slot(s) to "
        f"more than one layer of the same compute group: {detail}"
        + (" ..." if len(aliased) > 4 else "")
    )


def build_fp8_ds_mla_wire_config(
    kv_cache_groups: list[KVCacheGroupSpec],
    wire_groups: list[KVCacheGroupSpec],
    num_blocks: int,
) -> KVCacheConfig:
    """Build a packed wire-cache config from preplanned wire groups."""
    block_stride = packed_bytes_per_block(wire_groups)
    total_size = block_stride * num_blocks
    tensors: list[KVCacheTensor] = []
    for group in wire_groups:
        offset = 0
        for layer_name in group.layer_names:
            group_spec = group.kv_cache_spec
            spec = (
                group_spec.kv_cache_specs[layer_name]
                if isinstance(group_spec, UniformTypeKVCacheSpecs)
                else group_spec
            )
            tensors.append(
                KVCacheTensor(
                    size=total_size,
                    layers=[layer_name],
                    layer_stride=spec.page_size_bytes,
                    block_stride=block_stride,
                    offset=offset,
                )
            )
            offset += spec.page_size_bytes

    _assert_no_aliased_wire_slots(kv_cache_groups, tensors)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=wire_groups,
    )
