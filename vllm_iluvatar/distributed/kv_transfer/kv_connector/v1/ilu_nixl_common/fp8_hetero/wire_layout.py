# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-side mirror of D's packed FP8 KV cache byte layout (vLLM 0.31.0)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

HETERO_WIRE_CONNECTORS = ("IluNixlPushConnector",)

SCRATCH_BUFFER_COUNT = 2

COPY_ONLY = "COPY_ONLY"
QUANT_PACK = "QUANT_PACK"
QUANT_PACK_COMPRESSED = "QUANT_PACK_COMPRESSED"
QUANT_INDEXER = "QUANT_INDEXER"
REPACK_ONLY = "REPACK_ONLY"

_COMPRESSED_MAIN_DTYPES = ("i8g64", "i4g32")
_SPLIT_INDEXER_DTYPE = "int8_indexer_split"

ByteSpan = tuple[int, int]


def page_states(spec: KVCacheSpec) -> int:
    """Rows (stored states) in one physical page of ``spec`` on vLLM 0.31."""
    storage_block_size = getattr(spec, "storage_block_size", None)
    if storage_block_size is None:
        return int(spec.num_states)
    return int(spec.get_num_kernel_states(storage_block_size))


def spec_tokens_per_state(spec: KVCacheSpec) -> int:
    """Compression ratio (tokens per stored state) of ``spec`` on vLLM 0.31."""
    tps = getattr(spec, "tokens_per_state", None)
    if isinstance(tps, int) and not isinstance(tps, bool) and tps > 0:
        return tps
    cr = getattr(spec, "compress_ratio", None)
    if isinstance(cr, int) and not isinstance(cr, bool) and cr > 0:
        return cr
    return 1


@dataclass(frozen=True)
class HeteroDtypeConfig:
    """``kv_connector_extra_config`` knobs for P BF16 -> D FP8 push."""

    enabled: bool = False
    decode_cache_dtype: str = ""
    decode_block_size: int = 0
    staging_blocks: int = 8
    layerwise: bool = False

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> HeteroDtypeConfig:
        kv_transfer_config = _unwrap_push_connector(
            getattr(vllm_config, "kv_transfer_config", None)
        )
        extra = (
            getattr(kv_transfer_config, "kv_connector_extra_config", None)
            if kv_transfer_config is not None
            else None
        ) or {}
        connector = getattr(kv_transfer_config, "kv_connector", None)
        default_lw = connector == "IluNixlPushLayerwiseConnector"
        is_layerwise = (
            False
            if connector == "IluNixlPushConnector"
            else _as_bool(extra.get("prefer_layer_wise", default_lw))
        )
        if "fp8_staging_mode" in extra or "fp8_h2h_threads" in extra:
            raise ValueError(
                "fp8_staging_mode / fp8_h2h_threads have been removed; host-wire "
                "staging is direct-only (scratch D2H into pinned HostKVPool)"
            )
        staging_blocks = int(extra.get("fp8_staging_blocks", 8))
        if staging_blocks <= 0:
            raise ValueError(
                f"fp8_staging_blocks must be positive, got {staging_blocks}"
            )
        common = {
            "staging_blocks": staging_blocks,
            "layerwise": is_layerwise,
        }
        if not _as_bool(extra.get("hetero_kv_dtype", False)):
            return cls(**common)

        if connector not in HETERO_WIRE_CONNECTORS:
            raise ValueError(
                "hetero_kv_dtype is only implemented for "
                f"{HETERO_WIRE_CONNECTORS}, got {connector!r}"
            )
        decode_cache_dtype = str(extra.get("decode_cache_dtype", ""))
        if decode_cache_dtype != "fp8_ds_mla":
            raise ValueError(
                "hetero_kv_dtype requires decode_cache_dtype='fp8_ds_mla', "
                f"got {decode_cache_dtype!r}"
            )
        is_consumer = getattr(kv_transfer_config, "kv_role", None) == "kv_consumer"
        if not is_consumer:
            decode_block_size = int(extra.get("decode_block_size", 0))
            if decode_block_size <= 0:
                raise ValueError(
                    "hetero_kv_dtype requires a positive decode_block_size, got "
                    f"{decode_block_size!r}"
                )
            local_block_size = _local_wire_block_size(vllm_config, kv_cache_config)
            if decode_block_size != local_block_size:
                raise ValueError(
                    "hetero_kv_dtype requires identical P/D block_size, got "
                    f"P={local_block_size}, D={decode_block_size}"
                )
            if vllm_config.cache_config.num_gpu_blocks_override is not None:
                raise ValueError(
                    "num_gpu_blocks_override bypasses FP8 wire scratch reservation "
                    "and is not supported with hetero_kv_dtype"
                )

        return cls(
            enabled=True,
            decode_cache_dtype=decode_cache_dtype,
            decode_block_size=int(extra.get("decode_block_size", 0))
            if not is_consumer
            else 0,
            **common,
        )


def _group_member_specs(group: KVCacheGroupSpec) -> tuple[KVCacheSpec, ...]:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return tuple(spec.kv_cache_specs.values())
    return (spec,)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _local_wire_block_size(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None
) -> int:
    """Token grid shared by the HMA pages.

    After KV planning, ``cache_config.block_size`` is the minimum
    prefix-cacheable page and no longer the configured block size. SWA pages
    can be smaller than that, so the P/D block size is the GCD of the pages.
    """
    if kv_cache_config is None:
        return int(vllm_config.cache_config.block_size)
    page_sizes = [
        int(spec.block_size)
        for group in kv_cache_config.kv_cache_groups
        for spec in _group_member_specs(group)
        if isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec))
        and int(spec.block_size) > 0
    ]
    if not page_sizes:
        return int(vllm_config.cache_config.block_size)
    logical = page_sizes[0]
    for size in page_sizes[1:]:
        logical = _gcd(logical, size)
    return logical


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def use_hetero_fp8_wire(vllm_config: VllmConfig) -> bool:
    """Whether this process plans and allocates the FP8 wire mirror."""
    return HeteroDtypeConfig.from_vllm_config(vllm_config).enabled


def _unwrap_push_connector(cfg):
    """Use the IluNixlPush* child when the outer connector is MultiConnector."""
    if cfg is None or getattr(cfg, "kv_connector", None) != "MultiConnector":
        return cfg
    children = (getattr(cfg, "kv_connector_extra_config", None) or {}).get(
        "connectors"
    ) or ()
    push = [
        child
        for child in children
        if child.get("kv_connector") in HETERO_WIRE_CONNECTORS
    ]
    if len(push) != 1:
        return cfg
    child = push[0]
    role = child.get("kv_role", getattr(cfg, "kv_role", None))
    return SimpleNamespace(
        kv_connector=child["kv_connector"],
        kv_role=role,
        kv_buffer_device=child.get("kv_buffer_device"),
        kv_connector_extra_config=child.get("kv_connector_extra_config") or {},
        is_kv_producer=lambda: role in ("kv_producer", "kv_both"),
    )


def use_host_staging_wire(vllm_config: VllmConfig) -> bool:
    """Whether this process is a CPU-buffered push producer."""
    cfg = _unwrap_push_connector(getattr(vllm_config, "kv_transfer_config", None))
    if cfg is None:
        return False
    is_producer = getattr(cfg, "is_kv_producer", None)
    if is_producer is None:
        is_producer = getattr(cfg, "kv_role", None) in ("kv_producer", "kv_both")
    enabled = (
        getattr(cfg, "kv_connector", None) in HETERO_WIRE_CONNECTORS
        and getattr(cfg, "kv_buffer_device", None) == "cpu"
        and bool(is_producer)
    )
    cache_config = getattr(vllm_config, "cache_config", None)
    if enabled and getattr(cache_config, "num_gpu_blocks_override", None) is not None:
        raise ValueError(
            "num_gpu_blocks_override bypasses host wire scratch reservation "
            "and is not supported with CPU-buffered push staging"
        )
    return enabled


def _to_wire_spec(
    spec: KVCacheSpec,
    layer_name: str | None = None,
    *,
    prefer_v41: bool = False,
) -> KVCacheSpec:
    """Return the D-side wire-layout counterpart of a producer cache spec."""
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return replace(
            spec,
            kv_cache_specs={
                name: _to_wire_spec(layer_spec, name, prefer_v41=prefer_v41)
                for name, layer_spec in spec.kv_cache_specs.items()
            },
        )
    if isinstance(spec, CircularBufferSpec):
        # Ring / compressor state stays local under swa_bounded_replay and is
        # never scheduled onto the FP8 host-wire path.
        return spec
    if isinstance(spec, MLAAttentionSpec):
        if _is_indexer_cache(layer_name):
            return _indexer_wire_spec(spec, prefer_v41=prefer_v41)
        if _is_compressed_main_cache(spec) or _is_main_ds_v41_cache(spec):
            return _main_fp8_wire_spec(spec, model_version="deepseek_v4", alignment=64)
        if prefer_v41 and _is_plain_main_mla_cache(spec):
            return _main_fp8_wire_spec(spec, model_version="deepseek_v4", alignment=64)
        if _is_main_ds_v4_cache(spec):
            return _main_fp8_wire_spec(spec, model_version="deepseek_v4", alignment=576)
        raise ValueError(
            "Unsupported MLA cache for FP8 DS MLA wire staging: "
            f"layer={layer_name}, spec={spec}"
        )

    if isinstance(spec, SlidingWindowMLASpec):
        if (
            (_is_compressed_main_cache(spec) or _is_main_ds_v41_cache(spec))
            and (layer_name is None or ".state_cache" not in layer_name)
        ):
            return _main_fp8_wire_spec(spec, model_version="deepseek_v4", alignment=64)
        if (
            prefer_v41
            and _is_plain_main_mla_cache(spec)
            and (layer_name is None or ".state_cache" not in layer_name)
        ):
            return _main_fp8_wire_spec(spec, model_version="deepseek_v4", alignment=64)
        if _is_main_ds_v4_cache(spec) and (
            layer_name is None or ".state_cache" not in layer_name
        ):
            return _main_fp8_wire_spec(spec, model_version="deepseek_v4", alignment=576)
        if spec.sliding_window == 8:
            block_size = 4
            align = 576
        elif spec.sliding_window == 128:
            block_size = 8
            align = 576
        elif int(getattr(spec, "sliding_window", 0) or 0) > 0:
            tps = max(1, int(spec_tokens_per_state(spec)))
            block_size = max(
                1,
                min(
                    int(spec.block_size),
                    int(spec.sliding_window) // (2 * tps) or 1,
                ),
            )
            align = int(getattr(spec, "alignment", 576) or 576)
        else:
            raise ValueError(
                "Unsupported compressor-state sliding window for FP8 DS MLA "
                f"wire staging: layer={layer_name}, "
                f"sliding_window={spec.sliding_window}"
            )
        return replace(
            spec, block_size=block_size, alignment=align, page_size_padded=None
        )

    raise ValueError(
        "FP8 DS MLA wire staging only supports MLA, indexer, and compressor "
        f"state cache specs, got {type(spec).__name__}"
    )


def _main_fp8_wire_spec(
    spec: KVCacheSpec, *, model_version: str, alignment: int
) -> KVCacheSpec:
    return replace(
        spec,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        model_version=model_version,
        state_content_bytes=584,
        alignment=alignment,
        page_size_padded=None,
    )


def _indexer_wire_spec(
    spec: MLAAttentionSpec, *, prefer_v41: bool = False
) -> KVCacheSpec:
    """Map BF16 / INT8 / split-INT8 indexer pages to D's 132-byte FP8 records."""
    is_v41 = prefer_v41 or (
        getattr(spec, "model_version", None) == "deepseek_v41"
        or getattr(spec, "cache_dtype_str", None) == _SPLIT_INDEXER_DTYPE
        or (
            getattr(spec, "alignment", 576) != 576
            and spec.head_size not in (128, 132)
        )
    )
    if spec.head_size in (128, 132):
        wire_head_size = 132
    elif spec.dtype == torch.bfloat16:
        wire_head_size = int(spec.head_size) + 4
    else:
        wire_head_size = int(spec.head_size)
    wire_align = 64 if is_v41 else 576
    return replace(
        spec,
        head_size=wire_head_size,
        dtype=torch.uint8,
        cache_dtype_str=None,
        model_version=None,
        state_content_bytes=wire_head_size,
        alignment=wire_align,
        page_size_padded=None,
    )


def _is_indexer_cache(layer_name: str | None) -> bool:
    return layer_name is not None and (
        layer_name.endswith(".indexer.k_cache") or ".indexer." in layer_name
    )


def _is_compressed_main_cache(spec: KVCacheSpec) -> bool:
    return getattr(spec, "cache_dtype_str", None) in _COMPRESSED_MAIN_DTYPES


def _is_main_ds_v41_cache(spec: KVCacheSpec) -> bool:
    if getattr(spec, "model_version", None) == "deepseek_v41":
        return True
    if _is_compressed_main_cache(spec):
        return True
    return (
        getattr(spec, "state_content_bytes", None) == 584
        and getattr(spec, "model_version", None) != "deepseek_v4"
        and int(getattr(spec, "alignment", 576) or 576) == 64
        and int(getattr(spec, "head_size", 0) or 0) == 512
    )


def _is_plain_main_mla_cache(spec: KVCacheSpec) -> bool:
    return int(getattr(spec, "head_size", 0) or 0) == 512 and getattr(
        spec, "cache_dtype_str", None
    ) in ("auto", "bfloat16", "fp8_ds_mla", None)


def _is_main_ds_v4_cache(spec: KVCacheSpec) -> bool:
    if _is_compressed_main_cache(spec) or _is_main_ds_v41_cache(spec):
        return False
    return spec.model_version == "deepseek_v4" or _is_plain_main_mla_cache(spec)


def quant_kind_for(
    layer_name: str | None,
    wire_spec: KVCacheSpec,
    compute_spec: KVCacheSpec | None = None,
) -> str:
    if _is_indexer_cache(layer_name):
        return QUANT_INDEXER
    if compute_spec is not None and _is_compressed_main_cache(compute_spec):
        return QUANT_PACK_COMPRESSED
    if (
        isinstance(wire_spec, (MLAAttentionSpec, SlidingWindowMLASpec))
        and wire_spec.model_version in ("deepseek_v4", "deepseek_v41")
        and wire_spec.cache_dtype_str == "fp8_ds_mla"
    ):
        return QUANT_PACK
    if compute_spec is not None and compute_spec == wire_spec:
        return COPY_ONLY
    return REPACK_ONLY


def page_ratio_for(
    layer_name: str | None,
    compute_spec: KVCacheSpec,
    target_spec: KVCacheSpec,
) -> int:
    """Number of D wire pages carried by one P compute page."""
    source_block_size = int(compute_spec.block_size)
    target_block_size = int(target_spec.block_size)
    if source_block_size % target_block_size:
        raise ValueError(
            "P/D cache block sizes must have an integral page ratio: "
            f"layer={layer_name}, P={source_block_size}, D={target_block_size}"
        )
    ratio = source_block_size // target_block_size
    if ratio not in (1, 2):
        raise ValueError(
            "FP8 wire staging only supports one or two D pages per P page: "
            f"layer={layer_name}, ratio={ratio}"
        )
    if ratio > 1 and quant_kind_for(layer_name, target_spec) != REPACK_ONLY:
        raise ValueError(
            "Only compressor-state pages may be split on the FP8 wire: "
            f"layer={layer_name}, ratio={ratio}"
        )
    return ratio


def group_page_ratios(
    compute_groups: list[KVCacheGroupSpec],
    target_groups: list[KVCacheGroupSpec],
) -> tuple[int, ...]:
    """One P:D page ratio per compute group."""
    compute_specs = flatten_layer_specs(compute_groups)
    target_specs = flatten_layer_specs(target_groups)
    ratios: list[int] = []
    for group, target_group in zip(compute_groups, target_groups):
        if not group.layer_names:
            ratios.append(
                page_ratio_for(
                    None, group.kv_cache_spec, target_group.kv_cache_spec
                )
            )
            continue
        group_ratios = {
            page_ratio_for(name, compute_specs[name], target_specs[name])
            for name in group.layer_names
        }
        if len(group_ratios) != 1:
            raise ValueError(
                "Every cache in one compute group must use the same wire page "
                f"ratio: layers={group.layer_names}, ratios={sorted(group_ratios)}"
            )
        ratios.append(group_ratios.pop())
    return tuple(ratios)


def flatten_layer_specs(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> dict[str, KVCacheSpec]:
    specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        for layer_name in group.layer_names:
            specs[layer_name] = (
                group_spec.kv_cache_specs[layer_name]
                if isinstance(group_spec, UniformTypeKVCacheSpecs)
                else group_spec
            )
    return specs


def get_wire_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    kv_cache_spec: dict[str, KVCacheSpec],
    global_kv_cache_groups: list[KVCacheGroupSpec] | None = None,
) -> list[KVCacheGroupSpec]:
    """Keep P's projected groups; swap each layer to its D-side wire spec."""
    flattened = flatten_layer_specs(kv_cache_groups)
    missing = sorted(set(flattened) - set(kv_cache_spec))
    if missing:
        raise ValueError(
            "FP8 DS MLA wire staging is missing these grouped layers in the "
            "KV cache spec: " + ", ".join(missing[:8])
        )
    hetero = use_hetero_fp8_wire(vllm_config)
    if global_kv_cache_groups is not None and len(global_kv_cache_groups) != len(
        kv_cache_groups
    ):
        raise ValueError(
            "Global and projected KV cache groups must correspond one to one: "
            f"{len(global_kv_cache_groups)} != {len(kv_cache_groups)}"
        )
    from ..dsv41 import is_dsv41_config

    prefer_v41 = is_dsv41_config(vllm_config)
    globals_by_index = global_kv_cache_groups or [None] * len(kv_cache_groups)
    order_index = {name: idx for idx, name in enumerate(kv_cache_spec)}
    return [
        _wire_group(
            group,
            (
                (
                    lambda name, spec, _prefer_v41=prefer_v41: _to_wire_spec(
                        spec, name, prefer_v41=_prefer_v41
                    )
                )
                if hetero
                else (lambda name, spec: spec)
            ),
            global_group,
            order_index=order_index,
        )
        for group, global_group in zip(kv_cache_groups, globals_by_index)
    ]


def _wire_group(
    group: KVCacheGroupSpec,
    convert: Callable[[str | None, KVCacheSpec], KVCacheSpec],
    global_group: KVCacheGroupSpec | None = None,
    *,
    order_index: dict[str, int] | None = None,
) -> KVCacheGroupSpec:
    names = list(group.layer_names)
    if not names:
        if global_group is None or not global_group.layer_names:
            return group
        return replace(
            group, kv_cache_spec=_wire_group(global_group, convert).kv_cache_spec
        )
    if order_index is not None and all(n in order_index for n in names):
        names = sorted(names, key=order_index.__getitem__)
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        converted = {name: convert(name, spec.kv_cache_specs[name]) for name in names}
        wire_spec = UniformTypeKVCacheSpecs.from_specs(converted)
        if wire_spec is None:
            raise ValueError(
                "FP8 DS MLA wire staging could not keep a uniform wire spec "
                f"for compute group layers={names[:8]}"
            )
        return replace(group, layer_names=names, kv_cache_spec=wire_spec)
    converted_specs = {name: convert(name, spec) for name in names}
    wire_spec = next(iter(converted_specs.values()))
    if any(layer_spec != wire_spec for layer_spec in converted_specs.values()):
        raise ValueError(
            "FP8 DS MLA wire staging needs one wire spec per non-uniform "
            f"compute group: layers={names[:8]}"
        )
    return replace(group, layer_names=names, kv_cache_spec=wire_spec)


def packed_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """Return the v0.31 block-outermost shared-pool stride."""
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


def build_wire_config(
    kv_cache_groups: list[KVCacheGroupSpec],
    wire_groups: list[KVCacheGroupSpec],
    num_blocks: int,
) -> KVCacheConfig:
    """Build the packed D-layout ``KVCacheConfig`` mirrored by the host pool."""
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
    cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=wire_groups,
    )
    cfg._keep_on_scheduler = True
    return cfg


def _tensor_layers(tensor: KVCacheTensor) -> list[str]:
    layers = getattr(tensor, "layers", None)
    if layers is not None:
        return list(layers)
    return list(getattr(tensor, "shared_by", ()))


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
        layer_stride = int(getattr(tensor, "layer_stride", 0) or 0)
        for idx, layer_name in enumerate(_tensor_layers(tensor)):
            if layer_name in owner:
                slot_offset = tensor.offset + idx * layer_stride
                slots.setdefault((owner[layer_name], slot_offset), []).append(
                    layer_name
                )
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
    )


def _wire_page_bytes(wire_groups: list[KVCacheGroupSpec]) -> dict[str, int]:
    return {
        name: spec.page_size_bytes
        for name, spec in flatten_layer_specs(wire_groups).items()
    }


def stage_unit_bytes(
    compute_groups: list[KVCacheGroupSpec],
    wire_groups: list[KVCacheGroupSpec],
    layerwise: bool,
) -> tuple[int, ...]:
    """Bytes of one dense scratch row, per compute group."""
    page_bytes = _wire_page_bytes(wire_groups)
    compute_specs = flatten_layer_specs(compute_groups)
    target_specs = flatten_layer_specs(wire_groups)
    per_batch: dict[tuple[int, int], int] = {}
    for group_idx, group in enumerate(compute_groups):
        for layer_name in group.layer_names:
            batch = _transformer_index(layer_name) if layerwise else 0
            key = (group_idx, batch)
            ratio = page_ratio_for(
                layer_name, compute_specs[layer_name], target_specs[layer_name]
            )
            per_batch[key] = (
                per_batch.get(key, 0) + ratio * page_bytes[layer_name]
            )
    units = [0] * len(compute_groups)
    for (group_idx, _), value in per_batch.items():
        units[group_idx] = max(units[group_idx], value)
    return tuple(units)


def stage_rows(
    vllm_config: VllmConfig,
    compute_groups: list[KVCacheGroupSpec],
    layerwise: bool,
    staging_blocks: int,
) -> tuple[int, ...]:
    """Blocks held by one scratch/staging buffer, per compute group."""
    if not layerwise:
        return (staging_blocks,) * len(compute_groups)
    max_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
    rows: list[int] = []
    for group in compute_groups:
        block_size = int(group.kv_cache_spec.block_size)
        rows.append(-(-max_tokens // block_size) + 1)
    return tuple(rows)


def wire_scratch_bytes(
    vllm_config: VllmConfig,
    compute_groups: list[KVCacheGroupSpec],
    wire_groups: list[KVCacheGroupSpec],
) -> int:
    """GPU bytes to carve out of the KV budget for quantization scratch."""
    cfg = HeteroDtypeConfig.from_vllm_config(vllm_config)
    units = stage_unit_bytes(compute_groups, wire_groups, cfg.layerwise)
    rows = stage_rows(
        vllm_config, compute_groups, cfg.layerwise, cfg.staging_blocks
    )
    return SCRATCH_BUFFER_COUNT * sum(r * u for r, u in zip(rows, units))


def _transformer_index(layer_name: str) -> int:
    try:
        return extract_layer_index(layer_name)
    except (AssertionError, ValueError, TypeError):
        return -1


def subpage_ratio_for(wire_spec: KVCacheSpec, decode_block_size: int) -> int:
    """Number of D scheduler-block sub-pages packed inside one wire page."""
    if decode_block_size <= 0:
        return 1
    wire_bs = int(getattr(wire_spec, "block_size", 0) or 0)
    if wire_bs > decode_block_size and wire_bs % decode_block_size == 0:
        return wire_bs // decode_block_size
    return 1


@dataclass(frozen=True)
class WirePlan:
    """Everything the worker needs to mirror P's HBM KV into D's byte layout."""

    wire_config: KVCacheConfig
    block_stride: int
    page_bytes: dict[str, int]
    target_page_bytes: dict[str, int]
    page_ratios: dict[str, int]
    byte_offset: dict[str, int]
    quant_kind: dict[str, str]
    compute_specs: dict[str, KVCacheSpec]
    wire_specs: dict[str, KVCacheSpec]
    layer_to_group: dict[str, int]
    group_layers: tuple[tuple[str, ...], ...]
    decode_block_size: int = 0

    @classmethod
    def build(cls, vllm_config: VllmConfig, compute_config: KVCacheConfig) -> WirePlan:
        hetero_cfg = HeteroDtypeConfig.from_vllm_config(vllm_config, compute_config)
        hetero = hetero_cfg.enabled
        if hetero and vllm_config.model_config.dtype is not torch.bfloat16:
            raise ValueError(
                "hetero_kv_dtype requires a BF16 producer, got "
                f"{vllm_config.model_config.dtype}"
            )
        target_config = getattr(compute_config, "wire_kv_cache_config", None)
        if target_config is None:
            raise ValueError(
                "CPU-buffered push staging is enabled but KV planning did not attach a "
                "wire_kv_cache_config; the ILU kv_cache_utils patch must be "
                "active in the engine core process"
            )
        if target_config.num_blocks != compute_config.num_blocks:
            raise ValueError(
                "Compute and wire KV cache block counts must match: "
                f"{compute_config.num_blocks} != {target_config.num_blocks}"
            )
        if not target_config.kv_cache_tensors:
            raise ValueError("FP8 DS MLA wire staging has no cache tensors")

        target_block_stride = target_config.kv_cache_tensors[0].block_stride
        if target_block_stride <= 0 or any(
            tensor.block_stride != target_block_stride
            for tensor in target_config.kv_cache_tensors
        ):
            raise ValueError("FP8 DS MLA wire staging must use one packed layout")

        target_specs = flatten_layer_specs(target_config.kv_cache_groups)
        compute_specs = flatten_layer_specs(compute_config.kv_cache_groups)
        page_ratios = {
            name: page_ratio_for(name, compute_specs[name], target_specs[name])
            for name in compute_specs
        }
        if any(ratio > 1 for ratio in page_ratios.values()):
            pool_specs = {
                name: (
                    replace(
                        spec,
                        alignment=None,
                        page_size_padded=spec.page_size_bytes * page_ratios[name],
                    )
                    if page_ratios[name] > 1
                    else spec
                )
                for name, spec in target_specs.items()
            }
            pool_groups = []
            for group in target_config.kv_cache_groups:
                if not group.layer_names:
                    pool_groups.append(group)
                    continue
                group_spec = group.kv_cache_spec
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    group_spec = replace(
                        group_spec,
                        kv_cache_specs={
                            name: pool_specs[name]
                            for name in group.layer_names
                        },
                    )
                else:
                    group_spec = pool_specs[group.layer_names[0]]
                    if any(
                        pool_specs[name] != group_spec
                        for name in group.layer_names[1:]
                    ):
                        raise ValueError(
                            "A non-uniform pool page expansion requires "
                            f"per-layer specs: layers={group.layer_names}"
                        )
                pool_groups.append(replace(group, kv_cache_spec=group_spec))
            wire_config = build_wire_config(
                compute_config.kv_cache_groups,
                pool_groups,
                compute_config.num_blocks,
            )
        else:
            wire_config = target_config
        block_stride = wire_config.kv_cache_tensors[0].block_stride
        byte_offset: dict[str, int] = {}
        for tensor in wire_config.kv_cache_tensors:
            layer_stride = int(getattr(tensor, "layer_stride", 0) or 0)
            layers = _tensor_layers(tensor)
            for idx, layer_name in enumerate(layers):
                byte_offset[layer_name] = (
                    tensor.offset + idx * layer_stride
                    if len(layers) > 1
                    else tensor.offset
                )
        pool_specs = flatten_layer_specs(wire_config.kv_cache_groups)
        page_bytes = {name: spec.page_size_bytes for name, spec in pool_specs.items()}
        target_page_bytes = {
            name: spec.page_size_bytes for name, spec in target_specs.items()
        }

        layer_to_group: dict[str, int] = {}
        group_layers: list[tuple[str, ...]] = []
        for group_idx, group in enumerate(compute_config.kv_cache_groups):
            for layer_name in group.layer_names:
                if layer_name in layer_to_group:
                    raise ValueError(
                        f"Compute cache layer {layer_name} belongs to multiple groups"
                    )
                layer_to_group[layer_name] = group_idx
            group_layers.append(tuple(group.layer_names))

        missing = sorted(set(layer_to_group) - set(byte_offset))
        if missing:
            raise ValueError(
                "FP8 wire tensors do not cover compute cache layers: "
                + ", ".join(missing[:8])
            )
        for layer_name in layer_to_group:
            end = byte_offset[layer_name] + page_bytes[layer_name]
            if end > block_stride:
                raise ValueError(
                    "FP8 wire span is outside the packed block: "
                    f"layer={layer_name}, end={end}, block_stride={block_stride}"
                )

        return cls(
            wire_config=wire_config,
            block_stride=block_stride,
            page_bytes=page_bytes,
            target_page_bytes=target_page_bytes,
            page_ratios=page_ratios,
            byte_offset=byte_offset,
            quant_kind={
                name: (
                    quant_kind_for(
                        name, target_specs[name], compute_specs[name]
                    )
                    if hetero
                    else COPY_ONLY
                )
                for name in layer_to_group
            },
            compute_specs=compute_specs,
            wire_specs=target_specs,
            layer_to_group=layer_to_group,
            group_layers=tuple(group_layers),
            decode_block_size=hetero_cfg.decode_block_size,
        )

    def layer_views(
        self, backing: torch.Tensor, num_blocks: int
    ) -> dict[str, torch.Tensor]:
        """D-layout per-layer views into one packed backing tensor."""
        return {
            layer_name: torch.as_strided(
                backing,
                size=(num_blocks, self.page_bytes[layer_name]),
                stride=(self.block_stride, 1),
                storage_offset=self.byte_offset[layer_name],
            )
            for layer_name in self.byte_offset
        }
