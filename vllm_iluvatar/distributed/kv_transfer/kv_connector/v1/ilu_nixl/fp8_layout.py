# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-side planning for the DeepSeek-V4 packed FP8 wire cache."""

from dataclasses import replace

import torch
from vllm.config import VllmConfig
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

from ..ilu_nixl_common.dsv41 import is_dsv41_config

_ILU_WIRE_CONNECTORS = ("IluNixlConnector", "IluNixlPullConnector")
FP8_WIRE_SCRATCH_BUFFER_COUNT = 2
_COMPRESSED_MAIN_FORMATS = frozenset({"i8g64", "i4g32"})
_SPLIT_INDEXER_FORMAT = "int8_indexer_split"
_MISSING = object()


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _child_connectors(extra: dict) -> list[dict]:
    connectors = extra.get("connectors", ())
    if not isinstance(connectors, (list, tuple)):
        return []
    return [child for child in connectors if isinstance(child, dict)]


def _walk_producer_wire_extras(kv_transfer_config):
    """Yield ``(extra_dict, get_from_extra_config)`` for producer IluNixl endpoints.

    MultiConnector children keep their own extra_config. Planning still sees the
    outer MultiConnector, so the staging gate has to walk those children.
    """
    if kv_transfer_config is None:
        return

    def emit(name, role, extra, get_extra):
        if name in _ILU_WIRE_CONNECTORS and str(role).lower() == "kv_producer":
            yield extra, get_extra

    name = getattr(kv_transfer_config, "kv_connector", None)
    role = getattr(kv_transfer_config, "kv_role", "")
    extra = _mapping(getattr(kv_transfer_config, "kv_connector_extra_config", None))
    get_extra = getattr(kv_transfer_config, "get_from_extra_config", None)
    yield from emit(name, role, extra, get_extra)
    if name != "MultiConnector":
        return

    def visit(children, inherited_role):
        for child in children:
            child_name = child.get("kv_connector")
            child_role = child.get("kv_role", inherited_role)
            child_extra = _mapping(child.get("kv_connector_extra_config"))
            yield from emit(child_name, child_role, child_extra, None)
            if child_name == "MultiConnector":
                yield from visit(_child_connectors(child_extra), child_role)

    yield from visit(_child_connectors(extra), role)


def _wire_extra_get(kv_transfer_config, key, default):
    """Read a wire-staging key from the IluNixl producer, including Multi children."""
    if kv_transfer_config is None:
        return default
    for extra, get_extra in _walk_producer_wire_extras(kv_transfer_config):
        if get_extra is not None:
            value = get_extra(key, _MISSING)
            if value is not _MISSING:
                return value
        elif key in extra:
            return extra[key]
    parent_extra = _mapping(
        getattr(kv_transfer_config, "kv_connector_extra_config", None)
    )
    if getattr(kv_transfer_config, "kv_connector", None) == "MultiConnector":
        get_extra = getattr(kv_transfer_config, "get_from_extra_config", None)
        if get_extra is not None:
            value = get_extra(key, _MISSING)
            if value is not _MISSING:
                return value
        if key in parent_extra:
            return parent_extra[key]
    return default


def use_fp8_ds_mla_wire_staging(vllm_config: VllmConfig) -> bool:
    """Whether this Ilu producer publishes a D-compatible FP8 staging cache."""
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        return False
    if not any(True for _ in _walk_producer_wire_extras(kv_transfer_config)):
        return False
    enabled = _wire_extra_get(
        kv_transfer_config, "enable_fp8_ds_mla_wire_staging", False
    )
    return str(enabled).lower() == "true"


def page_states(spec: KVCacheSpec) -> int:
    """Rows (stored states) in one physical page of ``spec``.

    v0.31 ``storage_block_size`` is an optional token width that defaults to
    None and does not exist on ``SlidingWindowMLASpec``; without it a page is
    one ``block_size`` block.
    """
    storage_block_size = getattr(spec, "storage_block_size", None)
    if storage_block_size is None:
        return int(spec.num_states)
    return int(spec.get_num_kernel_states(storage_block_size))


def get_fp8_wire_scratch_blocks(vllm_config: VllmConfig) -> int:
    """Return and validate blocks per GPU wire scratch buffer."""
    kv_transfer_config = vllm_config.kv_transfer_config
    raw_value = _wire_extra_get(kv_transfer_config, "fp8_wire_scratch_blocks", 4)
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


def get_fp8_wire_block_size(vllm_config: VllmConfig) -> int | None:
    """Return optional target wire block size (tokens per D block) if configured."""
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    raw_value = _wire_extra_get(kv_transfer_config, "fp8_wire_block_size", None)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise ValueError("fp8_wire_block_size must be a positive integer")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"fp8_wire_block_size must be a positive integer, got {raw_value!r}"
        ) from exc
    if value <= 0 or str(raw_value).strip() != str(value):
        raise ValueError(
            f"fp8_wire_block_size must be a positive integer, got {raw_value!r}"
        )
    return value


def get_fp8_wire_alignment(vllm_config: VllmConfig) -> int | None:
    """Return optional wire page byte alignment override if configured."""
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    raw_value = _wire_extra_get(kv_transfer_config, "fp8_wire_alignment", None)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise ValueError("fp8_wire_alignment must be a positive integer")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"fp8_wire_alignment must be a positive integer, got {raw_value!r}"
        ) from exc
    if value <= 0 or str(raw_value).strip() != str(value):
        raise ValueError(
            f"fp8_wire_alignment must be a positive integer, got {raw_value!r}"
        )
    return value


def _group_layer_spec(group: KVCacheGroupSpec, layer_name: str) -> KVCacheSpec:
    group_spec = group.kv_cache_spec
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        return group_spec.kv_cache_specs[layer_name]
    return group_spec


def wire_sub_blocks_per_compute_block(
    compute_spec: KVCacheSpec,
    wire_spec: KVCacheSpec,
) -> int:
    """Return how many wire blocks correspond to one compute block."""
    c_bs = int(getattr(compute_spec, "block_size", 0) or 0)
    w_bs = int(getattr(wire_spec, "block_size", 0) or 0)
    if c_bs <= 0 or w_bs <= 0 or c_bs == w_bs:
        return 1
    if c_bs < w_bs or c_bs % w_bs != 0:
        raise ValueError(
            "Compute block size must be a positive multiple of wire block size: "
            f"compute={c_bs}, wire={w_bs}"
        )
    return c_bs // w_bs


def _v41_fp8_wire_spec(
    spec: MLAAttentionSpec | SlidingWindowMLASpec,
    *,
    alignment: int,
    block_size: int | None = None,
) -> KVCacheSpec:
    """D-facing DeepSeek-V4.1 packed FP8 MLA page (584 B content).

    ``model_version`` is vLLM's packed-MLA *page dialect*, not HF
    ``model_type``. Community ``SlidingWindowMLASpec`` only accepts
    ``None`` / ``deepseek_v4``; the 584 B ``fp8_ds_mla`` record is that
    V4 dialect (V4.1 compute fp8 pages already tag ``deepseek_v4``).
    V4 vs V4.1 wire differences are ``alignment`` (576 vs 64 on main)
    and the caller branch, not a new ``model_version`` string.
    """
    kwargs = dict(
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
        state_content_bytes=584,
        alignment=alignment,
        page_size_padded=None,
    )
    if block_size is not None:
        kwargs["block_size"] = block_size
    # Compressed compute pages pin storage_block_size; wire pages follow
    # tokens_per_state / num_states like native D fp8_ds_mla.
    if getattr(spec, "storage_block_size", None) is not None:
        kwargs["storage_block_size"] = None
    return replace(spec, **kwargs)


def _resolve_target_wire_block_size(
    spec: MLAAttentionSpec | SlidingWindowMLASpec,
    layer_name: str | None,
    wire_block_size: int | None,
) -> int | None:
    if wire_block_size is None:
        return None
    compute_bs = int(getattr(spec, "block_size", 0) or 0)
    tps = max(1, int(getattr(spec, "tokens_per_state", 1) or 1))
    if compute_bs < wire_block_size or compute_bs % wire_block_size != 0:
        raise ValueError(
            f"Layer {layer_name}: compute block_size={compute_bs} is not divisible "
            f"by fp8_wire_block_size={wire_block_size}"
        )
    if wire_block_size % tps != 0:
        raise ValueError(
            f"Layer {layer_name}: fp8_wire_block_size={wire_block_size} is not "
            f"divisible by tokens_per_state={tps}"
        )
    return wire_block_size


def _to_fp8_ds_mla_wire_spec(
    spec: KVCacheSpec,
    layer_name: str | None = None,
    *,
    is_dsv41: bool = False,
    wire_block_size: int | None = None,
    wire_alignment: int | None = None,
) -> KVCacheSpec:
    """Return the D-side wire-layout counterpart of a producer cache spec."""
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return replace(
            spec,
            kv_cache_specs={
                name: _to_fp8_ds_mla_wire_spec(
                    layer_spec,
                    name,
                    is_dsv41=is_dsv41,
                    wire_block_size=wire_block_size,
                    wire_alignment=wire_alignment,
                )
                for name, layer_spec in spec.kv_cache_specs.items()
            },
        )
    if isinstance(spec, CircularBufferSpec):
        # Bounded-replay skips compressor rings from transfer; keep geometry.
        return spec
    if isinstance(spec, MLAAttentionSpec):
        target_bs = _resolve_target_wire_block_size(spec, layer_name, wire_block_size)
        is_indexer_cache = layer_name is not None and ".indexer." in layer_name
        if is_indexer_cache:
            wire_head_size = (
                132
                if spec.head_size in (128, 132)
                or spec.cache_dtype_str == _SPLIT_INDEXER_FORMAT
                else (
                    spec.head_size + 4
                    if spec.dtype == torch.bfloat16
                    else spec.head_size
                )
            )
            default_idx_align = (
                64
                if (
                    target_bs is not None
                    and (is_dsv41 or spec.cache_dtype_str == _SPLIT_INDEXER_FORMAT)
                )
                else 576
            )
            idx_kwargs = dict(
                head_size=wire_head_size,
                dtype=torch.uint8,
                cache_dtype_str=None,
                model_version=None,
                alignment=wire_alignment or default_idx_align,
                page_size_padded=None,
            )
            if target_bs is not None:
                idx_kwargs["block_size"] = target_bs
            if getattr(spec, "state_content_bytes", None) is not None:
                idx_kwargs["state_content_bytes"] = None
            if getattr(spec, "storage_block_size", None) is not None:
                idx_kwargs["storage_block_size"] = None
            return replace(spec, **idx_kwargs)

        compressed = spec.cache_dtype_str in _COMPRESSED_MAIN_FORMATS
        is_v41_main = (
            is_dsv41
            or spec.model_version == "deepseek_v41"
            or compressed
            or (
                getattr(spec, "state_content_bytes", None) == 584
                and spec.model_version != "deepseek_v4"
            )
        )
        if is_v41_main:
            return _v41_fp8_wire_spec(
                spec,
                alignment=wire_alignment or 64,
                block_size=target_bs,
            )

        is_main_ds_v4_cache = spec.model_version == "deepseek_v4" or (
            spec.head_size == 512
            and spec.cache_dtype_str in ("auto", "bfloat16", "fp8_ds_mla")
        )
        if is_main_ds_v4_cache:
            kwargs = dict(
                dtype=torch.uint8,
                cache_dtype_str="fp8_ds_mla",
                model_version="deepseek_v4",
                alignment=wire_alignment or 576,
                page_size_padded=None,
            )
            if target_bs is not None:
                kwargs["block_size"] = target_bs
            return replace(spec, **kwargs)
        raise ValueError(
            "Unsupported MLA cache for FP8 DS MLA wire staging: "
            f"layer={layer_name}, spec={spec}"
        )

    if isinstance(spec, SlidingWindowMLASpec):
        is_state_cache = layer_name is not None and ".state_cache" in layer_name
        target_bs = (
            _resolve_target_wire_block_size(spec, layer_name, wire_block_size)
            if not is_state_cache
            else None
        )
        is_v41_swa = (
            is_dsv41
            or spec.model_version == "deepseek_v41"
            or (
                getattr(spec, "state_content_bytes", None) == 584
                and spec.model_version != "deepseek_v4"
                and not is_state_cache
            )
        )
        if is_v41_swa and not is_state_cache:
            # SWA pages keep the wider 576 alignment used by NV D layouts.
            return _v41_fp8_wire_spec(
                spec,
                alignment=wire_alignment or getattr(spec, "alignment", 576) or 576,
                block_size=target_bs,
            )

        is_main_ds_v4_cache = spec.model_version == "deepseek_v4" or (
            spec.head_size == 512
            and spec.cache_dtype_str in ("auto", "bfloat16", "fp8_ds_mla")
            and not is_state_cache
        )
        if is_main_ds_v4_cache:
            kwargs = dict(
                dtype=torch.uint8,
                cache_dtype_str="fp8_ds_mla",
                model_version="deepseek_v4",
                alignment=wire_alignment or 576,
                page_size_padded=None,
            )
            if target_bs is not None:
                kwargs["block_size"] = target_bs
            return replace(spec, **kwargs)

        if spec.sliding_window == 8:
            block_size = 4
            align = 576
        elif spec.sliding_window == 128:
            block_size = 8
            align = 576
        elif (
            getattr(spec, "sliding_window", 0)
            and getattr(spec, "sliding_window", 0) > 0
        ):
            tps = max(1, int(getattr(spec, "tokens_per_state", 1) or 1))
            block_size = max(
                1,
                min(
                    int(spec.block_size),
                    int(spec.sliding_window) // (2 * tps) or 1,
                ),
            )
            align = getattr(spec, "alignment", 576) or 576
        else:
            raise ValueError(
                "Unsupported compressor-state sliding window for FP8 DS MLA "
                f"wire staging: layer={layer_name}, sliding_window="
                f"{spec.sliding_window}"
            )
        return replace(
            spec,
            block_size=block_size,
            alignment=align,
            page_size_padded=None,
        )

    raise ValueError(
        "FP8 DS MLA wire staging only supports MLA, indexer, compressor "
        f"state, and circular-buffer cache specs, got {type(spec).__name__}"
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
    projected groups) and only rewrite specs.
    """
    is_dsv41 = is_dsv41_config(vllm_config)
    wire_block_size = get_fp8_wire_block_size(vllm_config)
    wire_alignment = get_fp8_wire_alignment(vllm_config)

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
            name: _to_fp8_ds_mla_wire_spec(
                flattened[name],
                name,
                is_dsv41=is_dsv41,
                wire_block_size=wire_block_size,
                wire_alignment=wire_alignment,
            )
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


def packed_bytes_per_block(
    kv_cache_groups: list[KVCacheGroupSpec],
    compute_groups: list[KVCacheGroupSpec] | None = None,
) -> int:
    """Return the per-compute-block scratch/pool stride across groups."""
    if compute_groups is None:
        group_sizes = [
            sum(
                _group_layer_spec(group, layer_name).page_size_bytes
                for layer_name in group.layer_names
            )
            for group in kv_cache_groups
            if group.layer_names
        ]
    else:
        group_sizes = [
            sum(
                wire_sub_blocks_per_compute_block(
                    _group_layer_spec(compute_group, layer_name),
                    _group_layer_spec(wire_group, layer_name),
                )
                * _group_layer_spec(wire_group, layer_name).page_size_bytes
                for layer_name in wire_group.layer_names
            )
            for compute_group, wire_group in zip(compute_groups, kv_cache_groups)
            if wire_group.layer_names
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
    has_sub_blocks = any(
        wire_sub_blocks_per_compute_block(
            _group_layer_spec(compute_group, layer_name),
            _group_layer_spec(wire_group, layer_name),
        )
        > 1
        for compute_group, wire_group in zip(kv_cache_groups, wire_groups)
        for layer_name in wire_group.layer_names
    )
    if has_sub_blocks:
        offset = 0
        raw_tensors: list[KVCacheTensor] = []
        for compute_group, wire_group in zip(kv_cache_groups, wire_groups):
            for layer_name in wire_group.layer_names:
                c_spec = _group_layer_spec(compute_group, layer_name)
                w_spec = _group_layer_spec(wire_group, layer_name)
                sub_blocks = wire_sub_blocks_per_compute_block(c_spec, w_spec)
                page_bytes = int(w_spec.page_size_bytes)
                pool_bytes = num_blocks * sub_blocks * page_bytes
                raw_tensors.append(
                    KVCacheTensor(
                        size=0,
                        layers=[layer_name],
                        layer_stride=pool_bytes,
                        block_stride=page_bytes,
                        offset=offset,
                    )
                )
                offset += pool_bytes
        total_size = offset
        tensors = [replace(t, size=total_size) for t in raw_tensors]
        _assert_no_aliased_wire_slots(kv_cache_groups, tensors)
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=tensors,
            kv_cache_groups=wire_groups,
        )

    block_stride = packed_bytes_per_block(wire_groups)
    total_size = block_stride * num_blocks
    tensors = []
    for group in wire_groups:
        offset = 0
        for layer_name in group.layer_names:
            spec = _group_layer_spec(group, layer_name)
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
