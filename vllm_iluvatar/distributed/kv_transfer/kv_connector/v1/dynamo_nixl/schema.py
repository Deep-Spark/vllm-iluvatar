# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic KV-cache schema metadata for Dynamo NIXL handshakes."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec
from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind


class KVCacheStorageView(msgspec.Struct, frozen=True):
    """One planned or runtime storage view used by a cache payload."""

    shared_by: tuple[str, ...]
    storage_base_ptr: int | None
    storage_offset: int
    stride: int
    identity: str


class KVCacheSchemaEntry(msgspec.Struct, frozen=True):
    """Semantic cache entry within a scheduler group."""

    cache_family: str
    cache_role: str
    layer_names: tuple[str, ...]
    logical_block_size: int
    storage_block_size: int
    compress_ratio: int
    page_bytes: int
    stride: int
    storage_offset: int
    storage_base_ptr: int | None
    shared_by: tuple[str, ...]
    sliding_window: int | None
    shared_storage_identity: str
    dtype: str | None
    external: bool
    reusable: bool
    storage_views: tuple[KVCacheStorageView, ...] = ()


class KVCacheGroupSchema(msgspec.Struct, frozen=True):
    """Wire contract for one scheduler KV-cache group."""

    group_id: str
    group_idx: int
    kind: str
    block_size: int
    cache_family: str
    cache_role: str
    layer_names: tuple[str, ...]
    logical_block_size: int
    storage_block_size: int
    compress_ratio: int
    page_bytes: int
    stride: int
    storage_offset: int
    storage_base_ptr: int | None
    shared_by: tuple[str, ...]
    sliding_window: int | None
    shared_storage_identity: str
    dtype: str | None
    external: bool
    reusable: bool
    entries: tuple[KVCacheSchemaEntry, ...]
    storage_views: tuple[KVCacheStorageView, ...] = ()


class HmaCacheSchema(msgspec.Struct, frozen=True):
    """Versioned, canonicalizable schema carried in the NIXL handshake."""

    groups: tuple[KVCacheGroupSchema, ...]
    version: int = 1
    model_revision: str | None = None
    model_dtype: str | None = None
    kv_cache_layout: str | None = None
    tp_size: int | None = None
    pp_size: int | None = None
    pp_rank: int | None = None


def _model_config_field(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _identity_component(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_model_id(model_id: Any) -> str:
    normalized = _identity_component(model_id)
    if normalized is None:
        return "<unknown-model>"
    model_path = Path(normalized).expanduser()
    if (
        model_path.is_absolute()
        or normalized.startswith((".", "~"))
        or model_path.exists()
    ):
        return str(model_path.resolve(strict=False))
    return normalized


def resolve_model_identity(model_config: Any) -> str:
    """Return canonical JSON identifying model source and resolved revision.

    The stable format has ``identity_version``, normalized ``model_id``,
    requested ``revision``, resolved HF ``commit_hash``, and ``resolution``.
    Only ``resolution="resolved_commit"`` identifies immutable model content;
    ``"unresolved"`` explicitly tells external caches to fail closed.
    """
    hf_config = _model_config_field(model_config, "hf_config")
    model_id = _model_config_field(model_config, "model")
    if model_id is None:
        model_id = _model_config_field(hf_config, "_name_or_path")
    commit_hash = _identity_component(_model_config_field(hf_config, "_commit_hash"))
    if not commit_hash:
        # Local checkpoints carry no HF commit hash. An operator who vouches
        # for the checkpoint's immutability may declare the identity
        # explicitly; without it external caches keep failing closed.
        override = os.environ.get("VLLM_ILU_HMA_MODEL_COMMIT_HASH", "").strip()
        if override:
            commit_hash = _identity_component(override)
    identity = {
        "commit_hash": commit_hash,
        "identity_version": 1,
        "model_id": _normalize_model_id(model_id),
        "resolution": "resolved_commit" if commit_hash else "unresolved",
        "revision": _identity_component(_model_config_field(model_config, "revision")),
    }
    return msgspec.json.encode(identity).decode()


def _classify_role(
    layer_names: Iterable[str],
    compress_ratio: int,
    sliding_window: int | None = None,
    *,
    glm_dsa: bool = False,
) -> tuple[str, str, bool, bool]:
    names = tuple(name.lower() for name in layer_names)
    if not names:
        return "unknown", "unknown", False, False

    classifications: set[tuple[str, str, bool, bool]] = set()
    for name in names:
        family = (
            "CSA" if compress_ratio == 4 else "HCA" if compress_ratio == 128 else ""
        )
        if ".compressor.state_cache" in name or ".indexer.compressor" in name:
            # CompressorStateCache keeps no ratio on its spec (setting one
            # would corrupt its storage geometry), but its window is
            # coff * ratio — 8 for C4, 128 for C128 — which identifies the
            # family safely. Treat it as an externalizable junction payload
            # (sliding_window != None), same contract as SWA.
            compressor_family = {8: "CSA", 128: "HCA"}.get(sliding_window)
            classifications.add(
                (
                    compressor_family or family or "unknown",
                    "compressor_state",
                    True,
                    True,
                )
            )
        elif ".swa_cache" in name:
            classifications.add(("SWA", "SWA", True, True))
        elif ".indexer.k_cache" in name and (
            family == "CSA" or (not family and glm_dsa)
        ):
            # GLM has no ratio; DeepSeek-V3.2 shares this name, so gate fallback.
            classifications.add(("CSA", "K_IComp", True, True))
        elif family and (
            ".self_attn" in name
            or name.endswith(".attn")
            or ".ilu_deepseek_v4_multi_head_latent_attention" in name
        ):
            classifications.add((family, "K_Comp", True, True))
        elif (
            glm_dsa and not family and ".self_attn" in name and ".indexer." not in name
        ):
            # GLM-5.2 MLA: ``model.layers.N.self_attn.attn``, tokens_per_state=1.
            # Reuse CSA/K_Comp so KVBM's existing whitelist and canonical
            # block-size picker apply without a runtime fork.
            classifications.add(("CSA", "K_Comp", True, True))
        else:
            classifications.add(("unknown", "unknown", False, False))

    if len(classifications) != 1:
        return "unknown", "unknown", False, False
    return classifications.pop()


def _storage_views(
    layer_names: tuple[str, ...],
    tensors: Iterable[Any],
    kv_caches: Mapping[str, Any] | None,
    page_bytes_by_layer: Mapping[str, int],
) -> tuple[tuple[KVCacheStorageView, ...], bool]:
    """Resolve every allocator view intersecting ``layer_names``.

    A packed scheduler group can contain several allocator records with
    different offsets, strides, and backing storages. At runtime, layers from
    one planned ``shared_by`` record may use independent host transfer buffers;
    equal geometries are aggregated while distinct geometries remain separate.
    """
    group_layers = set(layer_names)
    views: dict[
        tuple[tuple[str, ...], int | None, int, int],
        KVCacheStorageView,
    ] = {}
    storage_known = True
    matched = False
    for tensor in tensors:
        layers = tuple(sorted(getattr(tensor, "layers", None) or ()))
        if not group_layers.intersection(layers):
            continue
        matched = True
        if kv_caches is None:
            page_sizes = {
                page_bytes_by_layer[name]
                for name in layers
                if name in page_bytes_by_layer
            }
            configured_stride = int(getattr(tensor, "block_stride", 0) or 0)
            if configured_stride > 0:
                stride = configured_stride
            elif len(page_sizes) == 1:
                stride = page_sizes.pop()
            else:
                storage_known = False
                continue
            offset = int(getattr(tensor, "offset", 0) or 0)
            resolved_views = ((layers, None, offset, stride),)
        else:
            associated_layers = tuple(
                layer_name for layer_name in layers if layer_name in group_layers
            )
            runtime_views = _runtime_storage_views(
                associated_layers,
                kv_caches,
            )
            if runtime_views is None:
                storage_known = False
                continue
            resolved_views = runtime_views
        for associated_layers, base_ptr, offset, stride in resolved_views:
            ptr_identity = "unknown" if base_ptr is None else str(base_ptr)
            identity = (
                f"{'|'.join(associated_layers)}|"
                f"storage={ptr_identity}@{offset}:{stride}"
            )
            key = (associated_layers, base_ptr, offset, stride)
            views[key] = KVCacheStorageView(
                shared_by=associated_layers,
                storage_base_ptr=base_ptr,
                storage_offset=offset,
                stride=stride,
                identity=identity,
            )

    ordered = tuple(
        views[key]
        for key in sorted(
            views,
            key=lambda item: (
                item[0],
                -1 if item[1] is None else item[1],
                item[2],
                item[3],
            ),
        )
    )
    return ordered, matched and storage_known and bool(ordered)


def _scalar_storage_metadata(
    views: tuple[KVCacheStorageView, ...],
) -> tuple[str, int, int, int | None, tuple[str, ...]]:
    """Project views onto the scalar handshake fields without losing ambiguity."""
    shared_by = tuple(
        sorted({layer_name for view in views for layer_name in view.shared_by})
    )
    if len(views) == 1:
        view = views[0]
        return (
            view.identity,
            view.stride,
            view.storage_offset,
            view.storage_base_ptr,
            shared_by,
        )
    if len(views) > 1:
        return "multiple", 0, -1, None, shared_by
    return "unknown", 0, -1, None, ()


def _merge_storage_views(
    entries: Iterable[KVCacheSchemaEntry],
) -> tuple[KVCacheStorageView, ...]:
    views: dict[
        tuple[tuple[str, ...], int | None, int, int],
        KVCacheStorageView,
    ] = {}
    for entry in entries:
        for view in entry.storage_views:
            key = (
                view.shared_by,
                view.storage_base_ptr,
                view.storage_offset,
                view.stride,
            )
            views[key] = view
    return tuple(
        views[key]
        for key in sorted(
            views,
            key=lambda item: (
                item[0],
                -1 if item[1] is None else item[1],
                item[2],
                item[3],
            ),
        )
    )


def _runtime_storage_views(
    shared_by: tuple[str, ...], kv_caches: Mapping[str, Any]
) -> tuple[tuple[tuple[str, ...], int, int, int], ...] | None:
    """Resolve actual tensors and aggregate layers with equal geometry.

    Runtime KV tensors are block-major in the connector contract: axis 0 is
    the block axis, including DSV4 tensors shaped
    ``[num_blocks, storage_block_size, head_dim]``. A layer can expose multiple
    tensors. Every advertised layer must resolve completely, but layers are
    allowed to use independent host allocations.
    """
    layers_by_geometry: dict[tuple[int, int, int], set[str]] = {}
    for layer_name in shared_by:
        cache = kv_caches.get(layer_name)
        if cache is None:
            return None
        caches = cache if isinstance(cache, (list, tuple)) else (cache,)
        geometries: set[tuple[int, int, int]] = set()
        for tensor in caches:
            geometry = _tensor_storage_geometry(tensor)
            if geometry is None:
                return None
            geometries.add(geometry)
        if not geometries:
            return None
        for geometry in geometries:
            layers_by_geometry.setdefault(geometry, set()).add(layer_name)
    if not layers_by_geometry:
        return None
    return tuple(
        (tuple(sorted(layer_names)), base_ptr, offset, stride)
        for (base_ptr, offset, stride), layer_names in sorted(
            layers_by_geometry.items()
        )
    )


def _tensor_storage_geometry(tensor: Any) -> tuple[int, int, int] | None:
    """Derive block-major runtime geometry from a torch-like tensor."""
    try:
        base_ptr = int(tensor.untyped_storage().data_ptr())
        view_ptr = int(tensor.data_ptr())
        offset = view_ptr - base_ptr
        stride = int(tensor.stride(0)) * int(tensor.element_size())
    except (AttributeError, TypeError, RuntimeError):
        return None
    if offset < 0 or stride <= 0:
        return None
    return base_ptr, offset, stride


def _spec_kind(spec: Any) -> str:
    try:
        return get_kv_cache_spec_kind(spec).value
    except (AttributeError, TypeError):
        return "unknown"


def _dtype_name(spec: Any) -> str | None:
    dtype = getattr(spec, "dtype", None)
    return None if dtype is None else str(dtype)


def _spec_compress_ratio(spec: Any) -> int:
    # 0.31 carries the ratio on tokens_per_state. The KVCacheSpec base
    # property raises NotImplementedError; ignore that, Fraction, and
    # non-positive sentinels (Mamba uses -1).
    try:
        value = spec.tokens_per_state
    except (AttributeError, NotImplementedError, TypeError):
        return 1
    if isinstance(value, bool):
        return 1
    if isinstance(value, int) and value > 0:
        return value
    return 1


def _spec_storage_block_size(
    spec: Any, logical_block_size: int, compress_ratio: int
) -> int:
    # 0.31 MLAAttentionSpec.storage_block_size is optional; derive when absent.
    value = getattr(spec, "storage_block_size", None)
    if value is not None:
        return int(value)
    if compress_ratio > 1:
        return logical_block_size // compress_ratio
    return logical_block_size


def _entry_group_signature(entry: KVCacheSchemaEntry) -> tuple[Any, ...]:
    return (
        entry.cache_family,
        entry.cache_role,
        entry.logical_block_size,
        entry.storage_block_size,
        entry.compress_ratio,
        entry.page_bytes,
        entry.sliding_window,
        entry.dtype,
        entry.external,
        entry.reusable,
    )


def _entry_requires_payload(entry: KVCacheSchemaEntry) -> bool:
    return (entry.cache_family, entry.cache_role) in {
        ("CSA", "K_IComp"),
        ("CSA", "K_Comp"),
        ("CSA", "compressor_state"),
        ("HCA", "K_Comp"),
        ("HCA", "compressor_state"),
        ("SWA", "SWA"),
    }


def _group_is_junction(group: KVCacheGroupSchema) -> bool:
    """Return whether the group retains a sliding window instead of a prefix."""
    return any(
        _entry_requires_payload(entry) and entry.sliding_window is not None
        for entry in group.entries
    )


def _build_entry(
    layer_names: tuple[str, ...],
    spec: Any,
    tensors: Iterable[Any],
    kv_caches: Mapping[str, Any] | None,
    glm_dsa: bool = False,
) -> KVCacheSchemaEntry:
    logical_block_size = int(spec.block_size)
    compress_ratio = _spec_compress_ratio(spec)
    storage_block_size = _spec_storage_block_size(
        spec, logical_block_size, compress_ratio
    )
    family, role, external, reusable = _classify_role(
        layer_names,
        compress_ratio,
        getattr(spec, "sliding_window", None),
        glm_dsa=glm_dsa,
    )
    storage_views, _storage_known = _storage_views(
        layer_names,
        tensors,
        kv_caches,
        {layer_name: int(spec.page_size_bytes) for layer_name in layer_names},
    )
    (
        storage_identity,
        stride,
        storage_offset,
        storage_base_ptr,
        shared_by,
    ) = _scalar_storage_metadata(storage_views)
    # NOTE: external/reusable describe the *contract* (may this cache be
    # externalized), not runtime storage availability. The scheduler side
    # builds the schema without tensors and both sides must agree on the
    # hash, so tensor-derived storage knowledge must not flip these flags;
    # registration validates the actual views separately and fails closed.
    return KVCacheSchemaEntry(
        cache_family=family,
        cache_role=role,
        layer_names=layer_names,
        logical_block_size=logical_block_size,
        storage_block_size=storage_block_size,
        compress_ratio=compress_ratio,
        page_bytes=int(spec.page_size_bytes),
        stride=stride,
        storage_offset=storage_offset,
        storage_base_ptr=storage_base_ptr,
        shared_by=shared_by,
        storage_views=storage_views,
        sliding_window=getattr(spec, "sliding_window", None),
        shared_storage_identity=storage_identity,
        dtype=_dtype_name(spec),
        external=external,
        reusable=reusable,
    )


def _build_entries_from_shared_spec(
    layer_names: tuple[str, ...],
    spec: Any,
    tensors: Iterable[Any],
    kv_caches: Mapping[str, Any] | None,
    glm_dsa: bool = False,
) -> tuple[KVCacheSchemaEntry, ...]:
    """Split a single-spec group into per-role entries by layer name.

    A vLLM cache group whose layers share one KV cache spec may still carry
    several payload roles — DSV4's CSA group holds both the compressed
    attention cache (``.attn``) and the indexer cache (``.attn.indexer.k_cache``)
    under one MLA spec. Classifying the whole group at once collapses to
    "unknown", so bucket the layer names by their individual classification
    and build one entry per role; the shared spec supplies the geometry,
    which vLLM guarantees is uniform within the group.
    """
    buckets: dict[tuple[str, str], list[str]] = {}
    compress_ratio = _spec_compress_ratio(spec)
    for name in layer_names:
        family, role, _, _ = _classify_role(
            (name,),
            compress_ratio,
            getattr(spec, "sliding_window", None),
            glm_dsa=glm_dsa,
        )
        buckets.setdefault((family, role), []).append(name)
    return tuple(
        _build_entry(tuple(sorted(names)), spec, tensors, kv_caches, glm_dsa=glm_dsa)
        for (_, _), names in sorted(buckets.items())
    )


def identity_kv_cache_config(kv_cache_config: Any, *, pp_size: int | None) -> Any:
    """View whose ``kv_cache_groups`` are the unprojected global set.

    Pipeline-parallel projection empties ``layer_names`` on stages that do
    not own those layers. HMA schema identity (EngineCore metadata, handshake
    hash) must stay the full-model group set. Local registration still reads
    the original ``kv_cache_config.kv_cache_groups``.
    """
    groups = getattr(kv_cache_config, "global_kv_cache_groups", None)
    if groups:
        return SimpleNamespace(
            kv_cache_groups=groups,
            kv_cache_tensors=getattr(kv_cache_config, "kv_cache_tensors", ()) or (),
            num_blocks=getattr(kv_cache_config, "num_blocks", 0) or 0,
        )
    resolved_pp = (
        pp_size
        if isinstance(pp_size, int) and not isinstance(pp_size, bool) and pp_size >= 1
        else 1
    )
    if resolved_pp > 1:
        raise RuntimeError(
            "HMA pipeline-parallel serving requires global KV cache groups "
            "on KVCacheConfig."
        )
    return kv_cache_config


def build_kv_cache_schema(
    kv_cache_config: Any,
    *,
    kv_caches: Mapping[str, Any] | None = None,
    model_revision: str | None = None,
    kv_cache_layout: str | None = None,
    tp_size: int | None = None,
    pp_size: int | None = None,
    pp_rank: int | None = None,
    model_dtype: str | None = None,
    model_type: str | None = None,
) -> HmaCacheSchema:
    """Build semantic HMA cache metadata plus optional runtime storage identity."""
    tensors = getattr(kv_cache_config, "kv_cache_tensors", ()) or ()
    glm_dsa = isinstance(model_type, str) and model_type.lower() == "glm_moe_dsa"
    groups: list[KVCacheGroupSchema] = []
    for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        layer_names = tuple(sorted(group.layer_names))
        per_layer_specs = getattr(spec, "kv_cache_specs", None)
        if per_layer_specs is None:
            entries = _build_entries_from_shared_spec(
                layer_names, spec, tensors, kv_caches, glm_dsa=glm_dsa
            )
        else:
            entries = tuple(
                _build_entry(
                    (layer_name,),
                    per_layer_specs[layer_name],
                    tensors,
                    kv_caches,
                    glm_dsa=glm_dsa,
                )
                for layer_name in layer_names
            )

        required_entries = tuple(
            entry for entry in entries if _entry_requires_payload(entry)
        )
        unsupported_entries = tuple(
            entry for entry in entries if not _entry_requires_payload(entry)
        )
        storage_views = _merge_storage_views(entries)
        (
            storage_identity,
            stride,
            storage_offset,
            storage_base_ptr,
            shared_by,
        ) = _scalar_storage_metadata(storage_views)
        if len({_entry_group_signature(entry) for entry in entries}) == 1:
            entry = entries[0]
            family = entry.cache_family
            role = entry.cache_role
            logical_block_size = entry.logical_block_size
            storage_block_size = entry.storage_block_size
            compress_ratio = entry.compress_ratio
            page_bytes = entry.page_bytes
            sliding_window = entry.sliding_window
            dtype = entry.dtype
        else:
            family = "mixed"
            role = "mixed"
            logical_block_size = int(spec.block_size)
            compress_ratio = 1
            storage_block_size = _spec_storage_block_size(
                spec, logical_block_size, compress_ratio
            )
            page_bytes = int(spec.page_size_bytes)
            sliding_window = getattr(spec, "sliding_window", None)
            dtype = None
        external = (
            bool(required_entries)
            and not unsupported_entries
            and all(entry.external for entry in required_entries)
        )
        reusable = (
            bool(required_entries)
            and not unsupported_entries
            and all(entry.reusable for entry in required_entries)
        )

        group_fields = {
            "kind": _spec_kind(spec),
            "cache_family": family,
            "cache_role": role,
            "layer_names": layer_names,
            "logical_block_size": logical_block_size,
            "storage_block_size": storage_block_size,
            "compress_ratio": compress_ratio,
            "page_bytes": page_bytes,
            "stride": stride,
            "storage_offset": storage_offset,
            "storage_base_ptr": storage_base_ptr,
            "shared_by": shared_by,
            "storage_views": storage_views,
            "sliding_window": sliding_window,
            "shared_storage_identity": storage_identity,
            "dtype": dtype,
            "external": external,
            "reusable": reusable,
            "entries": entries,
        }
        group_id = hashlib.sha256(
            msgspec.json.encode(_group_semantic_fields(group_fields))
        ).hexdigest()
        groups.append(
            KVCacheGroupSchema(
                group_id=group_id,
                group_idx=group_idx,
                block_size=logical_block_size,
                **group_fields,
            )
        )

    return HmaCacheSchema(
        groups=tuple(groups),
        model_revision=model_revision,
        model_dtype=model_dtype,
        kv_cache_layout=kv_cache_layout,
        tp_size=tp_size,
        pp_size=pp_size,
        pp_rank=pp_rank,
    )


def _entry_semantic_fields(entry: KVCacheSchemaEntry) -> dict[str, Any]:
    return {
        "cache_family": entry.cache_family,
        "cache_role": entry.cache_role,
        "logical_block_size": entry.logical_block_size,
        "storage_block_size": entry.storage_block_size,
        "compress_ratio": entry.compress_ratio,
        "page_bytes": entry.page_bytes,
        "sliding_window": entry.sliding_window,
        "dtype": entry.dtype,
        "external": entry.external,
        "reusable": entry.reusable,
    }


def _group_field(group: Any, name: str) -> Any:
    return group[name] if isinstance(group, dict) else getattr(group, name)


def _canonical_entry_semantics(
    entries: Iterable[KVCacheSchemaEntry],
) -> tuple[Any, ...]:
    """Return unique entry signatures without PP-local layer multiplicity."""
    by_encoding: dict[bytes, dict[str, Any]] = {}
    for entry in entries:
        fields = _entry_semantic_fields(entry)
        by_encoding[msgspec.json.encode(fields)] = fields
    return tuple(by_encoding[encoding] for encoding in sorted(by_encoding))


def _group_semantic_fields(group: Any) -> dict[str, Any]:
    # NOTE: group-level page_bytes is deliberately excluded. For a mixed
    # group vLLM derives it from the *local* layers, so it varies across PP
    # stages and would break the cross-stage contract hash. Per-role geometry
    # stays hashed at entry level, which is stage-invariant.
    return {
        "kind": _group_field(group, "kind"),
        "cache_family": _group_field(group, "cache_family"),
        "cache_role": _group_field(group, "cache_role"),
        "logical_block_size": _group_field(group, "logical_block_size"),
        "storage_block_size": _group_field(group, "storage_block_size"),
        "compress_ratio": _group_field(group, "compress_ratio"),
        "sliding_window": _group_field(group, "sliding_window"),
        "dtype": _group_field(group, "dtype"),
        "external": _group_field(group, "external"),
        "reusable": _group_field(group, "reusable"),
        "entries": _canonical_entry_semantics(_group_field(group, "entries")),
    }


def hash_kv_cache_schema(schema: HmaCacheSchema) -> str:
    """Hash only stable model/cache semantics, excluding runtime topology."""
    groups = [_group_semantic_fields(group) for group in schema.groups]
    groups.sort(key=msgspec.json.encode)
    canonical = {
        "version": schema.version,
        "model_revision": schema.model_revision,
        "model_dtype": schema.model_dtype,
        "kv_cache_layout": schema.kv_cache_layout,
        "groups": groups,
    }
    return hashlib.sha256(msgspec.json.encode(canonical)).hexdigest()


def canonical_prefix_coverage(
    schema: HmaCacheSchema,
    full_logical_blocks_by_group: Mapping[tuple[str, int], int],
) -> int:
    """Return publishable token coverage shared by all required cache groups.

    Keys are ``(group_id, group_idx)`` so duplicate semantic groups remain
    distinct local instances. Values count complete logical blocks; callers
    must exclude partial blocks. Compression ratios are already represented by
    each group's logical block size and are never applied again here. Missing
    instances, incomplete required entries, or conflicting logical block sizes
    fail closed. Physical storage geometry and compression may differ across
    entries because the block count is already a scheduler-group logical count.
    """
    required = [
        (
            group,
            tuple(entry for entry in group.entries if _entry_requires_payload(entry)),
        )
        for group in schema.groups
        if any(_entry_requires_payload(entry) for entry in group.entries)
        and not _group_is_junction(group)
    ]
    if not required:
        return 0

    coverage: list[int] = []
    for group, required_entries in required:
        if (
            not group.external
            or not group.reusable
            or not all(entry.external and entry.reusable for entry in required_entries)
        ):
            return 0
        logical_block_sizes = {entry.logical_block_size for entry in required_entries}
        if logical_block_sizes != {group.logical_block_size}:
            return 0
        group_instance = (group.group_id, group.group_idx)
        full_blocks = full_logical_blocks_by_group.get(group_instance)
        if not isinstance(full_blocks, int) or full_blocks < 0:
            return 0
        coverage.append(full_blocks * group.logical_block_size)
    return min(coverage)


HMA_MODEL_TYPES = frozenset({"deepseek_v4", "glm_moe_dsa"})


def is_hma_model_config(vllm_config: Any) -> bool:
    """Return whether EngineCore should publish the HMA group-metadata contract."""
    model_config = _model_config_field(vllm_config, "model_config")
    hf_config = _model_config_field(model_config, "hf_config")
    types = {
        value.lower()
        for value in (
            _model_config_field(model_config, "model_type"),
            _model_config_field(hf_config, "model_type"),
        )
        if isinstance(value, str)
    }
    return bool(types & HMA_MODEL_TYPES)


def export_engine_group_metadata(schema: HmaCacheSchema) -> list[dict[str, Any]]:
    """Flatten a built schema into EngineCore ``get_kv_cache_group_metadata`` rows."""
    schema_hash = hash_kv_cache_schema(schema)
    metadata: list[dict[str, Any]] = []
    for group in schema.groups:
        item = msgspec.json.decode(msgspec.json.encode(group))
        item.update(
            version=schema.version,
            model_revision=schema.model_revision,
            model_dtype=schema.model_dtype,
            kv_cache_layout=schema.kv_cache_layout,
            tp_size=schema.tp_size,
            pp_size=schema.pp_size,
            pp_rank=schema.pp_rank,
            schema_hash=schema_hash,
        )
        metadata.append(item)
    return metadata


def _resolve_kv_cache_layout(vllm_config: Any) -> str:
    cache_config = getattr(vllm_config, "cache_config", None)
    try:
        layout = cache_config.get_resolved_kv_cache_layout()
    except Exception as exc:
        raise RuntimeError(
            "Failed to resolve KV cache layout for HMA metadata."
        ) from exc
    name = getattr(layout, "name", None)
    if not isinstance(name, str) or not name:
        raise RuntimeError(
            f"Failed to resolve KV cache layout for HMA metadata: {layout!r}."
        )
    return name


def build_engine_group_metadata(
    vllm_config: Any, kv_cache_config: Any
) -> list[dict[str, Any]]:
    """Build the HMA EngineCore payload from the scheduler KV cache config."""
    model_config = _model_config_field(vllm_config, "model_config")
    hf_config = _model_config_field(model_config, "hf_config")
    parallel_config = _model_config_field(vllm_config, "parallel_config")
    model_dtype = _model_config_field(model_config, "dtype")
    pp_size = _model_config_field(parallel_config, "pipeline_parallel_size")
    model_type = _model_config_field(hf_config, "model_type") or _model_config_field(
        model_config, "model_type"
    )
    schema = build_kv_cache_schema(
        identity_kv_cache_config(kv_cache_config, pp_size=pp_size),
        model_revision=resolve_model_identity(model_config),
        model_dtype=str(model_dtype) if model_dtype is not None else None,
        kv_cache_layout=_resolve_kv_cache_layout(vllm_config),
        tp_size=_model_config_field(parallel_config, "tensor_parallel_size"),
        pp_size=pp_size,
        model_type=model_type,
    )
    return export_engine_group_metadata(schema)
