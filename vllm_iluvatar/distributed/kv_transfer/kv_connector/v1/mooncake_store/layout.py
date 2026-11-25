# SPDX-License-Identifier: Apache-2.0
"""KV cache layouts, paired-scale registration, and key namespaces for Mooncake."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
    group_kernel_blocks,
)

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.skip_layers import (
    get_skip_layers,
)

from .protocol import TailKeyBoundary

logger = init_logger(__name__)

STORE_GROUP_PP_OWNERS_ATTR = "_ilu_store_group_pp_owners"

# Fallback DSv4.1 compressed KV geometry (matches ixformer quantized_kv_format)
# so CPU-only layout planning & unit tests work even without ixformer loaded.
_FALLBACK_QUANT_FORMATS: dict[str, tuple[int, int, int, int]] = {
    # name -> (rows_per_page, scale_groups, payload_bytes, scale_page_bytes) for head_dim=512
    "i4g32": (256, 16, 256, 8192),
    "i8g64": (128, 8, 512, 2048),
}


@dataclass(frozen=True)
class _QuantFormatFallback:
    name: str
    rows: int
    groups: int
    payload_row_bytes: int
    scale_bytes_per_page: int

    def rows_per_page(self, head_dim: int = 512) -> int:
        assert head_dim == 512
        return self.rows

    def scale_groups(self, head_dim: int = 512) -> int:
        assert head_dim == 512
        return self.groups

    def payload_bytes(self, head_dim: int = 512) -> int:
        assert head_dim == 512
        return self.payload_row_bytes

    def scale_page_bytes(self, head_dim: int = 512) -> int:
        assert head_dim == 512
        return self.scale_bytes_per_page


def _resolve_dsv41_quant_format(fmt_name: str) -> Any:
    try:
        from ixformer.inference.functions.models.deepseek_v41.kv.common.quantized_kv_format import (
            resolve_quant_format,
        )

        return resolve_quant_format(fmt_name)
    except ImportError:
        if fmt_name not in _FALLBACK_QUANT_FORMATS:
            raise ValueError(f"Unsupported compressed KV format: {fmt_name!r}") from None
        rows, groups, payload_b, scale_b = _FALLBACK_QUANT_FORMATS[fmt_name]
        return _QuantFormatFallback(fmt_name, rows, groups, payload_b, scale_b)


def _is_dense_row(row: torch.Tensor) -> bool:
    if row.numel() == 0:
        return True
    strides = row.stride()
    shapes = row.shape
    if any(s < 0 for s in strides):
        return False
    extent = 1 + sum((sz - 1) * st for sz, st in zip(shapes, strides) if sz > 0)
    return extent == row.numel()


class BlobBlockHashes(Sequence[BlockHash]):
    """Zero-copy view over concatenated fixed-width block hashes."""

    def __init__(self, blob: memoryview | bytes, hash_len: int) -> None:
        self._blob = memoryview(blob)
        self._hash_len = hash_len
        self._n = len(self._blob) // hash_len if hash_len > 0 else 0

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int | slice) -> Any:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._n))]
        if idx < 0:
            idx += self._n
        if not 0 <= idx < self._n:
            raise IndexError(idx)
        off = idx * self._hash_len
        return BlockHash(bytes(self._blob[off : off + self._hash_len]))


class _CompactChunkHashList(Sequence[BlockHash]):
    """Maps ``block_size`` chunks to the last constituent ``hash_block_size`` digest."""

    def __init__(
        self,
        block_hashes: Sequence[BlockHash],
        hash_block_size: int,
        target_block_size: int,
    ) -> None:
        if target_block_size % hash_block_size != 0:
            raise ValueError(
                f"target_block_size ({target_block_size}) must be divisible by "
                f"hash_block_size ({hash_block_size})"
            )
        self.block_hashes = block_hashes
        self.scale_factor = target_block_size // hash_block_size

    def __len__(self) -> int:
        return len(self.block_hashes) // self.scale_factor

    def __getitem__(self, idx: int | slice) -> Any:
        n = len(self)
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(n))]
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(idx)
        return self.block_hashes[idx * self.scale_factor + self.scale_factor - 1]


def chunk_hashes_for_block_size(
    block_hashes: Sequence[BlockHash],
    hash_block_size: int,
    block_size: int,
) -> Sequence[BlockHash]:
    if block_size == hash_block_size:
        return block_hashes
    return _CompactChunkHashList(block_hashes, hash_block_size, block_size)


def resolve_store_tp_size(extra_config: Mapping[str, Any]) -> int | None:
    """Resolve optional shared Store TP size across heterogeneous TP instances."""
    if extra_config.get("enable_store_tp_lcm") is True:
        prefill_tp_sizes = extra_config.get("prefill_tp_sizes")
        if not isinstance(prefill_tp_sizes, list) or not prefill_tp_sizes:
            return None
        if any(
            type(tp_size) is not int or tp_size <= 0 for tp_size in prefill_tp_sizes
        ):
            return None
        return math.lcm(*prefill_tp_sizes)

    store_tp_size = extra_config.get("store_tp_size")
    return store_tp_size if type(store_tp_size) is int and store_tp_size > 0 else None


def _resolve_pp_layer_bounds(vllm_config: Any) -> list[list[int]]:
    parallel = vllm_config.parallel_config
    pp_size = int(getattr(parallel, "pipeline_parallel_size", 1))
    if pp_size <= 1:
        return [[0, 0]]

    try:
        from vllm_iluvatar.v1.worker.prefill_state.configuration import (
            config_pipeline_bounds,
        )

        bounds = config_pipeline_bounds(vllm_config)
        return [[int(a), int(b)] for a, b in bounds]
    except Exception:
        pass

    partition_env = os.getenv("VLLM_PP_LAYER_PARTITION", "").strip()
    if partition_env:
        counts = [int(x.strip()) for x in partition_env.split(",") if x.strip()]
        if len(counts) == pp_size:
            cursor = 0
            out: list[list[int]] = []
            for count in counts:
                out.append([cursor, cursor + count])
                cursor += count
            return out

    model_cfg = getattr(vllm_config, "model_config", None)
    if model_cfg is not None and hasattr(model_cfg, "get_layers_start_end_indices"):
        try:
            tp_size = int(getattr(parallel, "tensor_parallel_size", 1))
            bounds_list: list[list[int]] = []
            for stage in range(pp_size):
                stage_parallel = copy(parallel)
                stage_parallel.rank = stage * tp_size
                start, end = model_cfg.get_layers_start_end_indices(stage_parallel)
                bounds_list.append([int(start), int(end)])
            return bounds_list
        except Exception:
            pass

    return [[stage, stage + 1] for stage in range(pp_size)]


def compute_store_topology_namespace(
    vllm_config: Any,
    kv_cache_config: KVCacheConfig,
    *,
    hash_block_size: int,
    extra_namespace: str = "",
) -> str:
    """Compute a deterministic, process-invariant topology & KV-format namespace.

    Identical across independent 2P / xP Prefill instances (single-node or
    multi-node) when model, TP/PP/PCP/DCP geometry, PP layer bounds, cache
    dtype, group block sizes, and Iluvatar KV quantization formats match;
    strictly disjoint when any layout-affecting dimension differs.
    """
    parallel = vllm_config.parallel_config
    cache_cfg = getattr(vllm_config, "cache_config", None)
    tp_size = int(getattr(parallel, "tensor_parallel_size", 1))
    pp_size = int(getattr(parallel, "pipeline_parallel_size", 1))
    pcp_size = int(getattr(parallel, "prefill_context_parallel_size", 1))
    dcp_size = int(getattr(parallel, "decode_context_parallel_size", 1))

    cache_dtype = str(getattr(cache_cfg, "cache_dtype", "auto"))
    layout_name = "default"
    if kv_cache_config.kv_cache_layout:
        layout_name = getattr(
            kv_cache_config.kv_cache_layout,
            "name",
            str(kv_cache_config.kv_cache_layout),
        )
    elif cache_cfg is not None and callable(
        getattr(cache_cfg, "get_resolved_kv_cache_layout", None)
    ):
        resolved = cache_cfg.get_resolved_kv_cache_layout()
        layout_name = getattr(resolved, "name", str(resolved))

    group_block_sizes = [
        int(group.kv_cache_spec.block_size)
        for group in kv_cache_config.prefix_cacheable_groups
    ]
    pp_bounds = _resolve_pp_layer_bounds(vllm_config)

    ilu_kv_factors = {
        "ds41_compressed_kv": os.getenv(
            "VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "none"
        ).strip(),
        "ds41_indexer_layout": os.getenv(
            "VLLM_ILU_DS41_INDEXER_KV_LAYOUT", "legacy"
        ).strip(),
        "ds4_indexer_dtype": os.getenv(
            "VLLM_ILU_DS4_INDEXER_KV_DTYPE",
            os.getenv("VLLM_ILU_INDEXER_KV_DTYPE", "bf16"),
        ).strip(),
    }

    identity = {
        "v": 1,
        "tp": tp_size if not extra_namespace else 0,
        "pp": pp_size,
        "pcp": pcp_size,
        "dcp": dcp_size,
        "pp_bounds": pp_bounds,
        "cache_dtype": cache_dtype,
        "layout": layout_name,
        "group_block_sizes": group_block_sizes,
        "hash_block_size": int(hash_block_size),
        "ilu": ilu_kv_factors,
        "extra": extra_namespace,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    if extra_namespace:
        return f"{extra_namespace}@ilu_mc_v1:{digest}"
    return f"@ilu_mc_v1:{digest}"


def _extract_layer_id(layer_name: str) -> int | None:
    marker = ".layers."
    idx = layer_name.find(marker)
    if idx >= 0:
        rest = layer_name[idx + len(marker) :]
        head = rest.split(".", 1)[0]
        if head.isdigit():
            return int(head)
    if layer_name.startswith("layers."):
        head = layer_name.split(".", 2)[1]
        if head.isdigit():
            return int(head)
    return None


def resolve_group_pp_owners(
    vllm_config: Any,
    kv_cache_config: KVCacheConfig,
    store_group_ids: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Determine which PP stages own non-empty KV layers for each store group.

    On non-uniform PP models such as DeepSeek-V4.1-Flash (PP=8 with
    ``kv_source_layer_ids=[2, 8, 14, 20]``), only a subset of PP stages own
    Ratio-2 or Ratio-1 main/indexer KV layers. Rank-0 prefix lookup must query
    only the owning PP stages for each group rather than ``range(pp_size)``.
    """
    parallel = vllm_config.parallel_config
    pp_size = int(getattr(parallel, "pipeline_parallel_size", 1))
    if pp_size <= 1:
        return tuple((0,) for _ in store_group_ids)

    propagated = getattr(kv_cache_config, STORE_GROUP_PP_OWNERS_ATTR, None)
    if propagated is not None and len(propagated) == len(kv_cache_config.kv_cache_groups):
        return tuple(tuple(int(r) for r in propagated[gid]) for gid in store_group_ids)

    extra = getattr(vllm_config.kv_transfer_config, "kv_connector_extra_config", None) or {}
    explicit = extra.get("group_pp_owners")
    if isinstance(explicit, Sequence) and len(explicit) == len(store_group_ids):
        return tuple(tuple(int(r) for r in owners) for owners in explicit)

    # Fallback: map global layer indices onto PP bounds when layer names carry indices.
    pp_bounds = _resolve_pp_layer_bounds(vllm_config)
    all_stages = tuple(range(pp_size))
    inferred: list[tuple[int, ...]] = []
    for gid in store_group_ids:
        group = kv_cache_config.kv_cache_groups[gid]
        layer_ids = [
            lid
            for name in group.layer_names
            if (lid := _extract_layer_id(name)) is not None
        ]
        if layer_ids and len(pp_bounds) == pp_size and pp_bounds != [[s, s + 1] for s in range(pp_size)]:
            owners = tuple(
                stage
                for stage, (start, end) in enumerate(pp_bounds)
                if any(start <= lid < end for lid in layer_ids)
            )
            if owners:
                inferred.append(owners)
                continue
        inferred.append(all_stages)
    return tuple(inferred)


@dataclass
class KeyMetadata:
    """Metadata encoded in every Mooncake Store object key."""

    model_name: str
    tp_rank: int
    pcp_rank: int
    dcp_rank: int
    pp_rank: int
    group_id: int = 0
    cache_prefix: str = ""
    store_namespace: str = ""


@dataclass(order=True)
class PoolKey:
    """Object key for a KV cache block shard in MooncakeDistributedStore."""

    key_metadata: KeyMetadata
    chunk_hash: str

    def __hash__(self) -> int:
        return hash(
            (
                self.key_metadata.cache_prefix,
                self.key_metadata.model_name,
                self.key_metadata.store_namespace,
                self.key_metadata.tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_rank,
                self.key_metadata.group_id,
                self.chunk_hash,
            )
        )

    @staticmethod
    def build_prefix(
        key_metadata: KeyMetadata,
        *,
        tp_rank: int | None = None,
        pcp_rank: int | None = None,
        dcp_rank: int | None = None,
        pp_rank: int | None = None,
    ) -> str:
        prefix = f"{key_metadata.cache_prefix}@" if key_metadata.cache_prefix else ""
        return (
            f"{prefix}"
            f"{key_metadata.model_name}"
            f"{key_metadata.store_namespace}"
            f"@tp_rank:{key_metadata.tp_rank if tp_rank is None else tp_rank}"
            f"@pcp{key_metadata.pcp_rank if pcp_rank is None else pcp_rank}"
            f"@dcp{key_metadata.dcp_rank if dcp_rank is None else dcp_rank}"
            f"@pp_rank:{key_metadata.pp_rank if pp_rank is None else pp_rank}"
            f"@group:{key_metadata.group_id}"
        )

    @staticmethod
    def build_key_string(key_prefix: str, chunk_hash: str) -> str:
        return f"{key_prefix}@{chunk_hash}"

    def to_string(self) -> str:
        return self.build_key_string(
            self.build_prefix(self.key_metadata), self.chunk_hash
        )


StoreShardId = int
# Physical ranks encoded in a rank-local key: (TP, PCP, DCP, PP).
RankNamespace = tuple[int, int, int, int]


@dataclass(frozen=True)
class StoreRegion:
    """Physical memory region and per-block stride/payload for one plane."""

    plane: str
    layer: str
    storage_addr: int
    storage_nbytes: int
    base_addr: int
    block_stride_bytes: int
    page_bytes: int


def _coalesce_adjacent_regions(regions: Sequence[StoreRegion]) -> tuple[StoreRegion, ...]:
    """Merge contiguous per-layer slices inside one block sharing the same stride."""
    if len(regions) <= 1:
        return tuple(regions)
    merged: list[StoreRegion] = [regions[0]]
    for cur in regions[1:]:
        prev = merged[-1]
        if (
            cur.plane == prev.plane
            and cur.storage_addr == prev.storage_addr
            and cur.storage_nbytes == prev.storage_nbytes
            and cur.block_stride_bytes == prev.block_stride_bytes
            and cur.base_addr == prev.base_addr + prev.page_bytes
            and (prev.base_addr - prev.storage_addr) % prev.block_stride_bytes
            + prev.page_bytes
            + cur.page_bytes
            <= prev.block_stride_bytes
        ):
            merged[-1] = StoreRegion(
                plane=prev.plane,
                layer=f"{prev.layer}+{cur.layer}",
                storage_addr=prev.storage_addr,
                storage_nbytes=prev.storage_nbytes,
                base_addr=prev.base_addr,
                block_stride_bytes=prev.block_stride_bytes,
                page_bytes=prev.page_bytes + cur.page_bytes,
            )
        else:
            merged.append(cur)
    return tuple(merged)


def _has_dsv41_paired_pages(kv_cache_config: KVCacheConfig) -> bool:
    from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
        is_quantized_main_spec,
        layer_specs,
    )
    from vllm_iluvatar.models.deepseek_v41.indexer_kv import is_split_indexer_spec

    specs = layer_specs(kv_cache_config.kv_cache_groups)
    return any(
        is_quantized_main_spec(spec) or is_split_indexer_spec(spec)
        for spec in specs.values()
    )


def _plan_dsv41_paired_group_regions(
    kv_cache_config: KVCacheConfig,
    store_group_ids: Sequence[int],
    kv_caches: Mapping[str, Any],
    forward_context: Mapping[str, Any] | None,
    kv_cache_scales: Mapping[str, torch.Tensor] | None = None,
    *,
    coalesce: bool = True,
    honor_skip_layers: bool = False,
) -> tuple[list[tuple[StoreRegion, ...]], list[tuple[int, int]]]:
    """Plan payload + FP16 main_scale + FP32 index_scale regions per store group."""
    from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
        is_quantized_main_spec,
        layer_specs,
        plan_scale_layout,
    )
    from vllm_iluvatar.models.deepseek_v41.indexer_kv import plan_indexer_scales

    skip_layers = get_skip_layers() if honor_skip_layers else frozenset()
    specs = layer_specs(kv_cache_config.kv_cache_groups)
    main_formats = {
        spec.cache_dtype_str for spec in specs.values() if is_quantized_main_spec(spec)
    }
    if len(main_formats) > 1:
        raise ValueError(f"Worker cannot mix main KV quant formats: {main_formats}")

    main_layout = (
        plan_scale_layout(
            kv_cache_config, _resolve_dsv41_quant_format(next(iter(main_formats)))
        )
        if main_formats
        else None
    )
    index_layout = plan_indexer_scales(kv_cache_config)

    payload_placements: dict[str, tuple[int, int, int]] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        for pos, name in enumerate(tensor.layers):
            if name not in specs:
                continue
            payload_placements[name] = (
                tensor.offset + pos * tensor.layer_stride,
                tensor.block_stride,
                specs[name].page_size_bytes,
            )

    num_blocks = kv_cache_config.num_blocks
    storage_registry: dict[int, int] = {}

    def _resolve_scale_tensor(layer_name: str) -> torch.Tensor:
        if kv_cache_scales is not None and layer_name in kv_cache_scales:
            scale = kv_cache_scales[layer_name]
            if isinstance(scale, torch.Tensor):
                return scale
        if forward_context is not None and layer_name in forward_context:
            scale = getattr(forward_context[layer_name], "kv_cache_scales", None)
            if isinstance(scale, torch.Tensor):
                return scale
        raise ValueError(f"{layer_name}: missing paired kv_cache_scales tensor")

    def _bind_region(
        plane: str,
        layer_name: str,
        source: torch.Tensor,
        offset_bytes: int,
        stride_bytes: int,
        page_bytes: int,
    ) -> StoreRegion:
        if offset_bytes < 0 or page_bytes <= 0 or stride_bytes < page_bytes:
            raise ValueError(
                f"{layer_name}: invalid {plane} geometry "
                f"(offset={offset_bytes}, stride={stride_bytes}, page={page_bytes})"
            )
        physical = group_kernel_blocks(source, num_blocks)
        elem = source.element_size()
        actual_offset = physical.storage_offset() * elem
        actual_stride = physical.stride(0) * elem
        if actual_offset != offset_bytes or actual_stride != stride_bytes:
            raise ValueError(
                f"{layer_name}: {plane} tensor view (offset={actual_offset}, "
                f"stride={actual_stride}) differs from planned ({offset_bytes}, {stride_bytes})"
            )
        if not physical[0].is_contiguous() or physical[0].numel() * elem != page_bytes:
            raise ValueError(f"{layer_name}: {plane} kernel block is not dense")
        storage = source.untyped_storage()
        s_addr = int(storage.data_ptr())
        s_len = int(storage.nbytes())
        prev_len = storage_registry.get(s_addr)
        if prev_len is None:
            storage_registry[s_addr] = s_len
        elif prev_len != s_len:
            raise ValueError(
                f"{layer_name}: inconsistent storage lengths at {s_addr:#x}: "
                f"{prev_len} vs {s_len}"
            )
        return StoreRegion(
            plane=plane,
            layer=layer_name,
            storage_addr=s_addr,
            storage_nbytes=s_len,
            base_addr=s_addr + offset_bytes,
            block_stride_bytes=stride_bytes,
            page_bytes=page_bytes,
        )

    per_group_regions: list[tuple[StoreRegion, ...]] = []
    for gid in store_group_ids:
        group = kv_cache_config.kv_cache_groups[gid]
        seen_keys: set[tuple[str, int, int, int]] = set()
        group_list: list[StoreRegion] = []
        for name in group.layer_names:
            if name in skip_layers or name not in kv_caches:
                continue
            off, stride, page = payload_placements[name]
            key = ("payload", off, stride, page)
            if key not in seen_keys:
                seen_keys.add(key)
                group_list.append(
                    _bind_region("payload", name, kv_caches[name], off, stride, page)
                )

            if main_layout is not None and name in main_layout.placements:
                mp = main_layout.placements[name]
                m_off = mp.offset_elements * 2
                m_stride = mp.block_stride_elements * mp.blocks_per_page * 2
                m_page = mp.rows * mp.groups * mp.blocks_per_page * 2
                m_key = ("main_scale", m_off, m_stride, m_page)
                if m_key not in seen_keys:
                    seen_keys.add(m_key)
                    group_list.append(
                        _bind_region(
                            "main_scale",
                            name,
                            _resolve_scale_tensor(name),
                            m_off,
                            m_stride,
                            m_page,
                        )
                    )

            if name in index_layout.placements:
                ip = index_layout.placements[name]
                i_off = ip.offset * 4
                i_stride = ip.rows_per_manager_block * 4
                i_page = ip.rows_per_manager_block * 4
                i_key = ("index_scale", i_off, i_stride, i_page)
                if i_key not in seen_keys:
                    seen_keys.add(i_key)
                    group_list.append(
                        _bind_region(
                            "index_scale",
                            name,
                            _resolve_scale_tensor(name),
                            i_off,
                            i_stride,
                            i_page,
                        )
                    )

        per_group_regions.append(
            _coalesce_adjacent_regions(group_list) if coalesce else tuple(group_list)
        )

    registrations = list(storage_registry.items())
    return per_group_regions, registrations


def _iter_cache_tensors(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,) if value.numel() > 0 else ()
    if isinstance(value, (list, tuple)):
        return tuple(t for t in value if isinstance(t, torch.Tensor) and t.numel() > 0)
    return ()


def _layer_spec_for_group(group: KVCacheGroupSpec, layer_name: str) -> KVCacheSpec:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.kv_cache_specs.get(layer_name, spec)
    return spec


def plan_group_store_regions(
    kv_cache_config: KVCacheConfig,
    store_group_ids: Sequence[int],
    kv_caches: Mapping[str, Any],
    forward_context: Mapping[str, Any] | None = None,
    kv_cache_scales: Mapping[str, torch.Tensor] | None = None,
    *,
    coalesce: bool = True,
    honor_skip_layers: bool = False,
) -> tuple[list[tuple[StoreRegion, ...]], list[tuple[int, int]]]:
    """Plan per-group Mooncake transfer regions and unique buffer registrations.

    Handles:
      - DeepSeek-V4.1-Flash paired quantized main KV + FP16 scales and split
        INT8 indexer KV + FP32 scales (including PP forwarder replicas across
        rank-symmetric Prefill instances when ``honor_skip_layers=False``).
      - Standard MHA / GQA / MQA / MLA (single tensor or K/V tuple).
      - Hybrid Mamba + Attention (list of conv/ssm state tensors per layer).
      - Non-uniform PP where a worker rank owns empty ``layer_names`` in some
        or all ``store_group_ids``.
    """
    if _has_dsv41_paired_pages(kv_cache_config):
        return _plan_dsv41_paired_group_regions(
            kv_cache_config,
            store_group_ids,
            kv_caches,
            forward_context,
            kv_cache_scales,
            coalesce=coalesce,
            honor_skip_layers=honor_skip_layers,
        )

    skip_layers = get_skip_layers() if honor_skip_layers else frozenset()
    storage_registry: dict[int, int] = {}
    per_group_regions: list[tuple[StoreRegion, ...]] = []

    for gid in store_group_ids:
        group = kv_cache_config.kv_cache_groups[gid]
        is_host = bool(getattr(group, "host_resident", False))
        num_blocks = (
            int(kv_cache_config.hisparse_host_num_blocks)
            if is_host and kv_cache_config.hisparse_host_num_blocks is not None
            else int(kv_cache_config.num_blocks)
        )

        seen_group_regions: set[tuple[int, int, int]] = set()
        group_list: list[StoreRegion] = []
        for layer_name in group.layer_names:
            if layer_name in skip_layers or layer_name not in kv_caches:
                continue
            spec = _layer_spec_for_group(group, layer_name)
            raw_tensors = _iter_cache_tensors(kv_caches[layer_name])
            for sub_idx, raw in enumerate(raw_tensors):
                cache = (
                    raw
                    if isinstance(spec, MambaSpec) or raw.ndim < 2
                    else group_kernel_blocks(raw, num_blocks)
                )
                elem = cache.element_size()
                storage = cache.untyped_storage()
                s_addr = int(storage.data_ptr())
                s_len = int(storage.nbytes())
                if is_host and s_len == 0:
                    s_addr = int(cache.data_ptr())
                    s_len = int(cache.nbytes)

                prev_len = storage_registry.get(s_addr)
                if prev_len is None:
                    storage_registry[s_addr] = s_len
                elif prev_len != s_len:
                    storage_registry[s_addr] = max(prev_len, s_len)

                if cache.ndim >= 2 and not _is_dense_row(cache[0]):
                    # Split non-dense head views (e.g., separate K/V views strided on dim 1)
                    for head_idx in range(cache.shape[1]):
                        head_cache = cache[:, head_idx]
                        if not _is_dense_row(head_cache[0]):
                            raise ValueError(
                                f"{layer_name}[{sub_idx}] head {head_idx} is not dense"
                            )
                        r_addr = int(head_cache.data_ptr())
                        r_stride = int(head_cache.stride(0) * elem)
                        r_page = int(head_cache[0].numel() * elem)
                        r_key = (r_addr, r_stride, r_page)
                        if r_key not in seen_group_regions:
                            seen_group_regions.add(r_key)
                            group_list.append(
                                StoreRegion(
                                    plane="payload",
                                    layer=f"{layer_name}.h{head_idx}",
                                    storage_addr=s_addr,
                                    storage_nbytes=storage_registry[s_addr],
                                    base_addr=r_addr,
                                    block_stride_bytes=r_stride,
                                    page_bytes=r_page,
                                )
                            )
                    continue

                r_addr = int(cache.data_ptr())
                if cache.ndim == 1:
                    r_page = int(spec.page_size_bytes)
                    r_stride = r_page
                else:
                    r_stride = int(cache.stride(0) * elem)
                    r_page = int(cache[0].numel() * elem)

                if r_stride < r_page or r_page <= 0:
                    raise ValueError(
                        f"{layer_name}[{sub_idx}]: invalid stride/page "
                        f"({r_stride} < {r_page})"
                    )
                r_key = (r_addr, r_stride, r_page)
                if r_key not in seen_group_regions:
                    seen_group_regions.add(r_key)
                    group_list.append(
                        StoreRegion(
                            plane="payload",
                            layer=layer_name if len(raw_tensors) == 1 else f"{layer_name}.{sub_idx}",
                            storage_addr=s_addr,
                            storage_nbytes=storage_registry[s_addr],
                            base_addr=r_addr,
                            block_stride_bytes=r_stride,
                            page_bytes=r_page,
                        )
                    )

        per_group_regions.append(
            _coalesce_adjacent_regions(group_list) if coalesce else tuple(group_list)
        )

    return per_group_regions, list(storage_registry.items())


class StoreLayout:
    """Base store payload layout for one KV-cache group on one worker rank."""

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        hash_block_size: int,
    ) -> None:
        self.metadata = metadata
        self.block_size = block_size
        self.hash_block_size = hash_block_size

    @property
    def local_shard_ids(self) -> tuple[StoreShardId, ...]:
        raise NotImplementedError

    @property
    def has_regions(self) -> bool:
        raise NotImplementedError

    def key_for(self, shard_id: StoreShardId, chunk_hash: BlockHash) -> str:
        raise NotImplementedError

    def lookup_key_prefixes(
        self,
        rank_namespaces: Sequence[RankNamespace],
    ) -> tuple[str, ...]:
        raise NotImplementedError

    def prepare_values(
        self,
        chunks: Sequence[tuple[int, int]],
        block_ids: Sequence[int],
        shard_ids: Sequence[StoreShardId],
    ) -> tuple[list[list[int]], list[list[int]], list[int]]:
        raise NotImplementedError

    def prepare_value_for_block(self, block_id: int) -> tuple[list[int], list[int]]:
        raise NotImplementedError


class GroupStoreLayout(StoreLayout):
    """Per-group strided + paired-scale store layout (distinguishes stride from page_bytes)."""

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        hash_block_size: int,
    ) -> None:
        super().__init__(metadata, block_size, hash_block_size)
        self._key_prefix = PoolKey.build_prefix(metadata)
        self._regions: tuple[StoreRegion, ...] = ()
        self._base_addrs = np.empty((0,), dtype=np.uint64)
        self._block_strides = np.empty((0,), dtype=np.uint64)
        self._page_bytes = np.empty((0,), dtype=np.uint64)

    @property
    def local_shard_ids(self) -> tuple[StoreShardId, ...]:
        return (0,)

    @property
    def has_regions(self) -> bool:
        return len(self._base_addrs) > 0

    @property
    def regions(self) -> tuple[StoreRegion, ...]:
        return self._regions

    @property
    def bytes_per_block(self) -> int:
        return int(self._page_bytes.sum()) if len(self._page_bytes) else 0

    def bind_regions(self, regions: Sequence[StoreRegion]) -> None:
        self._regions = tuple(regions)
        self._base_addrs = np.asarray(
            [r.base_addr for r in self._regions], dtype=np.uint64
        )
        self._block_strides = np.asarray(
            [r.block_stride_bytes for r in self._regions], dtype=np.uint64
        )
        self._page_bytes = np.asarray(
            [r.page_bytes for r in self._regions], dtype=np.uint64
        )

    def set_raw_descriptors(
        self,
        base_addrs: Sequence[int],
        block_lens: Sequence[int],
        block_strides: Sequence[int] | None = None,
    ) -> None:
        strides = block_lens if block_strides is None else block_strides
        if not (len(base_addrs) == len(block_lens) == len(strides)):
            raise ValueError("base_addrs, block_lens, and block_strides must align")
        self._regions = tuple(
            StoreRegion(
                plane="payload",
                layer=f"seg_{i}",
                storage_addr=int(addr),
                storage_nbytes=0,
                base_addr=int(addr),
                block_stride_bytes=int(stride),
                page_bytes=int(length),
            )
            for i, (addr, stride, length) in enumerate(
                zip(base_addrs, strides, block_lens, strict=True)
            )
        )
        self._base_addrs = np.asarray(base_addrs, dtype=np.uint64)
        self._block_strides = np.asarray(strides, dtype=np.uint64)
        self._page_bytes = np.asarray(block_lens, dtype=np.uint64)

    def key_for(self, shard_id: StoreShardId, chunk_hash: BlockHash) -> str:
        assert shard_id == 0
        return PoolKey.build_key_string(self._key_prefix, chunk_hash.hex())

    def lookup_key_prefixes(
        self,
        rank_namespaces: Sequence[RankNamespace],
    ) -> tuple[str, ...]:
        return tuple(
            PoolKey.build_prefix(
                self.metadata,
                tp_rank=tp_rank,
                pcp_rank=pcp_rank,
                dcp_rank=dcp_rank,
                pp_rank=pp_rank,
            )
            for tp_rank, pcp_rank, dcp_rank, pp_rank in rank_namespaces
        )

    def prepare_values(
        self,
        chunks: Sequence[tuple[int, int]],
        block_ids: Sequence[int],
        shard_ids: Sequence[StoreShardId],
    ) -> tuple[list[list[int]], list[list[int]], list[int]]:
        if not chunks or not self.has_regions:
            return [], [], []
        assert len(chunks) == len(shard_ids) and all(s == 0 for s in shard_ids)

        n = len(chunks)
        starts = np.fromiter((c[0] for c in chunks), dtype=np.int64, count=n)
        spans = np.fromiter((c[1] for c in chunks), dtype=np.int64, count=n) - starts
        assert np.all(spans % self.hash_block_size == 0)

        bids = np.fromiter(
            (block_ids[i] for i in (starts // self.block_size).tolist()),
            dtype=np.uint64,
            count=n,
        )
        addrs = self._base_addrs[None, :] + bids[:, None] * self._block_strides[None, :]
        block_counts = ((spans + self.block_size - 1) // self.block_size).astype(
            np.uint64
        )
        # Multi-block span is only contiguous when block_stride == page_bytes;
        # for standard single-block chunks block_counts == 1 always.
        if np.any(block_counts > 1) and not np.array_equal(
            self._block_strides, self._page_bytes
        ):
            out_addrs: list[list[int]] = []
            out_sizes: list[list[int]] = []
            for start_tok, span_tok in zip(starts.tolist(), spans.tolist(), strict=True):
                first_idx = start_tok // self.block_size
                cnt = (span_tok + self.block_size - 1) // self.block_size
                seg_addrs: list[int] = []
                seg_sizes: list[int] = []
                for sub in range(cnt):
                    b_id = int(block_ids[first_idx + sub])
                    for base, stride, page in zip(
                        self._base_addrs.tolist(),
                        self._block_strides.tolist(),
                        self._page_bytes.tolist(),
                        strict=True,
                    ):
                        seg_addrs.append(int(base + b_id * stride))
                        seg_sizes.append(int(page))
                out_addrs.append(seg_addrs)
                out_sizes.append(seg_sizes)
            return out_addrs, out_sizes, [int(x) for x in bids.tolist()]

        sizes = self._page_bytes[None, :] * block_counts[:, None]
        return (
            addrs.tolist(),
            sizes.tolist(),
            [int(x) for x in bids.tolist()],
        )

    def prepare_value_for_block(self, block_id: int) -> tuple[list[int], list[int]]:
        if not self.has_regions:
            return [], []
        bid = int(block_id)
        return (
            [
                int(base + bid * stride)
                for base, stride in zip(
                    self._base_addrs.tolist(),
                    self._block_strides.tolist(),
                    strict=True,
                )
            ],
            [int(page) for page in self._page_bytes.tolist()],
        )


class TPShardedStoreLayout(StoreLayout):
    """Store layout shared across divisible TP sizes for dense FullAttention."""

    store_format: str

    @classmethod
    def shared_namespace(cls, store_tp_size: int, pp_size: int) -> str:
        return (
            f"@store_tp:{store_tp_size}@store_pp:{pp_size}"
            f"@store_format:{cls.store_format}"
        )

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        hash_block_size: int,
        local_tp_size: int,
        store_tp_size: int,
        tp_rank: int,
        num_kv_heads: int,
    ) -> None:
        super().__init__(metadata, block_size, hash_block_size)
        self.shards_per_rank = store_tp_size // local_tp_size
        first_shard = tp_rank * self.shards_per_rank
        self.store_shard_ids = tuple(
            range(first_shard, first_shard + self.shards_per_rank)
        )
        self.store_tp_size = store_tp_size
        self.heads_per_store_shard = num_kv_heads // store_tp_size
        self.local_num_kv_heads = num_kv_heads // local_tp_size
        self._shard_key_prefixes = {
            shard_id: PoolKey.build_prefix(metadata, tp_rank=shard_id)
            for shard_id in self.store_shard_ids
        }
        self._shard_addr_bases = np.empty((self.shards_per_rank, 0), dtype=np.uint64)
        self._shard_block_strides = np.empty((self.shards_per_rank, 0), dtype=np.uint64)
        self._shard_sizes = np.empty((self.shards_per_rank, 0), dtype=np.uint64)

    @property
    def local_shard_ids(self) -> tuple[StoreShardId, ...]:
        return self.store_shard_ids

    @property
    def has_regions(self) -> bool:
        return self._shard_addr_bases.shape[1] > 0

    def key_for(self, shard_id: StoreShardId, chunk_hash: BlockHash) -> str:
        return PoolKey.build_key_string(
            self._shard_key_prefixes[shard_id], chunk_hash.hex()
        )

    def lookup_key_prefixes(
        self,
        rank_namespaces: Sequence[RankNamespace],
    ) -> tuple[str, ...]:
        pp_ranks = sorted({namespace[3] for namespace in rank_namespaces})
        return tuple(
            PoolKey.build_prefix(self.metadata, tp_rank=shard_id, pp_rank=pp_rank)
            for shard_id in range(self.store_tp_size)
            for pp_rank in pp_ranks
        )

    def _has_packed_strides(
        self, head_stride: int, token_stride: int, content_bytes: int
    ) -> bool:
        raise NotImplementedError

    def _shard_segments(
        self, head_start: int, head_stride: int, token_stride: int, content_bytes: int
    ) -> Iterable[tuple[int, int]]:
        raise NotImplementedError

    def register_kv_caches(
        self,
        kv_caches: Sequence[torch.Tensor],
        num_blocks: int,
    ) -> None:
        templates: list[tuple[list[int], list[int], list[int]]] = [
            ([], [], []) for _ in range(self.shards_per_rank)
        ]
        for cache in kv_caches:
            if cache.ndim != 4 or tuple(cache.shape[:3]) != (
                num_blocks,
                self.local_num_kv_heads,
                self.block_size,
            ):
                raise ValueError(
                    "TP-shared Mooncake store requires packed KV caches with "
                    "logical shape (num_blocks, local_kv_heads, block_size, content)"
                )
            element_size = cache.element_size()
            block_stride, head_stride, token_stride, content_stride = (
                stride * element_size for stride in cache.stride()
            )
            content_bytes = cache.shape[3] * element_size
            if content_stride != element_size or not self._has_packed_strides(
                head_stride, token_stride, content_bytes
            ):
                raise ValueError(
                    f"TP-shared Mooncake store requires packed {self.store_format} KV layout"
                )
            for local_shard, (addr_bases, block_strides, sizes) in enumerate(templates):
                head_start = local_shard * self.heads_per_store_shard
                for offset, size in self._shard_segments(
                    head_start, head_stride, token_stride, content_bytes
                ):
                    addr_bases.append(int(cache.data_ptr()) + offset)
                    block_strides.append(block_stride)
                    sizes.append(size)

        self._shard_addr_bases = np.asarray([t[0] for t in templates], dtype=np.uint64)
        self._shard_block_strides = np.asarray(
            [t[1] for t in templates], dtype=np.uint64
        )
        self._shard_sizes = np.asarray([t[2] for t in templates], dtype=np.uint64)

    def prepare_values(
        self,
        chunks: Sequence[tuple[int, int]],
        block_ids: Sequence[int],
        shard_ids: Sequence[StoreShardId],
    ) -> tuple[list[list[int]], list[list[int]], list[int]]:
        if not chunks or not self.has_regions:
            return [], [], []
        if len(chunks) != len(shard_ids):
            raise ValueError("Each Store chunk must have one shard ID")

        count = len(chunks)
        starts = np.fromiter((c[0] for c in chunks), dtype=np.int64, count=count)
        ends = np.fromiter((c[1] for c in chunks), dtype=np.int64, count=count)
        if np.any(ends - starts != self.block_size):
            raise ValueError("TP-shared Mooncake store requires full KV blocks")

        first_shard = self.store_shard_ids[0]
        local_shards = np.fromiter(
            (s - first_shard for s in shard_ids), dtype=np.int64, count=count
        )
        if np.any(local_shards < 0) or np.any(local_shards >= self.shards_per_rank):
            raise ValueError("Store shard is not owned by this TP rank")
        chunk_block_ids = np.fromiter(
            (block_ids[idx] for idx in (starts // self.block_size).tolist()),
            dtype=np.uint64,
            count=count,
        )
        addr_bases = self._shard_addr_bases[local_shards]
        block_strides = self._shard_block_strides[local_shards]
        addrs = addr_bases + chunk_block_ids[:, None] * block_strides
        sizes = self._shard_sizes[local_shards]
        return (
            addrs.tolist(),
            sizes.tolist(),
            [int(x) for x in chunk_block_ids.tolist()],
        )

    def prepare_value_for_block(self, block_id: int) -> tuple[list[int], list[int]]:
        raise RuntimeError("prepare_value_for_block requires rank-local GroupStoreLayout")


class LBHNCStoreLayout(TPShardedStoreLayout):
    store_format = "tp_shared_lbhnc"

    def _has_packed_strides(
        self, head_stride: int, token_stride: int, content_bytes: int
    ) -> bool:
        return (
            token_stride == content_bytes
            and head_stride == self.block_size * content_bytes
        )

    def _shard_segments(
        self, head_start: int, head_stride: int, token_stride: int, content_bytes: int
    ) -> Iterable[tuple[int, int]]:
        yield head_start * head_stride, self.heads_per_store_shard * head_stride


class LBNHCStoreLayout(TPShardedStoreLayout):
    store_format = "tp_shared_lbnhc"

    def _has_packed_strides(
        self, head_stride: int, token_stride: int, content_bytes: int
    ) -> bool:
        return (
            head_stride == content_bytes
            and token_stride == self.local_num_kv_heads * content_bytes
        )

    def _shard_segments(
        self, head_start: int, head_stride: int, token_stride: int, content_bytes: int
    ) -> Iterable[tuple[int, int]]:
        for token_idx in range(self.block_size):
            yield (
                token_idx * token_stride + head_start * head_stride,
                self.heads_per_store_shard * content_bytes,
            )


class ChunkedTokenDatabase:
    """Enumerates logical token chunks, hashes, and multi-buffer descriptors."""

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        hash_block_size: int | None = None,
        store_layout: StoreLayout | None = None,
    ) -> None:
        self.metadata = metadata
        self.block_size = block_size
        self.hash_block_size = hash_block_size or block_size
        if self.block_size % self.hash_block_size != 0:
            raise ValueError(
                f"block_size ({self.block_size}) must be a multiple of "
                f"hash_block_size ({self.hash_block_size})"
            )
        self.store_layout: StoreLayout = store_layout or GroupStoreLayout(
            metadata, block_size, self.hash_block_size
        )

    def _group_layout(self) -> GroupStoreLayout:
        if not isinstance(self.store_layout, GroupStoreLayout):
            raise RuntimeError("Operation requires GroupStoreLayout")
        return self.store_layout

    def key_for(self, chunk_hash: BlockHash) -> str:
        return self._group_layout().key_for(0, chunk_hash)

    def prepare_value_for_block(self, block_id: int) -> tuple[list[int], list[int]]:
        return self._group_layout().prepare_value_for_block(block_id)

    def process_tokens(
        self,
        token_len: int,
        block_hashes: Sequence[BlockHash],
        mask_num: int = 0,
        *,
        chunk_mask: Sequence[bool] | None = None,
        put_step: int = 1,
        put_step_rank: int = 0,
    ) -> Iterable[tuple[int, int, BlockHash]]:
        assert put_step > 0
        if not block_hashes or token_len <= 0:
            return
        assert token_len % self.hash_block_size == 0
        assert token_len // self.hash_block_size <= len(block_hashes)
        start_chunk = max(0, cdiv(mask_num, self.block_size))
        max_chunks = cdiv(token_len, self.block_size)
        if chunk_mask is not None:
            max_chunks = min(max_chunks, start_chunk + len(chunk_mask))
        for chunk_id in range(start_chunk, max_chunks):
            if chunk_mask is not None and not chunk_mask[chunk_id - start_chunk]:
                continue
            if chunk_id % put_step != put_step_rank:
                continue
            start_idx = chunk_id * self.block_size
            end_idx = min(start_idx + self.block_size, token_len)
            h = block_hashes[end_idx // self.hash_block_size - 1]
            yield start_idx, end_idx, h


@dataclass
class LoadSpec:
    """Specification for loading KV cache blocks from Mooncake Store."""

    vllm_cached_tokens: int
    kvpool_cached_tokens: int
    can_load: bool
    token_len: int = 0
    tail_key_boundaries: tuple[TailKeyBoundary, ...] = ()


@dataclass
class RequestTracker:
    """Tracks per-request state across scheduler steps."""

    req_id: str
    token_len: int
    allocated_block_ids: tuple[list[int], ...]
    num_saved_tokens: int = 0
    token_ids: list[int] | None = None
    has_pending_offload: bool = False
    prefill_end_tokens: int = 0

    def reset(self) -> None:
        self.token_len = 0
        self.allocated_block_ids = ()
        self.num_saved_tokens = 0
        self.token_ids = None
        self.has_pending_offload = False
        self.prefill_end_tokens = 0

    def update(self, new_block_ids: tuple[list[int], ...] | list[int]) -> None:
        if isinstance(new_block_ids, list):
            new_block_ids = (new_block_ids,)
        if len(new_block_ids) != len(self.allocated_block_ids):
            raise ValueError(
                f"Group count mismatch: tracker has "
                f"{len(self.allocated_block_ids)} groups, update has {len(new_block_ids)}"
            )
        for existing, new in zip(self.allocated_block_ids, new_block_ids, strict=True):
            if new:
                existing.extend(new)


@dataclass
class ReqMeta:
    """Per-request metadata passed from Scheduler to Worker."""

    req_id: str
    token_len_chunk: int
    block_ids: tuple[list[int], ...]
    block_hashes: list[BlockHash]

    can_save: bool | None = None
    load_spec: LoadSpec | None = None
    current_event: Any = None

    token_ids: list[int] | None = None
    token_ids_start: int = 0
    num_prompt_tokens: int | None = None
    store_job_id: int | None = None
    boundary_state_offloads: list[tuple[int, int, int]] | None = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: LoadSpec | None = None,
        skip_save: bool | None = False,
        block_hashes: Sequence[BlockHash] | None = None,
    ) -> ReqMeta | None:
        resolved_hashes = list(block_hashes) if block_hashes is not None else []
        input_token_len = tracker.token_len

        token_ids_start = tracker.num_saved_tokens
        chunk_boundary = cdiv(token_ids_start + 1, block_size) * block_size
        num_tokens_to_save = input_token_len // block_size * block_size

        skip_save = bool(skip_save) or num_tokens_to_save < chunk_boundary
        if load_spec is not None and load_spec.can_load:
            skip_save = True
        if skip_save and load_spec is None:
            return None

        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        token_ids = None
        if tracker.token_ids and not skip_save:
            token_ids = tracker.token_ids[token_ids_start:num_tokens_to_save]

        effective_load = (
            load_spec if (load_spec is not None and load_spec.can_load) else None
        )
        if skip_save and effective_load is None:
            return None

        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=num_tokens_to_save,
            block_ids=tracker.allocated_block_ids,
            can_save=not skip_save,
            load_spec=effective_load,
            block_hashes=resolved_hashes,
            token_ids=token_ids,
            token_ids_start=token_ids_start,
            num_prompt_tokens=tracker.prefill_end_tokens,
        )


@dataclass
class MooncakeStoreWorkerMetadata(KVConnectorWorkerMetadata):
    """Maps ``store_job_id`` to number of worker ranks that finished reading its blocks."""

    completed_saves: dict[int, int] = field(default_factory=dict)

    def aggregate(
        self, other: KVConnectorWorkerMetadata
    ) -> MooncakeStoreWorkerMetadata:
        other_meta = cast(MooncakeStoreWorkerMetadata, other)
        for store_job_id, count in other_meta.completed_saves.items():
            self.completed_saves[store_job_id] = (
                self.completed_saves.get(store_job_id, 0) + count
            )
        return self


class MooncakeStoreConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from Scheduler to Worker on each step."""

    def __init__(
        self,
        unfinished_request_ids: set[str],
        preempted_req_ids: set[str],
    ) -> None:
        self.requests: list[ReqMeta] = []
        self.unfinished_request_ids = set(unfinished_request_ids)
        self.preempted_req_ids = set(preempted_req_ids)

    def add_request(self, req_meta: ReqMeta) -> None:
        self.requests.append(req_meta)
