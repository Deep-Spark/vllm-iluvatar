# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side PP extensions for the Ilu NIXL connector."""

import time
from collections import defaultdict
from concurrent.futures import Future
from contextlib import suppress
from typing import Any

import msgspec
import numpy as np
import zmq
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import EngineTransferInfo
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    _is_attention_spec,
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs

from .metadata import IluNixlAgentMetadata, IluNixlTransferRegion
from .pp_utils import infer_pp_split

logger = init_logger(__name__)


class IluNixlWorkerMixin:
    """Mixin that makes the official NIXL worker homogeneous-PP aware."""

    def __init__(self, *args, **kwargs):
        vllm_config = args[0] if args else kwargs.get("vllm_config")
        kv_cache_config = args[2] if len(args) > 2 else kwargs.get("kv_cache_config")
        parallel_config = getattr(vllm_config, "parallel_config", None)
        original_pp_size = getattr(parallel_config, "pipeline_parallel_size", 1)
        is_pp = original_pp_size > 1
        is_consumer = (
            vllm_config is not None
            and vllm_config.kv_transfer_config is not None
            and vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        requires_hma = (
            vllm_config is not None
            and kv_cache_config is not None
            and not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(group.kv_cache_spec, FullAttentionSpec)
                for group in kv_cache_config.kv_cache_groups
            )
        )

        # Upstream rejects both consumer PP and PP with HMA during worker
        # construction. IluNixl implements PP-aware notification accounting and
        # region metadata for those cases, so hide PP only while constructing
        # the upstream worker, then restore the real topology.
        mask_pp_during_init = is_pp and (is_consumer or requires_hma)

        # Upstream KV-cache registration can handle uniform PP layouts. Only
        # HMA needs IluNixl's semantic PP/HMA registration path below.
        self._mask_pp_during_register = is_pp and requires_hma
        if mask_pp_during_init:
            try:
                parallel_config.pipeline_parallel_size = 1
            except Exception:
                object.__setattr__(parallel_config, "pipeline_parallel_size", 1)
        try:
            super().__init__(*args, **kwargs)
        finally:
            if mask_pp_during_init:
                try:
                    parallel_config.pipeline_parallel_size = original_pp_size
                except Exception:
                    object.__setattr__(
                        parallel_config, "pipeline_parallel_size", original_pp_size
                    )

        if mask_pp_during_init:
            self.pp_size = original_pp_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_world_size = get_pp_group().world_size
        self.pp_split = envs.VLLM_PP_LAYER_PARTITION or infer_pp_split(self.vllm_config)
        assert len(self.pp_split.split(",")) == self.pp_world_size, (
            "PP split count must equal pp_size"
        )
        self.layer_region_ids: dict[str, list[int]] = {}
        self.region_ids_by_group: list[np.ndarray] = []
        self.transfer_regions: list[IluNixlTransferRegion] = []
        self.local_region_ids_by_group: dict[str, dict[int, list[np.ndarray]]] = {}
        self.remote_region_ids_by_group: dict[str, dict[int, list[np.ndarray]]] = {}
        self._remote_pp_size_by_engine: dict[str, int] = {}
        self._remote_pp_split_by_engine: dict[str, str] = {}

    @staticmethod
    def _pp_layer_range(pp_split: str, pp_rank: int) -> tuple[int, int]:
        split = [int(x) for x in pp_split.split(",")]
        start = sum(split[:pp_rank])
        return start, start + split[pp_rank]

    @classmethod
    def _target_pp_ranks(
        cls,
        local_pp_split: str,
        local_pp_rank: int,
        remote_pp_split: str,
    ) -> list[int]:
        local_start, local_end = cls._pp_layer_range(local_pp_split, local_pp_rank)
        remote_split = [int(x) for x in remote_pp_split.split(",")]
        target_pp_ranks: list[int] = []
        remote_start = 0
        for remote_pp_rank, num_layers in enumerate(remote_split):
            remote_end = remote_start + num_layers
            if max(local_start, remote_start) < min(local_end, remote_end):
                target_pp_ranks.append(remote_pp_rank)
            remote_start = remote_end
        return target_pp_ranks

    @classmethod
    def _overlap_pp_count(
        cls,
        producer_pp_rank: int,
        producer_pp_split: str,
        consumer_pp_split: str,
    ) -> int:
        producer_start, producer_end = cls._pp_layer_range(
            producer_pp_split, producer_pp_rank
        )
        consumer_split = [int(x) for x in consumer_pp_split.split(",")]
        count = 0
        consumer_start = 0
        for num_layers in consumer_split:
            consumer_end = consumer_start + num_layers
            if max(producer_start, consumer_start) < min(producer_end, consumer_end):
                count += 1
            consumer_start = consumer_end
        return count

    @staticmethod
    def _cache_kind_from_layer_name(layer_name: str) -> str:
        if ".swa_cache" in layer_name:
            return "swa_cache"
        if ".indexer.k_cache" in layer_name:
            return "indexer_k_cache"
        if ".indexer.compressor" in layer_name:
            return "indexer_compressor"
        if ".compressor.state_cache" in layer_name:
            return "compressor_state_cache"
        if ".ilu_deepseek_v4_multi_head_latent_attention" in layer_name:
            return "main_mla"
        if layer_name.endswith(".attn"):
            return "main_mla"
        return "unknown"

    @classmethod
    def _transfer_identity(cls, layer_name: str, group_idx: int) -> str:
        try:
            layer_idx = extract_layer_index(layer_name)
        except (AssertionError, ValueError):
            layer_idx = -1
        kind = cls._cache_kind_from_layer_name(layer_name)
        return f"{layer_idx}:{kind}:group{group_idx}"

    @staticmethod
    def _regions_by_id(
        transfer_regions: list[IluNixlTransferRegion] | None,
    ) -> dict[int, IluNixlTransferRegion]:
        return {region.region_id: region for region in transfer_regions or []}

    @classmethod
    def _region_ids_for_layer_range(
        cls,
        layer_region_ids: dict[str, list[int]],
        layer_start: int,
        layer_end: int,
    ) -> list[int]:
        selected: list[int] = []
        layers: list[tuple[int, str, list[int]]] = []
        for layer_name, region_ids in layer_region_ids.items():
            try:
                layer_idx = extract_layer_index(layer_name)
            except (AssertionError, ValueError):
                continue
            if layer_start <= layer_idx < layer_end:
                layers.append((layer_idx, layer_name, region_ids))
        for _, _, region_ids in sorted(layers):
            for region_id in region_ids:
                if region_id not in selected:
                    selected.append(region_id)
        return selected

    def _actual_to_virtual_region_ids(self) -> dict[int, list[int]]:
        mapping: dict[int, list[int]] = {}
        next_region_id = 0
        for actual_region_id in range(len(self.block_len_per_layer)):
            mapping[actual_region_id] = [next_region_id]
            next_region_id += 1
            if (
                self.transfer_topo is not None
                and self.transfer_topo.virtually_split_kv_in_blocks
                and not self._is_region_replicated(actual_region_id)
            ):
                mapping[actual_region_id].append(next_region_id)
                next_region_id += 1
        return mapping

    def _build_layer_region_ids_from_kv_caches(
        self, kv_caches: dict[str, Any]
    ) -> dict[str, list[int]]:
        assert self.transfer_topo is not None
        xfer_buffers = self.host_xfer_buffers if self.use_host_buffer else kv_caches
        base_addrs = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        base_addr_to_actual_region = {
            base_addr: region_id for region_id, base_addr in enumerate(base_addrs)
        }
        actual_to_virtual = self._actual_to_virtual_region_ids()

        layer_region_ids: dict[str, list[int]] = defaultdict(list)
        for layer_name, cache_or_caches in xfer_buffers.items():
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                continue
            if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                layer_spec = layer_spec.kv_cache_specs[layer_name]
            cache_list = self.transfer_topo.get_transfer_cache_regions(
                cache_or_caches, layer_spec
            )
            for cache in cache_list:
                actual_region_id = base_addr_to_actual_region.get(cache.data_ptr())
                if actual_region_id is None:
                    continue
                for region_id in actual_to_virtual[actual_region_id]:
                    if region_id not in layer_region_ids[layer_name]:
                        layer_region_ids[layer_name].append(region_id)
        return dict(layer_region_ids)

    def _build_region_ids_by_group(
        self, layer_region_ids: dict[str, list[int]]
    ) -> list[np.ndarray]:
        region_ids_by_group: list[np.ndarray] = []
        for group in self.kv_cache_config.kv_cache_groups:
            group_region_ids: list[int] = []
            for layer_name in group.layer_names:
                group_region_ids.extend(layer_region_ids.get(layer_name, []))
            region_ids_by_group.append(
                np.asarray(list(dict.fromkeys(group_region_ids)), dtype=np.int64)
            )
        if not region_ids_by_group:
            region_ids_by_group = [np.arange(self.num_regions, dtype=np.int64)]
        return region_ids_by_group

    @staticmethod
    def _filter_group_region_ids(
        group_region_ids: list[list[int]] | list[np.ndarray],
        selected_region_ids: list[int],
        *,
        compact: bool,
    ) -> list[np.ndarray]:
        selected = set(selected_region_ids)
        compact_map = {
            region_id: compact_id
            for compact_id, region_id in enumerate(selected_region_ids)
        }
        result: list[np.ndarray] = []
        for group in group_region_ids:
            if compact:
                ids = [compact_map[int(r)] for r in group if int(r) in compact_map]
            else:
                ids = [int(r) for r in group if int(r) in selected]
            result.append(np.asarray(ids, dtype=np.int64))
        return result

    def _block_len_for_virtual_region(self, region_id: int) -> int:
        next_region_id = 0
        for actual_region_id, block_len in enumerate(self.block_len_per_layer):
            if region_id == next_region_id:
                if (
                    self.transfer_topo is not None
                    and self.transfer_topo.virtually_split_kv_in_blocks
                    and not self._is_region_replicated(actual_region_id)
                ):
                    return block_len // 2
                return block_len
            next_region_id += 1
            if (
                self.transfer_topo is not None
                and self.transfer_topo.virtually_split_kv_in_blocks
                and not self._is_region_replicated(actual_region_id)
            ):
                if region_id == next_region_id:
                    return block_len // 2
                next_region_id += 1
        return self.block_len_per_layer[region_id]

    def _build_hma_transfer_regions(
        self,
        layer_region_ids: dict[str, list[int]],
        region_ids_by_group: list[np.ndarray],
    ) -> list[IluNixlTransferRegion]:
        region_to_layers: dict[int, list[str]] = defaultdict(list)
        for layer_name, region_ids in layer_region_ids.items():
            for region_id in region_ids:
                region_to_layers[int(region_id)].append(layer_name)

        region_to_groups: dict[int, list[int]] = defaultdict(list)
        for group_idx, group_region_ids in enumerate(region_ids_by_group):
            for region_id in group_region_ids:
                region_to_groups[int(region_id)].append(group_idx)

        layer_to_groups: dict[str, list[int]] = defaultdict(list)
        for group_idx, group in enumerate(self.kv_cache_config.kv_cache_groups):
            for layer_name in group.layer_names:
                layer_to_groups[layer_name].append(group_idx)

        transfer_regions: list[IluNixlTransferRegion] = []
        for region_id in sorted(region_to_layers):
            shared_by = sorted(set(region_to_layers[region_id]))
            layer_indices: list[int] = []
            cache_kinds: set[str] = set()
            identities: set[str] = set()
            for layer_name in shared_by:
                with suppress(AssertionError, ValueError):
                    layer_indices.append(extract_layer_index(layer_name))
                cache_kinds.add(self._cache_kind_from_layer_name(layer_name))
                for group_idx in layer_to_groups.get(layer_name, []):
                    identities.add(self._transfer_identity(layer_name, group_idx))
            transfer_regions.append(
                IluNixlTransferRegion(
                    region_id=region_id,
                    block_len=self._block_len_for_virtual_region(region_id),
                    shared_by=shared_by,
                    layer_indices=sorted(set(layer_indices)),
                    group_ids=sorted(set(region_to_groups.get(region_id, []))),
                    cache_kinds=sorted(cache_kinds),
                    identities=sorted(identities),
                )
            )
        return transfer_regions

    def register_kv_caches(self, kv_caches: dict[str, Any]):
        if self._mask_pp_during_register:
            original_pp_size = self.pp_size
            # vLLM 0.24's generic NIXL register path assumes uniform regions per
            # PP layer slice. HMA violates that assumption, and IluNixl builds
            # semantic PP/HMA region metadata below instead.
            self.pp_size = 1
            try:
                super().register_kv_caches(kv_caches)
            finally:
                self.pp_size = original_pp_size
        else:
            super().register_kv_caches(kv_caches)

        self.layer_region_ids = self._build_layer_region_ids_from_kv_caches(kv_caches)
        self.region_ids_by_group = self._build_region_ids_by_group(
            self.layer_region_ids
        )
        self.transfer_regions = self._build_hma_transfer_regions(
            self.layer_region_ids, self.region_ids_by_group
        )

        agent_metadata = IluNixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=(
                self.kv_caches_base_addr[self.engine_id][self.tp_rank]
            ),
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            kv_cache_layout=self.kv_cache_layout
            if not self.use_host_buffer
            else self.host_buffer_kv_cache_layout,
            block_size=self.block_size,
            ssm_sizes=self._mamba_ssm_size,
            attn_backend_name=self.backend_name,
            physical_blocks_per_logical_kv_block=(
                self._physical_blocks_per_logical_kv_block
            ),
            pp_split=self.pp_split,
            pp_rank=self.pp_rank,
            tp_rank=self.tp_rank,
            layer_region_ids=self.layer_region_ids,
            region_ids_by_group=[
                region_ids.tolist() for region_ids in self.region_ids_by_group
            ],
            transfer_regions=self.transfer_regions,
        )
        assert self.compat_hash is not None
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )

    @classmethod
    def _align_hma_transfer_regions(
        cls,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
        local_transfer_regions: list[IluNixlTransferRegion],
        remote_transfer_regions: list[IluNixlTransferRegion],
        num_groups: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]:
        local_regions = cls._regions_by_id(local_transfer_regions)
        remote_regions = cls._regions_by_id(remote_transfer_regions)

        selected_local = [
            local_regions[region_id]
            for region_id in selected_local_region_ids
            if region_id in local_regions
        ]
        selected_remote = [
            remote_regions[region_id]
            for region_id in selected_remote_region_ids
            if region_id in remote_regions
        ]

        local_by_identity: dict[str, IluNixlTransferRegion] = {}
        for region in selected_local:
            for identity in region.identities:
                local_by_identity[identity] = region

        remote_by_identity: dict[str, IluNixlTransferRegion] = {}
        for region in selected_remote:
            for identity in region.identities:
                remote_by_identity[identity] = region

        def identity_sort_key(identity: str) -> tuple[int, str, int]:
            layer_idx, kind, group = identity.split(":", 2)
            return (int(layer_idx), kind, int(group.removeprefix("group")))

        common_identities = sorted(
            set(local_by_identity).intersection(remote_by_identity),
            key=identity_sort_key,
        )
        if not common_identities:
            raise RuntimeError(
                "Heterogeneous PP HMA semantic alignment found no shared "
                "cache identities between local and remote selected regions. "
                f"local_regions={selected_local_region_ids}, "
                f"remote_regions={selected_remote_region_ids}"
            )

        group_pairs: list[list[tuple[int, int, str]]] = [[] for _ in range(num_groups)]
        for identity in common_identities:
            local_region = local_by_identity[identity]
            remote_region = remote_by_identity[identity]
            group_idx = int(identity.rsplit("group", 1)[1])
            if group_idx >= num_groups:
                continue
            if local_region.block_len != remote_region.block_len:
                raise RuntimeError(
                    "Heterogeneous PP HMA semantic alignment found a KV "
                    "region block_len mismatch for identity "
                    f"{identity}: local_region={local_region.region_id} "
                    f"local_block_len={local_region.block_len}, "
                    f"remote_region={remote_region.region_id} "
                    f"remote_block_len={remote_region.block_len}."
                )
            group_pairs[group_idx].append(
                (local_region.region_id, remote_region.region_id, identity)
            )

        conflicts: list[dict[str, Any]] = []
        local_group_region_ids: list[np.ndarray] = []
        remote_group_region_ids: list[np.ndarray] = []
        selected_local_ids: list[int] = []
        selected_remote_ids: list[int] = []

        for group_idx, pairs in enumerate(group_pairs):
            local_to_remote: dict[int, int] = {}
            remote_to_local: dict[int, int] = {}
            local_group: list[int] = []
            remote_group: list[int] = []
            for local_region_id, remote_region_id, identity in pairs:
                prior_remote = local_to_remote.get(local_region_id)
                prior_local = remote_to_local.get(remote_region_id)
                if prior_remote is not None and prior_remote != remote_region_id:
                    conflicts.append(
                        {
                            "group": group_idx,
                            "local_region": local_region_id,
                            "remote_regions": sorted({prior_remote, remote_region_id}),
                            "identity": identity,
                        }
                    )
                    continue
                if prior_local is not None and prior_local != local_region_id:
                    conflicts.append(
                        {
                            "group": group_idx,
                            "remote_region": remote_region_id,
                            "local_regions": sorted({prior_local, local_region_id}),
                            "identity": identity,
                        }
                    )
                    continue
                local_to_remote[local_region_id] = remote_region_id
                remote_to_local[remote_region_id] = local_region_id
                if local_region_id not in local_group:
                    local_group.append(local_region_id)
                if remote_region_id not in remote_group:
                    remote_group.append(remote_region_id)
                if local_region_id not in selected_local_ids:
                    selected_local_ids.append(local_region_id)
                if remote_region_id not in selected_remote_ids:
                    selected_remote_ids.append(remote_region_id)

            local_group_region_ids.append(np.asarray(local_group, dtype=np.int64))
            remote_group_region_ids.append(np.asarray(remote_group, dtype=np.int64))

        if conflicts:
            raise RuntimeError(
                "Heterogeneous PP HMA needs slice/scatter transfer inside a "
                "single KV group because one semantic owner maps to multiple "
                f"physical regions. conflicts={conflicts[:8]}"
            )

        compact_remote_map = {
            region_id: compact_id
            for compact_id, region_id in enumerate(selected_remote_ids)
        }
        remote_group_region_ids = [
            np.asarray(
                [compact_remote_map[int(region_id)] for region_id in group],
                dtype=np.int64,
            )
            for group in remote_group_region_ids
        ]
        return (
            local_group_region_ids,
            remote_group_region_ids,
            selected_local_ids,
            selected_remote_ids,
        )

    def _validate_pp_selected_region_lengths(
        self,
        nixl_agent_meta: IluNixlAgentMetadata,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
        remote_tp_size: int,
    ) -> None:
        assert self.transfer_topo is not None
        assert len(selected_local_region_ids) == len(selected_remote_region_ids), (
            "Heterogeneous PP selected region count mismatch: "
            f"local={len(selected_local_region_ids)}, "
            f"remote={len(selected_remote_region_ids)}"
        )
        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )
        model_replicated = self.use_mla or self.transfer_topo.is_kv_replicated(
            nixl_agent_meta.engine_id
        )
        for pair_idx, (local_region_id, remote_region_id) in enumerate(
            zip(selected_local_region_ids, selected_remote_region_ids)
        ):
            local_len = self.block_len_per_layer[local_region_id]
            remote_len = nixl_agent_meta.block_lens[remote_region_id]
            replicated = model_replicated or self._is_region_replicated(local_region_id)
            if replicated:
                expected_remote_len = local_len // block_size_ratio
            elif tp_ratio > 0:
                expected_remote_len = (local_len * tp_ratio) // block_size_ratio
            else:
                assert block_size_ratio == 1, (
                    "Different local/remote block sizes are not supported "
                    "when P TP > D TP."
                )
                expected_remote_len = local_len // (-tp_ratio)
            assert remote_len == expected_remote_len, (
                "Heterogeneous PP selected region size mismatch at "
                f"pair={pair_idx}: local_region={local_region_id}, "
                f"remote_region={remote_region_id}, local={local_len}, "
                f"remote={remote_len}, expected_remote={expected_remote_len}, "
                f"tp_ratio={tp_ratio}, block_size_ratio={block_size_ratio}"
            )
        assert self.dst_num_blocks[nixl_agent_meta.engine_id] == (
            nixl_agent_meta.num_blocks
        )

    def _validate_pp_hma_region_lengths(
        self,
        nixl_agent_meta: IluNixlAgentMetadata,
        local_region_ids_by_group: list[np.ndarray],
        remote_region_ids_by_group: list[np.ndarray],
        selected_remote_region_ids: list[int],
    ) -> None:
        assert self.transfer_topo is not None
        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )
        assert block_size_ratio == 1, (
            "HMA does not support different remote block size yet"
        )
        assert len(local_region_ids_by_group) == len(remote_region_ids_by_group), (
            "KV cache HMA group count must match after semantic alignment: "
            f"local={len(local_region_ids_by_group)}, "
            f"remote={len(remote_region_ids_by_group)}"
        )
        for group_idx, (local_group, remote_group) in enumerate(
            zip(local_region_ids_by_group, remote_region_ids_by_group)
        ):
            if len(local_group) != len(remote_group):
                raise AssertionError(
                    "KV cache selected region count must match within each "
                    "HMA group after semantic alignment: "
                    f"group={group_idx}, local={len(local_group)}, "
                    f"remote={len(remote_group)}"
                )
            for pair_idx, (local_region_id, remote_compact_id) in enumerate(
                zip(local_group, remote_group)
            ):
                remote_region_id = selected_remote_region_ids[int(remote_compact_id)]
                local_block_len = self.block_len_per_layer[int(local_region_id)]
                remote_block_len = nixl_agent_meta.block_lens[int(remote_region_id)]
                assert local_block_len == remote_block_len, (
                    "KV cache selected region size must match between P and D "
                    "when using HMA with heterogeneous PP at "
                    f"group={group_idx}, pair={pair_idx}: "
                    f"local_region={int(local_region_id)} "
                    f"local={local_block_len}, "
                    f"remote_region={remote_region_id} "
                    f"remote={remote_block_len}"
                )
        assert self.dst_num_blocks[nixl_agent_meta.engine_id] == (
            nixl_agent_meta.num_blocks
        )

    def _select_pp_regions(
        self,
        nixl_agent_meta: IluNixlAgentMetadata,
        remote_pp_rank: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]:
        local_start, local_end = self._pp_layer_range(self.pp_split, self.pp_rank)
        remote_start, remote_end = self._pp_layer_range(
            nixl_agent_meta.pp_split, remote_pp_rank
        )
        overlap_start = max(local_start, remote_start)
        overlap_end = min(local_end, remote_end)
        if overlap_start >= overlap_end:
            raise RuntimeError(
                "Heterogeneous PP handshake selected a remote PP stage "
                "with no overlapping layers: "
                f"local=[{local_start}, {local_end}), "
                f"remote=[{remote_start}, {remote_end})."
            )

        selected_local_region_ids = self._region_ids_for_layer_range(
            self.layer_region_ids, overlap_start, overlap_end
        )
        assert nixl_agent_meta.layer_region_ids is not None
        selected_remote_region_ids = self._region_ids_for_layer_range(
            nixl_agent_meta.layer_region_ids, overlap_start, overlap_end
        )

        if self._is_hma_required:
            assert nixl_agent_meta.transfer_regions is not None
            return self._align_hma_transfer_regions(
                selected_local_region_ids,
                selected_remote_region_ids,
                self.transfer_regions,
                nixl_agent_meta.transfer_regions,
                len(self.region_ids_by_group),
            )

        assert nixl_agent_meta.region_ids_by_group is not None
        return (
            self._filter_group_region_ids(
                self.region_ids_by_group,
                selected_local_region_ids,
                compact=False,
            ),
            self._filter_group_region_ids(
                nixl_agent_meta.region_ids_by_group,
                selected_remote_region_ids,
                compact=True,
            ),
            selected_local_region_ids,
            selected_remote_region_ids,
        )

    def _remote_transfer_regions(
        self,
        engine_id: str,
        nixl_agent_meta: IluNixlAgentMetadata,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
    ) -> list[tuple[int, int, int]]:
        """Return (remote_region_id, transfer_len, rank_offset)."""
        if self._is_hma_required:
            return [
                (region_id, nixl_agent_meta.block_lens[region_id], 0)
                for region_id in selected_remote_region_ids
            ]

        assert self.transfer_topo is not None
        fa_group_idx = next(
            i for i, t in enumerate(self._group_spec_types) if _is_attention_spec(t)
        )
        plan = self.tp_mappings[engine_id]
        split_reads = len(plan.source_ranks_per_group[fa_group_idx])
        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )

        regions: list[tuple[int, int, int]] = []
        for local_region_id, remote_region_id in zip(
            selected_local_region_ids, selected_remote_region_ids
        ):
            replicated = self._is_region_replicated(local_region_id)
            local_block_len = self.get_backend_aware_kv_block_len(
                layer_idx=local_region_id,
                first_split=True,
                mamba_view=False,
            )
            remote_kv_block_len = local_block_len // block_size_ratio
            if block_size_ratio > 1:
                local_block_len = remote_kv_block_len

            num_reads = 1 if replicated else split_reads
            rank_offset = (
                0 if replicated else plan.rank_offset_factor * remote_kv_block_len
            )
            regions.append(
                (remote_region_id, local_block_len // num_reads, rank_offset)
            )
        return regions

    def _register_pp_remote_xfer_handle(
        self,
        engine_id: str,
        remote_key: int,
        remote_agent_name: str,
        nixl_agent_meta: IluNixlAgentMetadata,
        selected_local_region_ids: list[int],
        selected_remote_region_ids: list[int],
    ) -> None:
        blocks_data = []
        for region_id, transfer_len, rank_offset in self._remote_transfer_regions(
            engine_id,
            nixl_agent_meta,
            selected_local_region_ids,
            selected_remote_region_ids,
        ):
            base_addr = nixl_agent_meta.kv_caches_base_addr[region_id]
            page_size = nixl_agent_meta.block_lens[region_id]
            for block_id in range(nixl_agent_meta.num_blocks):
                blocks_data.append(
                    (
                        base_addr + block_id * page_size + rank_offset,
                        transfer_len,
                        nixl_agent_meta.device_id,
                    )
                )

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        self.dst_xfer_side_handles[engine_id][remote_key] = (
            self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        )

    def _ensure_pp_local_split_handles(
        self, engine_id: str, remote_tp_size: int
    ) -> None:
        assert self.transfer_topo is not None
        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        if (
            tp_ratio >= 0
            or self.use_mla
            or tp_ratio in self.src_xfer_handles_by_tp_ratio
        ):
            return

        self.src_xfer_handles_by_tp_ratio[tp_ratio] = []
        for handle_data in self._build_local_splits_from_plan(
            self.tp_mappings[engine_id],
            self.src_blocks_data,
            self.num_descs,
        ):
            local_descs = self.nixl_wrapper.get_xfer_descs(
                handle_data, self.nixl_memory_type
            )
            self.src_xfer_handles_by_tp_ratio[tp_ratio].append(
                self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", local_descs)
            )

    def add_remote_agent(
        self,
        nixl_agent_meta: IluNixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
        remote_pp_rank: int = 0,
        remote_pp_size: int = 1,
    ) -> str:
        if remote_pp_size == self.pp_world_size:
            return super().add_remote_agent(
                nixl_agent_meta, remote_tp_rank, remote_tp_size
            )

        if (
            nixl_agent_meta.layer_region_ids is None
            or nixl_agent_meta.region_ids_by_group is None
        ):
            raise RuntimeError(
                "Heterogeneous PP requires layer_region_ids and "
                "region_ids_by_group in the "
                "IluNixl handshake metadata."
            )
        if self._is_hma_required and nixl_agent_meta.transfer_regions is None:
            raise RuntimeError(
                "Heterogeneous PP with HMA requires transfer_regions in the "
                "IluNixl handshake metadata."
            )
        if self._is_hma_required and not self.use_mla:
            raise RuntimeError(
                "IluNixl heterogeneous PP HMA currently supports the "
                "DeepSeekV4/MLA replicated KV path only."
            )

        engine_id = nixl_agent_meta.engine_id
        remote_key = remote_pp_rank * remote_tp_size + remote_tp_rank
        if remote_key in self._remote_agents.get(engine_id, {}):
            return self._remote_agents[engine_id][remote_key]

        assert self.transfer_topo is not None
        physical_blocks_per_logical = (
            nixl_agent_meta.physical_blocks_per_logical_kv_block
        )
        transfer_info = EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=nixl_agent_meta.block_size,
            remote_block_len=nixl_agent_meta.block_lens[0],
            remote_physical_blocks_per_logical=physical_blocks_per_logical,
        )
        self.transfer_topo.register_remote_engine(engine_id, transfer_info)
        self.tp_mappings[engine_id] = compute_tp_mapping(
            transfer_topology=self.transfer_topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
        )
        self._remote_pp_size_by_engine[engine_id] = remote_pp_size
        self._remote_pp_split_by_engine[engine_id] = nixl_agent_meta.pp_split

        (
            local_region_ids_by_group,
            remote_region_ids_by_group,
            selected_local_region_ids,
            selected_remote_region_ids,
        ) = self._select_pp_regions(nixl_agent_meta, remote_pp_rank)

        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        if self._is_hma_required:
            self._validate_pp_hma_region_lengths(
                nixl_agent_meta,
                local_region_ids_by_group,
                remote_region_ids_by_group,
                selected_remote_region_ids,
            )
        else:
            self._validate_pp_selected_region_lengths(
                nixl_agent_meta,
                selected_local_region_ids,
                selected_remote_region_ids,
                remote_tp_size,
            )
        self.local_region_ids_by_group.setdefault(engine_id, {})[remote_key] = (
            local_region_ids_by_group
        )
        self.remote_region_ids_by_group.setdefault(engine_id, {})[remote_key] = (
            remote_region_ids_by_group
        )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )
        self.kv_caches_base_addr[engine_id][remote_key] = [
            nixl_agent_meta.kv_caches_base_addr[i] for i in selected_remote_region_ids
        ]

        self._register_pp_remote_xfer_handle(
            engine_id,
            remote_key,
            remote_agent_name,
            nixl_agent_meta,
            selected_local_region_ids,
            selected_remote_region_ids,
        )
        self._ensure_pp_local_split_handles(engine_id, remote_tp_size)
        return remote_agent_name

    def _make_pp_aware_notif_id(
        self,
        remote_request_id: str,
        dst_engine_id: str,
        remote_key: int,
    ) -> bytes:
        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        remote_pp_rank = remote_key // remote_info.remote_tp_size
        remote_pp_split = self._remote_pp_split_by_engine[dst_engine_id]
        pp_consumers = self._overlap_pp_count(
            remote_pp_rank,
            remote_pp_split,
            self.pp_split,
        )
        return f"{remote_request_id}:{self.world_size}:{pp_consumers}".encode()

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        remote_pp_size: int,
        remote_pp_split: str,
        expected_engine_id: str,
    ) -> dict[int, str]:
        if not self.use_host_buffer:
            current_platform.set_device(self.device_id)

        assert self.transfer_topo is not None
        p_remote_ranks = self.transfer_topo.handshake_target_ranks(remote_tp_size)
        if remote_pp_split:
            remote_pp_ranks = self._target_pp_ranks(
                self.pp_split, self.pp_rank, remote_pp_split
            )
        elif remote_pp_size == self.pp_world_size:
            remote_pp_ranks = [self.pp_rank]
        else:
            raise RuntimeError(
                "IluNixl heterogeneous PP handshake requires remote "
                "pp_split in kv_transfer_params."
            )
        remote_rank_to_agent_name = {}
        path = make_zmq_path("tcp", host, port)

        with zmq_ctx(zmq.REQ, path) as sock:
            for remote_pp_rank in remote_pp_ranks:
                for remote_rank in p_remote_ranks:
                    target_rank = remote_pp_rank * remote_tp_size + remote_rank
                    logger.debug(
                        "Querying metadata on path: %s at remote pp rank %s, "
                        "tp rank %s (global rank %s)",
                        path,
                        remote_pp_rank,
                        remote_rank,
                        target_rank,
                    )

                    start_time = time.perf_counter()
                    msg = msgspec.msgpack.encode((GET_META_MSG, target_rank))
                    sock.setsockopt(zmq.RCVTIMEO, 5000)
                    sock.send(msg)
                    handshake_bytes = sock.recv()

                    handshake_decoder = msgspec.msgpack.Decoder(NixlHandshakePayload)
                    try:
                        handshake_payload = handshake_decoder.decode(handshake_bytes)
                    except (msgspec.DecodeError, msgspec.ValidationError) as e:
                        raise RuntimeError(
                            "Failed to decode NixlHandshakePayload. This likely "
                            "indicates an incompatible connector version. "
                            f"Error: {e}"
                        ) from e

                    got_metadata_time = time.perf_counter()
                    logger.debug(
                        "NIXL handshake: get metadata took: %s",
                        got_metadata_time - start_time,
                    )

                    assert self.compat_hash is not None
                    if (
                        self.enforce_compat_hash
                        and handshake_payload.compatibility_hash != self.compat_hash
                    ):
                        raise RuntimeError(
                            "NIXL compatibility hash mismatch. "
                            f"Local: {self.compat_hash}, "
                            f"Remote: {handshake_payload.compatibility_hash}."
                        )

                    metadata_decoder = msgspec.msgpack.Decoder(IluNixlAgentMetadata)
                    try:
                        metadata = metadata_decoder.decode(
                            handshake_payload.agent_metadata_bytes
                        )
                    except (msgspec.DecodeError, msgspec.ValidationError) as e:
                        raise RuntimeError(
                            f"Failed to decode IluNixlAgentMetadata. Error: {e}"
                        ) from e

                    if metadata.engine_id != expected_engine_id:
                        raise RuntimeError(
                            "Remote NIXL agent engine ID mismatch. "
                            f"Expected {expected_engine_id}, "
                            f"received {metadata.engine_id}."
                        )

                    remote_agent_name = self.add_remote_agent(
                        metadata,
                        remote_rank,
                        remote_tp_size,
                        remote_pp_rank,
                        remote_pp_size,
                    )
                    logger.debug(
                        "NIXL handshake: add agent took: %s",
                        time.perf_counter() - got_metadata_time,
                    )
                    remote_key = (
                        target_rank
                        if remote_pp_size != self.pp_world_size
                        else remote_rank
                    )
                    remote_rank_to_agent_name[remote_key] = remote_agent_name

        return remote_rank_to_agent_name

    def _ensure_handshake(
        self,
        engine_id,
        host: str,
        port: int,
        tp_size: int,
        pp_size: int = 1,
        pp_split: str = "",
    ) -> Future[dict[int, str]] | None:
        self._evict_stale_engines()
        with self._handshake_lock:
            if engine_id in self._remote_agents:
                return None
            fut = self._handshake_futures.get(engine_id)
            if fut is not None:
                return fut
            fut = self._handshake_initiation_executor.submit(
                self._nixl_handshake,
                host,
                port,
                tp_size,
                pp_size,
                pp_split,
                engine_id,
            )
            self._handshake_futures[engine_id] = fut

            def done_callback(f: Future[dict[int, str]], eid=engine_id):
                with self._handshake_lock:
                    del self._handshake_futures[eid]
                    try:
                        self._remote_agents[eid] = f.result()
                        self._engine_last_active[eid] = time.perf_counter()
                    except Exception as e:
                        self._log_failure(
                            failure_type="handshake_setup_failed",
                            req_id=None,
                            error=e,
                            remote_engine_id=eid,
                        )

            fut.add_done_callback(done_callback)
            return fut

    def _background_nixl_handshake(self, req_id: str, remote_engine_id, meta):
        assert meta.remote is not None
        fut = self._ensure_handshake(
            remote_engine_id,
            meta.remote.host,
            meta.remote.port,
            meta.tp_size,
            getattr(meta, "remote_pp_size", 1),
            getattr(meta, "remote_pp_split", ""),
        )
        if fut is None:
            self._ready_requests.put((req_id, meta))
            return

        def request_ready(f: Future[Any], entry=(req_id, meta)):
            try:
                f.result()
                self._ready_requests.put(entry)
            except Exception as e:
                self._log_failure(
                    failure_type="handshake_failed",
                    req_id=req_id,
                    error=e,
                    meta=meta,
                )
                self._handle_failed_transfer(req_id, None)

        fut.add_done_callback(request_ready)

    def _is_heterogeneous_pp_engine(self, engine_id: str) -> bool:
        return (
            self._remote_pp_size_by_engine.get(engine_id, self.pp_world_size)
            != self.pp_world_size
        )

    def _compute_desc_ids_for_regions(
        self,
        block_ids: list[int],
        region_ids: np.ndarray,
        num_blocks: int,
        block_size_ratio: float | None = None,
    ) -> np.ndarray:
        if block_size_ratio is not None:
            num_blocks = int(num_blocks * block_size_ratio)
        if len(block_ids) == 0 or len(region_ids) == 0:
            return np.asarray([], dtype=np.int64)
        return (
            region_ids[:, None] * num_blocks
            + np.asarray(block_ids, dtype=np.int64)[None, :]
        ).flatten()

    def _read_blocks_for_req(self, req_id: str, meta):
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        if not self._is_heterogeneous_pp_engine(engine_id):
            return super()._read_blocks_for_req(req_id, meta)

        self._engine_last_active[engine_id] = time.perf_counter()
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        num_groups = len(local_block_ids)

        remote_keys = sorted(self._remote_agents[engine_id])
        for remote_key in remote_keys:
            remote_tp_rank = remote_key % remote_info.remote_tp_size
            read_spec = ReadSpec(
                remote_rank=remote_key,
                local_block_ids=[
                    list(local_block_ids[g])
                    if remote_tp_rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
                remote_block_ids=[
                    list(remote_block_ids[g])
                    if remote_tp_rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
            )
            if not any(read_spec.local_block_ids):
                if self.use_mla:
                    notif_id = self._make_pp_aware_notif_id(
                        meta.remote.request_id, engine_id, remote_key
                    )
                    self.nixl_wrapper.send_notif(
                        self._remote_agents[engine_id][remote_key],
                        notif_msg=notif_id,
                    )
                continue

            if tp_ratio < 0 and not self.use_mla:
                split_idx = plan.all_source_ranks.index(remote_tp_rank)
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[tp_ratio][
                    split_idx
                ]
            else:
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_info.remote_block_size
                ]
            remote_xfer_side_handle = self.dst_xfer_side_handles[engine_id][remote_key]
            self._read_blocks(
                read_spec=read_spec,
                request_id=req_id,
                dst_engine_id=engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
            )

    def _read_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ):
        if not self._is_heterogeneous_pp_engine(dst_engine_id):
            return super()._read_blocks(
                read_spec,
                dst_engine_id,
                request_id,
                remote_request_id,
                local_xfer_side_handle,
                remote_xfer_side_handle,
            )

        assert self.transfer_topo is not None
        remote_key = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids
        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )

        notif_id = self._make_pp_aware_notif_id(
            remote_request_id, dst_engine_id, remote_key
        )
        if len(local_block_ids) == 0 or not any(local_block_ids):
            agent_name = self._remote_agents[dst_engine_id][remote_key]
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            return

        local_block_ids, remote_block_ids = self._apply_prefix_caching(
            local_block_ids,
            remote_block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )

        local_region_ids_by_group = self.local_region_ids_by_group[dst_engine_id][
            remote_key
        ]
        remote_region_ids_by_group = self.remote_region_ids_by_group[dst_engine_id][
            remote_key
        ]

        local_descs_list: list[np.ndarray] = []
        remote_descs_list: list[np.ndarray] = []
        for group_idx, (group_local_blocks, group_remote_blocks) in enumerate(
            zip(local_block_ids, remote_block_ids)
        ):
            if not group_local_blocks:
                continue
            local_descs_list.append(
                self._compute_desc_ids_for_regions(
                    group_local_blocks,
                    local_region_ids_by_group[group_idx],
                    self.dst_num_blocks[self.engine_id],
                    block_size_ratio,
                )
            )
            remote_descs_list.append(
                self._compute_desc_ids_for_regions(
                    group_remote_blocks,
                    remote_region_ids_by_group[group_idx],
                    self.dst_num_blocks[dst_engine_id],
                    None,
                )
            )

        if not local_descs_list:
            agent_name = self._remote_agents[dst_engine_id][remote_key]
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            return

        local_block_descs_ids = np.concatenate(local_descs_list)
        remote_block_descs_ids = np.concatenate(remote_descs_list)
        assert len(local_block_descs_ids) == len(remote_block_descs_ids), (
            "Heterogeneous PP HMA descriptor count mismatch: "
            f"local={len(local_block_descs_ids)}, "
            f"remote={len(remote_block_descs_ids)}"
        )

        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )
            self.nixl_wrapper.transfer(handle)
            self._recving_transfers[request_id].append(handle)
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Marking blocks as invalid",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_key,
            )
            self._handle_failed_transfer(request_id, handle)

    def _send_heartbeats(self, metadata) -> None:
        for engine_id, hb_info in metadata.heartbeat_by_engine.items():
            if (
                self._ensure_handshake(
                    engine_id,
                    hb_info.host,
                    hb_info.port,
                    hb_info.tp_size,
                    getattr(hb_info, "pp_size", 1),
                    getattr(hb_info, "pp_split", ""),
                )
                is not None
            ):
                continue

            hb_msg = ("HB:" + ",".join(hb_info.req_ids)).encode()
            for agent_name in self._remote_agents[engine_id].values():
                try:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=hb_msg)
                except Exception:
                    logger.debug(
                        "Failed to send heartbeat to engine %s",
                        engine_id,
                        exc_info=True,
                    )
