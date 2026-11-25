# SPDX-License-Identifier: Apache-2.0
"""External-store cache-hit coordinator for Iluvatar MooncakeStoreConnector."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, cast

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from .layout import chunk_hashes_for_block_size


class StoreSpecGroup(NamedTuple):
    spec: KVCacheSpec
    group_ids: list[int]
    manager_cls: type[SingleTypeKVCacheManager]
    use_eagle: bool


class ExternalCachedBlockPool:
    """Duck-typed BlockPool backed by a ``(group_id, hash_bytes)`` exists set."""

    def __init__(
        self,
        hash_block_size: int,
        exists: set[tuple[int, bytes]] | None = None,
    ) -> None:
        self._exists = exists
        self.hash_block_size = hash_block_size
        self.null_block = KVCacheBlock(block_id=0)
        self._present_block = KVCacheBlock(block_id=1)

    def get_cached_block(
        self,
        block_hash: BlockHash,
        group_ids: list[int],
    ) -> list[KVCacheBlock] | None:
        if self._exists is None:
            return [self._present_block] * len(group_ids)
        h = bytes(block_hash)
        if all((g, h) in self._exists for g in group_ids):
            return [self._present_block] * len(group_ids)
        return None

    def contains(self, group_id: int, block_hash: BlockHash) -> bool:
        return (
            self._exists is not None and (group_id, bytes(block_hash)) in self._exists
        )


def _unwrap_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        if spec.kv_cache_specs:
            return next(iter(spec.kv_cache_specs.values()))
    return spec


def partial_hash_hits_enabled(
    kv_cache_groups: Sequence[KVCacheGroupSpec],
    hash_block_size: int,
    dcp_world_size: int = 1,
) -> bool:
    """Match vLLM core's DCP-aware Mamba partial-hit condition."""
    return any(
        isinstance(spec := _unwrap_spec(g.kv_cache_spec), MambaSpec)
        and spec.mamba_cache_mode == "align"
        and (
            (dcp_world_size == 1 and spec.block_size > hash_block_size)
            or (dcp_world_size > 1 and spec.block_size >= hash_block_size)
        )
        for g in kv_cache_groups
    )


class MooncakeStoreCoordinator:
    """Mirror of ``HybridKVCacheCoordinator.find_longest_cache_hit`` over
    ``ExternalCachedBlockPool``."""

    def __init__(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        scheduler_block_size: int,
        hash_block_size: int,
        use_eagle: bool = False,
        retention_interval: int | None = None,
        dcp_world_size: int = 1,
    ) -> None:
        assert all(
            g.kv_cache_spec.block_size % hash_block_size == 0
            for g in kv_cache_groups
            if g.kv_cache_spec.prefix_cacheable
        ), "block_size must be divisible by hash_block_size"
        assert scheduler_block_size % hash_block_size == 0, (
            f"scheduler_block_size ({scheduler_block_size}) must be a multiple of "
            f"hash_block_size ({hash_block_size})"
        )
        assert all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_groups
            if g.kv_cache_spec.prefix_cacheable
        ), "scheduler_block_size must be a multiple of each group's block_size"
        self.kv_cache_groups = kv_cache_groups
        self.mamba_group_ids = {
            group_id
            for group_id, group in enumerate(kv_cache_groups)
            if isinstance(_unwrap_spec(group.kv_cache_spec), MambaSpec)
        }
        self.hash_block_size = hash_block_size
        self.lcm_block_size = scheduler_block_size
        self.enable_partial_hash_hits = partial_hash_hits_enabled(
            kv_cache_groups, hash_block_size, dcp_world_size
        )
        self.use_eagle = use_eagle
        self.retention_interval = retention_interval
        self._verify_and_split_kv_cache_groups()

    def align_lookup_length(self, length: int) -> int:
        alignment = (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.lcm_block_size
        )
        return length // alignment * alignment

    def _verify_and_split_kv_cache_groups(self) -> None:
        attention_groups: list[StoreSpecGroup] = []
        for i, g in enumerate(self.kv_cache_groups):
            if not g.kv_cache_spec.prefix_cacheable:
                continue
            spec = _unwrap_spec(g.kv_cache_spec)
            manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
            assert manager_cls is not None, (
                f"No manager registered for KVCacheSpec {spec}"
            )
            for idx, group in enumerate(attention_groups):
                if group.spec == spec:
                    assert manager_cls is group.manager_cls
                    group.group_ids.append(i)
                    if g.is_eagle_group and not group.use_eagle:
                        attention_groups[idx] = group._replace(use_eagle=True)
                    break
            else:
                attention_groups.append(
                    StoreSpecGroup(spec, [i], manager_cls, g.is_eagle_group)
                )
        attention_groups.sort(key=lambda g: not isinstance(g.spec, FullAttentionSpec))
        if self.use_eagle and not any(g.use_eagle for g in attention_groups):
            attention_groups = [g._replace(use_eagle=True) for g in attention_groups]
        self.attention_groups = attention_groups
        self.eagle_group_ids = {
            gid for g in attention_groups if g.use_eagle for gid in g.group_ids
        }

    def find_longest_cache_hit(
        self,
        block_hashes: Sequence[BlockHash],
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
        *,
        apply_eagle: bool = True,
    ) -> tuple[tuple[list[bool], ...], int]:
        blocks_per_group, hit_length = self._find_hit_blocks(
            block_hashes, max_length, cached_block_pool, apply_eagle=apply_eagle
        )
        masks = tuple(
            [blk is not cached_block_pool.null_block for blk in blocks]
            for blocks in blocks_per_group
        )
        return masks, hit_length

    def load_mask(
        self,
        block_hashes: Sequence[BlockHash],
        token_len: int,
    ) -> tuple[list[bool], ...]:
        masks, _ = self.find_longest_cache_hit(
            block_hashes,
            token_len,
            ExternalCachedBlockPool(self.hash_block_size),
            apply_eagle=False,
        )
        return masks

    def store_mask(
        self,
        aligned_token_len: int,
        start_token: int = 0,
        num_prompt_tokens: int | None = None,
    ) -> tuple[list[bool] | None, ...]:
        return self._reachable_masks(
            aligned_token_len,
            start_token,
            retention_interval=self.retention_interval,
            num_prompt_tokens=num_prompt_tokens,
            exclude_mamba=True,
        )

    def lookup_mask(
        self,
        aligned_token_len: int,
    ) -> tuple[list[bool] | None, ...]:
        return self._reachable_masks(
            aligned_token_len,
            0,
            retention_interval=None,
            num_prompt_tokens=None,
        )

    def _reachable_masks(
        self,
        aligned_token_len: int,
        start_token: int,
        *,
        retention_interval: int | None,
        num_prompt_tokens: int | None,
        exclude_mamba: bool = False,
    ) -> tuple[list[bool] | None, ...]:
        mask_alignment = (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.lcm_block_size
        )
        assert aligned_token_len % mask_alignment == 0, (
            f"aligned_token_len ({aligned_token_len}) must be a multiple of "
            f"{mask_alignment}"
        )
        masks: list[list[bool] | None] = []
        for g_idx, g in enumerate(self.kv_cache_groups):
            spec = _unwrap_spec(g.kv_cache_spec)
            end_chunk = aligned_token_len // spec.block_size
            start_chunk = min(end_chunk, max(0, cdiv(start_token, spec.block_size)))
            if exclude_mamba and isinstance(spec, MambaSpec):
                masks.append([False] * (end_chunk - start_chunk))
                continue
            manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
            assert manager_cls is not None
            use_eagle = g_idx in self.eagle_group_ids
            reachable_boundaries = (
                () if num_prompt_tokens is None else (num_prompt_tokens - 1,)
            )
            mask = manager_cls.reachable_block_mask(
                start_block=start_chunk,
                end_block=end_chunk,
                alignment_tokens=self.lcm_block_size,
                kv_cache_spec=spec,
                use_eagle=use_eagle,
                retention_interval=retention_interval,
                reachable_boundaries=reachable_boundaries,
                dcp_world_size=1,
            )
            if mask is not None:
                assert len(mask) == end_chunk - start_chunk
            masks.append(mask)
        return tuple(masks)

    def block_hashes_for_spec(
        self, block_hashes: Sequence[BlockHash], spec: KVCacheSpec
    ) -> Sequence[BlockHash]:
        return chunk_hashes_for_block_size(
            block_hashes, self.hash_block_size, spec.block_size
        )

    def _find_hit_blocks(
        self,
        block_hashes: Sequence[BlockHash],
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
        *,
        apply_eagle: bool = True,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        alignment_tokens = (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.lcm_block_size
        )
        if len(self.attention_groups) == 1:
            spec, group_ids, manager_cls, group_eagle = self.attention_groups[0]
            hit_blocks, hit_length = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,  # type: ignore[arg-type]
                max_length=max_length,
                kv_cache_group_ids=group_ids,
                block_pool=cast(BlockPool, cached_block_pool),
                kv_cache_spec=spec,
                drop_eagle_block=apply_eagle and group_eagle,
                alignment_tokens=alignment_tokens,
            )
            num_groups = len(self.kv_cache_groups)
            blocks_by_group: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
            for gid, blks in zip(group_ids, hit_blocks, strict=True):
                blocks_by_group[gid] = blks
            return tuple(blocks_by_group), hit_length

        num_groups = len(self.kv_cache_groups)
        hit_length = max_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        hit_length_by_group: list[int] = [0] * num_groups

        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0].spec, FullAttentionSpec
        )
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls, group_eagle) in enumerate(
                self.attention_groups
            ):
                first_group_id = group_ids[0]
                cached = hit_blocks_by_group[first_group_id]
                if isinstance(spec, FullAttentionSpec) and cached is not None:
                    curr_hit_length = min(
                        curr_hit_length, hit_length_by_group[first_group_id]
                    )
                    continue

                drop_eagle_block = (
                    apply_eagle and group_eagle and idx not in eagle_verified
                )
                _max_length = curr_hit_length
                if drop_eagle_block and not isinstance(spec, MambaSpec):
                    supports_fg = getattr(
                        manager_cls, "supports_fine_grained_hash_lookup", False
                    )
                    eagle_margin = (
                        self.hash_block_size
                        if self.enable_partial_hash_hits
                        and supports_fg
                        and spec.block_size > self.hash_block_size
                        else spec.block_size
                    )
                    _max_length = min(curr_hit_length + eagle_margin, max_length)
                hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,  # type: ignore[arg-type]
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=cast(BlockPool, cached_block_pool),
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=alignment_tokens,
                )
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for gid, blocks in zip(group_ids, hit_blocks, strict=True):
                    hit_blocks_by_group[gid] = blocks
                    hit_length_by_group[gid] = _new_hit_length

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        for group in self.attention_groups:
            if not isinstance(group.spec, FullAttentionSpec):
                continue
            num_blocks = cdiv(hit_length, group.spec.block_size)
            for group_id in group.group_ids:
                full_blks = hit_blocks_by_group[group_id]
                assert full_blks is not None
                del full_blks[num_blocks:]
                hit_length_by_group[group_id] = hit_length

        return (
            tuple(blks if blks is not None else [] for blks in hit_blocks_by_group),
            hit_length,
        )
