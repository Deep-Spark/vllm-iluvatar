"""Lay out one dlist per side for a remote target and index it by block id.

Every writing pointer segment of the overlap becomes a region pair (one local,
one remote region).  Identical pairs are registered once, so aliasing views
(V4.1 registers every KV tensor of a rank from one storage) and a per-rank
transfer spanning several layers do not duplicate descriptors.  A dlist is
laid out ``[region][block]``: descriptor id = exclusive cumsum of the region
block counts + block id.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RegionGeometry:
    """``num_blocks`` descriptors of ``length`` bytes at base[ptr] + offset + b * block_stride."""

    ptr: int
    offset: int
    length: int
    block_stride: int
    num_blocks: int


@dataclass(frozen=True)
class RegionPair:
    layer_index: int
    local: RegionGeometry
    remote: RegionGeometry
    local_group_id: int
    remote_group_id: int
    # Attention KV: D may ask for a sub-range of the prompt's blocks (prefix
    # hit at the head, external tokens aligned down at the tail); local ids
    # follow D's start offset, or the tail when D sent none.
    # Mamba state is a fixed set of blocks read from the head.
    tail_aligned: bool


@dataclass(frozen=True)
class XferRegionGroup:
    """Regions of one transfer that index the same (local, remote) groups."""

    local_group_id: int
    remote_group_id: int
    tail_aligned: bool
    # [regions, 1] first descriptor id of each region in the dlist.
    local_offsets: np.ndarray = field(repr=False)
    remote_offsets: np.ndarray = field(repr=False)

    @property
    def num_regions(self) -> int:
        return int(self.local_offsets.shape[0])

    def descriptor_ids(
        self,
        local_block_ids: list[np.ndarray],
        remote_block_ids: list[np.ndarray],
        remote_block_starts: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Descriptor ids for this request, or None when a side has no blocks.

        Block ids are ``[1, n]`` arrays per group.  ``remote_block_starts``
        (per remote group, -1 = unknown) is the index in the prompt's block
        list of the first block remote asks for; attention groups without one
        are aligned on the tail.
        """

        local_ids = local_block_ids[self.local_group_id]
        remote_ids = remote_block_ids[self.remote_group_id]
        num_local = local_ids.shape[1]
        num_remote = remote_ids.shape[1]
        if num_local == 0 or num_remote == 0:
            return None
        if self.tail_aligned:
            if num_local < num_remote:
                raise ValueError(
                    f"local group {self.local_group_id} has {num_local} blocks, "
                    f"fewer than the {num_remote} blocks remote group "
                    f"{self.remote_group_id} expects"
                )
            start = (
                -1
                if remote_block_starts is None
                else remote_block_starts[self.remote_group_id]
            )
            if start < 0:
                start = num_local - num_remote
            elif start + num_remote > num_local:
                raise ValueError(
                    f"remote group {self.remote_group_id} asks for blocks "
                    f"[{start}, {start + num_remote}) but local group "
                    f"{self.local_group_id} has {num_local}"
                )
            local_ids = local_ids[:, start : start + num_remote]
        else:
            count = min(num_local, num_remote)
            local_ids = local_ids[:, :count]
            remote_ids = remote_ids[:, :count]
        return (
            (self.local_offsets + local_ids).reshape(-1),
            (self.remote_offsets + remote_ids).reshape(-1),
        )


@dataclass(frozen=True)
class XferPlan:
    local_regions: tuple[RegionGeometry, ...]
    remote_regions: tuple[RegionGeometry, ...]
    # Global layer index -> region groups written when that layer is ready.
    per_layer: dict[int, tuple[XferRegionGroup, ...]]
    # Region groups written once the whole rank is ready.
    per_rank: tuple[XferRegionGroup, ...]


@dataclass(frozen=True)
class RemoteXfer:
    """One cell of the per-layer ``[layer][remote tp]`` or per-rank
    ``[remote pp][remote tp]`` mapping: at most one WRITE plus one notify.

    A cell without region groups only sends the notify: a replica that does
    not write, or an overlap with nothing registered to transfer.
    """

    remote_agent_name: str
    notify_msg: str
    local_handle: object | None = None
    remote_handle: object | None = None
    region_groups: tuple[XferRegionGroup, ...] = ()

    @property
    def notify_only(self) -> bool:
        return not self.region_groups


def _exclusive_cumsum(counts: list[int]) -> np.ndarray:
    offsets = np.zeros(len(counts), dtype=np.int64)
    if counts:
        np.cumsum(counts[:-1], out=offsets[1:])
    return offsets


def build_xfer_plan(pairs: list[RegionPair]) -> XferPlan:
    """Deduplicate region pairs into dlist layouts and group them per cell."""

    index_of: dict[tuple[RegionGeometry, RegionGeometry], int] = {}
    for pair in pairs:
        index_of.setdefault((pair.local, pair.remote), len(index_of))
    local_regions = tuple(local for local, _ in index_of)
    remote_regions = tuple(remote for _, remote in index_of)
    local_offsets = _exclusive_cumsum([region.num_blocks for region in local_regions])
    remote_offsets = _exclusive_cumsum([region.num_blocks for region in remote_regions])

    def group(selected: Iterable[RegionPair]) -> tuple[XferRegionGroup, ...]:
        buckets: dict[tuple[int, int, bool], list[int]] = {}
        for pair in selected:
            region = index_of[(pair.local, pair.remote)]
            bucket = buckets.setdefault(
                (pair.local_group_id, pair.remote_group_id, pair.tail_aligned), []
            )
            if region not in bucket:
                bucket.append(region)
        return tuple(
            XferRegionGroup(
                local_group_id=local_group_id,
                remote_group_id=remote_group_id,
                tail_aligned=tail_aligned,
                local_offsets=local_offsets[regions].reshape(-1, 1),
                remote_offsets=remote_offsets[regions].reshape(-1, 1),
            )
            for (local_group_id, remote_group_id, tail_aligned), regions in (
                buckets.items()
            )
        )

    by_layer: dict[int, list[RegionPair]] = {}
    for pair in pairs:
        by_layer.setdefault(pair.layer_index, []).append(pair)
    return XferPlan(
        local_regions=local_regions,
        remote_regions=remote_regions,
        per_layer={
            layer_index: group(layer_pairs)
            for layer_index, layer_pairs in sorted(by_layer.items())
        },
        per_rank=group(pairs),
    )
