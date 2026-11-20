# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side ledger of which host-pool rows hold a fresh FP8 mirror."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class PoolDirectory:
    """Map ``(group_idx, block_id) -> block_hash`` mirrored in the pinned host pool."""

    def __init__(self) -> None:
        self._hash_by_block: dict[tuple[int, int], Any] = {}

    def is_fresh(self, group_idx: int, block_id: int, block_hash: Any) -> bool:
        """Return True iff ``block_id`` has a non-None hash matching the host pool."""
        if block_hash is None:
            return False
        return self._hash_by_block.get((group_idx, int(block_id))) == block_hash

    def stale_hits(self, blocks: Any) -> tuple[tuple[int, ...], ...]:
        """Cached blocks whose host-pool row is missing or holds older content."""
        raw_groups = getattr(blocks, "blocks", ())
        result: list[tuple[int, ...]] = []
        for g_idx, group in enumerate(raw_groups):
            stale: list[int] = []
            for blk in group:
                if getattr(blk, "is_null", False):
                    continue
                b_hash = getattr(blk, "block_hash", None)
                if b_hash is not None and not self.is_fresh(
                    g_idx, blk.block_id, b_hash
                ):
                    stale.append(int(blk.block_id))
            result.append(tuple(stale))
        return tuple(result)

    def filter_blocks_needing_export(
        self,
        group_idx: int,
        block_ids: Sequence[int],
        block_hashes: dict[int, Any] | None = None,
    ) -> list[int]:
        """Keep only unhashed blocks or hashed blocks whose pool mirror is stale."""
        if not block_hashes:
            return list(block_ids)
        return [
            int(bid)
            for bid in block_ids
            if not self.is_fresh(group_idx, int(bid), block_hashes.get(int(bid)))
        ]

    def record_exported(
        self,
        exported_block_ids: Sequence[Sequence[int]],
        block_hashes_by_group: Sequence[dict[int, Any]] | None = None,
    ) -> None:
        """Record newly exported hashed blocks as fresh in the host pool."""
        if not block_hashes_by_group:
            return
        for g_idx, (group_ids, hmap) in enumerate(
            zip(exported_block_ids, block_hashes_by_group)
        ):
            for bid in group_ids:
                b_int = int(bid)
                b_hash = hmap.get(b_int)
                if b_hash is not None:
                    self._hash_by_block[(g_idx, b_int)] = b_hash

    def invalidate_blocks(
        self, group_idx: int, block_ids: Sequence[int]
    ) -> None:
        """Drop invalidated or overwritten block IDs from the ledger."""
        for bid in block_ids:
            self._hash_by_block.pop((group_idx, int(bid)), None)

    def clear(self) -> None:
        self._hash_by_block.clear()
