# SPDX-License-Identifier: Apache-2.0
"""Iluvatar override for ``vllm.v1.core.block_pool.BlockPool.free_blocks``.

Corresponds to: vllm/v1/core/block_pool.py
Target   : BlockPool.free_blocks
Baseline : main_4cd213ead
Why      : Community commit a82f1b388fe changed the free-queue ordering for
           blocks released while prefix caching is disabled.
How      : Restore the pre-a82f1b388fe hash split and its original comments.
Scope    : KV-cache block release and reuse ordering.
Remove   : When community vLLM reverts a82f1b388fe or provides an equivalent fix.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock


def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    """Free a list of blocks. The blocks should be ordered by their
    eviction priority, where the first block will be evicted first.

    Args:
        ordered_blocks: A list of blocks to free ordered by their eviction
            priority.
    """
    # >>> ILU MODIFICATION >>>
    # Identify blocks with hash (LRU cache) and without it (will never match in APC)
    # <<< ILU MODIFICATION <<<
    blocks_with_hash = []
    blocks_without_hash = []
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            # >>> ILU MODIFICATION >>>
            if block.block_hash is None:
                # <<< ILU MODIFICATION <<<
                blocks_without_hash.append(block)
            else:
                blocks_with_hash.append(block)

    # >>> ILU MODIFICATION >>>
    # Blocks without hash always get evicted first - prepend them last to the tail
    # <<< ILU MODIFICATION <<<
    self.free_block_queue.prepend_n(blocks_without_hash)
    self.free_block_queue.append_n(blocks_with_hash)
