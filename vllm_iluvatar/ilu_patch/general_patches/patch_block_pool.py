# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar KV-cache block-pool fix.

Corresponds to: vllm/v1/core/block_pool.py
Target   : BlockPool.free_blocks
Baseline : v0.30.0 (re-synced from community 0.30.0)
Why      : Community commit a82f1b388fe changed the free-queue ordering for
           blocks released while prefix caching is disabled; 0.30.0 added
           multi-pool routing (block.pool) to free_blocks.
How      : Fingerprint the current community ``BlockPool.free_blocks``
           (0.30.0, with other_pools routing), then rebind it to the
           pre-a82f1b388fe hash split in ``vllm_iluvatar.v1.core.block_pool``,
           preserving the 0.30.0 multi-pool routing.
Scope    : KV-cache block release and reuse ordering.
Remove   : When community vLLM reverts a82f1b388fe or provides an equivalent fix.
Fingerprint: the live community ``BlockPool.free_blocks`` source is checked
           against ``FREE_BLOCKS`` before rebinding; gated by
           VLLM_ILU_PATCH_CHECK_FINGERPRINT.
"""

from __future__ import annotations

import inspect

from vllm_iluvatar.ilu_patch import general_patch as p

FREE_BLOCKS = '''\
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    """Free a list of blocks. The blocks should be ordered by their
    eviction priority, where the first block will be evicted first.

    Args:
        ordered_blocks: A list of blocks to free ordered by their eviction
            priority.
    """
    # Identify blocks with hash (LRU cache) and without it (never match APC)
    blocks_to_evict_last = []
    blocks_to_evict_first = []
    other_pools: dict[BlockPool, list[KVCacheBlock]] = {}
    for block in ordered_blocks:
        if block.pool is not None and block.pool is not self:
            other_pools.setdefault(block.pool, []).append(block)
            continue
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            if block.block_hash is None or not self.enable_caching:
                # LIFO reuse of non-cached blocks for better GPU locality.
                blocks_to_evict_first.append(block)
            else:
                # FIFO reuse of cached blocks for LRU eviction behavior.
                blocks_to_evict_last.append(block)

    # Blocks to reuse first are prepended to the front of the free queue.
    self.free_block_queue.prepend_n(blocks_to_evict_first)
    # Blocks to reuse last are appended to the end of the free queue.
    self.free_block_queue.append_n(blocks_to_evict_last)
    for pool, blocks in other_pools.items():
        pool.free_blocks(blocks)
'''


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="BlockPool.free_blocks pre-a82f1b388fe behavior",
    )


def _apply() -> None:
    from vllm.v1.core.block_pool import BlockPool

    import vllm_iluvatar.v1.core.block_pool as impl

    p.check_fingerprint(
        "vllm.v1.core.block_pool.BlockPool.free_blocks",
        inspect.getsource(BlockPool.free_blocks),
        FREE_BLOCKS,
        resync_hint=(
            "Refresh FREE_BLOCKS to the new community source, then review "
            "whether the pre-a82f1b388fe override still applies."
        ),
    )
    p.check_signatures(BlockPool, impl, ["free_blocks"], strict=True)

    # [ILU override] BlockPool.free_blocks — delta: restore the pre-a82f1b388fe
    # hash split so unhashed blocks are the next reusable free-queue entries.
    BlockPool.free_blocks = impl.free_blocks
