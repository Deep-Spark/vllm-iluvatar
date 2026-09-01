# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar KV-cache block-pool fix.

Corresponds to: vllm/v1/core/block_pool.py
Target   : BlockPool.free_blocks
Baseline : main_6e448d0
Why      : Community commit a82f1b388fe changed the free-queue ordering for
           blocks released while prefix caching is disabled.
How      : Fingerprint the current community ``BlockPool.free_blocks``
           (post-a82f1b388fe), then rebind it to the pre-a82f1b388fe hash
           split in ``vllm_iluvatar.v1.core.block_pool``.
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
    blocks_with_hash = []
    blocks_without_hash = []
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            # When caching is disabled we always append for better
            # GPU cache locality from reusing recently used blocks
            if block.block_hash is None and self.enable_caching:
                blocks_without_hash.append(block)
            else:
                blocks_with_hash.append(block)

    # Blocks without hash get evicted first - prepend them last to the tail
    self.free_block_queue.prepend_n(blocks_without_hash)
    self.free_block_queue.append_n(blocks_with_hash)
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
