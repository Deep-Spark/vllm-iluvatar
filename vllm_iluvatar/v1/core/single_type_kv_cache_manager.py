# SPDX-License-Identifier: Apache-2.0
"""Iluvatar override for Mamba align skip retirement (APC-safe).

Corresponds to: vllm/v1/core/single_type_kv_cache_manager.py
Target   : MambaManager._remove_blocks_in_range
Baseline : v0.31.0 (341bf7d4d1)
Why      : Community align-mode walks null gaps with ``continue`` and frees
           every non-null block, including hashed APC checkpoints. Under
           agent40k-prefix c8 those checkpoints land on the free queue and are
           reused for unique suffixes, collapsing hybrid FA+Mamba hits
           (~32% vs ~48% on v0.27).
How      : Same loop as community; skip freeing blocks that still have
           ``block_hash`` so APC checkpoints stay held until request end.
Scope    : Mamba ``align`` mode skip retirement only; non-align delegates to
           ``SingleTypeKVCacheManager._remove_blocks_in_range``.
Remove   : When upstream align retirement is APC-safe (does not free hashed
           checkpoints mid-request).
"""

from __future__ import annotations

from vllm.v1.core.kv_cache_utils import KVCacheBlock


def _remove_blocks_in_range(
    self, request_id: str, first_block: int, last_block: int
) -> None:
    if self.mamba_cache_mode != "align":
        from vllm.v1.core.single_type_kv_cache_manager import (
            SingleTypeKVCacheManager,
        )

        return SingleTypeKVCacheManager._remove_blocks_in_range(
            self, request_id, first_block, last_block
        )
    blocks = self.req_to_blocks.get(request_id, [])
    first_block = max(first_block, self._num_retired_blocks.get(request_id, 0))
    last_block = min(last_block, len(blocks))
    if first_block >= last_block:
        return
    freed: list[KVCacheBlock] = []
    # Mamba prefill leaves null gaps between states awaiting retirement.
    for i in range(last_block - 1, first_block - 1, -1):
        if blocks[i].is_null:
            continue
        # >>> ILU MODIFICATION >>>
        # Keep hashed APC checkpoints held until request end. Freeing them
        # mid-request exposes them to eviction under high concurrency and
        # collapses hybrid (FA+Mamba) prefix hits.
        if blocks[i].block_hash is not None:
            continue
        # <<< ILU MODIFICATION <<<
        freed.append(blocks[i])
        blocks[i] = self._null_block
    if freed:
        self.block_pool.free_blocks(freed)
    self._num_retired_blocks[request_id] = last_block
