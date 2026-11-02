# SPDX-License-Identifier: Apache-2.0
"""Wiring for APC-safe Mamba align skip retirement.

Corresponds to: vllm/v1/core/single_type_kv_cache_manager.py
Target   : MambaManager._remove_blocks_in_range (align mode)
Baseline : v0.31.0 (341bf7d4d1)
Why      : Community align-mode walks null gaps with ``continue`` and frees
           every non-null block, including hashed APC checkpoints. Under
           agent40k-prefix c8 those checkpoints land on the free queue and are
           reused for unique suffixes, collapsing hybrid FA+Mamba hits
           (~32% vs ~48% on v0.27).
How      : Fingerprint the live community method, then rebind to the mirrored
           override in ``vllm_iluvatar.v1.core.single_type_kv_cache_manager``
           which skips freeing blocks that still have ``block_hash``.
Scope    : Mamba align-mode skip retirement only.
Remove   : When upstream align retirement is APC-safe (does not free hashed
           checkpoints mid-request).
Fingerprint: live community ``MambaManager._remove_blocks_in_range`` is checked
           against ``REMOVE_BLOCKS_IN_RANGE``; gated by
           VLLM_ILU_PATCH_CHECK_FINGERPRINT.
"""

from __future__ import annotations

import inspect

from vllm_iluvatar.ilu_patch import general_patch as p

# Community baseline snapshot (v0.31.0 / 341bf7d4d1) — no ILU delta.
REMOVE_BLOCKS_IN_RANGE = '''\
    def _remove_blocks_in_range(
        self, request_id: str, first_block: int, last_block: int
    ) -> None:
        if self.mamba_cache_mode != "align":
            return super()._remove_blocks_in_range(request_id, first_block, last_block)
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
            freed.append(blocks[i])
            blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)
        self._num_retired_blocks[request_id] = last_block
'''


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="Mamba align retain hashed APC checkpoints on skip",
    )


def _apply() -> None:
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    import vllm_iluvatar.v1.core.single_type_kv_cache_manager as impl

    p.check_fingerprint(
        "vllm.v1.core.single_type_kv_cache_manager.MambaManager._remove_blocks_in_range",
        inspect.getsource(MambaManager._remove_blocks_in_range),
        REMOVE_BLOCKS_IN_RANGE,
        resync_hint=(
            "Refresh REMOVE_BLOCKS_IN_RANGE to the new community source, then "
            "confirm hashed-checkpoint retention is still required."
        ),
    )
    p.check_signatures(MambaManager, impl, ["_remove_blocks_in_range"], strict=True)

    # [ILU override] MambaManager._remove_blocks_in_range — delta: skip freeing
    # blocks with block_hash so APC checkpoints survive align skip retirement.
    MambaManager._remove_blocks_in_range = impl._remove_blocks_in_range
