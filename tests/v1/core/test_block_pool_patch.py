# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar KV-cache block-pool fix."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from vllm.v1.core.block_pool import BlockPool

from vllm_iluvatar.ilu_patch import general_patch
from vllm_iluvatar.ilu_patch.general_patches import patch_block_pool
from vllm_iluvatar.v1.core import block_pool


class _Queue:
    def __init__(self) -> None:
        self.prepended = []
        self.appended = []

    def prepend_n(self, blocks) -> None:
        self.prepended.extend(blocks)

    def append_n(self, blocks) -> None:
        self.appended.extend(blocks)


def test_community_free_blocks_fingerprint_matches() -> None:
    assert general_patch.source_sha(
        inspect.getsource(BlockPool.free_blocks)
    ) == general_patch.source_sha(patch_block_pool.FREE_BLOCKS)


def test_caching_disabled_reuses_unhashed_block_first() -> None:
    queue = _Queue()
    pool = SimpleNamespace(enable_caching=False, free_block_queue=queue)
    freed = SimpleNamespace(ref_cnt=1, is_null=False, block_hash=None)

    block_pool.free_blocks(pool, [freed])

    assert freed.ref_cnt == 0
    assert queue.prepended == [freed]
    assert queue.appended == []


def test_caching_enabled_preserves_hash_priority() -> None:
    queue = _Queue()
    pool = SimpleNamespace(enable_caching=True, free_block_queue=queue)
    unhashed = SimpleNamespace(ref_cnt=1, is_null=False, block_hash=None)
    hashed = SimpleNamespace(ref_cnt=1, is_null=False, block_hash=object())

    block_pool.free_blocks(pool, [unhashed, hashed])

    assert queue.prepended == [unhashed]
    assert queue.appended == [hashed]
