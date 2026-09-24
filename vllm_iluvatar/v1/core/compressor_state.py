# SPDX-License-Identifier: Apache-2.0
"""One-block ring for the V4.1 compressor state.

The prepare kernel addresses ``block_table[req, 0]`` and ``pos % capacity``.
The scheduler must therefore hand each request exactly one block for the
whole lifetime. A sliding-window admission would reserve one block per
in-flight token and cap concurrency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec


@dataclass(frozen=True, kw_only=True)
class IluCompressorStateSpec(SlidingWindowMLASpec):
    """Sliding-window subclass so DeepSeek-V4 grouping still picks it up.

    Memory and the block table are a single page. ``sliding_window`` only
    keeps this group from merging with the real SWA layers.
    """

    def max_memory_usage_bytes(self, vllm_config) -> int:
        del vllm_config
        return self.page_size_bytes

    def max_num_blocks_per_req(self, vllm_config, max_len: int) -> int:
        del vllm_config, max_len
        return 1

    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        del max_in_flight_tokens, max_model_len
        return 1


class CompressorStateManager(SingleTypeKVCacheManager):
    """Claims the ring's single block per request. Prefix caching is a no-op."""

    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    def _claim_ring_block(self, request_id: str) -> list:
        req_blocks = self.req_to_blocks[request_id]
        if req_blocks:
            return []
        new_blocks = self.block_pool.get_new_blocks(1)
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(block.block_id for block in new_blocks)
        return new_blocks

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        del (
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        return 0 if self.req_to_blocks.get(request_id) else 1

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list:
        del num_tokens, num_tokens_main_model
        return self._claim_ring_block(request_id)

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        del num_local_computed_tokens, num_external_computed_tokens
        self._claim_ring_block(request_id)

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ):
        del (
            block_hashes,
            max_length,
            block_pool,
            kv_cache_spec,
            drop_eagle_block,
            alignment_tokens,
            dcp_world_size,
            pcp_world_size,
        )
        return tuple([] for _ in kv_cache_group_ids), 0

    def cache_blocks(self, request, num_tokens: int, retention_interval=None, **kwargs) -> None:
        del request, num_tokens, retention_interval, kwargs
        return

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        del (
            request_id,
            new_computed_blocks,
            num_local_computed_tokens,
            num_external_computed_tokens,
        )
        return

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        del request_id, processed_computed_tokens, num_prompt_tokens
        return

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        del running_request_id
        return 0

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        del num_computed_tokens
        return 0
