# SPDX-License-Identifier: Apache-2.0
"""Exercise real hybrid cache lookup, deferred publication and private COW."""

from __future__ import annotations

from math import lcm
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.request import Request, RequestStatus

from vllm_iluvatar.models.deepseek_v41.indexer_kv import SPLIT_INDEXER_FORMAT
from vllm_iluvatar.models.deepseek_v41.indexer_prefix_cache import (
    configure_indexer_prefix_hits,
)


def scheduler(*, layout="split", role="kv_producer", caching=True, hash_size=256):
    groups = [
        KVCacheGroupSpec(
            ["main"],
            MLAAttentionSpec(
                block_size=512,
                tokens_per_state=2,
                num_kv_heads=1,
                head_size=512,
                state_content_bytes=256,
                dtype=torch.uint8,
                cache_dtype_str="i4g32",
            ),
        ),
        KVCacheGroupSpec(
            ["index"],
            MLAAttentionSpec(
                block_size=1024 if layout == "split" else 512,
                tokens_per_state=2,
                num_kv_heads=1,
                head_size=128 if layout == "split" else 256,
                dtype=torch.uint8,
                cache_dtype_str=SPLIT_INDEXER_FORMAT if layout == "split" else None,
            ),
        ),
        KVCacheGroupSpec(
            ["swa"],
            SlidingWindowMLASpec(
                block_size=64,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.bfloat16,
                sliding_window=128,
                bounded_replay=True,
            ),
        ),
        KVCacheGroupSpec(
            ["ring"],
            CircularBufferSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=1024,
                dtype=torch.float32,
            ),
        ),
    ]
    config = KVCacheConfig(num_blocks=256, kv_cache_tensors=[], kv_cache_groups=groups)
    manager = KVCacheManager(
        config,
        max_model_len=8192,
        scheduler_block_size=lcm(*(g.kv_cache_spec.block_size for g in groups)),
        hash_block_size=hash_size,
        enable_caching=caching,
    )
    return SimpleNamespace(
        kv_cache_manager=manager,
        prefix_replay_tokens=128,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_v41")
            ),
            kv_transfer_config=SimpleNamespace(kv_role=role),
            speculative_config=None,
        ),
    )


def request(name, tokens, hash_size=256):
    return Request(
        request_id=name,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_size, sha256),
    )


@pytest.fixture(autouse=True)
def hash_and_gate(monkeypatch):
    init_none_hash(sha256)
    monkeypatch.delenv("VLLM_ILU_DS41_INDEXER_PREFIX_SUBPAGE", raising=False)


@pytest.mark.parametrize("length", [1290, 1733, 2311, 3337])
def test_completed_prefix_reuses_partial_indexer_page_with_cow(length):
    sched = scheduler()
    manager = sched.kv_cache_manager
    assert configure_indexer_prefix_hits(sched)
    owner = request("owner", list(range(length)))
    assert manager.allocate_slots(owner, length) is not None
    owner.num_computed_tokens = length
    owner.status = RequestStatus.FINISHED_LENGTH_CAPPED
    manager.free(owner)
    manager.new_step_starts()
    before = manager.block_pool.get_num_free_blocks()
    consumer = request("consumer", list(range(length)))
    blocks, hit, _ = manager.get_computed_blocks(consumer)
    assert hit == (length - 1) // 256 * 256
    assert hit > (length - 1) // 1024 * 1024
    source = blocks.blocks[1][-1]
    assert manager.allocate_slots(consumer, length - hit, hit, blocks) is not None
    copies, retained = manager.take_kv_cache_block_copies()
    matching = [copy for copy in copies if copy.src_block_id == source.block_id]
    assert len(matching) == 1
    copy = matching[0]
    assert copy.dst_block_id != copy.src_block_id
    assert all(block.ref_cnt > 0 for block in retained)
    # The original source remains published; new appends use a private tail.
    again = request("again", list(range(length)))
    again_blocks, again_hit, _ = manager.get_computed_blocks(again)
    assert again_hit == hit and again_blocks.blocks[1][-1] is source
    manager.block_pool.free_blocks(retained)
    consumer.num_computed_tokens = length
    consumer.status = RequestStatus.FINISHED_LENGTH_CAPPED
    manager.free(consumer)
    assert manager.block_pool.get_num_free_blocks() == before


def test_allocation_does_not_publish_unproduced_partial_tail():
    sched = scheduler()
    manager = sched.kv_cache_manager
    assert configure_indexer_prefix_hits(sched)
    owner = request("owner", list(range(1733)))
    assert manager.allocate_slots(owner, 1733) is not None
    clone = request("same-step", list(range(1733)))
    _, hit, _ = manager.get_computed_blocks(clone)
    # Coarse blocks retain upstream semantics. A COW must not read the
    # unfinished tail before the producer's KV insertion has executed.
    assert hit == 1024
    owner.num_computed_tokens = 1733
    owner.num_in_flight_tokens = 1733
    owner.status = RequestStatus.FINISHED_ABORTED
    manager.free(owner)
    assert manager.get_computed_blocks(clone)[1] == 1024


def test_success_with_pending_forward_does_not_publish_tail():
    sched = scheduler()
    manager = sched.kv_cache_manager
    assert configure_indexer_prefix_hits(sched)
    owner = request("owner", list(range(1733)))
    assert manager.allocate_slots(owner, 1733) is not None
    owner.num_computed_tokens = 1733
    owner.num_in_flight_tokens = 1733
    owner.status = RequestStatus.FINISHED_LENGTH_CAPPED
    manager.free(owner)
    assert manager.get_computed_blocks(request("clone", list(range(1733))))[1] == 1024


@pytest.mark.parametrize("length", [1024, 1536, 2048, 2051, 3077])
def test_exact_hash_boundaries_preserve_original_valid_hit(length):
    sched = scheduler()
    manager = sched.kv_cache_manager
    assert configure_indexer_prefix_hits(sched)
    owner = request("owner", list(range(length)))
    assert manager.allocate_slots(owner, length) is not None
    owner.num_computed_tokens = length
    owner.status = RequestStatus.FINISHED_STOPPED
    manager.free(owner)
    hit = manager.get_computed_blocks(request("clone", list(range(length))))[1]
    assert hit >= (length - 1) // 1024 * 1024
    assert hit <= length - 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"layout": "legacy"},
        {"role": "kv_consumer"},
        {"role": "kv_both"},
        {"caching": False},
    ],
)
def test_other_policies_are_unchanged(overrides):
    sched = scheduler(**overrides)
    assert not configure_indexer_prefix_hits(sched)
    assert not sched.kv_cache_manager.coordinator.enable_partial_hash_hits


def test_optout_and_idempotence(monkeypatch):
    sched = scheduler()
    monkeypatch.setenv("VLLM_ILU_DS41_INDEXER_PREFIX_SUBPAGE", "0")
    assert not configure_indexer_prefix_hits(sched)
    monkeypatch.setenv("VLLM_ILU_DS41_INDEXER_PREFIX_SUBPAGE", "1")
    assert configure_indexer_prefix_hits(sched)
    free = sched.kv_cache_manager.free
    assert not configure_indexer_prefix_hits(sched)
    assert sched.kv_cache_manager.free is free


def test_reject_speculative_producer():
    sched = scheduler()
    sched.vllm_config.speculative_config = object()
    assert not configure_indexer_prefix_hits(sched)


def test_reject_unsafe_swa_alignment():
    # Hash32 is legal for the dense groups, but replayed SWA retires64-token
    # blocks. Do not enable an alignment smaller than that allocator handles.
    sched = scheduler(hash_size=32)
    assert not configure_indexer_prefix_hits(sched)
