"""Disabled block dropping must work with explicit draft group metadata."""

import pytest
import torch
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

from vllm_iluvatar.ilu_patch.general_patches.patch_eagle_block_drop import register


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("drop", [False, True])
def test_explicit_draft_group_respects_block_drop_switch(hybrid, drop):
    register()
    full = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
    )
    groups = [KVCacheGroupSpec(["target"], full, is_eagle_group=not hybrid)]
    if hybrid:
        swa = SlidingWindowSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
            sliding_window=16,
        )
        groups.append(KVCacheGroupSpec(["draft"], swa, is_eagle_group=True))
    config = KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=groups)
    coordinator = get_kv_cache_coordinator(
        kv_cache_config=config,
        max_model_len=1024,
        max_in_flight_tokens=64,
        use_eagle=drop,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=16,
        hash_block_size=16,
    )
    assert coordinator.eagle_group_ids == ({len(groups) - 1} if drop else set())
    assert coordinator.single_type_managers[-1].use_eagle is drop
    # Clearing the coordinator's policy must preserve source draft metadata.
    assert config.kv_cache_groups[-1].is_eagle_group is True


@pytest.mark.parametrize("drop, expected_hit", [(False, 64), (True, 48)])
def test_cached_prefix_plus_one_token_matches_full_boundary(drop, expected_hit):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    register()
    init_none_hash(sha256)
    full = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
    )
    swa = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        sliding_window=16,
    )
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        prefix_cache_retention_interval=16,
        kv_cache_groups=[
            KVCacheGroupSpec(["target"], full),
            KVCacheGroupSpec(["draft"], swa, is_eagle_group=True),
        ],
    )
    manager = KVCacheManager(
        config,
        max_model_len=1024,
        scheduler_block_size=16,
        hash_block_size=16,
        use_eagle=drop,
    )

    def make_request(name, ids):
        return Request(
            request_id=name,
            prompt_token_ids=ids,
            sampling_params=SamplingParams(max_tokens=17),
            pooling_params=None,
            block_hasher=get_request_block_hasher(16, sha256),
        )

    prefix = list(range(64))
    filled = make_request("fill", prefix)
    assert manager.allocate_slots(filled, num_new_tokens=64) is not None
    filled.num_computed_tokens = 64
    manager.cache_blocks(filled, 64)
    manager.free(filled)
    sibling = make_request("sibling", prefix + [1000])
    _, hit, _ = manager.get_computed_blocks(sibling)
    assert hit == expected_hit
