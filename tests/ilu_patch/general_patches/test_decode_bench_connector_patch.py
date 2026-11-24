# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import decode_bench_connector
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KpoolTailSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_iluvatar.ilu_patch.general_patches import patch_decode_bench_connector


@pytest.mark.parametrize("startup_fill", [False, True])
@pytest.mark.parametrize("ring_type", ["circular_buffer", "kpool_tail"])
def test_decode_bench_zero_fills_ring_per_request(
    monkeypatch, startup_fill, ring_type
):
    monkeypatch.setattr(decode_bench_connector, "CircularBufferSpec", CircularBufferSpec)
    patch_decode_bench_connector._apply()

    extra_config = {"startup_fill": startup_fill}
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(get_from_extra_config=extra_config.get),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    if ring_type == "kpool_tail":
        tail = KpoolTailSpec(
            block_size=4,
            num_kv_heads=2,
            head_size=128,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=4,
        )
        ring = UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs={"ring": tail})
    else:
        ring = CircularBufferSpec(
            block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32
        )
    attention = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], attention),
            KVCacheGroupSpec(["ring"], ring),
        ],
    )
    scheduler = decode_bench_connector.DecodeBenchConnectorScheduler(
        vllm_config, cache_config
    )
    request = SimpleNamespace(request_id="request", num_tokens=18)
    blocks = KVCacheBlocks(
        ([KVCacheBlock(block_id=1), KVCacheBlock(block_id=2)], [KVCacheBlock(block_id=4)])
    )
    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=17)
    metadata = scheduler.build_connector_meta(SchedulerOutput.make_empty())
    attention_at_startup = startup_fill and not cache_config.needs_kv_cache_zeroing
    assert metadata.reqs_to_fill == {
        request.request_id: (([] if attention_at_startup else [1, 2], [4]), 17)
    }

    worker = decode_bench_connector.DecodeBenchConnectorWorker(vllm_config, cache_config)
    attention_cache = torch.zeros(8, 2)
    ring_cache = torch.ones(8, 2)
    worker.register_kv_caches({"attention": attention_cache, "ring": ring_cache})
    torch.testing.assert_close(ring_cache, torch.ones_like(ring_cache))
    worker.start_fill_kv(metadata)

    expected_ring = torch.ones_like(ring_cache)
    expected_ring[4] = 0
    torch.testing.assert_close(ring_cache, expected_ring)
