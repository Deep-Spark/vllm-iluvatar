# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
import vllm_iluvatar._C  # noqa: F401  (registers torch.ops._C_cache_ops)


def test_swap_blocks_batch_schema_and_dispatch() -> None:
    expected = torch._C.parse_schema(
        "_C_cache_ops::swap_blocks_batch("
        "Tensor src_ptrs, Tensor dst_ptrs, Tensor sizes, "
        "bool is_src_access_order_any=False) -> ()"
    )
    actual = torch._C._dispatch_find_schema_or_throw(
        "_C_cache_ops::swap_blocks_batch", ""
    ).schema()
    assert actual == expected
    has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key
    assert has_kernel("_C_cache_ops::swap_blocks_batch", "CPU")
    assert not has_kernel("_C_cache_ops::swap_blocks_batch", "CUDA")


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _swap(
    sources: list[tuple[torch.Tensor, int]],
    destinations: list[tuple[torch.Tensor, int]],
    sizes: list[int],
) -> None:
    src_ptrs = torch.tensor(
        [tensor.data_ptr() + offset for tensor, offset in sources],
        dtype=torch.int64,
    )
    dst_ptrs = torch.tensor(
        [tensor.data_ptr() + offset for tensor, offset in destinations],
        dtype=torch.int64,
    )
    torch.ops._C_cache_ops.swap_blocks_batch(
        src_ptrs,
        dst_ptrs,
        torch.tensor(sizes, dtype=torch.int64),
    )
    torch.cuda.synchronize()


@requires_cuda
@pytest.mark.parametrize("direction", ["d2d", "h2d", "d2h"])
def test_swap_blocks_batch(direction: str) -> None:
    source_host = torch.arange(96, dtype=torch.uint8, pin_memory=True)
    destination_host = torch.full((96,), 0xEE, dtype=torch.uint8, pin_memory=True)
    source_device = source_host.cuda()
    destination_device = torch.full((96,), 0xEE, dtype=torch.uint8, device="cuda")

    if direction == "d2d":
        source, destination = source_device, destination_device
    elif direction == "h2d":
        source, destination = source_host, destination_device
    else:
        source, destination = source_device, destination_host

    _swap(
        [(source, 3), (source, 41), (source, 80)],
        [(destination, 7), (destination, 37), (destination, 72)],
        [13, 17, 0],
    )
    actual = destination.cpu()
    expected = torch.full((96,), 0xEE, dtype=torch.uint8)
    expected[7:20] = source_host[3:16]
    expected[37:54] = source_host[41:58]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_cuda
def test_swap_blocks_batch_empty_and_validation() -> None:
    empty = torch.empty(0, dtype=torch.int64)
    torch.ops._C_cache_ops.swap_blocks_batch(empty, empty, empty)

    noncontiguous = torch.arange(6, dtype=torch.int64)[::2]
    with pytest.raises(RuntimeError, match="src_ptrs must be contiguous 1D"):
        torch.ops._C_cache_ops.swap_blocks_batch(
            noncontiguous,
            torch.empty(3, dtype=torch.int64),
            torch.zeros(3, dtype=torch.int64),
        )


@requires_cuda
@pytest.mark.parametrize("blocks_per_chunk", [1, 2])
def test_group_cpu_arenas_roundtrip_with_shared_gpu_pages(blocks_per_chunk):
    """Same slot number in two groups must retain different KV payloads."""
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCacheTensor,
        CanonicalKVCaches,
        GPULoadStoreSpec,
    )
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
    from vllm_iluvatar.v1.worker.grouped_cpu_offloading_worker import (
        GroupCPUOffloadingWorker,
    )
    from vllm_iluvatar.v1.worker.grouped_cpu_offloading import (
        CPUGroupAllocation,
        compact_worker_cache,
    )

    b = blocks_per_chunk
    sizes = [256, 32, 8]
    originals = [
        ((torch.arange(8 * n).reshape(8, n) + i * 31) % 251).to(torch.uint8)
        for i, n in enumerate(sizes)
    ]
    gpu = [x.cuda() for x in originals]
    caches = CanonicalKVCaches(
        [CanonicalKVCacheTensor(t, n) for t, n in zip(gpu, sizes)],
        [
            [CanonicalKVCacheRef(0, 256), CanonicalKVCacheRef(1, 32)],
            [CanonicalKVCacheRef(0, 256), CanonicalKVCacheRef(2, 8)],
            [],
        ],
    )
    allocations = (
        CPUGroupAllocation(3, b, (288 * b, 0), 1),
        CPUGroupAllocation(7, b, (264 * b, 0), 2),
        CPUGroupAllocation(9, b, (0, 1), 1),
    )
    compact, counts = compact_worker_cache(caches, allocations, 0, b)
    worker = GroupCPUOffloadingWorker(
        compact,
        blocks_per_chunk=b,
        cpu_chunks_per_tensor=counts,
    )
    try:
        src = GPULoadStoreSpec(
            list(range(b)) + list(range(2 * b, 3 * b)),
            group_sizes=[b, b, 0],
            block_indices=[0, 0, 0],
        )
        assert worker.submit_store(1, src, CPULoadStoreSpec([0, 0]))
        torch.cuda.synchronize()
        finished = worker.get_finished()
        assert [t.job_id for t in finished] == [1]
        assert all(t.success for t in finished)
        assert [t.shape[0] for t in worker._store_handler.dst_tensors] == counts
        for t in gpu:
            t.zero_()
        dst = GPULoadStoreSpec(
            list(range(b, 2 * b)) + list(range(3 * b, 4 * b)),
            group_sizes=[b, b, 0],
            block_indices=[0, 0, 0],
        )
        assert worker.submit_load(2, CPULoadStoreSpec([0, 0]), dst)
        torch.cuda.synchronize()
        finished = worker.get_finished()
        assert [t.job_id for t in finished] == [2]
        assert all(t.success for t in finished)
        for index in [0, 1]:
            torch.testing.assert_close(
                gpu[index][b : 2 * b].cpu(), originals[index][:b], rtol=0, atol=0
            )
        for index in [0, 2]:
            torch.testing.assert_close(
                gpu[index][3 * b : 4 * b].cpu(),
                originals[index][2 * b : 3 * b],
                rtol=0,
                atol=0,
            )
    finally:
        worker.shutdown()


@requires_cuda
@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_group_arena_bytes_survive_eviction_and_reset(policy):
    """Exercise full small arenas and reused slots through real GPU transfers."""
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCacheTensor,
        CanonicalKVCaches,
        GPULoadStoreSpec,
        ReqContext,
        make_offload_key,
    )
    from vllm_iluvatar.v1.worker.grouped_cpu_offloading_worker import (
        GroupCPUOffloadingWorker,
    )
    from vllm_iluvatar.v1.worker.grouped_cpu_offloading import (
        CPUGroupAllocation,
        GroupCPUOffloadingManager,
        compact_worker_cache,
    )

    gpu = [torch.zeros(2, n, dtype=torch.uint8, device="cuda") for n in [64, 8, 2]]
    caches = CanonicalKVCaches(
        [CanonicalKVCacheTensor(t, t.shape[1]) for t in gpu],
        [
            [CanonicalKVCacheRef(0, 64), CanonicalKVCacheRef(1, 8)],
            [CanonicalKVCacheRef(0, 64), CanonicalKVCacheRef(2, 2)],
        ],
    )
    allocations = (
        CPUGroupAllocation(3, 1, (72,), 2),
        CPUGroupAllocation(7, 1, (66,), 3),
    )
    compact, counts = compact_worker_cache(caches, allocations, 0, 1)
    manager = GroupCPUOffloadingManager(allocations, cache_policy=policy)
    worker = GroupCPUOffloadingWorker(compact, 1, cpu_chunks_per_tensor=counts)
    expected = {}
    job = 0

    def transfer(store, src, dst):
        nonlocal job
        job += 1
        submit = worker.submit_store if store else worker.submit_load
        assert submit(job, src, dst)
        torch.cuda.synchronize()
        done = worker.get_finished()
        assert len(done) == 1 and done[0].job_id == job and done[0].success

    def gpu_spec(group, block):
        return GPULoadStoreSpec(
            [block],
            group_sizes=[int(group == 3), int(group == 7)],
            block_indices=[0, 0],
        )

    try:
        for i in range(80):
            if i == 40:
                manager.reset_cache()
                expected.clear()
            group = 3 if i % 2 == 0 else 7
            key = make_offload_key(i.to_bytes(32, "big"), group)
            ctx = ReqContext(str(i))
            manager.on_new_request(ctx)
            output = manager.prepare_store([key], ctx)
            assert output is not None
            for evicted in output.evicted_keys:
                expected.pop(evicted)
            value = i + 1
            for t in gpu:
                t[0].fill_(value)
            transfer(True, gpu_spec(group, 0), output.store_spec)
            manager.complete_store(output.keys_to_store, ctx)
            expected[key] = (group, value)
            for old, (old_group, old_value) in list(expected.items()):
                load = manager.prepare_load([old], ctx)
                for t in gpu:
                    t[1].zero_()
                transfer(False, load, gpu_spec(old_group, 1))
                manager.complete_load([old], ctx)
                for plane in [0, 1 if old_group == 3 else 2]:
                    assert bool((gpu[plane][1] == old_value).all())
            manager.on_request_finished(ctx)
    finally:
        worker.shutdown()
