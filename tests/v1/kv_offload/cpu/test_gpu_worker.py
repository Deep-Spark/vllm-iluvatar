from types import SimpleNamespace

import pytest

from vllm_iluvatar.v1.kv_offload.cpu import gpu_worker as gpu_impl


def test_private_worker_init_pins_exact_anonymous_mmap_and_unregisters_on_shutdown(
    monkeypatch,
) -> None:
    import mmap

    import torch
    from vllm.v1.kv_offload.cpu import gpu_worker

    registered: list[tuple[int, int, int]] = []
    unregistered: list[int] = []

    class FakeCudart:
        def cudaHostRegister(self, ptr, size, flags):
            registered.append((ptr, size, flags))
            return SimpleNamespace(value=0)

        def cudaHostUnregister(self, ptr):
            unregistered.append(ptr)
            return SimpleNamespace(value=0)

    class FakeHandler:
        def __init__(self, *, gpu_tensors, cpu_tensors, host_memory_is_pinned, **kw):
            self.gpu_tensors = list(gpu_tensors)
            self.cpu_tensors = list(cpu_tensors)
            self.host_memory_is_pinned = host_memory_is_pinned

        def shutdown(self) -> None:
            self.gpu_tensors.clear()
            self.cpu_tensors.clear()

    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(
        "vllm.platforms.current_platform.is_cuda_alike",
        lambda: True,
    )
    monkeypatch.setattr(gpu_worker, "PIN_MEMORY", True)
    monkeypatch.setattr(gpu_worker, "MAX_HOST_REGISTER_CHUNK_BYTES", 2 * mmap.PAGESIZE)
    monkeypatch.setattr(gpu_worker, "SingleDirectionOffloadingHandler", FakeHandler)
    monkeypatch.setattr(
        gpu_impl,
        "_ORIGINAL_WORKER_SHUTDOWN",
        lambda self: (self._store_handler.shutdown(), self._load_handler.shutdown()),
    )

    gpu_t0 = torch.zeros((4, 3000), dtype=torch.int8)
    gpu_t1 = torch.zeros((4, 1500), dtype=torch.int8)
    kv_caches = SimpleNamespace(
        tensors=[
            SimpleNamespace(tensor=gpu_t0, page_size_bytes=3000),
            SimpleNamespace(tensor=gpu_t1, page_size_bytes=1500),
        ],
        group_data_refs=[[SimpleNamespace(tensor_idx=0, page_size_bytes=3000)]],
    )

    worker = SimpleNamespace()
    gpu_impl._ilu_cpu_offloading_worker_init(
        worker,
        kv_caches=kv_caches,
        blocks_per_chunk=1,
        num_cpu_chunks=3,
        mmap_region=None,
    )

    assert worker._mmap_region is None
    assert len(worker._ilu_private_regions) == 2
    assert worker._store_handler.host_memory_is_pinned is True
    assert worker._load_handler.host_memory_is_pinned is True

    # Tensor 0: 3 * 3000 = 9000 bytes -> 3 pages (12288 bytes), chunked at 2 pages + 1 page.
    # Tensor 1: 3 * 1500 = 4500 bytes -> 2 pages (8192 bytes), 1 chunk of 2 pages.
    assert [size for _, size, _ in registered] == [
        2 * mmap.PAGESIZE,
        1 * mmap.PAGESIZE,
        2 * mmap.PAGESIZE,
    ]
    assert all(ptr % mmap.PAGESIZE == 0 for ptr, _, _ in registered)

    # Verify private anonymous mapping (rw-p, not /dev/shm) in /proc/self/maps.
    ptr0 = worker._store_handler.cpu_tensors[0].data_ptr()
    with open("/proc/self/maps") as maps:
        for line in maps:
            lo, hi = (int(addr, 16) for addr in line.split()[0].split("-"))
            if lo <= ptr0 < hi:
                assert line.split()[1] == "rw-p"
                assert "/dev/shm" not in line
                break
        else:
            raise AssertionError("private anonymous mmap not found in /proc/self/maps")

    mmaps = [region.mmap_obj for region in worker._ilu_private_regions]
    gpu_impl._ilu_cpu_offloading_worker_shutdown(worker)

    assert worker._ilu_private_regions == []
    assert unregistered == [ptr for ptr, _, _ in reversed(registered)]
    assert all(m.closed for m in mmaps)


def test_private_worker_shutdown_synchronizes_device_when_handler_fails(
    monkeypatch,
) -> None:
    import torch
    from vllm.v1.kv_offload.cpu import gpu_worker

    events: list[str] = []

    class FakeCudart:
        def cudaHostRegister(self, ptr, size, flags):
            return SimpleNamespace(value=0)

        def cudaHostUnregister(self, ptr):
            events.append("unregister")
            return SimpleNamespace(value=0)

    class FailingStoreHandler:
        def __init__(self, *, gpu_tensors, cpu_tensors, gpu_to_cpu, **kw):
            self.src_tensors = list(gpu_tensors if gpu_to_cpu else cpu_tensors)
            self.dst_tensors = list(cpu_tensors if gpu_to_cpu else gpu_tensors)
            self.gpu_to_cpu = gpu_to_cpu

        def shutdown(self) -> None:
            if self.gpu_to_cpu:
                events.append("store_shutdown_fail")
                raise RuntimeError("transfer event sync failed")
            events.append("load_shutdown_ok")
            self.src_tensors.clear()
            self.dst_tensors.clear()

    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(
        "vllm.platforms.current_platform.is_cuda_alike",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.accelerator,
        "synchronize",
        lambda: events.append("device_sync"),
    )
    monkeypatch.setattr(gpu_worker, "PIN_MEMORY", True)
    monkeypatch.setattr(
        gpu_worker, "SingleDirectionOffloadingHandler", FailingStoreHandler
    )

    gpu_t0 = torch.zeros((2, 1024), dtype=torch.int8)
    kv_caches = SimpleNamespace(
        tensors=[SimpleNamespace(tensor=gpu_t0, page_size_bytes=1024)],
        group_data_refs=[[SimpleNamespace(tensor_idx=0, page_size_bytes=1024)]],
    )

    worker = SimpleNamespace()
    gpu_impl._ilu_cpu_offloading_worker_init(
        worker,
        kv_caches=kv_caches,
        blocks_per_chunk=1,
        num_cpu_chunks=2,
        mmap_region=None,
    )
    mmap_obj = worker._ilu_private_regions[0].mmap_obj

    gpu_impl._ilu_cpu_offloading_worker_shutdown(worker)

    assert events == [
        "store_shutdown_fail",
        "load_shutdown_ok",
        "device_sync",
        "unregister",
    ]
    assert mmap_obj.closed
    assert worker._ilu_private_regions == []


def test_private_worker_init_rollback_closes_mmaps_and_unregisters_on_error(
    monkeypatch,
) -> None:
    import mmap

    import torch
    from vllm.v1.kv_offload.cpu import gpu_worker

    registered: list[int] = []
    unregistered: list[int] = []
    created_mmaps: list[mmap.mmap] = []
    real_mmap = mmap.mmap

    def tracking_mmap(*args, **kwargs):
        m = real_mmap(*args, **kwargs)
        created_mmaps.append(m)
        return m

    class FakeCudart:
        def cudaHostRegister(self, ptr, size, flags):
            registered.append(ptr)
            return SimpleNamespace(value=0)

        def cudaHostUnregister(self, ptr):
            unregistered.append(ptr)
            return SimpleNamespace(value=0)

    class FailOnLoadHandler:
        def __init__(self, *, gpu_tensors, cpu_tensors, gpu_to_cpu, **kw):
            self.src_tensors = list(gpu_tensors if gpu_to_cpu else cpu_tensors)
            self.dst_tensors = list(cpu_tensors if gpu_to_cpu else gpu_tensors)
            if not gpu_to_cpu:
                raise RuntimeError("load handler init boom")

        def shutdown(self) -> None:
            self.src_tensors.clear()
            self.dst_tensors.clear()

    monkeypatch.setattr(mmap, "mmap", tracking_mmap)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(
        "vllm.platforms.current_platform.is_cuda_alike",
        lambda: True,
    )
    monkeypatch.setattr(gpu_worker, "PIN_MEMORY", True)
    monkeypatch.setattr(
        gpu_worker, "SingleDirectionOffloadingHandler", FailOnLoadHandler
    )

    gpu_t0 = torch.zeros((2, 1024), dtype=torch.int8)
    gpu_t1 = torch.zeros((2, 2048), dtype=torch.int8)
    kv_caches = SimpleNamespace(
        tensors=[
            SimpleNamespace(tensor=gpu_t0, page_size_bytes=1024),
            SimpleNamespace(tensor=gpu_t1, page_size_bytes=2048),
        ],
        group_data_refs=[[SimpleNamespace(tensor_idx=0, page_size_bytes=1024)]],
    )

    worker = SimpleNamespace()
    with pytest.raises(RuntimeError, match="load handler init boom"):
        gpu_impl._ilu_cpu_offloading_worker_init(
            worker,
            kv_caches=kv_caches,
            blocks_per_chunk=1,
            num_cpu_chunks=2,
            mmap_region=None,
        )

    assert len(created_mmaps) == 2
    assert all(m.closed for m in created_mmaps)
    assert unregistered == list(reversed(registered))
    assert worker._ilu_private_regions == []


def test_private_offload_buffer_gc_finalizer_closes_mmap_without_buffer_error(
    monkeypatch,
) -> None:
    import gc

    import torch

    events: list[str] = []

    class FakeCudart:
        def cudaHostRegister(self, ptr, size, flags):
            events.append("register")
            return SimpleNamespace(value=0)

        def cudaHostUnregister(self, ptr):
            events.append("unregister")
            return SimpleNamespace(value=0)

    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(
        "vllm.platforms.current_platform.is_cuda_alike",
        lambda: True,
    )

    tensor, region = gpu_impl._allocate_private_cpu_tensor(
        2,
        1024,
        pin_memory=True,
        max_register_chunk_bytes=1 << 20,
    )
    mmap_obj = region.mmap_obj
    assert mmap_obj is not None and not mmap_obj.closed
    assert events == ["register"]

    del tensor, region
    gc.collect()

    assert events == ["register", "unregister"]
    assert mmap_obj.closed


def test_shared_mmap_constructor_and_shutdown_delegate_unchanged(monkeypatch):
    calls = []
    region = object()
    caches = object()
    worker = SimpleNamespace()
    monkeypatch.setattr(
        gpu_impl, "_ORIGINAL_WORKER_INIT", lambda self, **kw: calls.append((self, kw))
    )
    monkeypatch.setattr(
        gpu_impl, "_ORIGINAL_WORKER_SHUTDOWN", lambda self: calls.append(self)
    )
    gpu_impl._ilu_cpu_offloading_worker_init(
        worker, caches, 2, 3, mmap_region=region, canonical_layout=True
    )
    assert calls == [
        (
            worker,
            dict(
                kv_caches=caches,
                blocks_per_chunk=2,
                num_cpu_chunks=3,
                mmap_region=region,
                canonical_layout=True,
            ),
        )
    ]
    gpu_impl._ilu_cpu_offloading_worker_shutdown(worker)
    assert calls[-1] is worker


@pytest.mark.parametrize("explicit_cleanup", [False, True])
def test_private_mapping_survives_retained_tensor_view(monkeypatch, explicit_cleanup):
    import gc

    import torch

    monkeypatch.setattr(gpu_impl, "_cudart_host_register", lambda *_: 0)
    unregistered = []
    monkeypatch.setattr(
        gpu_impl,
        "_cudart_host_unregister",
        lambda address: unregistered.append(address) or 0,
    )
    tensor, region = gpu_impl._allocate_private_cpu_tensor(
        2, 1024, pin_memory=True, max_register_chunk_bytes=1 << 20
    )
    mapping = region.mmap_obj
    view = tensor.view(-1)
    view.fill_(37)
    if explicit_cleanup:
        region.cleanup()
    del tensor, region
    gc.collect()
    assert not mapping.closed
    assert not unregistered
    assert torch.all(view == 37)
    del view
    gc.collect()
    assert mapping.closed
    assert len(unregistered) == 1


@pytest.mark.parametrize("fail_at", [1, 2])
def test_failed_host_registration_rolls_back_to_pageable(monkeypatch, fail_at):
    import mmap

    registered, unregistered = [], []

    def register(address, size):
        registered.append((address, size))
        return 1 if len(registered) == fail_at else 0

    monkeypatch.setattr(gpu_impl, "_cudart_host_register", register)
    monkeypatch.setattr(
        gpu_impl,
        "_cudart_host_unregister",
        lambda address: unregistered.append(address) or 0,
    )
    monkeypatch.setattr(gpu_impl, "_clear_cuda_last_error", lambda: None)
    tensor, region = gpu_impl._allocate_private_cpu_tensor(
        3, mmap.PAGESIZE, pin_memory=True, max_register_chunk_bytes=mmap.PAGESIZE
    )
    mapping = region.mmap_obj
    assert not region.is_pinned
    assert not region.pinned_addresses
    assert unregistered == [address for address, _ in reversed(registered[:-1])]
    tensor.fill_(19)
    del tensor  # Release the returned view and its base before owner cleanup.
    region.cleanup()
    assert mapping.closed
