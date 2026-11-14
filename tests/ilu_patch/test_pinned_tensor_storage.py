# SPDX-License-Identifier: Apache-2.0
"""Host storage packing, byte isolation, and native Torch view ownership."""
import gc
import random

import pytest
import torch

from vllm_iluvatar.v1.worker.pinned_tensor_storage import (
    allocate_pinned_tensor_views,
    plan_pinned_tensors,
)


def check_layout(sizes):
    layout = plan_pinned_tensors(sizes)
    assert len(layout.placements) == len(sizes)
    assert layout.storage_bytes <= sum(
        1 << (max(size, 64) - 1).bit_length() for size in sizes
    )
    assert layout == plan_pinned_tensors(sizes)
    for arena, capacity in enumerate(layout.arena_bytes):
        assert capacity > 0 and capacity & (capacity - 1) == 0
        spans = []
        for size, place in zip(sizes, layout.placements):
            assert place.nbytes == size
            if place.arena == arena:
                assert place.offset % 64 == 0
                assert place.offset + size <= capacity
                spans.append((place.offset, place.offset + size))
        spans.sort()
        assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
    return layout


def test_ragged_sizes_and_random_plane_counts():
    rng = random.Random(1114)
    for count in range(1, 65):
        for _ in range(8):
            check_layout([rng.randrange(1, 1 << 18) for _ in range(count)])
    assert plan_pinned_tensors([]).storage_bytes == 0


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None])
def test_bad_plane_sizes_rejected_before_allocation(value):
    with pytest.raises(ValueError, match="positive integers"):
        plan_pinned_tensors([value])


def test_bad_alignment_rejected():
    with pytest.raises(ValueError, match="power of two"):
        plan_pinned_tensors([1], alignment=3)


def test_service_geometry_capacity_preserved_without_full_host_allocations():
    n = 284058
    regular = [n * 65536, n * 8192, (n // 2) * 65536, (n // 2) * 2048]
    tail = regular + [size * 2 for size in regular]
    assert sum(regular) == 30541916160
    assert sum(tail) == 91625748480
    assert check_layout(regular).storage_bytes == 32 * 2**30
    assert check_layout(tail).storage_bytes == 96 * 2**30
    assert sum(1 << (size - 1).bit_length() for size in regular) == 52.5 * 2**30
    per_producer = (
        6 * check_layout(regular).storage_bytes + check_layout(tail).storage_bytes
    )
    assert 2 * per_producer == 576 * 2**30


def test_cpu_views_are_dense_disjoint_and_keep_torch_storage_alive():
    shapes = [(7, 19), (3, 65), (11, 2), (1, 99)]
    planes, layout = allocate_pinned_tensor_views(shapes, pin_memory=False)
    for i, (plane, shape, place) in enumerate(zip(planes, shapes, layout.placements)):
        assert tuple(plane.shape) == shape
        assert plane.dtype == torch.int8 and plane.is_contiguous()
        assert plane.stride(0) == shape[1]
        assert plane.untyped_storage().nbytes() == layout.arena_bytes[place.arena]
        assert plane.data_ptr() - plane.untyped_storage().data_ptr() == place.offset
        assert torch.count_nonzero(plane) == 0
        plane.fill_(i + 1)
    for i, plane in enumerate(planes):
        assert torch.all(plane == i + 1)
    survivor = planes[1]
    del planes
    gc.collect()
    assert torch.all(survivor == 2)
    survivor.fill_(17)
    assert torch.all(survivor == 17)


def test_copy_directions_keep_independent_view_lists(monkeypatch):
    from vllm.v1.kv_offload.base import CanonicalKVCacheTensor, CanonicalKVCaches
    from vllm_iluvatar.v1.worker import grouped_cpu_offloading_worker as worker

    calls = []

    class Handler:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(worker, "PIN_MEMORY", False)
    monkeypatch.setattr(worker, "SingleDirectionOffloadingHandler", Handler)
    caches = CanonicalKVCaches(
        [CanonicalKVCacheTensor(torch.zeros(2, 64, dtype=torch.int8), 64)], [[]]
    )
    worker.GroupCPUOffloadingWorker(caches, 1, [7])
    store, load = calls
    assert store["gpu_tensors"] is not load["gpu_tensors"]
    assert store["cpu_tensors"] is not load["cpu_tensors"]
    assert store["cpu_tensors"][0] is load["cpu_tensors"][0]
    store["cpu_tensors"].clear()
    store["gpu_tensors"].clear()
    gc.collect()
    assert len(load["cpu_tensors"]) == len(load["gpu_tensors"]) == 1
    assert torch.count_nonzero(load["cpu_tensors"][0]) == 0
