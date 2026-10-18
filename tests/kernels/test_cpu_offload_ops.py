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
