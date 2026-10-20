"""Compare published split keys with the original production quantizer."""

import pytest
import torch
import vllm_iluvatar._C  # noqa: F401
from ixformer.inference.functions.deepseek.int8_split_cache import (
    indexer_norm_rope_int8_store_split,
)
from vllm.v1.worker.utils import KVBlockZeroer, copy_kv_cache_blocks_inplace

from vllm_iluvatar.models.deepseek_v41.indexer_kv import extend_kv_zeroer


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("tokens", [129, 2048])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_split_key_publication_matches_original_production_quantizer(
    ratio,
    tokens,
    index_dtype,
):
    torch.manual_seed(1021)
    block = 64
    count = (tokens + block - 1) // block + 2
    stride = block * 256
    raw = torch.full((count * stride,), 37, dtype=torch.uint8, device="cuda")
    packed = raw.as_strided((count, block, 132), (stride, 132, 1))
    values = torch.full((count, block, 128), 37, dtype=torch.uint8, device="cuda")
    scales = torch.full(
        (count, block), 0x25252525, dtype=torch.int32, device="cuda"
    ).view(torch.float32)
    keys = torch.randn(tokens, 256, dtype=torch.bfloat16, device="cuda")[:, :128]
    weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(tokens, dtype=index_dtype, device="cuda")
    slots = positions // ratio + block
    slots[::7] = -1
    cos_sin = torch.randn(tokens, 64, dtype=torch.float32, device="cuda")
    legacy_mirror = torch.full((tokens, 132), 37, dtype=torch.uint8, device="cuda")
    split_mirror = legacy_mirror.clone()
    torch.ops._C.v41_indexer_k_norm_rope_store(
        keys,
        positions.to(torch.int64),
        cos_sin,
        weight,
        1e-6,
        packed,
        slots.to(torch.int64),
        ratio,
        "int8",
        legacy_mirror,
    )
    indexer_norm_rope_int8_store_split(
        keys,
        positions,
        cos_sin,
        weight,
        1e-6,
        values,
        scales,
        slots,
        ratio,
        split_mirror,
    )
    physical = raw.view(count, stride)
    legacy_values = physical[:, : block * 128].reshape(count, block, 128)
    legacy_scales = physical[:, block * 128 : block * 132].view(torch.float32)
    torch.cuda.synchronize()
    assert torch.equal(values, legacy_values)
    assert torch.equal(scales.view(torch.uint8), legacy_scales.view(torch.uint8))
    assert torch.equal(split_mirror, legacy_mirror)


def test_physical_cow_and_zeroing_cover_both_planes_after_kernel_page_split():
    """Exercise the actual V2 helpers with eight kernel pages per manager page."""
    count, ratio, rows = 3, 8, 64
    values = torch.randint(
        0, 256, (count * ratio, rows, 128), dtype=torch.uint8, device="cuda"
    )
    scales = torch.randn(count * ratio, rows, dtype=torch.float32, device="cuda")
    expected_values, expected_scales = values.clone(), scales.clone()
    expected_values[2 * ratio : 3 * ratio].copy_(values[ratio : 2 * ratio])
    expected_scales[2 * ratio : 3 * ratio].copy_(scales[ratio : 2 * ratio])
    copy_kv_cache_blocks_inplace((values, scales), count, [(1, 2)])
    torch.cuda.synchronize()
    assert torch.equal(values, expected_values)
    assert torch.equal(scales, expected_scales)
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = values.device
    zeroer._meta = None
    extend_kv_zeroer(zeroer, count, (values, scales))
    zeroer.zero_block_ids([2])
    expected_values[2 * ratio : 3 * ratio].zero_()
    expected_scales[2 * ratio : 3 * ratio].zero_()
    torch.cuda.synchronize()
    assert torch.equal(values, expected_values)
    assert torch.equal(scales, expected_scales)
