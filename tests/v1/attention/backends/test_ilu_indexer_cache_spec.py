# SPDX-License-Identifier: Apache-2.0
"""Geometry of the paged index-key cache.

A quantized record is ``index_head_dim + 4`` = 132 bytes, which is an awkward
width to page: it shares no factor with the power-of-two rows around it. The
page is therefore allocated at ``_INDEX_PAGE_ROW_BYTES`` and narrowed back to
the record when it is bound, so the record width the kernels see and the pitch
the allocator hands out can differ.

These tests cover that split -- the declared allocation, the absence of
padding, and the strided view bind produces. They deliberately do *not* assert
how the page relates to the main KV page: that relationship is decided by
``unify_kv_cache_spec_page_size`` against the whole group (four spec kinds in
this model, including a sliding-window and a ring-buffer spec), and modelling
it from two specs in isolation gives the wrong answer.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.models.deepseek_v41 import attention as attn

INDEX_HEAD_DIM = 128
RECORD_BYTES = INDEX_HEAD_DIM + attn._INDEX_SCALE_BYTES
BLOCK_SIZES = [16, 32, 64, 128, 256]
RECORDS = ["bf16", "fp8", "int8"]


@pytest.fixture(autouse=True)
def _fake_vllm_config(monkeypatch):
    """The cache only touches the config to register itself for compilation."""
    monkeypatch.setattr(
        attn,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
    )


def _cache(record: str, block_size: int, compress_ratio: int):
    row_bytes, row_dtype = attn._indexer_k_cache_row(INDEX_HEAD_DIM, record)
    return attn.IluDeepseekV41IndexerCache(
        head_dim=row_bytes,
        dtype=row_dtype,
        prefix=f"model.layers.0.indexer.k_cache.{record}.{block_size}",
        cache_config=SimpleNamespace(block_size=block_size),
        compress_ratio=compress_ratio,
    )


# compress_ratio 1 and 2 are the only values the compressor emits.
@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("record", RECORDS)
def test_the_page_scales_linearly_with_the_block_size(
    record: str, block_size: int, compress_ratio: int
):
    """No padding anywhere, at any size.

    ``unify_kv_cache_spec_page_size`` rebuilds a spec at a scaled block size
    and asserts the page scaled with it, so a page that is rounded up at one
    size and not at another cannot take part. Keeping every page exactly
    ``states * row`` is what makes that hold.
    """
    spec = _cache(record, block_size, compress_ratio).get_kv_cache_spec(None)
    states = block_size // compress_ratio
    row = spec.head_size * (1 if spec.dtype is torch.uint8 else 2)

    assert spec.page_size_padded is None
    assert spec.page_size_bytes == states * row
    rescaled = replace(spec, block_size=spec.block_size * 8)
    assert rescaled.page_size_bytes == spec.page_size_bytes * 8


@pytest.mark.parametrize("record", ["fp8", "int8"])
def test_a_quantized_page_is_allocated_wider_than_its_record(record: str):
    """A 132-byte record in a row wide enough for the page to divide."""
    assert attn._indexer_k_cache_row(INDEX_HEAD_DIM, record) == (
        RECORD_BYTES,
        torch.uint8,
    )
    spec = _cache(record, block_size=128, compress_ratio=2).get_kv_cache_spec(None)
    assert spec.head_size == attn._INDEX_PAGE_ROW_BYTES
    assert spec.dtype is torch.uint8
    # Padding here would be fatal twice over: the rescale above would stop
    # holding, and ``compute_layer_kv_cache_shape`` refuses to split a padded
    # page into kernel blocks at all.
    assert spec.page_size_padded is None


@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("record", ["fp8", "int8"])
def test_a_quantized_page_is_a_power_of_two(
    record: str, block_size: int, compress_ratio: int
):
    """The group's largest page is a power of two, so this one must be too.

    ``unify_kv_cache_spec_page_size`` needs the index page to divide the
    largest page in the group, which in this model is the sliding-window
    cache's 65536 bytes. A page only divides a power of two by being one, and
    the state count already is, so the burden falls on the row width.
    """
    page = (
        _cache(record, block_size, compress_ratio)
        .get_kv_cache_spec(None)
        .page_size_bytes
    )
    assert page & (page - 1) == 0, f"{page} is not a power of two"
    assert 65536 % page == 0


@pytest.mark.parametrize("block_size", [64, 128])
def test_bind_narrows_a_quantized_page_to_the_record_at_the_page_pitch(
    block_size: int,
):
    """What the kernels actually receive.

    Both our store kernel and ixformer's readers take the page pitch from
    ``stride(0)`` and require ``stride(1)`` to be the record width, so the
    allocated tail is never addressed.
    """
    cache = _cache("int8", block_size, compress_ratio=2)
    states = block_size // 2
    allocated = torch.zeros(
        (4, 1, states, attn._INDEX_PAGE_ROW_BYTES), dtype=torch.uint8
    )
    cache.bind_kv_cache(allocated)

    bound = cache.kv_cache
    assert bound.shape == (4, states, RECORD_BYTES)
    assert bound.stride() == (states * attn._INDEX_PAGE_ROW_BYTES, RECORD_BYTES, 1)
    # A view, not a copy: the kernels write through it into the real pages.
    assert bound.data_ptr() == allocated.data_ptr()


def test_bind_leaves_a_bf16_page_alone():
    cache = _cache("bf16", block_size=128, compress_ratio=2)
    allocated = torch.zeros((4, 1, 64, INDEX_HEAD_DIM), dtype=torch.bfloat16)
    cache.bind_kv_cache(allocated)

    assert cache.kv_cache.shape == (4, 64, INDEX_HEAD_DIM)
    assert cache.kv_cache.is_contiguous()
