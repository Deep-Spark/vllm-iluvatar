# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness of the plugin paged-cache write operators."""

from __future__ import annotations

import pytest
import torch
import torch.library
import vllm_iluvatar._C  # noqa: F401

if not torch.cuda.is_available():
    pytest.skip("Iluvatar/CUDA device required", allow_module_level=True)

NUM_TOKENS = [1, 17, 128]
NUM_HEADS = [16]
HEAD_SIZES = [128]
BLOCK_SIZES = [16]
NUM_BLOCKS = [16]


def _make_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, strided: bool
) -> torch.Tensor:
    if strided:
        storage = torch.empty((shape[0], 2, *shape[1:]), dtype=dtype, device="cuda")
        tensor = storage[:, 0]
    else:
        tensor = torch.empty(shape, dtype=dtype, device="cuda")
    if dtype == torch.int8:
        tensor.random_(-128, 128)
    else:
        tensor.normal_()
    # PyTorch ignores the stride of singleton dimensions when determining
    # contiguity, so num_tokens=1 is logically strided but reports contiguous.
    if shape[0] > 1:
        assert tensor.is_contiguous() != strided
    return tensor


def _make_cache(
    shape: tuple[int, ...], dtype: torch.dtype, strided: bool
) -> torch.Tensor:
    if strided:
        storage = torch.empty((shape[0], 2, *shape[1:]), dtype=dtype, device="cuda")
        cache = storage[:, 0]
    else:
        cache = torch.empty(shape, dtype=dtype, device="cuda")
    if dtype == torch.int8:
        cache.random_(-128, 128)
    else:
        cache.normal_()
    assert cache.is_contiguous() != strided
    return cache


def _ref_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_scale: torch.Tensor,
    key_scale_cache: torch.Tensor,
) -> None:
    valid = slot_mapping >= 0
    token_indices = torch.where(valid)[0]
    slots = slot_mapping[valid]
    block_size = key_cache.shape[2]
    block_indices = torch.div(slots, block_size, rounding_mode="floor")
    block_offsets = slots % block_size
    key_cache[block_indices, :, block_offsets, :] = key[token_indices]
    value_cache[block_indices, :, block_offsets, :] = value[token_indices]
    key_scale_cache[block_indices, :, block_offsets] = key_scale[token_indices]


def _make_slot_mapping(
    num_tokens: int, num_blocks: int, block_size: int
) -> torch.Tensor:
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device="cuda"
    )[:num_tokens].contiguous()
    if num_tokens > 1:
        slot_mapping[0] = -1
    return slot_mapping


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("strided_input", [False, True])
@pytest.mark.parametrize("strided_cache", [False, True])
@torch.inference_mode()
def test_reshape_and_cache_with_scale(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    strided_input: bool,
    strided_cache: bool,
) -> None:
    torch.manual_seed(0)
    shape = (num_tokens, num_heads, head_size)
    cache_shape = (num_blocks, num_heads, block_size, head_size)
    scale_cache_shape = (num_blocks, num_heads, block_size)

    key = _make_tensor(shape, torch.int8, strided_input)
    value = _make_tensor(shape, torch.int8, strided_input)
    key_scale = torch.randn(num_tokens, num_heads, dtype=torch.float32, device="cuda")
    key_cache = _make_cache(cache_shape, torch.int8, strided_cache)
    value_cache = _make_cache(cache_shape, torch.int8, strided_cache)
    key_scale_cache = _make_cache(scale_cache_shape, torch.float32, strided_cache)
    slot_mapping = _make_slot_mapping(num_tokens, num_blocks, block_size)

    ref_key_cache = key_cache.clone()
    ref_value_cache = value_cache.clone()
    ref_key_scale_cache = key_scale_cache.clone()
    _ref_reshape_and_cache(
        key,
        value,
        ref_key_cache,
        ref_value_cache,
        slot_mapping,
        key_scale,
        ref_key_scale_cache,
    )

    torch.ops._C.reshape_and_cache_with_scale(
        key,
        value,
        key_scale,
        key_cache,
        value_cache,
        key_scale_cache,
        slot_mapping,
    )

    torch.testing.assert_close(key_cache, ref_key_cache, rtol=0, atol=0)
    torch.testing.assert_close(value_cache, ref_value_cache, rtol=0, atol=0)
    torch.testing.assert_close(key_scale_cache, ref_key_scale_cache, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize(
    "strided_input",
    [
        False,
    ],
)
@pytest.mark.parametrize("strided_cache", [False, True])
@torch.inference_mode()
def test_reshape_and_cache_mix(
    dtype: torch.dtype,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    strided_input: bool,
    strided_cache: bool,
) -> None:
    torch.manual_seed(0)
    shape = (num_tokens, num_heads, head_size)
    cache_shape = (num_blocks, num_heads, block_size, head_size)
    scale_cache_shape = (num_blocks, num_heads, block_size)

    key = _make_tensor(shape, torch.int8, strided_input)
    value = _make_tensor(shape, dtype, strided_input)
    key_scale = torch.randn(num_tokens, num_heads, dtype=torch.float32, device="cuda")
    key_cache = _make_cache(cache_shape, torch.int8, strided_cache)
    value_cache = _make_cache(cache_shape, dtype, strided_cache)
    key_scale_cache = _make_cache(scale_cache_shape, torch.float32, strided_cache)
    slot_mapping = _make_slot_mapping(num_tokens, num_blocks, block_size)

    ref_key_cache = key_cache.clone()
    ref_value_cache = value_cache.clone()
    ref_key_scale_cache = key_scale_cache.clone()
    _ref_reshape_and_cache(
        key,
        value,
        ref_key_cache,
        ref_value_cache,
        slot_mapping,
        key_scale,
        ref_key_scale_cache,
    )

    torch.ops._C.reshape_and_cache_mix(
        key,
        value,
        key_scale,
        key_cache,
        value_cache,
        key_scale_cache,
        slot_mapping,
    )

    torch.testing.assert_close(key_cache, ref_key_cache, rtol=0, atol=0)
    torch.testing.assert_close(value_cache, ref_value_cache, rtol=0, atol=0)
    torch.testing.assert_close(key_scale_cache, ref_key_scale_cache, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("op_name", "value_dtype"),
    [
        ("reshape_and_cache_with_scale", torch.int8),
        ("reshape_and_cache_mix", torch.float16),
    ],
)
@torch.inference_mode()
def test_reshape_and_cache_opcheck(op_name: str, value_dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    num_tokens, num_heads, head_size = 17, 4, 128
    num_blocks, block_size = 2, 16
    shape = (num_tokens, num_heads, head_size)
    cache_shape = (num_blocks, num_heads, block_size, head_size)

    key = _make_tensor(shape, torch.int8, False)
    value = _make_tensor(shape, value_dtype, False)
    key_cache = _make_cache(cache_shape, torch.int8, False)
    value_cache = _make_cache(cache_shape, value_dtype, False)
    slot_mapping = _make_slot_mapping(num_tokens, num_blocks, block_size)
    key_scale = torch.randn(num_tokens, num_heads, device="cuda")
    key_scale_cache = torch.randn(num_blocks, num_heads, block_size, device="cuda")

    torch.library.opcheck(
        getattr(torch.ops._C, op_name),
        (
            key,
            value,
            key_scale,
            key_cache,
            value_cache,
            key_scale_cache,
            slot_mapping,
        ),
    )
