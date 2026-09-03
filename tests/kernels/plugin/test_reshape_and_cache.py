# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness of the plugin paged-cache write operators."""

from __future__ import annotations

import pytest
import torch
import torch.library
import vllm_iluvatar._C  # noqa: F401
from conftest import COMPILED_CUDA_ARCH

if not torch.cuda.is_available():
    pytest.skip("Iluvatar/CUDA device required", allow_module_level=True)

# The FP8 E4M3 cache/quant kernels exist on ivcore30 only; elsewhere the ops
# stay bound but raise, so each direction gets its own gate.
ON_FP8_ARCH = COMPILED_CUDA_ARCH == "ivcore30"
requires_fp8_arch = pytest.mark.skipif(
    not ON_FP8_ARCH,
    reason=f"FP8 kernels need an ivcore30 build, got {COMPILED_CUDA_ARCH}",
)
requires_non_fp8_arch = pytest.mark.skipif(
    ON_FP8_ARCH,
    reason="only meaningful on a build without the ivcore30 FP8 kernels",
)

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
    elif dtype == torch.float8_e4m3fn:
        tensor.copy_(torch.randn(shape, dtype=torch.bfloat16, device="cuda"))
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
    elif dtype == torch.float8_e4m3fn:
        cache.copy_(torch.randn(shape, dtype=torch.bfloat16, device="cuda"))
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


@requires_fp8_arch
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_size", [13, 128])
@pytest.mark.parametrize("strided_input", [False, True])
@pytest.mark.parametrize("scale_value", [0.125, 0.001])
@torch.inference_mode()
def test_static_scaled_fp8_quant(
    dtype: torch.dtype,
    head_size: int,
    strided_input: bool,
    scale_value: float,
) -> None:
    torch.manual_seed(43)
    num_tokens, num_heads = 17, 16
    storage = torch.randn(
        num_tokens,
        2,
        num_heads,
        head_size,
        dtype=dtype,
        device="cuda",
    )
    input_tensor = storage[:, 0]
    if not strided_input:
        input_tensor = input_tensor.contiguous()

    scale = torch.tensor([scale_value], dtype=torch.float32, device="cuda")
    output = torch.empty(
        input_tensor.shape,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    reference = (
        (input_tensor.float() / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    )

    torch.ops._C.static_scaled_fp8_quant(output, input_tensor, scale)

    torch.testing.assert_close(output.float(), reference.float(), rtol=0, atol=0)


@requires_fp8_arch
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 17, 128, 1024])
@pytest.mark.parametrize("diff_kv", [True, False])
@pytest.mark.parametrize("strided_cache", [True, False])
@pytest.mark.parametrize("num_padded_tokens", [0, 7])
@torch.inference_mode()
def test_reshape_and_cache_fp8(
    dtype: torch.dtype,
    num_tokens: int,
    diff_kv: bool,
    strided_cache: bool,
    num_padded_tokens: int,
) -> None:
    torch.manual_seed(43)
    num_heads, head_size = 16, 128
    num_blocks, block_size = 64, 16
    fp8_dtype = torch.float8_e4m3fn

    # unbind keeps the ixformer test's padded token stride while the last two
    # dimensions remain contiguous. Extra rows reproduce V1 CUDA-graph padding:
    # slot_mapping still describes only the real tokens that should be cached.
    kv = torch.randn(
        num_tokens + num_padded_tokens,
        2,
        num_heads,
        head_size,
        dtype=dtype,
        device="cuda",
    )
    key, value = kv.unbind(dim=1)
    if diff_kv:
        value = value[..., :96]

    cache_shape = (num_blocks, num_heads, block_size, head_size)
    key_cache = _make_cache(cache_shape, fp8_dtype, strided_cache)
    value_cache = _make_cache(cache_shape, fp8_dtype, strided_cache)
    slot_mapping = _make_slot_mapping(num_tokens, num_blocks, block_size)
    key_scale = torch.tensor([0.125], dtype=torch.float32, device="cuda")
    value_scale = torch.tensor([0.25], dtype=torch.float32, device="cuda")

    ref_key_cache = key_cache.clone()
    ref_value_cache = value_cache.clone()
    fp8_max = torch.finfo(fp8_dtype).max
    quantized_key = (key.float() / key_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    quantized_value = (
        (value.float() / value_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    )
    valid = slot_mapping >= 0
    token_indices = torch.where(valid)[0]
    slots = slot_mapping[valid]
    block_indices = torch.div(slots, block_size, rounding_mode="floor")
    block_offsets = slots % block_size
    ref_key_cache[block_indices, :, block_offsets, :] = quantized_key[token_indices]
    ref_value_cache[block_indices, :, block_offsets, : value.size(2)] = quantized_value[
        token_indices
    ]

    torch.ops._C.reshape_and_cache_fp8(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "fp8",
        key_scale,
        value_scale,
    )

    torch.testing.assert_close(key_cache.float(), ref_key_cache.float(), rtol=0, atol=0)
    torch.testing.assert_close(
        value_cache.float(), ref_value_cache.float(), rtol=0, atol=0
    )


def test_fp8_ops_are_bound_on_every_arch() -> None:
    """The FP8 kernels are ivcore30-only, but their schemas never disappear."""
    assert COMPILED_CUDA_ARCH in ("ivcore11", "ivcore30")
    assert hasattr(torch.ops._C, "static_scaled_fp8_quant")
    assert hasattr(torch.ops._C, "reshape_and_cache_fp8")


@requires_non_fp8_arch
@torch.inference_mode()
def test_fp8_ops_raise_without_ivcore30_kernels() -> None:
    """Off ivcore30 the bound ops must fail loudly, naming the missing arch."""
    num_tokens, num_heads, head_size = 4, 2, 128
    num_blocks, block_size = 2, 16
    fp8_dtype = torch.float8_e4m3fn

    key = torch.randn(
        num_tokens, num_heads, head_size, dtype=torch.float16, device="cuda"
    )
    value = torch.randn_like(key)
    cache_shape = (num_blocks, num_heads, block_size, head_size)
    # Left uninitialised: writing FP8 values needs the very kernels this build
    # lacks, and the ops must reject the call before touching the buffers.
    key_cache = torch.empty(cache_shape, dtype=fp8_dtype, device="cuda")
    value_cache = torch.empty(cache_shape, dtype=fp8_dtype, device="cuda")
    quantized = torch.empty(key.shape, dtype=fp8_dtype, device="cuda")
    slot_mapping = _make_slot_mapping(num_tokens, num_blocks, block_size)
    scale = torch.tensor([0.125], dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="ivcore30"):
        torch.ops._C.static_scaled_fp8_quant(quantized, key, scale)

    with pytest.raises(RuntimeError, match="ivcore30"):
        torch.ops._C.reshape_and_cache_fp8(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            "fp8",
            scale,
            scale,
        )
