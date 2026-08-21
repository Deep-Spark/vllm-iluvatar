# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness of torch.ops._C.rotary_embedding (self-contained kernel).

Two layers of coverage:

A. Module level -- mirrors upstream tests/kernels/core/test_pos_encoding.py
   ::test_rotary_embedding. Builds a real rope module via ``get_rope`` and checks
   ``forward_cuda`` (community path -> ops.rotary_embedding -> our _C kernel)
   against ``forward_native``. This validates the full path after removing the
   plugin's 10 forward_cuda OOT overrides, across the flat / headed / padded
   layouts, is_neox styles, key=None, odd head counts, large seq and partial
   rotary_dim, plus the fp16 dtype upstream omits.

B. Bare op level -- exercises ``torch.ops._C.rotary_embedding`` directly against a
   pure-torch reference for the DeepSeek/ILU dimensions upstream does not cover:
   rope_dim_offset > 0, inverse=True, GQA (num_kv_heads < num_heads), a
   cos_sin_cache dtype that differs from q/k (fp32 cache + bf16/fp16 q/k), and
   strided (non-contiguous) query/key.

The native extension is a mandatory dep (importing it registers the op); only the
actual GPU device is gated via skip.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

# Native extension is a MANDATORY dep (importing it registers
# torch.ops._C.rotary_embedding): a missing build is an environment error
# (collection-time ImportError), never a silent skip.
import vllm_iluvatar._C  # noqa: F401
from vllm.model_executor.layers.rotary_embedding import get_rope

if not torch.cuda.is_available():
    pytest.skip("Iluvatar/CUDA device required", allow_module_level=True)

DTYPES = [torch.bfloat16, torch.float, torch.half]
HEAD_SIZES = [64, 80, 120, 256]
ROTARY_DIMS = [None, 32]  # None means rotary dim == head size
NUM_HEADS = [17]  # arbitrary / odd on purpose
BATCH_SIZES = [5]
SEQ_LENS = [11, 8192]
SEEDS = [0]
IS_NEOX_STYLE = [True, False]
USE_KEY = [True, False]


def _atol_rtol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float:
        return 1e-4, 1e-4
    if dtype == torch.half:
        return 1e-2, 1e-2
    return 2e-2, 2e-2  # bfloat16


# --- reference ---------------------------------------------------------------
def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _ref_rotary_embedding(
    positions: torch.Tensor,  # [num_tokens]
    query: torch.Tensor,  # [num_tokens, num_heads, head_size]
    key: torch.Tensor | None,
    cos_sin_cache: torch.Tensor,  # [max_pos, rot_dim]
    is_neox: bool,
    rope_dim_offset: int,
    inverse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    rot_dim = cos_sin_cache.shape[-1]
    cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)  # [T, embed]
    if is_neox:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    else:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
    cos = cos.unsqueeze(-2).float()  # [T, 1, rot_dim]
    sin = sin.unsqueeze(-2).float()
    if inverse:
        sin = -sin
    rotate_fn = _rotate_neox if is_neox else _rotate_gptj

    def apply(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None:
            return None
        out = t.clone()
        seg = out[..., rope_dim_offset : rope_dim_offset + rot_dim].float()
        rotated = seg * cos + rotate_fn(seg) * sin
        out[..., rope_dim_offset : rope_dim_offset + rot_dim] = rotated.to(t.dtype)
        return out

    return apply(query), apply(key)


# --- layer A: module level (mirrors upstream test_rotary_embedding) ----------
def _flat_shape(b, s, h, d):
    return (b, s, h * d)


def _headed_shape(b, s, h, d):
    return (b, s, h, d)


def _padded_shape(b, s, h, d):  # sliced later -> non-contiguous head stride
    return (b, s, h, d + 64)


TENSOR_SHAPE_FNS = [_flat_shape, _headed_shape, _padded_shape]


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("tensor_shape_fn", TENSOR_SHAPE_FNS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("use_key", USE_KEY)
@torch.inference_mode()
def test_rotary_embedding_module(
    default_vllm_config,
    is_neox_style: bool,
    tensor_shape_fn: Callable[[int, int, int, int], tuple[int, ...]],
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: int | None,
    dtype: torch.dtype,
    seed: int,
    use_key: bool,
) -> None:
    if rotary_dim is None:
        rotary_dim = head_size
    max_position = 8192

    torch.manual_seed(seed)
    device = "cuda:0"

    rope_parameters = {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": rotary_dim / head_size,
    }
    rope = get_rope(head_size, max_position, is_neox_style, rope_parameters)
    rope = rope.to(dtype=dtype, device=device)
    # forward_cuda must exercise the native kernel, not flashinfer.
    rope.use_flashinfer = False

    positions = torch.randint(0, max_position, (batch_size, seq_len), device=device)
    query_shape = tensor_shape_fn(batch_size, seq_len, num_heads, head_size)
    query = torch.randn(query_shape, dtype=dtype, device=device)[..., :head_size]
    key = torch.randn_like(query) if use_key else None

    # Reference first (custom kernel is in-place).
    ref_query, ref_key = rope.forward_native(
        positions, query.clone(), key.clone() if key is not None else None
    )
    out_query, out_key = rope.forward_cuda(
        positions, query.clone(), key.clone() if key is not None else None
    )

    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(out_query, ref_query, atol=atol, rtol=rtol)
    if use_key:
        torch.testing.assert_close(out_key, ref_key, atol=atol, rtol=rtol)
    else:
        assert ref_key is None and out_key is None


# --- layer B: bare op (ILU/DeepSeek dimensions upstream does not test) -------
def _make_cache(max_pos: int, rot_dim: int, dtype: torch.dtype) -> torch.Tensor:
    # cos_sin_cache = [cos(embed) || sin(embed)] with embed = rot_dim / 2.
    embed = rot_dim // 2
    angles = torch.randn(max_pos, embed, dtype=torch.float32) * 3.0
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return cache.to(dtype)


@pytest.mark.parametrize("is_neox", IS_NEOX_STYLE)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_size", [64, 128, 256, 120])
@pytest.mark.parametrize("rotary_dim_ratio", [0.5, 1.0])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (8, 2), (8, 1)])
@pytest.mark.parametrize("use_key", [True, False])
@torch.inference_mode()
def test_rotary_embedding_op_gqa_partial(
    is_neox: bool,
    dtype: torch.dtype,
    head_size: int,
    rotary_dim_ratio: float,
    num_heads: int,
    num_kv_heads: int,
    use_key: bool,
) -> None:
    torch.manual_seed(0)
    device = "cuda:0"
    num_tokens = 133
    max_pos = 4096
    rot_dim = int(head_size * rotary_dim_ratio)
    rot_dim -= rot_dim % 2  # rot_dim must be even
    if rot_dim == 0:
        pytest.skip("degenerate rot_dim")

    positions = torch.randint(0, max_pos, (num_tokens,), device=device)
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    key = (
        torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
        if use_key
        else None
    )
    cache = _make_cache(max_pos, rot_dim, dtype).to(device)

    ref_q, ref_k = _ref_rotary_embedding(
        positions, query, key, cache, is_neox, rope_dim_offset=0, inverse=False
    )

    out_q = query.clone()
    out_k = key.clone() if key is not None else None
    torch.ops._C.rotary_embedding(positions, out_q, out_k, head_size, cache, is_neox)

    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(out_q, ref_q, atol=atol, rtol=rtol)
    if use_key:
        torch.testing.assert_close(out_k, ref_k, atol=atol, rtol=rtol)


@pytest.mark.parametrize("is_neox", IS_NEOX_STYLE)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inverse", [False, True])
@torch.inference_mode()
def test_rotary_embedding_op_offset_inverse(
    is_neox: bool, dtype: torch.dtype, inverse: bool
) -> None:
    """DeepSeek indexer path: rope applied to the last rot_dim of the head."""
    torch.manual_seed(0)
    device = "cuda:0"
    num_tokens = 97
    num_heads = 4
    head_size = 128
    rot_dim = 64
    rope_dim_offset = head_size - rot_dim  # rotate the trailing slice
    max_pos = 4096

    positions = torch.randint(0, max_pos, (num_tokens,), device=device)
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    key = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    cache = _make_cache(max_pos, rot_dim, dtype).to(device)

    ref_q, ref_k = _ref_rotary_embedding(
        positions, query, key, cache, is_neox, rope_dim_offset, inverse
    )

    out_q = query.clone()
    out_k = key.clone()
    torch.ops._C.rotary_embedding(
        positions,
        out_q,
        out_k,
        head_size,
        cache,
        is_neox,
        rope_dim_offset,
        inverse,
    )

    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(out_q, ref_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_k, ref_k, atol=atol, rtol=rtol)


@pytest.mark.parametrize("is_neox", IS_NEOX_STYLE)
@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.half])
@torch.inference_mode()
def test_rotary_embedding_op_fp32_cache(is_neox: bool, q_dtype: torch.dtype) -> None:
    """fp32 cos_sin_cache with bf16/fp16 q/k (DeepSeek stores cache in fp32)."""
    torch.manual_seed(0)
    device = "cuda:0"
    num_tokens = 128
    num_heads = 8
    head_size = 128
    rot_dim = 128
    max_pos = 4096

    positions = torch.randint(0, max_pos, (num_tokens,), device=device)
    query = torch.randn(num_tokens, num_heads, head_size, dtype=q_dtype, device=device)
    key = torch.randn(num_tokens, num_heads, head_size, dtype=q_dtype, device=device)
    cache = _make_cache(max_pos, rot_dim, torch.float32).to(device)

    ref_q, ref_k = _ref_rotary_embedding(
        positions, query, key, cache, is_neox, rope_dim_offset=0, inverse=False
    )

    out_q = query.clone()
    out_k = key.clone()
    torch.ops._C.rotary_embedding(positions, out_q, out_k, head_size, cache, is_neox)

    atol, rtol = _atol_rtol(q_dtype)
    torch.testing.assert_close(out_q, ref_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_k, ref_k, atol=atol, rtol=rtol)


@pytest.mark.parametrize("is_neox", IS_NEOX_STYLE)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_rotary_embedding_op_strided(is_neox: bool, dtype: torch.dtype) -> None:
    """Non-contiguous query/key (token-dim slice of a [q|k] packed buffer)."""
    torch.manual_seed(0)
    device = "cuda:0"
    num_tokens = 120
    num_heads = 8
    head_size = 128
    rot_dim = 128
    max_pos = 4096

    positions = torch.randint(0, max_pos, (num_tokens,), device=device)
    # Allocate a wide buffer then slice the head dim -> non-contiguous head stride.
    q_wide = torch.randn(
        num_tokens, num_heads, head_size + 64, dtype=dtype, device=device
    )
    query = q_wide[..., :head_size]
    k_wide = torch.randn(
        num_tokens, num_heads, head_size + 64, dtype=dtype, device=device
    )
    key = k_wide[..., :head_size]
    assert not query.is_contiguous()
    cache = _make_cache(max_pos, rot_dim, dtype).to(device)

    ref_q, ref_k = _ref_rotary_embedding(
        positions, query, key, cache, is_neox, rope_dim_offset=0, inverse=False
    )

    # Operate on fresh strided views so the kernel must honour head_stride
    # (stride(-2) == head_size + 64, not head_size).
    q_test = q_wide.clone()[..., :head_size]
    k_test = k_wide.clone()[..., :head_size]
    assert not q_test.is_contiguous()
    torch.ops._C.rotary_embedding(positions, q_test, k_test, head_size, cache, is_neox)

    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(q_test, ref_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_test, ref_k, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_rotary_embedding_opcheck(dtype: torch.dtype) -> None:
    """torch.library.opcheck: schema / alias / fake consistency."""
    device = "cuda:0"
    num_tokens = 32
    num_heads = 4
    head_size = 64
    rot_dim = 64
    max_pos = 512

    positions = torch.randint(0, max_pos, (num_tokens,), device=device)
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    key = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    cache = _make_cache(max_pos, rot_dim, dtype).to(device)

    torch.library.opcheck(
        torch.ops._C.rotary_embedding,
        (positions, query, key, head_size, cache, True),
    )
