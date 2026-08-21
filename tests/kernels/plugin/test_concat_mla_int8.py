# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness of torch.ops._C.concat_and_cache_dsa_int8.

Exercises the plugin's native cache-write operator directly and checks it
against a plain PyTorch reference. The native build is a mandatory dependency
(missing -> hard import error); only the actual GPU device is gated via skip.
"""

from __future__ import annotations

import pytest

# Native extension is a MANDATORY dep (importing it registers
# torch.ops._C.concat_and_cache_dsa_int8): a missing build is an environment
# error (collection-time ImportError), never a silent skip.
import torch
import torch.library
import vllm_iluvatar._C  # noqa: F401

if not torch.cuda.is_available():
    pytest.skip("Iluvatar/CUDA device required", allow_module_level=True)

NUM_TOKENS = [1121]
KV_DIMS = [576]
BLOCK_SIZES = [64]
NUM_BLOCKS = [32]
SEEDS = [0]


def _ref_concat_and_cache_dsa_int8(
    kv_int8: torch.Tensor,
    nope_scale: torch.Tensor,
    rope_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    nope_scale_cache: torch.Tensor,
    rope_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Reference implementation of the paged-cache writes."""
    block_size = kv_cache.shape[1]
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_offsets = slot_mapping % block_size

    kv_cache[block_indices, block_offsets] = kv_int8
    nope_scale_cache[block_indices, block_offsets] = nope_scale
    rope_scale_cache[block_indices, block_offsets] = rope_scale


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("kv_dim", KV_DIMS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_concat_and_cache_dsa_int8(
    num_tokens: int,
    kv_dim: int,
    block_size: int,
    num_blocks: int,
    seed: int,
    strided_input: bool,
) -> None:
    torch.manual_seed(seed)
    device = "cuda:0"

    input_width = 2 * kv_dim if strided_input else kv_dim
    scale_width = 2 if strided_input else 1
    kv_int8 = torch.randint(
        -128, 128, (num_tokens, input_width), dtype=torch.int8, device=device
    )
    kv_int8 = kv_int8[..., :kv_dim]
    nope_scale = torch.randn(num_tokens, scale_width, device=device)[..., :1]
    rope_scale = torch.randn(num_tokens, scale_width, device=device)[..., :1]
    assert kv_int8.is_contiguous() != strided_input
    assert nope_scale.is_contiguous() != strided_input
    assert rope_scale.is_contiguous() != strided_input

    # Non-zero initialization also verifies that unmapped slots stay untouched.
    kv_cache = torch.randint(
        -128,
        128,
        (num_blocks, block_size, kv_dim),
        dtype=torch.int8,
        device=device,
    )
    nope_scale_cache = torch.randn(num_blocks, block_size, 1, device=device)
    rope_scale_cache = torch.randn(num_blocks, block_size, 1, device=device)

    # Duplicate slots would make parallel last-writer ordering ambiguous.
    total_slots = num_blocks * block_size
    slot_mapping = torch.randperm(total_slots, dtype=torch.int64, device=device)[
        :num_tokens
    ].contiguous()

    ref_kv_cache = kv_cache.clone()
    ref_nope_scale_cache = nope_scale_cache.clone()
    ref_rope_scale_cache = rope_scale_cache.clone()
    _ref_concat_and_cache_dsa_int8(
        kv_int8,
        nope_scale,
        rope_scale,
        ref_kv_cache,
        ref_nope_scale_cache,
        ref_rope_scale_cache,
        slot_mapping,
    )

    torch.ops._C.concat_and_cache_dsa_int8(
        kv_int8,
        nope_scale,
        rope_scale,
        kv_cache,
        nope_scale_cache,
        rope_scale_cache,
        slot_mapping,
    )

    torch.testing.assert_close(kv_cache, ref_kv_cache, rtol=0, atol=0)
    torch.testing.assert_close(nope_scale_cache, ref_nope_scale_cache, rtol=0, atol=0)
    torch.testing.assert_close(rope_scale_cache, ref_rope_scale_cache, rtol=0, atol=0)


@torch.inference_mode()
def test_concat_and_cache_dsa_int8_opcheck() -> None:
    """torch.library.opcheck: schema/alias/fake consistency for the cache op."""
    device = "cuda:0"
    num_tokens, kv_dim, block_size, num_blocks = 83, 576, 64, 2
    kv_int8 = torch.randint(
        -128, 128, (num_tokens, kv_dim), dtype=torch.int8, device=device
    )
    nope_scale = torch.randn(num_tokens, 1, device=device)
    rope_scale = torch.randn(num_tokens, 1, device=device)
    kv_cache = torch.zeros(
        num_blocks, block_size, kv_dim, dtype=torch.int8, device=device
    )
    nope_scale_cache = torch.zeros(num_blocks, block_size, 1, device=device)
    rope_scale_cache = torch.zeros(num_blocks, block_size, 1, device=device)
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device=device
    )[:num_tokens]

    torch.library.opcheck(
        torch.ops._C.concat_and_cache_dsa_int8,
        (
            kv_int8,
            nope_scale,
            rope_scale,
            kv_cache,
            nope_scale_cache,
            rope_scale_cache,
            slot_mapping,
        ),
    )
