# SPDX-License-Identifier: Apache-2.0
"""Numerical and state-preservation checks for the independent cache writer."""

import pytest
import torch

from tests.ixaite.backend_isolation.ixformer_guard import forbid_ixformer_calls
from vllm_iluvatar.backends.ixaite import ops as ilu_ops


# Calls from weight/load/forward helpers must never reach a vendor op.
@pytest.fixture(autouse=True)
def _strict_lifecycle():
    with forbid_ixformer_calls():
        yield


def _ref_reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping):
    valid = slot_mapping >= 0
    token_indices = torch.where(valid)[0]
    slots = slot_mapping[valid].long()
    blocks = slots // 16
    offsets = slots % 16
    key_cache[blocks, :, offsets, :] = key[token_indices]
    value_cache[blocks, :, offsets, :] = value[token_indices]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_qwen36_27b_reshape_and_cache_flash_ixaite_wrapper_matches_reference(
    monkeypatch, kv_heads
) -> None:
    torch.manual_seed(7)
    num_tokens, num_blocks = 11, 4
    key = torch.randn(num_tokens, kv_heads, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    packed = torch.randn(
        2, num_blocks, kv_heads, 16, 256, device="cuda", dtype=torch.bfloat16
    )
    key_cache, value_cache = packed.unbind(0)
    slots = torch.tensor(
        [0, 17, -1, 33, 5, 48, -1, 18, 31, 46, 63],
        device="cuda",
        dtype=torch.int32,
    )
    k_scale = torch.ones((), device="cuda", dtype=torch.float32)
    v_scale = torch.ones((), device="cuda", dtype=torch.float32)

    expected_key = key_cache.clone()
    expected_value = value_cache.clone()
    _ref_reshape_and_cache_flash(key, value, expected_key, expected_value, slots)

    ilu_ops.reshape_and_cache_flash(
        key, value, key_cache, value_cache, slots, "auto", k_scale, v_scale
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(key_cache, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache, expected_value, rtol=0, atol=0)
