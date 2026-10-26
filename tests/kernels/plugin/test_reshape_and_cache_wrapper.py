"""Plugin-to-ixformer unquantized cache ABI and real large packed backing."""

import pytest
import torch
from conftest import requires_corex30
from ixformer.core import config

from vllm_iluvatar import _custom_ops as ops


@requires_corex30
@pytest.mark.parametrize("torch_ops", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("packed", [False, True])
def test_wrapper_updates_exact_slots(dtype, packed, torch_ops, monkeypatch):
    monkeypatch.setattr(config, "IXFORMER_USE_TORCH_OPS", torch_ops)
    torch.manual_seed(713)
    source = torch.randn(3, 2, 4, 192, device="cuda", dtype=dtype)
    key, value = source[:, 0], source[:, 1]
    pair = torch.full(
        (5, 4, 16, 384 if packed else 192), 7.0, device="cuda", dtype=dtype
    )
    kc = pair[..., :192]
    vc = pair[..., 192:] if packed else torch.full_like(kc, 7.0)
    slots = torch.tensor([2, -1, 4 * 16 + 7], device="cuda", dtype=torch.int64)
    expected_k, expected_v = kc.clone(), vc.clone()
    for i, slot in [(0, 2), (2, 4 * 16 + 7)]:
        expected_k[slot // 16, :, slot % 16] = key[i]
        expected_v[slot // 16, :, slot % 16] = value[i]
    scale = torch.ones((), device="cuda")
    ops.reshape_and_cache_flash(key, value, kc, vc, slots, "auto", scale, scale)
    torch.testing.assert_close(kc, expected_k, atol=0, rtol=0)
    torch.testing.assert_close(vc, expected_v, atol=0, rtol=0)


@requires_corex30
@pytest.mark.parametrize("torch_ops", [False, True])
def test_wrapper_writes_above_4gib_without_wrapping_low_page(torch_ops, monkeypatch):
    monkeypatch.setattr(config, "IXFORMER_USE_TORCH_OPS", torch_ops)
    high = 65537
    backing = torch.empty((high + 1, 32768), device="cuda", dtype=torch.bfloat16)
    assert backing.untyped_storage().nbytes() > 4 * 1024**3
    shape, stride = (high + 1, 4, 16, 192), (32768, 6144, 384, 1)
    kc = backing.as_strided(shape, stride)
    vc = backing.as_strided(shape, stride, 192)
    assert high * kc.stride(0) * kc.element_size() > 4 * 1024**3
    for page in [0, 1, high]:
        kc[page].fill_(7)
        vc[page].fill_(9)
    key = (
        torch.arange(2 * 4 * 192, device="cuda", dtype=torch.float32)
        .reshape(2, 4, 192)
        .to(torch.bfloat16)
        / 256
    )
    value = -key
    slots = torch.tensor([2, high * 16 + 3], device="cuda", dtype=torch.int64)
    scale = torch.ones((), device="cuda")
    ops.reshape_and_cache_flash(key, value, kc, vc, slots, "auto", scale, scale)
    torch.testing.assert_close(kc[0, :, 2], key[0], atol=0, rtol=0)
    torch.testing.assert_close(vc[0, :, 2], value[0], atol=0, rtol=0)
    torch.testing.assert_close(kc[high, :, 3], key[1], atol=0, rtol=0)
    torch.testing.assert_close(vc[high, :, 3], value[1], atol=0, rtol=0)
    assert (kc[1] == 7).all() and (vc[1] == 9).all()


def test_wrapper_rejects_quantized_contract_before_any_write():
    x = torch.ones(1, 1, 1)
    cache = torch.full((1, 1, 1, 1), 7.0)
    with pytest.raises(ValueError, match="Unquantized cache writer"):
        ops.reshape_and_cache_flash(
            x, x, cache, cache, torch.zeros(1, dtype=torch.int64), "fp8", x, x
        )
    assert cache.item() == 7
