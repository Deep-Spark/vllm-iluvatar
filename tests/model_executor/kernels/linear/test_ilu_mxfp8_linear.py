# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar MXFP8 dense linear kernel and its wiring.

``IluPlatform._enum`` is CUDA, so every community MXFP8 kernel is a candidate on
Iluvatar and ``FlashInferCutlassMxfp8LinearKernel`` actually wins -- then fails
for want of CUTLASS. ``register_linear_kernel`` only appends, so the patch must
*prepend*; that priority is what these tests pin, along with the UE8M0 scale
handling (a zero-copy 1-byte view, never a numeric cast) and the guard against a
layer whose load-time requant did not run.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import vllm_iluvatar.model_executor.kernels.linear.mxfp8.ilu_mxfp8_linear as ilu_mod
from vllm_iluvatar.model_executor.kernels.linear.mxfp8.ilu_mxfp8_linear import (
    IluMxfp8LinearKernel,
)

_BLOCK = 32
_N = 64
_K = 128


def _make_layer(n=_N, k=_K, scale_dtype=torch.uint8):
    layer = nn.Module()
    weight = torch.randn(n, k).to(torch.float8_e4m3fn)
    scale = torch.full((n, k // _BLOCK), 0x7F, dtype=torch.uint8)
    if scale_dtype is not torch.uint8:
        scale = scale.view(scale_dtype)
    layer.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
    layer.register_parameter("weight_scale", nn.Parameter(scale, requires_grad=False))
    return layer


def _make_kernel(monkeypatch):
    monkeypatch.setattr(ilu_mod, "_is_ivcore30_fn", lambda: True)
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )

    return IluMxfp8LinearKernel(Mxfp8LinearLayerConfig())


def test_arch_gate(monkeypatch):
    monkeypatch.setattr(ilu_mod, "_is_ivcore30_fn", lambda: False)
    supported, reason = IluMxfp8LinearKernel.is_supported()
    assert supported is False
    assert "ivcore30" in reason

    monkeypatch.setattr(ilu_mod, "_is_ivcore30_fn", lambda: True)
    assert IluMxfp8LinearKernel.is_supported() == (True, None)


@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e8m0fnu])
def test_requant_uses_mxfp8_grid_and_zero_copy_scale_view(monkeypatch, scale_dtype):
    kernel = _make_kernel(monkeypatch)
    layer = _make_layer(scale_dtype=scale_dtype)
    raw_scale_bytes = layer.weight_scale.data.view(torch.uint8).clone()

    seen = {}

    def _fake_requant(weight, block_scale, group_n, group_k, fp8_output=None):
        seen["group"] = (group_n, group_k)
        seen["scale_dtype"] = block_scale.dtype
        seen["scale_bytes"] = block_scale.view(torch.uint8).clone()
        # ``weight.data`` yields a fresh view each access, so identity would not
        # hold even for a genuine in-place request; compare storage instead.
        seen["in_place"] = (
            fp8_output is not None and fp8_output.data_ptr() == weight.data_ptr()
        )
        return weight, torch.ones(weight.shape[0], dtype=torch.float32)

    # The kernel does ``import ixformer.inference.functions as ixfops``, whose
    # dotted-``as`` form resolves by attribute access on the parent package --
    # patching ``sys.modules`` would not be seen. Patch the real function.
    import ixformer.inference.functions as ixfops

    monkeypatch.setattr(
        ixfops, "fp8_weight_block_to_channel_requant", _fake_requant
    )

    kernel.process_weights_after_loading(layer)

    # MXFP8 is a [1, 32] grid; anything else silently mis-groups the scale.
    assert seen["group"] == (1, _BLOCK)
    # The op decodes float32 or float8_e8m0fnu; a uint8 checkpoint scale must be
    # reinterpreted, not converted (0x7F is 1.0, not 127.0).
    assert seen["scale_dtype"] is torch.float8_e8m0fnu
    assert torch.equal(seen["scale_bytes"], raw_scale_bytes)
    assert seen["in_place"] is True

    # Weight stays e4m3 [N, K]; only the scale becomes per-channel [N].
    assert layer.weight.dtype is torch.float8_e4m3fn
    assert tuple(layer.weight.shape) == (_N, _K)
    assert tuple(layer.weight_scale.shape) == (_N,)
    assert layer.weight_scale.dtype is torch.float32


def test_unaligned_k_is_rejected(monkeypatch):
    kernel = _make_kernel(monkeypatch)
    with pytest.raises(ValueError, match="K % 64 == 0"):
        kernel.process_weights_after_loading(_make_layer(k=_BLOCK * 3))


def test_scale_shape_mismatch_is_rejected(monkeypatch):
    kernel = _make_kernel(monkeypatch)
    layer = _make_layer()
    layer.register_parameter(
        "weight_scale",
        nn.Parameter(torch.zeros(_N, _K, dtype=torch.uint8), requires_grad=False),
    )
    with pytest.raises(ValueError, match="does not match"):
        kernel.process_weights_after_loading(layer)


def test_apply_rejects_a_layer_that_skipped_the_requant(monkeypatch):
    """A 2-D scale here means the load-time requant never ran for this layer."""
    kernel = _make_kernel(monkeypatch)
    layer = _make_layer()  # weight_scale is still [N, K/32]

    with pytest.raises(RuntimeError, match="per-channel"):
        kernel.apply_weights(layer, torch.zeros(1, _K, dtype=torch.bfloat16))


def test_patch_prepends_kernel_ahead_of_community_candidates(monkeypatch):
    import vllm.model_executor.kernels.linear as k
    from vllm.platforms import PlatformEnum

    import vllm_iluvatar.ilu_patch.general_patches.patch_mxfp8_linear_kernel as patch

    original = list(k._POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA])
    monkeypatch.setattr(
        "vllm_iluvatar.ilu_util.arch.is_ivcore30", lambda: True, raising=True
    )
    try:
        k._POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA][:] = [
            c for c in original if c is not IluMxfp8LinearKernel
        ]
        community_head = k._POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA][0]

        patch._apply()
        assert k._POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA][0] is IluMxfp8LinearKernel
        assert k._POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA][1] is community_head

        # Idempotent: apply_general_patches may run more than once.
        patch._apply()
        assert (
            k._POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA].count(IluMxfp8LinearKernel) == 1
        )
    finally:
        k._POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA][:] = original
