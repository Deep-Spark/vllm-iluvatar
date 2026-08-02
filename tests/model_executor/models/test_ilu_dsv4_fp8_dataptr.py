# SPDX-License-Identifier: Apache-2.0
"""CPU-only test (T4) for "the load path performs zero numeric transformation".

Design §4.3 ("加载期零数值变换的验证链") argues that on a CUDA-alike platform the
FP8 block-quantised weight and its ``float8_e8m0fnu`` scale reach the device
byte-for-byte as they sit in the checkpoint. The chain, verified against the
community sources vendored under ``vllm_plugin_dev/vllm`` (line numbers below are
the ones actually observed, not the ones quoted in design):

1. ``Fp8LinearMethod.process_weights_after_loading``
   (``vllm/model_executor/layers/quantization/fp8.py:398``) — in the
   ``block_quant`` branch the only statement is ``assert not self.act_q_static``
   (``:412-413``); it then hands over to
   ``self.fp8_linear.process_weights_after_loading(layer)`` (``:444``).
2. ``Fp8BlockScaledMMLinearKernel.process_weights_after_loading``
   (``vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:75``)
   picks ``weight_scale_inv`` when present, calls
   ``process_fp8_weight_block_strategy`` (``:89``) and writes the results back
   with ``replace_parameter`` (``:94-95``).
3. ``process_fp8_weight_block_strategy``
   (``vllm/model_executor/layers/quantization/utils/fp8_utils.py:1371``) — its
   only data-touching region is guarded by
   ``if current_platform.is_fp8_fnuz() and weight.dtype == torch.float8_e4m3fn:``
   (``:1380``), i.e. ROCm MI300x/MI325x only.
4. ``_maybe_pad_fp8_weight`` (same file, ``:1179``), invoked unconditionally at
   ``:1404``, is itself gated on
   ``envs.VLLM_ROCM_FP8_PADDING and current_platform.is_rocm()`` (``:1183-1184``).

Iluvatar's ``IluPlatform`` derives from ``CudaPlatform``, so both gates are
False and both functions degrade to identity.

Why ``data_ptr`` survives the write-back: ``replace_parameter``
(``vllm/model_executor/utils.py:47``) wraps the incoming tensor in a **fresh**
``torch.nn.Parameter`` (``:92``) and re-registers it on the layer (``:98``). It
never allocates or copies storage on this path — ``prefer_copy`` defaults to
False, and the ``old_param.copy_(new_data)`` arm (``:89``) is only reachable when
that flag is set. A new Python object over the same storage is exactly what
"``data_ptr`` unchanged, dtype unchanged" means.

``data_ptr`` equality alone is **not** sufficient evidence, which is why every
test below also compares the raw bytes: the ROCm arm at ``fp8_utils.py:1389-1390``
mutates the weight **in place** (``weight_as_int8[weight_as_int8 == -128] = 0``)
and would therefore keep ``data_ptr`` identical while silently rewriting bytes.

Scope notes:

* This file exercises **community code only** — no plugin model code, no vendor
  kernel library, no ``Fp8LinearMethod`` construction (which would need a quant
  config and a kernel selection). It is expected to pass as-is.
* Pure CPU. ``data_ptr()`` needs real storage, so a ``meta`` device is not usable
  here.

Validates: Requirements 1.1, 1.4
"""

from __future__ import annotations

import pytest
import torch

# --------------------------------------------------------------------------
# Fake layer geometry.
#
# Block-quantised FP8 linears hold ``weight`` as [N, K] =
# [output_size_per_partition, input_size_per_partition], and ``weight_scale_inv``
# as [ceil(N / block_n), ceil(K / block_k)] with block_shape = [128, 128]
# (the DS-V4-Flash checkpoint scale layout is
# [ceil(N/128), ceil(K/128)]).
#
# N = 256 -> 256 / 128 = 2 scale rows (>1, so a wrong row/col swap is visible)
# K = 128 -> 128 / 128 = 1 scale col
# => weight [256, 128], weight_scale_inv [2, 1]
#
# Kept tiny on purpose: the identity being tested is independent of size, and a
# small tensor keeps the byte-level comparison readable when it fails.
# --------------------------------------------------------------------------
N = 256
K = 128
BLOCK_N = 128
BLOCK_K = 128
SCALE_SHAPE = (N // BLOCK_N, K // BLOCK_K)  # (2, 1)

# e8m0 stores an exponent byte with bias 127, so 120 == 2^-7. That is the same
# sentinel value T3 (test_ilu_dsv4_fp8_e8m0.py) uses to show that a cross-dtype
# ``copy_()`` destroys the byte while ``view(torch.uint8)`` preserves it.
E8M0_EXP_2_POW_MINUS_7 = 120


def _make_fake_fp8_block_layer() -> torch.nn.Module:
    """A bare ``nn.Module`` carrying just the two params the chain reads.

    ``process_fp8_weight_block_strategy`` / ``replace_parameter`` only ever touch
    ``layer.weight`` and ``layer.weight_scale_inv``, so a real ``LinearBase`` (and
    with it a quant config, a TP group and a kernel choice) is unnecessary.
    """
    layer = torch.nn.Module()

    # float8_e4m3fn weight: build the bytes directly and reinterpret, so the test
    # never depends on CPU float -> fp8 conversion support. Values 0..255 cover
    # every possible byte, including the ROCm NaN sentinel 0x80 (-128 as int8)
    # that the fnuz arm would rewrite to 0.
    weight_bytes = torch.arange(N * K, dtype=torch.int64).remainder(256).to(torch.uint8)
    weight = weight_bytes.reshape(N, K).view(torch.float8_e4m3fn)

    # float8_e8m0fnu scale, again via a uint8 view (this is also exactly how the
    # loader hands e8m0 checkpoint bytes around — see Requirements 1.2).
    scale_bytes = torch.full(SCALE_SHAPE, E8M0_EXP_2_POW_MINUS_7, dtype=torch.uint8)
    weight_scale_inv = scale_bytes.view(torch.float8_e8m0fnu)

    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(weight_scale_inv, requires_grad=False)
    return layer


def _bytes_of(t: torch.Tensor) -> torch.Tensor:
    """Raw byte view, so fp8 tensors can be compared without any conversion."""
    return t.detach().view(torch.uint8).clone()


def _current_platform():
    from vllm.platforms import current_platform

    return current_platform


def _process_fn():
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        process_fp8_weight_block_strategy,
    )

    return process_fp8_weight_block_strategy


def _replace_parameter():
    from vllm.model_executor.utils import replace_parameter

    return replace_parameter


# --------------------------------------------------------------------------
# Platform premise.
#
# The whole identity is conditional on the platform, so state that up front:
# on ROCm / an FNUZ device the fnuz arm of process_fp8_weight_block_strategy is
# live and the bytes legitimately change. Failing here gives a readable reason
# instead of a baffling "bytes differ" further down.
# --------------------------------------------------------------------------


def test_platform_premise_is_non_rocm_non_fnuz():
    platform = _current_platform()

    assert not platform.is_rocm(), (
        "T4's data_ptr/dtype identity only holds on non-ROCm platforms: "
        "_maybe_pad_fp8_weight (fp8_utils.py:1179) pads the weight on ROCm. "
        f"Got platform {platform.device_name!r}."
    )
    assert not platform.is_fp8_fnuz(), (
        "T4's byte-level identity only holds on non-FNUZ platforms: "
        "process_fp8_weight_block_strategy (fp8_utils.py:1380) rewrites e4m3 "
        "bytes and bumps the e8m0 exponent on MI300x/MI325x. "
        f"Got platform {platform.device_name!r}."
    )


# --------------------------------------------------------------------------
# Requirements 1.1 / 1.4 — the strategy function itself is an identity.
# --------------------------------------------------------------------------


def test_process_fp8_weight_block_strategy_preserves_storage_and_dtype():
    layer = _make_fake_fp8_block_layer()
    weight_before, scale_before = layer.weight, layer.weight_scale_inv
    weight_ptr, scale_ptr = weight_before.data_ptr(), scale_before.data_ptr()
    weight_bytes, scale_bytes = _bytes_of(weight_before), _bytes_of(scale_before)

    new_weight, new_scale = _process_fn()(weight_before, scale_before)

    assert new_weight.data_ptr() == weight_ptr
    assert new_scale.data_ptr() == scale_ptr

    assert new_weight.dtype is torch.float8_e4m3fn
    assert new_scale.dtype is torch.float8_e8m0fnu

    assert tuple(new_weight.shape) == (N, K)
    assert tuple(new_scale.shape) == SCALE_SHAPE

    # data_ptr equality does not rule out in-place mutation (the fnuz arm mutates
    # through an int8 view), so compare the bytes too.
    assert torch.equal(_bytes_of(new_weight), weight_bytes)
    assert torch.equal(_bytes_of(new_scale), scale_bytes)


# --------------------------------------------------------------------------
# Requirements 1.1 / 1.4 — the strategy + replace_parameter pair, i.e. what the
# kernel's process_weights_after_loading actually performs.
# --------------------------------------------------------------------------


def test_replace_parameter_write_back_preserves_storage_and_dtype():
    layer = _make_fake_fp8_block_layer()
    weight_ptr = layer.weight.data_ptr()
    scale_ptr = layer.weight_scale_inv.data_ptr()
    weight_bytes = _bytes_of(layer.weight)
    scale_bytes = _bytes_of(layer.weight_scale_inv)
    old_weight_param = layer.weight
    old_scale_param = layer.weight_scale_inv

    new_weight, new_scale = _process_fn()(layer.weight, layer.weight_scale_inv)
    _replace_parameter()(layer, "weight", new_weight.data)
    _replace_parameter()(layer, "weight_scale_inv", new_scale.data)

    # The whole point of T4: the Parameter wrapper may be new, the storage is not.
    assert layer.weight.data_ptr() == weight_ptr
    assert layer.weight_scale_inv.data_ptr() == scale_ptr

    assert layer.weight.dtype is torch.float8_e4m3fn
    assert layer.weight_scale_inv.dtype is torch.float8_e8m0fnu

    assert tuple(layer.weight.shape) == (N, K)
    assert tuple(layer.weight_scale_inv.shape) == SCALE_SHAPE

    assert torch.equal(_bytes_of(layer.weight), weight_bytes)
    assert torch.equal(_bytes_of(layer.weight_scale_inv), scale_bytes)

    # Sanity: the params really did go through replace_parameter (a no-op test
    # that never called it would also satisfy every assertion above).
    assert layer.weight is not old_weight_param
    assert layer.weight_scale_inv is not old_scale_param
    assert isinstance(layer.weight, torch.nn.Parameter)
    assert isinstance(layer.weight_scale_inv, torch.nn.Parameter)


# --------------------------------------------------------------------------
# Same identity, driven through the community kernel method verbatim, so that a
# future edit to BlockScaledMMLinearKernel.py:75-95 (e.g. an added transpose or
# a scale relayout) is caught here rather than only on the 8-card run.
#
# The method reads no instance state — it only calls ``self._get_layer_params``,
# which forwards to ``FP8BlockParams.from_layer(layer)``. Borrowing both onto a
# shim avoids instantiating the ABC (whose __init__ wants a full
# FP8ScaledMMLinearLayerConfig plus an activation-quant descriptor).
# --------------------------------------------------------------------------


@pytest.fixture()
def kernel_self_shim():
    from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
        Fp8BlockScaledMMLinearKernel,
    )

    class _KernelSelfShim:
        _get_layer_params = Fp8BlockScaledMMLinearKernel._get_layer_params
        process_weights_after_loading = (
            Fp8BlockScaledMMLinearKernel.process_weights_after_loading
        )

    return _KernelSelfShim()


def test_kernel_process_weights_after_loading_preserves_storage_and_dtype(
    kernel_self_shim,
):
    layer = _make_fake_fp8_block_layer()
    weight_ptr = layer.weight.data_ptr()
    scale_ptr = layer.weight_scale_inv.data_ptr()
    weight_bytes = _bytes_of(layer.weight)
    scale_bytes = _bytes_of(layer.weight_scale_inv)

    kernel_self_shim.process_weights_after_loading(layer)

    assert layer.weight.data_ptr() == weight_ptr
    assert layer.weight_scale_inv.data_ptr() == scale_ptr

    assert layer.weight.dtype is torch.float8_e4m3fn
    assert layer.weight_scale_inv.dtype is torch.float8_e8m0fnu

    assert tuple(layer.weight.shape) == (N, K)
    assert tuple(layer.weight_scale_inv.shape) == SCALE_SHAPE

    assert torch.equal(_bytes_of(layer.weight), weight_bytes)
    assert torch.equal(_bytes_of(layer.weight_scale_inv), scale_bytes)

    # The scale must stay under its original attribute name: Fp8LinearMethod
    # registers it as ``weight_scale_inv`` ("intentional for deepseekv3",
    # fp8.py:379 area), and the kernel is expected to write it back there.
    assert not hasattr(layer, "weight_scale")
