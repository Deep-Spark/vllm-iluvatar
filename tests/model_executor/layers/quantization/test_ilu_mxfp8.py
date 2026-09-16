# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar MXFP8 routed-expert method and config.

The failure this file exists to prevent is the one that actually happened: the
community MXFP8 MoE selector lands on ``MARLIN`` (``MarlinExperts`` gates only on
"CUDA SM75+", and ``IluPlatform._enum`` is CUDA), and the layer then dies at load
time in ``prepare_mxfp8_moe_layer_for_marlin`` -> ``gptq_marlin_repack``, an op
the Iluvatar build does not define.

Covered here:

* ``__init__`` never calls ``select_mxfp8_moe_backend``, and leaves
  ``experts_cls`` unset so community ``is_monolithic()`` cannot dereference it.
* the UE8M0 scale is *decoded*, not cast -- a plain ``.to(float32)`` reads the
  exponent byte as an integer (0x7F -> 127.0 instead of 1.0), which would scale
  weights by ~1e38 rather than erroring.
* ``create_weights`` allocates the checkpoint MXFP8 grid in host memory with the
  community weight loader (no parent call), so EP4 never holds both the MXFP8
  grid and its FP4 image in HBM.
* ``process_weights_after_loading`` converts the MXFP8 tensors with the
  ``(1, 32)`` grid and UE8M0 decoding, requanting on the GPU.
* the config only swaps ``FusedMoEMethodCls``; Linear stays on the community
  method, which is what calls ``init_mxfp8_linear_kernel()``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8FusedMoE

import vllm_iluvatar.model_executor.layers.quantization.ilu_mxfp8 as ilu_mxfp8
from vllm_iluvatar.model_executor.layers.quantization.ilu_mxfp8 import (
    IluMxFp8Config,
    IluMxFp8FusedMoE,
    e8m0_to_fp32,
)

_BLOCK = 32
_H = 512
_I = 256
_E = 2


def _make_layer():
    """Routed-expert module carrying MXFP8 checkpoint parameters."""
    g = torch.Generator().manual_seed(0)
    layer = nn.Module()
    w13 = torch.randn(_E, 2 * _I, _H, generator=g).to(torch.float8_e4m3fn)
    w2 = torch.randn(_E, _H, _I, generator=g).to(torch.float8_e4m3fn)
    # Exponents around the 127 bias, i.e. scales around 1.0.
    s13 = torch.randint(
        125, 130, (_E, 2 * _I, _H // _BLOCK), dtype=torch.uint8, generator=g
    )
    s2 = torch.randint(125, 130, (_E, _H, _I // _BLOCK), dtype=torch.uint8, generator=g)
    layer.register_parameter("w13_weight", nn.Parameter(w13, requires_grad=False))
    layer.register_parameter("w2_weight", nn.Parameter(w2, requires_grad=False))
    layer.register_parameter("w13_weight_scale", nn.Parameter(s13, requires_grad=False))
    layer.register_parameter("w2_weight_scale", nn.Parameter(s2, requires_grad=False))
    layer.expert_map = None
    layer.activation = "silu"
    return layer


def _make_method():
    quant_config = MagicMock()
    quant_config.is_checkpoint_mxfp8_serialized = True
    return IluMxFp8FusedMoE(quant_config=quant_config, moe_config=MagicMock())


def test_init_skips_community_backend_selection(monkeypatch):
    import vllm.model_executor.layers.fused_moe.oracle.mxfp8 as oracle

    def _boom(*args, **kwargs):
        raise AssertionError("select_mxfp8_moe_backend must not be called")

    monkeypatch.setattr(oracle, "select_mxfp8_moe_backend", _boom)

    method = _make_method()

    assert method.mxfp8_backend is None
    assert method.moe_kernel is None
    assert method.weight_block_size == [1, _BLOCK]
    # Community ``FusedMoEMethodBase.is_monolithic`` dereferences ``experts_cls``;
    # leaving it unset (not None) keeps that path unreachable.
    assert not hasattr(method, "experts_cls")
    assert method.is_monolithic is False
    assert method.supports_eplb is False


def test_create_weights_allocates_mxfp8_grid_on_host(monkeypatch):
    """MXFP8 checkpoint layout, host memory, community loader -- own body."""

    def _boom(*args, **kwargs):
        raise AssertionError("must not call community create_weights")

    monkeypatch.setattr(ModelOptMxFp8FusedMoE, "create_weights", _boom)
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    method = _make_method()
    method.moe.is_act_and_mul = True
    layer = nn.Module()
    layer.intermediate_size_per_partition = _I
    layer.hidden_size = _H
    loader = object()
    method.create_weights(
        layer=layer,
        num_experts=_E,
        hidden_size=_H,
        intermediate_size_per_partition=_I,
        params_dtype=torch.bfloat16,
        weight_loader=loader,
    )

    # Host memory is what makes EP4 fit: the FP4 image is built on GPU a few
    # experts at a time in ``process_weights_after_loading``.
    assert layer.w13_weight.device.type == "cpu"
    assert layer.w2_weight.device.type == "cpu"
    assert tuple(layer.w13_weight.shape) == (_E, 2 * _I, _H)
    assert tuple(layer.w2_weight.shape) == (_E, _H, _I)
    assert tuple(layer.w13_weight_scale.shape) == (_E, 2 * _I, _H // _BLOCK)
    assert tuple(layer.w2_weight_scale.shape) == (_E, _H, _I // _BLOCK)
    assert layer.w13_weight.dtype is torch.float8_e4m3fn
    assert layer.w13_weight_scale.dtype is torch.uint8
    assert layer.w13_weight.weight_loader is loader
    # Without the BLOCK marker the generic loader treats these as per-tensor
    # scales and silently writes a single element per expert.
    assert layer.w13_weight_scale.quant_method == "block"
    assert layer.w2_weight_scale.quant_method == "block"
    assert layer.num_experts == _E


def test_e8m0_scale_is_decoded_not_cast():
    raw = torch.tensor([[0x7F, 0x80, 0x7E, 0x81]], dtype=torch.uint8)

    decoded = e8m0_to_fp32(raw)

    assert torch.equal(decoded, torch.tensor([[1.0, 2.0, 0.5, 4.0]]))
    # The silent-failure mode: ``.to(float32)`` yields 127/128/126/129.
    assert not torch.equal(decoded, raw.to(torch.float32))
    # A float8_e8m0fnu view of the same bytes must decode identically.
    assert torch.equal(e8m0_to_fp32(raw.view(torch.float8_e8m0fnu)), decoded)


def test_process_weights_publishes_ix30_layout(monkeypatch):
    """The conversion is driven with MXFP8's real (1, 32) grid and e8m0 decode."""
    monkeypatch.setattr(ilu_mxfp8, "_ixmm_backend_available", lambda: True)
    seen = []

    def _fake_preprocess(
        weight,
        scale,
        block,
        *,
        scale_transform=None,
        allow_k_padding=False,
        compute_device=None,
    ):
        decoded = scale if scale_transform is None else scale_transform(scale)
        seen.append(
            (
                tuple(weight.shape),
                tuple(scale.shape),
                block,
                decoded.dtype,
                compute_device,
            )
        )
        e, n, k = weight.shape
        return (
            torch.zeros((e, k // 2, n), dtype=torch.uint8),
            torch.ones((e, k // 256, n), dtype=torch.float32),
        )

    monkeypatch.setattr(
        ilu_mxfp8, "preprocess_fp8_moe_weight_for_ixmm", _fake_preprocess
    )

    layer = _make_layer()
    _make_method().process_weights_after_loading(layer)

    assert len(seen) == 2
    # ``compute_device`` follows the parameter, which the loader has already
    # staged on the worker's device. These bare-CPU parameters therefore convert
    # in place; ``test_fp4_image_survives_the_loader_device_context`` covers the
    # staged case, where the requant must be handed the GPU (doing it on the
    # host is what made loading take minutes per checkpoint shard).
    for _w_shape, _s_shape, block, decoded_dtype, compute_device in seen:
        assert block == (1, _BLOCK)
        assert decoded_dtype is torch.float32
        assert compute_device is None

    assert tuple(layer.w13_weight.shape) == (_E, _H // 2, 2 * _I)
    assert tuple(layer.w13_weight_scale.shape) == (_E, _H // 256, 2 * _I)
    assert tuple(layer.w2_weight.shape) == (_E, _I // 2, _H)
    assert tuple(layer.w2_weight_scale.shape) == (_E, _I // 256, _H)
    assert layer.w13_weight.dtype is torch.uint8
    assert layer.w13_weight_scale.dtype is torch.float32


def test_fp4_image_survives_the_loader_device_context(monkeypatch):
    """The converted weights must not be evicted back to host memory.

    The loader wraps ``process_weights_after_loading`` in
    ``device_loading_context``, which stages CPU parameters onto the device and,
    on exit, moves anything still carrying one of those *names* back to CPU.
    Publishing the FP4 image under the checkpoint's own names as a Parameter
    therefore hands it straight back to the host, and the first
    ``group_gemm_fp4`` dies with "weight must be on the same CUDA device as a".
    Buffers are invisible to that restore loop, which is why they are used.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a device for the loader staging context to target")

    from vllm.model_executor.model_loader.utils import device_loading_context

    monkeypatch.setattr(ilu_mxfp8, "_ixmm_backend_available", lambda: True)

    def _fake_preprocess(
        weight,
        scale,
        block,
        *,
        scale_transform=None,
        allow_k_padding=False,
        compute_device=None,
    ):
        # The real helper publishes on ``compute_device``; mirror that so the
        # test exercises the eviction, not a device mismatch of its own making.
        e, n, k = weight.shape
        dev = compute_device if compute_device is not None else weight.device
        return (
            torch.zeros((e, k // 2, n), dtype=torch.uint8, device=dev),
            torch.ones((e, k // 256, n), dtype=torch.float32, device=dev),
        )

    monkeypatch.setattr(
        ilu_mxfp8, "preprocess_fp8_moe_weight_for_ixmm", _fake_preprocess
    )

    target = torch.device("cuda", torch.cuda.current_device())
    layer = _make_layer()
    assert layer.w13_weight.device.type == "cpu"

    with device_loading_context(layer, target):
        # The context staged the checkpoint grid on the device, so the method
        # must read the device off the parameter rather than guessing.
        assert layer.w13_weight.device.type == "cuda"
        _make_method().process_weights_after_loading(layer)

    for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        tensor = getattr(layer, name)
        assert tensor.device.type == "cuda", f"{name} was evicted to host memory"
        # Buffers, not Parameters: that is what makes them invisible to the
        # context's restore loop.
        assert name in layer._buffers
        assert name not in layer._parameters
    # Derived tensors have no business in a checkpoint.
    assert "w13_weight" not in layer.state_dict()


def test_process_weights_is_idempotent(monkeypatch):
    """Community guards this too: mxfp8 reaches the hook more than once."""
    monkeypatch.setattr(ilu_mxfp8, "_ixmm_backend_available", lambda: True)
    calls = []

    def _fake_preprocess(
        weight,
        scale,
        block,
        *,
        scale_transform=None,
        allow_k_padding=False,
        compute_device=None,
    ):
        calls.append(block)
        e, n, k = weight.shape
        return (
            torch.zeros((e, k // 2, n), dtype=torch.uint8),
            torch.ones((e, k // 256, n), dtype=torch.float32),
        )

    monkeypatch.setattr(
        ilu_mxfp8, "preprocess_fp8_moe_weight_for_ixmm", _fake_preprocess
    )

    layer = _make_layer()
    method = _make_method()
    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(layer)

    assert len(calls) == 2, "second call must be a no-op, not a re-requant"


def test_missing_ixmm_backend_is_a_loud_failure(monkeypatch):
    monkeypatch.setattr(ilu_mxfp8, "_ixmm_backend_available", lambda: False)

    with pytest.raises(RuntimeError, match="ixmm backend"):
        _make_method().process_weights_after_loading(_make_layer())


@pytest.mark.parametrize("axis", ["hidden_size", "intermediate_size_per_partition"])
def test_create_weights_rejects_non_group_aligned_dims(monkeypatch, axis):
    """Both GEMM K axes must be multiples of 256, checked before allocation.

    ``allow_k_padding=False`` downstream means an unaligned dim would otherwise
    reach ``preprocess_fp8_block_weight_for_ixmm`` and assert there, after the
    whole grid has been allocated. Drive the real ``create_weights`` guard.
    """
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    bad = 3 * _BLOCK  # 96: a valid MXFP8 grid, but not a multiple of 256
    dims = {"hidden_size": _H, "intermediate_size_per_partition": _I}
    dims[axis] = bad

    method = _make_method()
    method.moe.is_act_and_mul = True
    layer = nn.Module()
    layer.intermediate_size_per_partition = dims["intermediate_size_per_partition"]
    layer.hidden_size = dims["hidden_size"]

    with pytest.raises(ValueError, match=f"divisible by {ilu_mxfp8._IXMM_GROUP_SIZE}"):
        method.create_weights(
            layer=layer,
            num_experts=_E,
            params_dtype=torch.bfloat16,
            weight_loader=object(),
            **dims,
        )


def test_unaligned_k_trips_the_real_ixmm_preprocess_guard(monkeypatch):
    """With ``allow_k_padding=False`` the leaf helper must reject an unpadded K.

    Complements the ``create_weights`` test above: this one lets the real
    ``preprocess_fp8_block_weight_for_ixmm`` run, so deleting its assertion
    cannot pass unnoticed.
    """
    monkeypatch.setattr(ilu_mxfp8, "_ixmm_backend_available", lambda: True)

    bad_k = 3 * _BLOCK
    weight = torch.zeros(_E, 2 * _I, bad_k).to(torch.float8_e4m3fn)
    scale = torch.full((_E, 2 * _I, bad_k // _BLOCK), 0x7F, dtype=torch.uint8)

    # ``match=`` is load-bearing: the helper carries more than one assert (the
    # scale-shape one would also fire), so a bare raises() would pass even if
    # the K-alignment check were deleted.
    with pytest.raises(AssertionError, match="multiple of 256"):
        ilu_mxfp8.preprocess_fp8_moe_weight_for_ixmm(
            weight,
            scale,
            (1, _BLOCK),
            scale_transform=ilu_mxfp8.e8m0_to_fp32,
            allow_k_padding=False,
        )


def test_minimax_mxfp8_is_shadowed_not_nvidia_modelopt():
    """MiniMax-M3 tags ``quant_method: "mxfp8"``. NVIDIA ``modelopt_mxfp8`` is
    left on the community class."""
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config

    assert get_quantization_config("mxfp8") is IluMxFp8Config
    assert get_quantization_config("modelopt_mxfp8") is ModelOptMxFp8Config


def test_config_only_swaps_the_moe_method():
    assert IluMxFp8Config.FusedMoEMethodCls is IluMxFp8FusedMoE
    # Linear must stay on the community method: it is the caller of
    # ``init_mxfp8_linear_kernel()``, which is where IluMxfp8LinearKernel wins.
    assert IluMxFp8Config.LinearMethodCls.__name__ == (
        "ModelOptMxFp8LinearMethod"
    )
    quant_config = IluMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    assert quant_config.get_name() == "mxfp8"


def test_unknown_activation_is_rejected_rather_than_mis_run():
    """Another SwiGLU flavour needs its own variant; silently reusing
    ``swigluoai`` would compute different math (alpha, ``up + 1``, interleave)."""
    method = _make_method()
    layer = _make_layer()
    layer.activation = "swiglustep"

    with pytest.raises(NotImplementedError, match="swigluoai_uninterleave"):
        method.apply(
            layer=layer,
            x=torch.zeros(1, _H, dtype=torch.bfloat16),
            topk_weights=torch.zeros(1, 1),
            topk_ids=torch.zeros(1, 1, dtype=torch.int32),
            shared_experts=None,
            shared_experts_input=None,
        )


def test_swigluoai_clamp_is_forwarded_to_fp4_grouped_path(monkeypatch):
    """layer.swiglu_* must reach the FP4 MoE pipeline, not module constants."""
    seen = {}

    def _fake_experts(**kwargs):
        seen.update(kwargs)
        return torch.zeros(1, _H, dtype=torch.bfloat16)

    monkeypatch.setattr(ilu_mxfp8, "ilu_mxfp4_fused_experts", _fake_experts)

    layer = _make_layer()
    layer.activation = "swigluoai_uninterleave"
    layer.routed_scaling_factor = 1.0
    layer.swiglu_limit = 5.5
    layer.swiglu_alpha = 1.1
    layer.swiglu_beta = 0.5
    _make_method().apply(
        layer=layer,
        x=torch.zeros(1, _H, dtype=torch.bfloat16),
        topk_weights=torch.zeros(1, 1),
        topk_ids=torch.zeros(1, 1, dtype=torch.int32),
        shared_experts=None,
        shared_experts_input=None,
    )

    assert seen["swiglu_limit"] == 5.5
    assert seen["swiglu_alpha"] == 1.1
    assert seen["swiglu_beta"] == 0.5
    assert seen["backend"] == "ixmm"
    assert seen["swiglu_variant"] == ilu_mxfp8._SWIGLUOAI_VARIANT


def test_missing_swiglu_limit_is_rejected_not_defaulted(monkeypatch):
    """Community asserts the clamp limit instead of choosing one; so must we.

    Substituting MiniMax-M3's 7.0 for a checkpoint that never declared a limit
    would compute a different activation with nothing in the log.
    """

    def _boom(**kwargs):
        raise AssertionError("must not reach the MoE pipeline without a limit")

    monkeypatch.setattr(ilu_mxfp8, "ilu_mxfp4_fused_experts", _boom)

    layer = _make_layer()
    layer.activation = "swigluoai_uninterleave"
    layer.swiglu_alpha = 1.702
    layer.swiglu_beta = 1.0
    assert not hasattr(layer, "swiglu_limit")

    with pytest.raises(ValueError, match="swiglu_limit"):
        _make_method().apply(
            layer=layer,
            x=torch.zeros(1, _H, dtype=torch.bfloat16),
            topk_weights=torch.zeros(1, 1),
            topk_ids=torch.zeros(1, 1, dtype=torch.int32),
            shared_experts=None,
            shared_experts_input=None,
        )


@pytest.mark.parametrize("missing", ["swiglu_alpha", "swiglu_beta"])
def test_missing_swiglu_alpha_beta_is_rejected_not_defaulted(monkeypatch, missing):
    """Same contract as the limit: no silent substitution.

    Falling back to ``SiluAndMulWithClamp``'s 1.0 / 0.0 would turn M3's
    ``gate * sigmoid(1.702*gate) * (up + beta)`` into plain ``silu(gate) * up``
    -- plausible-looking output, no error, no log line.
    """

    def _boom(**kwargs):
        raise AssertionError("must not reach the MoE pipeline without alpha/beta")

    monkeypatch.setattr(ilu_mxfp8, "ilu_mxfp4_fused_experts", _boom)

    layer = _make_layer()
    layer.activation = "swigluoai_uninterleave"
    layer.routed_scaling_factor = 1.0
    layer.swiglu_limit = 7.0
    layer.swiglu_alpha = 1.702
    layer.swiglu_beta = 1.0
    setattr(layer, missing, None)

    with pytest.raises(ValueError, match=missing):
        _make_method().apply(
            layer=layer,
            x=torch.zeros(1, _H, dtype=torch.bfloat16),
            topk_weights=torch.zeros(1, 1),
            topk_ids=torch.zeros(1, 1, dtype=torch.int32),
            shared_experts=None,
            shared_experts_input=None,
        )


def test_preprocess_moe_weight_chunks_onto_compute_device(monkeypatch):
    """Host MXFP8 grid is converted a few experts at a time on ``compute_device``."""
    seen = []

    def _fake_chunk(weight, scale, block, allow_k_padding=False):
        seen.append((weight.device.type, weight.shape[0], scale.clone(), block))
        e, n, k = weight.shape
        return (
            torch.zeros((e, k // 2, n), dtype=torch.uint8, device=weight.device),
            torch.zeros((e, k // 256, n), dtype=torch.float32, device=weight.device),
        )

    monkeypatch.setattr(ilu_mxfp8, "preprocess_fp8_block_weight_for_ixmm", _fake_chunk)
    weight = torch.ones((5, 2, 256))
    source_scale = torch.full((5, 1, 8), 3.0)

    packed, packed_scale = ilu_mxfp8.preprocess_fp8_moe_weight_for_ixmm(
        weight,
        source_scale,
        (1, 32),
        scale_transform=lambda x: x + 1,
        compute_device=torch.device("cpu"),
    )

    assert [chunk_size for _, chunk_size, *_ in seen] == [4, 1]
    assert all(dev == "cpu" for dev, *_ in seen)
    assert all(torch.all(scale == 4) for _, _, scale, _ in seen)
    assert packed.shape == (5, 128, 2)
    assert packed_scale.shape == (5, 1, 2)
    assert packed.device.type == "cpu"
