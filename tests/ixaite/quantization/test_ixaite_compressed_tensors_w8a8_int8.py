# SPDX-License-Identifier: Apache-2.0
"""Numeric and hot-path tests for the ixaite W8A8 compressed-tensors scheme."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn.parameter import Parameter

from tests.ixaite.backend_isolation.ixformer_guard import forbid_ixformer_calls
from vllm_iluvatar.backends.ixaite import ops
from vllm_iluvatar.backends.ixaite import ops as ilu_ops
from vllm_iluvatar.backends.ixaite.quantization import (
    is_w8a8_linear_scheme,
)
from vllm_iluvatar.backends.ixaite.quantization.w8a8 import (  # noqa: E501
    IxaiteCompressedTensorsW8A8Int8,
)


# Cover the entire weight-loading and execution lifecycle, including helper calls.
@pytest.fixture(autouse=True)
def _strict_lifecycle():
    with forbid_ixformer_calls():
        yield


class _DummyLinear(torch.nn.Module):
    def __init__(self, logical_widths: list[int] | None = None):
        super().__init__()
        self.logical_widths = logical_widths or [64]


def test_is_w8a8_linear_scheme_accepts_both_siblings():
    class IluCompressedTensorsW8A8Int8:
        pass

    class IxaiteCompressedTensorsW8A8Int8:
        pass

    assert is_w8a8_linear_scheme(IluCompressedTensorsW8A8Int8())
    assert is_w8a8_linear_scheme(IxaiteCompressedTensorsW8A8Int8())
    assert not is_w8a8_linear_scheme(object())
    assert not is_w8a8_linear_scheme(None)


def test_apply_reuses_cached_launch_when_m_is_stable(monkeypatch):
    # The launcher is mocked; this check does not require a physical stream.
    monkeypatch.setattr(
        "vllm_iluvatar.backends.ixaite.quantization.w8a8._current_stream", lambda: None
    )
    scheme = IxaiteCompressedTensorsW8A8Int8(
        strategy=QuantizationStrategy.CHANNEL,
        is_static_input_scheme=False,
        input_symmetric=True,
    )
    scheme.params_dtype = torch.bfloat16
    scheme._pad_n = 64
    scheme._pad_k = 64
    scheme._n = 64
    scheme._k = 64
    scheme._major = "tt"
    scheme._out_kind = "bf16"
    scheme._apply_bias = False
    first_launch = Mock()
    apply_fn = Mock(return_value=first_launch)
    scheme._apply = apply_fn

    x_q = torch.zeros((1, 64), dtype=torch.int8)
    x_s = torch.ones((1, 1), dtype=torch.float32)
    monkeypatch.setattr(
        ops,
        "scaled_int8_quant",
        lambda *args, **kwargs: (x_q, x_s, None),
    )
    layer = SimpleNamespace(
        weight=torch.zeros((64, 64), dtype=torch.int8),
        weight_scale=torch.ones(64, dtype=torch.float32),
        input_scale=None,
    )
    x = torch.zeros((1, 64), dtype=torch.bfloat16)

    scheme.apply_weights(layer, x, None)
    scheme.apply_weights(layer, x, None)

    assert apply_fn.call_count == 1
    assert apply_fn.call_args.kwargs["launch"] is None
    assert first_launch.call_count == 1
    assert apply_fn.call_args.kwargs["n"] == 64
    assert apply_fn.call_args.kwargs["k"] == 64
    assert apply_fn.call_args.kwargs["major"] == "tt"


@pytest.fixture(autouse=True)
def _release_gpu_memory():
    """Return the caching allocator's blocks between cases.

    Tests further down this directory compare providers by spawning worker
    subprocesses, which need their own context on the same pinned device. A
    pool grown across this file's GPU cases leaves them too little to start.
    """
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def _mock_tp_group(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )


def _quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 127.0
    q_weight = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)
    return q_weight, scale.float()


def _set_quantized_weight_params(
    layer: torch.nn.Module,
    in_out_features: tuple[int, int],
    *,
    dtype: torch.dtype,
    has_bias: bool,
) -> None:
    in_features, out_features = in_out_features
    weight_fp = torch.randn(out_features, in_features, dtype=torch.float32) * 0.5
    weight_q, weight_s = _quantize_weight(weight_fp)
    layer.weight.data.copy_(weight_q)
    layer.weight_scale.data.copy_(weight_s)
    if has_bias:
        bias = Parameter(torch.randn(out_features, dtype=dtype), requires_grad=False)
        layer.register_parameter("bias", bias)
    else:
        layer.register_parameter("bias", None)


def _ref_tn(x_q, w_tn, x_s, w_s, bias, out_dtype):
    acc = x_q.cpu().to(torch.int32) @ w_tn.cpu().to(torch.int32).T
    out = acc.to(torch.float32)
    out = out * x_s.cpu().reshape(-1, 1).to(torch.float32)
    out = out * w_s.cpu().reshape(1, -1).to(torch.float32)
    if bias is not None:
        out = out + bias.cpu().reshape(1, -1).to(torch.float32)
    return out.to(out_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize("in_out_features", [(64, 64), (128, 12)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_IxaiteCompressedTensorsW8A8Int8(
    in_out_features: tuple[int, int],
    with_bias: bool,
    monkeypatch,
):
    pytest.importorskip("ixaite")
    import vllm_iluvatar.backends.ixaite.runtime as kernel_provider

    kernel_provider._load_ixaite_runtime.cache_clear()
    kernel_provider.load_ixaite_python_op.cache_clear()

    torch.manual_seed(0)
    in_features, out_features = in_out_features
    test_dtype = torch.bfloat16
    layer = _DummyLinear(logical_widths=[out_features])
    scheme = IxaiteCompressedTensorsW8A8Int8(
        strategy=QuantizationStrategy.CHANNEL,
        is_static_input_scheme=False,
        input_symmetric=True,
    )
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=[out_features],
        input_size_per_partition=in_features,
        params_dtype=test_dtype,
        weight_loader=lambda *_args, **_kwargs: None,
    )
    _set_quantized_weight_params(
        layer,
        in_out_features,
        dtype=test_dtype,
        has_bias=with_bias,
    )
    layer = layer.cuda()
    weight_ref = layer.weight.detach().clone()
    w_scale_ref = layer.weight_scale.detach().clone()
    bias_ref = layer.bias.detach().clone() if with_bias else None
    scheme.process_weights_after_loading(layer)

    if in_out_features == (64, 64):
        assert scheme._major == "tt"
        assert tuple(layer.weight.shape) == (64, 64)
    else:
        assert scheme._major == "tn"
        assert tuple(layer.weight.shape) == (64, 128)

    for m in (1, 8, 16, 17):
        x = torch.randn(m, in_features, dtype=test_dtype, device="cuda")
        out = scheme.apply_weights(layer, x, layer.bias)
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), None, None, symmetric=True
        )
        assert x_zp is None
        ref = _ref_tn(x_q, weight_ref, x_s, w_scale_ref, bias_ref, test_dtype)
        assert out.shape == (m, out_features)
        torch.testing.assert_close(out.cpu(), ref, atol=2e-2, rtol=2e-2)


@pytest.fixture
def route_quant_to_ixaite(monkeypatch):
    """Point the routed op at ixaite, as VLLM_ILU_KERNEL_PROVIDER=ixaite does.

    The independent scheme imports its own quantizer directly.
    """
    return ilu_ops.scaled_int8_quant


def _build_scheme_on_gpu(
    in_features: int,
    out_features: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    static_scale: float | None = None,
    input_symmetric: bool = True,
    static_zero_point: int = 0,
) -> tuple[
    IxaiteCompressedTensorsW8A8Int8, torch.nn.Module, torch.Tensor, torch.Tensor
]:
    layer = _DummyLinear(logical_widths=[out_features])
    scheme = IxaiteCompressedTensorsW8A8Int8(
        strategy=QuantizationStrategy.CHANNEL,
        is_static_input_scheme=static_scale is not None,
        input_symmetric=input_symmetric,
    )
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=[out_features],
        input_size_per_partition=in_features,
        params_dtype=dtype,
        weight_loader=lambda *_args, **_kwargs: None,
    )
    _set_quantized_weight_params(
        layer, (in_features, out_features), dtype=dtype, has_bias=False
    )
    if static_scale is not None:
        layer.input_scale.data.copy_(torch.tensor([static_scale]))
        if not input_symmetric:
            layer.input_zero_point.data.copy_(
                torch.tensor([static_zero_point], dtype=torch.int8)
            )
    layer = layer.cuda()
    weight_ref = layer.weight.detach().clone()
    scale_ref = layer.weight_scale.detach().clone()
    scheme.process_weights_after_loading(layer)
    return scheme, layer, weight_ref, scale_ref


def _ref_tn_azp(x_q, w_tn, x_s, w_s, azp, azp_adj, bias, out_dtype):
    acc = x_q.cpu().to(torch.int32) @ w_tn.cpu().to(torch.int32).T
    out = acc.to(torch.float32)
    out = out - azp.cpu().reshape(-1, 1).float() * azp_adj.cpu().reshape(1, -1).float()
    out = out * x_s.cpu().reshape(-1, 1).to(torch.float32)
    out = out * w_s.cpu().reshape(1, -1).to(torch.float32)
    if bias is not None:
        out = out + bias.cpu().reshape(1, -1).to(torch.float32)
    return out.to(out_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize("in_features", [4096, 12288])
def test_qwen35_9b_widths_run_through_the_ixaite_quant_kernel(
    in_features: int,
    route_quant_to_ixaite,
):
    """The two activation widths a 9B W8A8 forward actually produces.

    4096 is every projection off the residual stream; 12288 is down_proj,
    whose input is the post-silu intermediate. Both divide 64, so neither
    exercises the pad path -- that is the point, this is the shipped shape.
    """
    pytest.importorskip("ixaite")
    torch.manual_seed(0)
    out_features = 4096
    scheme, layer, weight_ref, scale_ref = _build_scheme_on_gpu(
        in_features, out_features
    )
    assert scheme._pad_k == in_features
    assert scheme._pad_n == out_features

    for m in (1, 7, 128):
        x = torch.randn(m, in_features, dtype=torch.bfloat16, device="cuda")
        out = scheme.apply_weights(layer, x, None)
        x_q, x_s, x_zp = route_quant_to_ixaite(x.contiguous(), None, None, True)
        assert x_zp is None
        ref = _ref_tn(x_q, weight_ref, x_s, scale_ref, None, torch.bfloat16)
        assert out.shape == (m, out_features)
        torch.testing.assert_close(out.cpu(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_three_d_input_keeps_leading_dims(route_quant_to_ixaite):
    """[batch, seq, hidden] must survive quant, GEMM, and the unpad."""
    pytest.importorskip("ixaite")
    torch.manual_seed(0)
    in_features, out_features = 4096, 512
    scheme, layer, weight_ref, scale_ref = _build_scheme_on_gpu(
        in_features, out_features
    )

    x = torch.randn(2, 3, in_features, dtype=torch.bfloat16, device="cuda")
    out = scheme.apply_weights(layer, x, None)
    assert out.shape == (2, 3, out_features)

    x_q, x_s, _ = route_quant_to_ixaite(x.contiguous(), None, None, True)
    ref = _ref_tn(
        x_q.reshape(-1, in_features),
        weight_ref,
        x_s,
        scale_ref,
        None,
        torch.bfloat16,
    )
    torch.testing.assert_close(
        out.reshape(-1, out_features).cpu(), ref, atol=2e-2, rtol=2e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_static_input_scale_is_used_instead_of_a_per_token_scale(
    route_quant_to_ixaite,
):
    """process_weights_after_loading collapses input_scale to a 0-dim max.

    The kernel takes that as the whole-tensor scale, so every row must come
    back with the same scale rather than its own absmax.
    """
    pytest.importorskip("ixaite")
    torch.manual_seed(0)
    in_features, out_features = 4096, 512
    static_scale = 0.05
    scheme, layer, weight_ref, scale_ref = _build_scheme_on_gpu(
        in_features, out_features, static_scale=static_scale
    )
    assert layer.input_scale.dim() == 0
    assert layer.input_scale.dtype == torch.float32

    # Rows with very different magnitudes: a dynamic scale would differ per
    # row, a static one cannot.
    x = torch.randn(4, in_features, dtype=torch.bfloat16, device="cuda")
    x[0] *= 100.0
    x_q, x_s, x_zp = route_quant_to_ixaite(
        x.contiguous(), layer.input_scale, None, True
    )

    assert x_zp is None
    assert x_s.numel() == 1
    torch.testing.assert_close(
        x_s.reshape(()).cpu(),
        torch.tensor(static_scale, device="cpu", dtype=x_s.dtype),
        rtol=1e-6,
        atol=1e-8,
    )
    expected = torch.clamp(torch.round(x.float() / static_scale), -128, 127).to(
        torch.int8
    )
    torch.testing.assert_close(x_q.cpu(), expected.cpu(), atol=1, rtol=0.0)

    out = scheme.apply_weights(layer, x, None)
    assert out.shape == (4, out_features)
    # The epilogue indexes scale_a per row, so the one per-tensor scale has to
    # reach every row; checking only the shape would miss rows past the first
    # reading past the end of a length-1 scale_a.
    ref = _ref_tn(
        x_q, weight_ref, x_s.reshape(1).expand(4), scale_ref, None, torch.bfloat16
    )
    torch.testing.assert_close(out.cpu(), ref, atol=2e-2, rtol=2e-2)


def test_asymmetric_scheme_is_accepted():
    """input_symmetric=False must no longer be rejected outright."""
    scheme = IxaiteCompressedTensorsW8A8Int8(
        strategy=QuantizationStrategy.CHANNEL,
        is_static_input_scheme=False,
        input_symmetric=False,
    )

    assert scheme.input_symmetric is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize("in_out_features", [(4096, 512), (128, 12)])
def test_dynamic_asymmetric_matches_reference(in_out_features, route_quant_to_ixaite):
    """Per-token AZP: the correction must be applied inside the dequant."""
    pytest.importorskip("ixaite")
    torch.manual_seed(0)
    in_features, out_features = in_out_features
    scheme, layer, weight_ref, scale_ref = _build_scheme_on_gpu(
        in_features, out_features, input_symmetric=False
    )
    azp_adj = layer.azp_adj
    assert azp_adj.shape == (scheme._pad_n,)

    for m in (1, 7, 64):
        # Shift the activations off centre so the zero point is non-trivial.
        x = torch.randn(m, in_features, dtype=torch.bfloat16, device="cuda") + 4.0
        out = scheme.apply_weights(layer, x, None)
        x_q, x_s, x_zp = route_quant_to_ixaite(x.contiguous(), None, None, False)
        assert x_zp is not None

        # Reference reads the unpadded weight, so trim azp_adj to match.
        ref = _ref_tn_azp(
            x_q,
            weight_ref,
            x_s,
            scale_ref,
            x_zp,
            azp_adj[:out_features],
            None,
            torch.bfloat16,
        )
        assert out.shape == (m, out_features)
        torch.testing.assert_close(out.cpu(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_static_asymmetric_matches_reference(route_quant_to_ixaite):
    """Per-tensor AZP: the zero point is folded into azp_adj at load time."""
    pytest.importorskip("ixaite")
    torch.manual_seed(0)
    in_features, out_features = 4096, 512
    scheme, layer, weight_ref, scale_ref = _build_scheme_on_gpu(
        in_features,
        out_features,
        static_scale=0.05,
        input_symmetric=False,
        static_zero_point=-9,
    )
    assert layer.input_zero_point.dtype == torch.int32

    x = torch.randn(4, in_features, dtype=torch.bfloat16, device="cuda") + 2.0
    out = scheme.apply_weights(layer, x, None)
    x_q, x_s, x_zp = route_quant_to_ixaite(
        x.contiguous(), layer.input_scale, layer.input_zero_point, False
    )

    # azp_adj already carries the zero point, so the per-row term is 1.
    ones = torch.ones(4, dtype=torch.float32)
    ref = _ref_tn_azp(
        x_q,
        weight_ref,
        x_s,
        scale_ref,
        ones,
        layer.azp_adj[:out_features],
        None,
        torch.bfloat16,
    )
    assert out.shape == (4, out_features)
    torch.testing.assert_close(out.cpu(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_asymmetric_beats_dropping_the_correction(route_quant_to_ixaite):
    """A no-op AZP epilogue cannot hide inside the match budget.

    The correction is azp * azp_adj * scales and does not depend on the GEMM
    accumulator, so |uncorrected - corrected| is not required to exceed
    |corrected|. Compare the term, and the kernel-vs-no-AZP gap, to the same
    match budget used for the AZP reference.
    """
    pytest.importorskip("ixaite")
    torch.manual_seed(0)
    in_features, out_features = 4096, 512
    scheme, layer, weight_ref, scale_ref = _build_scheme_on_gpu(
        in_features, out_features, input_symmetric=False
    )

    x = torch.randn(32, in_features, dtype=torch.bfloat16, device="cuda") + 4.0
    out = scheme.apply_weights(layer, x, None).cpu().float()
    x_q, x_s, x_zp = route_quant_to_ixaite(x.contiguous(), None, None, False)

    corrected = _ref_tn_azp(
        x_q,
        weight_ref,
        x_s,
        scale_ref,
        x_zp,
        layer.azp_adj[:out_features],
        None,
        torch.bfloat16,
    ).float()
    uncorrected = _ref_tn(x_q, weight_ref, x_s, scale_ref, None, torch.bfloat16).float()

    match_tol = 2e-2 * corrected.abs().max().clamp(min=1.0)
    azp_term = (uncorrected - corrected).abs().max()
    # Setup: the correction must outrun rounding, or matching `corrected`
    # would not distinguish a dropped epilogue.
    assert azp_term > 10 * match_tol
    assert (out - corrected).abs().max() < match_tol
    assert (out - uncorrected).abs().max() > 10 * match_tol
