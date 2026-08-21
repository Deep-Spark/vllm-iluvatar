# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric parity tests for the Iluvatar W8A8 INT8 compressed-tensors scheme.

Compares ``IluCompressedTensorsW8A8Int8`` against the ixformer ``ref_w8a8``
kernel across representative (k, n) shapes (covering both the NN and TN weight
layouts plus output unpadding), bias on/off, and fp16/bf16. The ixformer
kernels are a mandatory dependency (a missing build is a collection-time import
error, never a silent skip); only the actual GPU device is gated via a
module-level skip.
"""

from __future__ import annotations

import pytest
import torch
import vllm_iluvatar._C  # noqa: F401
from compressed_tensors.quantization import QuantizationStrategy
from conftest import COMPILED_CUDA_ARCH
from ixformer.inference.functions.w8a8 import ref_w8a8
from torch.nn.parameter import Parameter
from vllm import _custom_ops as ops

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes.ilu_compressed_tensors_w8a8_int8 import (  # noqa: E501
    IluCompressedTensorsW8A8Int8,
)

supported_architectures = ["ivcore11"]
skip_case = COMPILED_CUDA_ARCH not in supported_architectures

if not torch.cuda.is_available():
    pytest.skip("Iluvatar/CUDA device required", allow_module_level=True)


class _DummyLinear(torch.nn.Module):
    def __init__(self, logical_widths: list[int] | None = None):
        super().__init__()
        self.logical_widths = logical_widths or [64]


@pytest.fixture(autouse=True)
def _mock_tp_group(monkeypatch):
    # These parameter classes read TP info during __init__.
    # Unit tests run without distributed init, so default to single-rank TP.
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )


def _quantize_weight(
    weight: torch.Tensor, strategy: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if strategy == QuantizationStrategy.CHANNEL:
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 127.0
    else:
        scale = torch.tensor(
            [weight.abs().amax().clamp(min=1e-6) / 127.0],
            dtype=torch.float32,
            device=weight.device,
        )

    q_weight = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)
    return q_weight, scale.float()


def _set_quantized_weight_params(
    layer: torch.nn.Module,
    in_out_features: tuple[int, int],
    *,
    strategy: str,
    dtype: torch.dtype,
    has_bias: bool = False,
) -> None:
    in_features, out_features = in_out_features
    weight_fp = torch.randn(out_features, in_features, dtype=torch.float32) * 0.5
    weight_q, weight_s = _quantize_weight(weight_fp, strategy)
    layer.weight.data.copy_(weight_q)
    layer.weight_scale.data.copy_(weight_s)
    if has_bias:
        bias = Parameter(torch.randn(out_features, dtype=dtype), requires_grad=False)
        layer.register_parameter("bias", bias)
    else:
        layer.register_parameter("bias", None)


@pytest.mark.parametrize(
    "strategy",
    [QuantizationStrategy.CHANNEL],
)
@pytest.mark.parametrize(
    "in_out_features",
    # (k, n)
    [
        (64, 64),  # NN path
        (4096, 1792),
        (4096, 12),
        (1536, 4096),
        (4096, 1280),
        (4096, 8),
        (1024, 4096),
        (4096, 320),
        (160, 4096),
        (4096, 2816),
        (1408, 4096),
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("test_dtype", [torch.half, torch.bfloat16])
def test_IluCompressedTensorsW8A8Int8(
    in_out_features: tuple[int, int],
    with_bias: bool,
    test_dtype: torch.dtype,
    strategy: str,
):
    torch.manual_seed(0)
    in_features, out_features = in_out_features

    if skip_case and in_out_features in [
        (4096, 12),
        (4096, 8),
    ]:
        pytest.skip("skip case: {in_out_features}, error on ixcore30, need fixed")
    # 1) Initialize layer/scheme and quantized parameters in (out, in) layout.
    layer = _DummyLinear(logical_widths=[out_features])
    scheme = IluCompressedTensorsW8A8Int8(
        strategy=strategy,
        is_static_input_scheme=False,
        input_symmetric=True,
        ilu_params={
            "mixed_strategy": "CHANNEL",
            "gemm_format": "TN",
            "gemm_mode": "PERF",
        },
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
        strategy=strategy,
        dtype=test_dtype,
        has_bias=with_bias,
    )
    # Keep the original checkpoint layout (TN, shape = [out_features, in_features])
    # as the golden reference for ref_w8a8.
    layer = layer.cuda()
    weight_ref = layer.weight.detach().clone()
    w_scale_ref = layer.weight_scale.detach().clone()
    bias_ref = layer.bias.detach().clone() if with_bias else None
    scheme.process_weights_after_loading(layer)

    # 2) Sweep m from 1..100 and compare real kernel output against ref_w8a8.
    diff_list = []
    for m in range(1, 101):
        x = torch.randn(m, in_features, dtype=test_dtype, device="cuda")

        # Real path: applies scaled_int8_quant + ixf_ops.w8a8 internally.
        out = scheme.apply_weights(layer, x, layer.bias)

        # Reference path: use the same activation quantization and ref_w8a8.
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), None, None, symmetric=True
        )
        assert x_zp is None

        ref = torch.empty(
            (m, out_features),
            dtype=test_dtype,
            device=x.device,
        )
        # Always use original TN-layout weight here. This validates that
        # process_weights_after_loading + apply_weights preserves correctness.
        ref_w8a8(
            input=x_q,
            weight=weight_ref,
            i_scales=x_s,
            w_scales=w_scale_ref,
            output=ref,
            format="TN",
            bias=bias_ref,
        )
        diff_list.append(torch.max(torch.abs(out - ref)).item())
        assert out.shape == (m, out_features)
        assert out.dtype == test_dtype
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    print(
        f"test_IluCompressedTensorsW8A8Int8 in_features:{in_features} "
        f"out_features:{out_features} with_bias:{with_bias} strategy:{strategy} "
        f"test_dtype:{test_dtype} max_diff: {max(diff_list):.4f} "
        f"min_diff: {min(diff_list):.4f} "
        f"avg_diff: {sum(diff_list) / len(diff_list):.4f}"
    )
