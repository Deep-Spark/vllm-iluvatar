# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric parity tests for the Iluvatar W4A8 compressed-tensors linear scheme.

Compares ``IluCompressedTensorsW4A8Int`` against the ixformer ``ref_w4a8``
kernel across representative (k, n) shapes, bias on/off, and fp16/bf16. The
ixformer kernels are a mandatory dependency (a missing build is a
collection-time import error, never a silent skip); only the actual GPU device
is gated via a module-level skip.
"""

from __future__ import annotations

import pytest
import torch
from compressed_tensors.quantization import QuantizationStrategy
from ixformer.inference.functions.w4a8 import (
    ref_w4a8,
    w4a8_gemm_quant_repack_int4,
)
from torch.nn.parameter import Parameter
from vllm import _custom_ops as ops

from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes.ilu_compressed_tensors_w4a8_int import (  # noqa: E501
    IluCompressedTensorsW4A8Int,
)

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


def _build_packed_w4_weight(
    in_out_features: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    in_features, out_features = in_out_features

    # Mimic realistic transformer linear stats with fan-in scaling.
    # This keeps output magnitude closer to production checkpoints.
    weight_std = 1.0 / (in_features**0.5)
    weight_fp16 = (
        torch.randn(
            size=(out_features, in_features),
            dtype=torch.float16,
        )
        * weight_std
    )
    weight_scale = (
        weight_fp16.abs().amax(dim=1, keepdim=True).to(torch.float32).clamp(min=1e-6)
        / 7.0
    )
    weight_i4 = torch.clamp(
        torch.round(weight_fp16.to(torch.float32) / weight_scale), -8, 7
    ).to(torch.int8)

    packed_weight, _, _ = w4a8_gemm_quant_repack_int4(
        # group_size == -1 path internally uses x / 18 -> int4, so q*18 keeps
        # exactly the quantized int4 values after packing.
        x=weight_i4.to(torch.float32) * 18.0,
        group_size=-1,
        version=2,
        format="NN",
    )
    return packed_weight.contiguous(), weight_scale.contiguous()


def _set_quantized_weight_params(
    layer: torch.nn.Module,
    in_out_features: tuple[int, int],
    *,
    dtype: torch.dtype,
    has_bias: bool = False,
) -> None:
    _in_features, out_features = in_out_features
    packed_weight, weight_scale = _build_packed_w4_weight(in_out_features)
    layer.weight.data.copy_(packed_weight)

    # Keep scales consistent with FP16->int4 quantization above.
    layer.weight_scale.data.copy_(weight_scale)

    if has_bias:
        # Bias is usually small compared to the matmul term.
        bias = Parameter(
            torch.randn(out_features, dtype=dtype) * 0.1, requires_grad=False
        )
        layer.register_parameter("bias", bias)
    else:
        layer.register_parameter("bias", None)


@pytest.mark.parametrize(
    "strategy",
    [QuantizationStrategy.CHANNEL],
)
@pytest.mark.parametrize(
    "in_out_features",
    # (k, n) both divisible by 32 to satisfy int4 pack layout.
    [
        (3584, 2304),
        (3584, 1792),
        (3584, 18944),
        (18944, 1792),
        (320, 256),
        (384, 128),
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("test_dtype", [torch.half, torch.bfloat16])
def test_IluCompressedTensorsW4A8Int(
    in_out_features: tuple[int, int],
    with_bias: bool,
    test_dtype: torch.dtype,
    strategy: str,
):
    torch.manual_seed(0)
    in_features, out_features = in_out_features

    layer = _DummyLinear(logical_widths=[out_features])
    scheme = IluCompressedTensorsW4A8Int(
        strategy=strategy,
        num_bits=4,
        group_size=-1,
        is_static_input_scheme=False,
        input_symmetric=True,
        ilu_params={"gemm_format": "NN"},
    )
    scheme.create_weights(
        layer=layer,
        output_size=out_features,
        input_size=in_features,
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

    diff_list = []
    for m in range(1, 101):
        # Approximate post-normalization activations from real networks:
        # near unit variance with limited tails.
        x = torch.randn(m, in_features, dtype=test_dtype, device="cuda").clamp_(
            min=-3.0, max=3.0
        )

        out = scheme.apply_weights(layer, x, layer.bias)

        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), None, None, symmetric=True
        )
        assert x_zp is None

        ref = torch.empty(
            (m, out_features),
            dtype=test_dtype,
            device=x.device,
        )
        ref_w4a8(
            input=x_q,
            weight=weight_ref,
            i_scales=x_s,
            w_scales=w_scale_ref.squeeze(-1),
            w_i8scales=None,
            w_i8zeros=None,
            bias=bias_ref,
            output=ref,
            format=0,
            group_size=-1,
            version=2,
            output_dtype=test_dtype,
        )

        diff_list.append(torch.max(torch.abs(out - ref)).item())
        assert out.shape == (m, out_features)
        assert out.dtype == test_dtype
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    print(
        f"test_IluCompressedTensorsW4A8Int "
        f"in_features:{in_features} out_features:{out_features} "
        f"with_bias:{with_bias} strategy:{strategy} test_dtype:{test_dtype} "
        f"max_diff:{max(diff_list):.4f} min_diff:{min(diff_list):.4f} "
        f"avg_diff:{sum(diff_list) / len(diff_list):.4f}"
    )
