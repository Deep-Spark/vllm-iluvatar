# SPDX-License-Identifier: Apache-2.0
"""Step5 producer -> shared FP8 linear, including local TP input shards."""

import pytest
import torch
from conftest import requires_corex30

from vllm_iluvatar.model_executor.layers.quantization.fp8 import (
    IluFp8PerChannelOnlineLinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.ilu_fp8_input import IluFP8Input


def method():
    value = object.__new__(IluFp8PerChannelOnlineLinearMethod)
    value.gemm_format = "TN"
    value._runtime_context = None
    return value


@requires_corex30
@pytest.mark.parametrize("projection", ["column", "row"])
@pytest.mark.parametrize("producer", ["gate", "swiglu"])
def test_fused_producer_linear_consumer(projection, producer):
    from ixformer.inference.functions.models.step5.norm.float.decode_fusions import (
        gate_fp8,
        swiglu_fp8,
    )

    torch.manual_seed(42)
    # Step5 TP2 attention output: 32 local heads x D192 = 6144.
    # Column projection owns output shards; row projection owns input shards.
    rows, width, out_features = 8, 6144, 128
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(rows, width // 192, device="cuda", dtype=torch.bfloat16)
    actual = torch.tensor([rows], device="cuda", dtype=torch.int32)
    if producer == "gate":
        q, scales = gate_fp8(x, gate, actual, backend="ixformer_native")
        expected = (
            (x.float().view(rows, -1, 192) * gate.float().sigmoid()[..., None])
            .bfloat16()
            .flatten(1)
            .float()
        )
    else:
        packed = torch.randn(rows, 2 * width, device="cuda", dtype=torch.bfloat16)
        mask = torch.arange(rows, device="cuda", dtype=torch.int32)
        q, scales = swiglu_fp8(packed, mask, 3.0, backend="ixformer_native")
        assert scales.shape == (rows,)
        scales = scales.unsqueeze(-1)
        first, second = packed.float().chunk(2, dim=-1)
        expected = (
            ((first * first.sigmoid()).clamp(max=3.0) * second.clamp(-3.0, 3.0))
            .bfloat16()
            .float()
        )
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    cls = ColumnParallelLinear if projection == "column" else RowParallelLinear
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.bias = None
    layer.skip_bias_add = False
    layer.return_bias = True
    layer.tp_size = 1
    layer.tp_rank = 0
    layer.gather_output = False
    layer.reduce_results = True
    layer.quant_method = method()
    layer.input_size_per_partition = width
    layer.output_size_per_partition = out_features
    if projection == "row":
        layer.input_is_parallel = True
    weights = torch.randn(out_features, width, device="cuda") * 0.02
    layer.weight_scale = (weights.abs().amax(-1) / 448).clamp_min(1e-12)
    layer.weight = (weights / layer.weight_scale[:, None]).to(torch.float8_e4m3fn)
    layer._can_use_gemm = True
    result, bias = layer(IluFP8Input(q, scales))
    assert bias is None
    # Independent FP32 oracle on identical quantized values, isolating consumer error.
    dequant = q.float() * scales.reshape(rows, 1)
    reference = dequant @ (layer.weight.float() * layer.weight_scale[:, None]).T
    torch.testing.assert_close(result.float(), reference, rtol=0.02, atol=0.02)
    # Compare quantization error separately from the linear consumer error.
    torch.testing.assert_close(dequant, expected, rtol=0.07, atol=0.015)
    # Missing activation scales must be detectable for these model-sized inputs.
    wrong = q.float() @ (layer.weight.float() * layer.weight_scale[:, None]).T
    assert not torch.allclose(result.float(), wrong, rtol=0.02, atol=0.02)


def test_prequantized_fallback_rejected_before_original_method():
    layer = torch.nn.Module()
    layer._ilu_fp8_online_fallback_reason = "unaligned shape"
    with pytest.raises(ValueError, match="cannot consume"):
        method().apply(layer, IluFP8Input(torch.empty(1, 16), torch.ones(1, 1)))
