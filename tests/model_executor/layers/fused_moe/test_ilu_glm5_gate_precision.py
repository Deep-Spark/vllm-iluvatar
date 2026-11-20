# SPDX-License-Identifier: Apache-2.0
"""GLM router gate preserves FP32 accumulation before expert selection."""

import pytest
import torch

from vllm_iluvatar.model_executor.layers.fused_moe.router.ilu_gate_linear import (
    IluGateLinear,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or getattr(torch, "corex_arch", None) != "ivcore30",
    reason="CQ FP32 gate",
)
@pytest.mark.parametrize("tokens", [1, 17, 128])
@pytest.mark.parametrize("experts", [256, 288])
@torch.inference_mode()
def test_glm_gate_fp32_logits(tokens, experts):
    torch.manual_seed(2099 + experts)
    gate = IluGateLinear.__new__(IluGateLinear)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(
        (torch.randn(experts, 6144, device="cuda") / 6144**0.5).bfloat16()
    )
    gate.bias = None
    gate.skip_bias_add = False
    gate.out_dtype = torch.float32
    x = torch.randn(tokens, 6144, device="cuda", dtype=torch.bfloat16)

    actual, bias = gate(x)
    expected = (x.cpu().double() @ gate.weight.cpu().double().T).float().cuda()
    assert bias is None and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
    assert (actual != actual.bfloat16().float()).any()
