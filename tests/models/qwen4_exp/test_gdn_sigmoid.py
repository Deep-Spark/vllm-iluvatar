# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)


def test_gdn_output_gate_accepts_sigmoid() -> None:
    hidden = 16
    x = torch.randn(2, hidden, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(2, hidden, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(hidden, dtype=torch.bfloat16, device="cuda")
    out = RMSNormGated.forward_static(
        x,
        gate,
        weight,
        1e-6,
        x.dtype,
        group_size=None,
        norm_before_gate=True,
        activation="sigmoid",
    )
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    expected = (x.float() * torch.rsqrt(variance + 1e-6) * weight.float()) * F.sigmoid(
        gate.float()
    )
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
    assert QwenGatedDeltaNetAttention is not None
