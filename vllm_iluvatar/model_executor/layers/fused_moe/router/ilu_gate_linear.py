# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false
"""Iluvatar OOT override for the MoE router gate linear layer.

Corresponds to: vllm/model_executor/layers/fused_moe/router/gate_linear.py

Target   : vllm.model_executor.layers.fused_moe.router.gate_linear.GateLinear
Baseline : main_6e448d0
Why      : community GateLinear enables CUDA/Hopper/Blackwell router GEMM
           branches that are NVIDIA-specific; Iluvatar should preserve the
           dispatch semantics but use equivalent torch implementations for
           unsupported kernels.
How      : @GateLinear.register_oot subclass; keep community initialization
           intact and override only forward.  The ixformer bf16×bf16→fp32
           GEMM path replaces community Tier 3 (cuBLAS bf16→fp32).
Scope    : imported during ``apply_oot_registrations()``.
Remove   : when Iluvatar provides compatible router GEMM kernels for these
           GateLinear dispatch branches.
"""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


@GateLinear.register_oot()
class IluGateLinear(GateLinear):
    """Use community GateLinear dispatch with Iluvatar-safe implementations."""

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.out_dtype is not None and x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)

        # [ILU override] GateLinear.forward — delta: ixformer bf16×bf16→fp32
        # GEMM for mixed-precision router logits (replaces community Tier 3
        # cuBLAS path on NVIDIA).
        if (
            self.out_dtype is not None
            and self.out_dtype == torch.float32
            and x.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
        ):
            import ixformer.inference.functions as ixfops

            output = ixfops.gemm_bf16_bf16_fp32(x, self.weight, format="TN")
            output_bias = self.bias if self.skip_bias_add else None
            return output, output_bias

        output, output_bias = super(GateLinear, self).forward(x)
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, output_bias
