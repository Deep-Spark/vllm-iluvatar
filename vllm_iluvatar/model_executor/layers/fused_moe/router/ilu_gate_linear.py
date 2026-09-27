# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false
"""Iluvatar OOT override for the MoE router gate linear layer.

Corresponds to: vllm/model_executor/layers/fused_moe/router/gate_linear.py

Target   : vllm.model_executor.layers.fused_moe.router.gate_linear.GateLinear
Baseline : v0.30.1rc0
Why      : community GateLinear enables CUDA/Hopper/Blackwell router GEMM
           branches that are NVIDIA-specific; Iluvatar should preserve the
           dispatch semantics but use equivalent torch implementations for
           unsupported kernels.
How      : @GateLinear.register_oot subclass; keep community initialization
           intact and override only forward.  The ixformer bf16×bf16→fp32
           GEMM path replaces community Tier 4 (``torch.mm`` with the cuBLAS
           fp32 out_dtype epilogue).  That tier became reachable on Iluvatar in
           0.30.1: its guard was relaxed from "Hopper/Blackwell" to "any
           CUDA/ROCm device", and corex has no ``torch.mm(out_dtype=...)``.
           The branch sits ahead of the community tiers, which is safe
           because tiers 1-3 are structurally unreachable on corex (tier 1
           needs cuteDSL, tiers 2/3 need an fp32 router weight); anything it
           does not claim falls through to community tier 5
           (``ReplicatedLinear``).
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
        if (
            self.out_dtype is not None
            and self.is_unquantized
            and x.dtype != self.weight.dtype
        ):
            x = x.to(self.weight.dtype)

        # [ILU override] GateLinear.forward — delta: ixformer bf16×bf16→fp32
        # GEMM for mixed-precision router logits, in place of community Tier 4
        # (``torch.mm(..., out_dtype=torch.float32)``). Guard mirrors
        # ``allow_cublas_router_gemm and x.dtype == torch.bfloat16`` without
        # the platform term, so it does not depend on how corex answers
        # ``current_platform.is_cuda()``.
        if (
            self.is_unquantized
            and getattr(self, "bias", None) is None
            and self.out_dtype == torch.float32
            and x.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
        ):
            import ixformer.inference.functions as ixfops

            output = ixfops.gemm_bf16_bf16_fp32(x, self.weight, format="TN")
            return self._return(output)

        return super().forward(x)
