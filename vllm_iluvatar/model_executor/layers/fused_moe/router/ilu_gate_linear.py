# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT override for the MoE router gate linear layer.

Corresponds to: vllm/model_executor/layers/fused_moe/router/gate_linear.py

Target   : vllm.model_executor.layers.fused_moe.router.gate_linear.GateLinear
Baseline : main_4875b4456b5
Why      : community GateLinear enables CUDA/Hopper/Blackwell router GEMM
           branches that are NVIDIA-specific; Iluvatar should use the safe
           ReplicatedLinear fallback unless a native plugin kernel is wired.
How      : @GateLinear.register_oot subclass; call the community initializer,
           then disable the specialized dispatch flags. No forward method copy.
Scope    : imported during ``apply_oot_registrations()``.
Remove   : when Iluvatar provides compatible router GEMM kernels for these
           GateLinear dispatch branches.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


@GateLinear.register_oot()
class IluGateLinear(GateLinear):
    """Disable community CUDA-specialized router GEMM branches on Iluvatar."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            out_dtype=out_dtype,
            params_dtype=params_dtype,
            force_fp32_compute=force_fp32_compute,
            prefix=prefix,
        )

        # [ILU override] GateLinear dispatch flags - delta: keep community
        # initialization/forward intact, but avoid NVIDIA-specific router GEMMs.
        self.allow_specialized_router_gemm = False
        self.allow_dsv3_router_gemm = False
        self.allow_fp32_router_gemm = False
        self.allow_cublas_router_gemm = False
