# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false
"""Iluvatar OOT override for the MoE router gate linear layer.

Corresponds to: vllm/model_executor/layers/fused_moe/router/gate_linear.py

Target   : vllm.model_executor.layers.fused_moe.router.gate_linear.GateLinear
Baseline : v0.31.0 (db9527a468)
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

from vllm_iluvatar.ilu_util.arch import IVCORE11, IVCORE30, cuda_arch


@GateLinear.register_oot()
class IluGateLinear(GateLinear):
    """Use community GateLinear dispatch with Iluvatar-safe implementations."""

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # MiniMax's gate is replicated under TP: K=6144, N=128 on every
        # rank. Keep the checkpoint FP32 weights and select only measured
        # shape ranges. ivcore11's M=7 is a near tie, so prefer torch there.
        arch = cuda_arch() if x.is_cuda else None
        if (
            x.is_cuda
            and x.ndim == 2
            and x.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
            and self.weight.dtype == torch.float32
            and self.weight.shape == (128, 6144)
            and self.bias is None
            and x.is_contiguous()
            and self.weight.is_contiguous()
            and arch in (IVCORE11, IVCORE30)
        ):
            import ixformer.inference.functions as ixfops

            native_max_m = 6 if arch == IVCORE11 else 32
            if (
                x.shape[0] <= native_max_m
                and x.data_ptr() % 16 == 0
                and self.weight.data_ptr() % 16 == 0
            ):
                return ixfops.router_gemm_bf16_fp32_fp32(
                    x, self.weight, backend="ixformer_native"
                ), None
            if arch == IVCORE11 and 7 <= x.shape[0] <= 32:
                return ixfops.router_gemm_bf16_fp32_fp32(
                    x, self.weight, backend="torch"
                ), None
            if arch == IVCORE30 and 128 <= x.shape[0] <= 4096:
                return ixfops.mixed_type_linear(x, self.weight), None

        # A model must explicitly opt in: the shape alone is not a model
        # identity and another model may also use a BF16 [256, 6144] gate.
        # GLM-5.2 decode batches here are one DP rank (bs <= 128), where
        # torch's BF16 GEMM beats the quantized dispatch. The 0.27 server
        # logged this path at bs 16.
        if (
            getattr(self, "use_bf16_small_batch_gate", False)
            and x.dtype == torch.bfloat16
            and self.weight.dtype == x.dtype
            and x.ndim == 2
            and 0 < x.shape[0] <= 128
            and self.weight.shape == (256, 6144)
            and (self.skip_bias_add or self.bias is None or self.bias.dtype == x.dtype)
            and self.out_dtype in (None, torch.bfloat16)
        ):
            bias = None if self.skip_bias_add else self.bias
            return torch.nn.functional.linear(x, self.weight, bias), (
                self.bias if self.skip_bias_add else None
            )

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
