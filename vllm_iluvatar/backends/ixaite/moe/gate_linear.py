# SPDX-License-Identifier: Apache-2.0
# Source: vllm_iluvatar/model_executor/layers/fused_moe/router/ilu_gate_linear.py at 47fb804; independent ixaite lifecycle.
# pyright: reportMissingImports=false
"""Iluvatar OOT override for the MoE router gate linear layer.

Corresponds to: vllm/model_executor/layers/fused_moe/router/gate_linear.py

Target   : vllm.model_executor.layers.fused_moe.router.gate_linear.GateLinear
Baseline : main_6e448d0
Why      : community GateLinear enables CUDA/Hopper/Blackwell router GEMM
           branches that are NVIDIA-specific; Iluvatar decode keeps router
           logits on ``gemm_bf16_bf16_fp32`` (MiniMax stores gate weights
           in fp32). Wider batches stay on the replicated linear: the
           ``t4x4_a2x2_k4`` write_c_fp32 epilogue fails IXCC.
How      : @GateLinear.register_oot subclass; keep community initialization
           intact and override only forward.  bf16 act x bf16 weight -> fp32
           logits replaces community Tier 3 (cuBLAS bf16->fp32). fp32 weights
           are cached as bf16; weight_loader drops the cache before copy.
Scope    : imported during ``apply_oot_registrations()``.
Remove   : when Iluvatar provides compatible router GEMM kernels for these
           GateLinear dispatch branches.
"""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

from vllm_iluvatar.backends.ixaite import ops as ilu_ops


def _router_tokens(x: torch.Tensor) -> int:
    """Token count the router GEMM consumes.

    The kernel uses ``numel // K``. A rank-3 activation whose leading
    dimension is small can still be a wide prefill.
    """
    if x.ndim == 0 or x.shape[-1] == 0:
        return 0
    return int(x.numel() // x.shape[-1])


class IluGateLinear(GateLinear):
    """Pure-ixaite MoE router GEMM, including MiniMax fp32-stored weights."""

    def _drop_router_bf16_cache(self) -> None:
        self._ixaite_weight_bf16_src = None

    def _router_weight_bf16(self) -> torch.Tensor:
        weight = self.weight
        if weight.dtype == torch.bfloat16:
            return weight
        # Checkpoint keeps fp32 router weights; ixaite gemm is bf16 x bf16 -> fp32.
        # data_ptr and _version stay put across Parameter.data.copy_. weight_loader
        # drops the cache before that copy; a raw data.copy_ does not.
        key = (
            weight.data_ptr(),
            int(weight._version),
            tuple(weight.shape),
            weight.device,
        )
        cached = getattr(self, "_ixaite_weight_bf16", None)
        if cached is None or getattr(self, "_ixaite_weight_bf16_src", None) != key:
            cached = weight.detach().to(dtype=torch.bfloat16)
            self.register_buffer("_ixaite_weight_bf16", cached, persistent=False)
            self._ixaite_weight_bf16_src = key
        return cached

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        # Clears the cache before the copy so a reload cannot keep serving
        # the previous bf16 snapshot.
        if param is getattr(self, "weight", None):
            self._drop_router_bf16_cache()
        super().weight_loader(param, loaded_weight)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # T<=1 is the warp GEMV. T=2..8 is the phys-16 MMA (k_atoms=8), which
        # compiles. Wider batches hit mr_hgemm t4x4_a2x2_k4 write_c_fp32 and
        # IXCC rejects that epilogue, so they stay on the replicated linear.
        tokens = _router_tokens(x)
        if (
            self.out_dtype is not None
            and self.out_dtype == torch.float32
            and tokens <= 8
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float32)
            and self.weight.dtype in (torch.bfloat16, torch.float32)
        ):
            x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
            output = ilu_ops.router_gemm_bf16_fp32(x_bf16, self._router_weight_bf16())
            output_bias = self.bias if self.skip_bias_add else None
            return output, output_bias

        if self.out_dtype is not None and x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)

        output, output_bias = super(GateLinear, self).forward(x)
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, output_bias
