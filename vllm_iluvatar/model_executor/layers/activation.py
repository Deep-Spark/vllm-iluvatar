# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT CustomOp overrides for activation layers.

Corresponds to: vllm/model_executor/layers/activation.py

Target   : vllm.model_executor.layers.activation activation CustomOps that call
           ``torch.ops._C`` on CUDA-like platforms.
Baseline : main_88e5e2c57be8
Why      : Iluvatar activation kernels are exposed through the plugin's
           ``vllm._custom_ops`` overrides rather than community ``torch.ops._C``
           registrations.
How      : @<Base>.register_oot subclasses override initialization to bind
           ``self.op`` to ``vllm._custom_ops``.  ``IluSiluAndMulWithClamp``
           additionally overrides ``forward_cuda`` (fuse_quant branch via
           ixformer ``swiglu_limit_dynamic_scaled_int8`` + alpha/beta op args)
           and exposes ``enable_fuse_quant`` for downstream W8A8 linears.
Scope    : activation CustomOps instantiated on Iluvatar.
Remove   : when community activation dispatch routes through a backend hook or
           Iluvatar implements the exact community ``torch.ops._C`` schemas.
"""

from __future__ import annotations

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import (
    FastGELU,
    FatreluAndMul,
    GeluAndMul,
    MulAndSilu,
    NewGELU,
    QuickGELU,
    SiluAndMul,
    SiluAndMulWithClamp,
)
from vllm.platforms import current_platform


@FatreluAndMul.register_oot
class IluFatreluAndMul(FatreluAndMul):
    def __init__(self, threshold: float = 0.0):
        CustomOp.__init__(self)
        self.threshold = threshold
        if current_platform.is_cuda_alike():
            from vllm import _custom_ops as ops

            # [ILU override] FatreluAndMul.op - delta: use vllm._custom_ops.
            self.op = ops.fatrelu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native


@SiluAndMul.register_oot
class IluSiluAndMul(SiluAndMul):
    def __init__(self, *, compile_native: bool = True):
        CustomOp.__init__(self, compile_native=compile_native)
        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            from vllm import _custom_ops as ops

            # [ILU override] SiluAndMul.op - delta: use vllm._custom_ops.
            self.op = ops.silu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native


@SiluAndMulWithClamp.register_oot
class IluSiluAndMulWithClamp(SiluAndMulWithClamp):
    def __init__(
        self,
        swiglu_limit: float,
        alpha: float = 1.0,
        beta: float = 0.0,
        *,
        compile_native: bool = True,
    ):
        CustomOp.__init__(self, compile_native=compile_native)
        self.fuse_quant = False
        self.fuse_quant_output_padding = 0
        self.swiglu_limit = float(swiglu_limit)
        self.alpha = float(alpha)
        self.beta = float(beta)
        if current_platform.is_rocm() or current_platform.is_xpu():
            self._forward_method = self.forward_native
        elif current_platform.is_cuda_alike():
            from vllm import _custom_ops as ops

            # [ILU override] SiluAndMulWithClamp.op - delta: use
            # vllm._custom_ops.
            self.op = ops.silu_and_mul_with_clamp
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native

    def enable_fuse_quant(self, output_padding: int = 0):
        """Enable fused activation + int8 quantization.  When enabled,
        forward_cuda returns (x_q, x_s, out_dtype) instead of a plain
        tensor so the downstream W8A8 linear can skip its own quant step."""
        self.fuse_quant = True
        self.fuse_quant_output_padding = output_padding

    # [ILU override] SiluAndMulWithClamp.forward_cuda — delta: fuse_quant
    # branch (ixformer swiglu_limit_dynamic_scaled_int8) + alpha/beta op args.
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if self.fuse_quant:
            import ixformer.inference.functions as ixf_ops

            out_dtype = x.dtype
            x_q, x_s = ixf_ops.swiglu_limit_dynamic_scaled_int8(
                input=x,
                limit=self.swiglu_limit,
                output_padding=self.fuse_quant_output_padding,
            )
            return (x_q, x_s, out_dtype)

        d = x.shape[-1] // 2
        out = torch.empty(
            x.shape[:-1] + (d,), dtype=x.dtype, device=x.device
        )
        self.op(out, x, self.swiglu_limit, self.alpha, self.beta)
        return out

@MulAndSilu.register_oot
class IluMulAndSilu(MulAndSilu):
    def __init__(self):
        CustomOp.__init__(self)
        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            from vllm import _custom_ops as ops

            # [ILU override] MulAndSilu.op - delta: use vllm._custom_ops.
            self.op = ops.mul_and_silu
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native


@GeluAndMul.register_oot
class IluGeluAndMul(GeluAndMul):
    def __init__(self, approximate: str = "none"):
        CustomOp.__init__(self)
        self.approximate = approximate
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Unknown approximate mode: {approximate}")
        if (
            current_platform.is_cuda_alike()
            or current_platform.is_cpu()
            or current_platform.is_xpu()
        ):
            from vllm import _custom_ops as ops

            if approximate == "none":
                # [ILU override] GeluAndMul.op - delta: use vllm._custom_ops.
                self.op = ops.gelu_and_mul
            elif approximate == "tanh":
                # [ILU override] GeluAndMul.op - delta: use vllm._custom_ops.
                self.op = ops.gelu_tanh_and_mul


@NewGELU.register_oot
class IluNewGELU(NewGELU):
    def __init__(self):
        CustomOp.__init__(self)
        if (
            current_platform.is_cuda_alike()
            or current_platform.is_cpu()
            or current_platform.is_xpu()
        ):
            from vllm import _custom_ops as ops

            # [ILU override] NewGELU.op - delta: use vllm._custom_ops.
            self.op = ops.gelu_new


@FastGELU.register_oot
class IluFastGELU(FastGELU):
    def __init__(self):
        CustomOp.__init__(self)
        if (
            current_platform.is_cuda_alike()
            or current_platform.is_cpu()
            or current_platform.is_xpu()
        ):
            from vllm import _custom_ops as ops

            # [ILU override] FastGELU.op - delta: use vllm._custom_ops.
            self.op = ops.gelu_fast


@QuickGELU.register_oot
class IluQuickGELU(QuickGELU):
    def __init__(self):
        CustomOp.__init__(self)
        if (
            current_platform.is_cuda_alike()
            or current_platform.is_cpu()
            or current_platform.is_xpu()
        ):
            from vllm import _custom_ops as ops

            # [ILU override] QuickGELU.op - delta: use vllm._custom_ops.
            self.op = ops.gelu_quick
