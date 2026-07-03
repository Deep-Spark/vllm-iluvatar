# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT CustomOp overrides for activation layers.

Corresponds to: vllm/model_executor/layers/activation.py

Target   : vllm.model_executor.layers.activation activation CustomOps that call
           ``torch.ops._C`` on CUDA-like platforms.
Baseline : main_4875b4456b5
Why      : Iluvatar activation kernels are exposed through the plugin's
           ``vllm._custom_ops`` overrides rather than community ``torch.ops._C``
           registrations.
How      : @<Base>.register_oot subclasses override initialization to bind
           ``self.op`` to ``vllm._custom_ops``. No forward body is copied except
           ``SiluAndMulWithClamp.forward_cuda``, whose community signature passes
           alpha/beta values unsupported by the Iluvatar custom op.
Scope    : activation CustomOps instantiated on Iluvatar.
Remove   : when community activation dispatch routes through a backend hook or
           Iluvatar implements the exact community ``torch.ops._C`` schemas.
"""

from __future__ import annotations

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import (
    FatreluAndMul,
    FastGELU,
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
