# SPDX-License-Identifier: Apache-2.0
# Source: vllm_iluvatar/model_executor/layers/activation.py at 47fb804.
# Decode storage belongs to ixaite; baseline allocation remains unchanged.
from __future__ import annotations

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import (
    SiluAndMul,
    SiluAndMulWithClamp,
    SwigluStepAndMul,
)
from vllm.platforms import current_platform

_ACT_ROW_ALIGN = 32


# Keep the decode padding contract paired with ixaite GEMM consumers.
class IluSiluAndMul(SiluAndMul):
    # Bind the published ABI once when the layer is constructed.
    def __init__(self, *, compile_native: bool = True):
        CustomOp.__init__(self, compile_native=compile_native)
        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            from vllm import _custom_ops as ops

            # [ILU override] SiluAndMul.op - delta: use vllm._custom_ops.
            self.op = ops.silu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native

    # [ILU override] SiluAndMul.forward_cuda - delta: over-allocate the row
    # dimension for short decode batches and return a view.
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        if x.dim() == 2 and x.shape[0] < _ACT_ROW_ALIGN:
            # Rows past x.shape[0] are never written and stay uninitialized.
            # A consumer that reads them must also drop them: each GEMM output
            # row depends only on the input row at the same index.
            buffer = torch.empty((_ACT_ROW_ALIGN, d), dtype=x.dtype, device=x.device)
            out = buffer[: x.shape[0]]
        else:
            out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out


# Reuse the community formula until ixaite has the fused quantized variant.
class IluSiluAndMulWithClamp(SiluAndMulWithClamp):
    # Preserve the public configuration consumed by the native formula.
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

    # Reject missing fusion before any output is written.
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if self.fuse_quant:
            raise RuntimeError(
                "ixaite does not support fused clamped activation quantization"
            )
        return self.forward_native(x)


# The inherited Torch formula preserves Step's clamp-after-silu semantics.
class IluSwigluStepAndMul(SwigluStepAndMul):
    """Step-3.5 shared-expert SwiGLU (clamp-after-silu), layers 43/44.

    Community ``SwigluStepAndMul.forward_cuda`` calls the triton kernel
    directly (not via ``self.op``), so this override uses the inherited Torch
    formula to retain clamp-after-silu semantics without entering ixformer.
    """

    # Validate the bound used by the inherited formula.
    def __init__(self, limit: float = 7.0, *, compile_native: bool = True):
        CustomOp.__init__(self, compile_native=compile_native)
        if limit is None:
            raise ValueError("SwigluStepAndMul requires limit to be set.")
        self.limit = float(limit)

    # Use the same independent formula for CUDA-alike dispatch.
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)
