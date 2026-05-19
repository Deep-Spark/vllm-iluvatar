"""Iluvatar-friendly subclass of ``SiluAndMulWithClamp``.

Stock vLLM's class hard-codes
``self.op = torch.ops._C.silu_and_mul_with_clamp`` for cuda_alike / xpu
platforms, but that custom op is not built on iluvatar. The
:func:`vllm._custom_ops.silu_and_mul_with_clamp` Python wrapper IS
provided (by ``vllm_iluvatar.patch.patch_custom_ops``) and routes to the
ixformer kernel; we wire ``self.op`` to that wrapper here.

Used in place of ``SiluAndMulWithClamp`` inside
``vllm_iluvatar.models.ilu_deepseek_v4`` (the MLP swiglu_limit branch).
"""

from __future__ import annotations

from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.platforms import current_platform


class IluSiluAndMulWithClamp(SiluAndMulWithClamp):
    """Reuses the upstream ``forward`` machinery but binds ``self.op`` to
    the ixformer-backed Python wrapper instead of ``torch.ops._C``."""

    def __init__(self, swiglu_limit: float, *, compile_native: bool = True):
        # Skip the upstream ``__init__`` body that would assign
        # ``self.op = torch.ops._C.silu_and_mul_with_clamp`` (kernel not
        # built on iluvatar); reach past it directly to ``CustomOp.__init__``.
        # ``super(SiluAndMulWithClamp, self).__init__`` resolves the next
        # class up the MRO from ``SiluAndMulWithClamp`` (i.e. ``CustomOp``).
        super(SiluAndMulWithClamp, self).__init__(compile_native=compile_native)
        self.swiglu_limit = float(swiglu_limit)

        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            from vllm import _custom_ops as ops

            self.op = ops.silu_and_mul_with_clamp
        elif current_platform.is_cpu():
            # Same fallback as upstream.
            self._forward_method = self.forward_native
