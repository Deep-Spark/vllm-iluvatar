# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/model_executor/layers/layernorm.py

Iluvatar OOT CustomOp overrides for RMS-style normalization layers, plus the
plugin-owned ``RMSNormQK`` op used by the fused Qwen3 attention.

Target   : vllm.model_executor.layers.layernorm.{RMSNorm, GemmaRMSNorm}
           (+ new RMSNormQK, plugin-owned)
Baseline : main_6e448d0
Why      : route RMSNorm residual / GemmaRMSNorm through ixformer-backed kernels
           (the fork's delta to community layernorm.py), and provide a fused
           q/k RMSNorm op for Qwen3 attention.
How      : @RMSNorm.register_oot / @GemmaRMSNorm.register_oot subclasses that
           override the dispatched forward; RMSNormQK is a fresh CustomOp.
Scope    : runs whenever these layers are instantiated on the Iluvatar platform.
Remove   : when community layernorm routes these paths through the device kernels
           natively.

Dispatch note: IluPlatform inherits CudaPlatform (``_enum == CUDA``), so
``CustomOp.dispatch_forward`` selects ``forward_cuda`` (NOT ``forward_oot``).
The fork's RMSNorm delta lives in ``forward_native`` (community ``forward_cuda``
delegates to it), so ``IluRMSNorm`` overrides ``forward_native``; GemmaRMSNorm's
delta is in both, so ``IluGemmaRMSNorm`` overrides both.

Registration: import-time (the decorators below) via
``vllm_iluvatar.ilu_register.apply_oot_registrations``. No general-patch
fingerprint applies — these are replacement subclasses, not method copies; only
a prefix-compatible signature check runs (see ilu_register / ilu_util.signature).
"""

from __future__ import annotations

import ixformer.inference.functions as ixf_ops
import torch
from vllm import envs, ir
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual + RMSNorm via the Iluvatar ``vllm._custom_ops`` override."""
    from vllm import _custom_ops as ops

    result = ops.fused_add_rms_norm(
        x,
        residual,
        weight,
        variance_epsilon,
    )
    if result is not None:
        return result
    return x, residual


def rms_norm_qk(
    input_q: torch.Tensor,
    input_k: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused q/k RMSNorm via the Iluvatar ``vllm._custom_ops`` addition."""
    from vllm import _custom_ops as ops

    output_q = torch.empty_like(input_q)
    output_k = torch.empty_like(input_k)
    ops.rms_norm_qk(output_q, output_k, input_q, input_k, weight_q, weight_k, epsilon)
    return output_q, output_k


@CustomOp.register("rms_norm_qk")
class RMSNormQK(CustomOp):
    """Root Mean Square Normalization for Query/Key tensors (plugin-owned).

    Computes:
        q -> w_q * q / sqrt(E[q^2] + eps)
        k -> w_k * k / sqrt(E[k^2] + eps)

    Community vLLM has no equivalent; this op is constructed only by the fused
    Iluvatar Qwen3 attention. On the CUDA-typed IluPlatform the dispatched method
    is ``forward_cuda``.
    """

    def __init__(
        self,
        hidden_size_q: int,
        hidden_size_k: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size_q = hidden_size_q
        self.hidden_size_k = hidden_size_k
        self.variance_epsilon = eps

    def forward_native(
        self,
        input_q: torch.Tensor,
        input_k: torch.Tensor,
        weight_q: torch.Tensor,
        weight_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xq = input_q.to(torch.float32)
        xk = input_k.to(torch.float32)
        var_q = xq.pow(2).mean(dim=-1, keepdim=True)
        var_k = xk.pow(2).mean(dim=-1, keepdim=True)
        out_q = xq * torch.rsqrt(var_q + self.variance_epsilon)
        out_k = xk * torch.rsqrt(var_k + self.variance_epsilon)
        out_q = out_q * weight_q
        out_k = out_k * weight_k
        return out_q.to(input_q.dtype), out_k.to(input_k.dtype)

    def forward_cuda(
        self,
        input_q: torch.Tensor,
        input_k: torch.Tensor,
        weight_q: torch.Tensor,
        weight_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return rms_norm_qk(
            input_q,
            input_k,
            weight_q,
            weight_k,
            self.variance_epsilon,
        )


@RMSNorm.register_oot
class IluRMSNorm(RMSNorm):
    """RMSNorm override: residual path through the Iluvatar fused kernel.

    Community ``forward_cuda`` (inherited) delegates to ``forward_native``, so
    overriding ``forward_native`` is sufficient and matches the fork's delta.
    """

    # [ILU override] RMSNorm.forward_native — delta: residual branch uses the
    # Iluvatar fused_add_rms_norm (vllm._custom_ops) instead of
    # ir.ops.fused_add_rms_norm.maybe_inplace. No-residual branch unchanged.
    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return ir.ops.rms_norm(
                x,
                self.weight.data if self.pass_weight else None,
                self.variance_epsilon,
                self.variance_size_override,
            )
        return fused_add_rms_norm(
            x,
            residual,
            self.weight.data if self.pass_weight_add else None,
            self.variance_epsilon,
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if envs.VLLM_BATCH_INVARIANT:
            assert self.variance_size_override is None, (
                "Batch invariance is not supported for variance_size_override"
            )
            return rms_norm_batch_invariant(
                x,
                self.weight.data,
                self.variance_epsilon,
                residual=residual,
            )
        return self.forward_native(x, residual)


@GemmaRMSNorm.register_oot
class IluGemmaRMSNorm(GemmaRMSNorm):
    """GemmaRMSNorm override: ixformer-backed gemma_rms_norm kernels."""

    # [ILU override] GemmaRMSNorm.forward_native — delta: route through the
    # Iluvatar fused_add_rms_norm with explicit float upcast / dtype restore.
    # residual is upcast together with x: residual_rms_norm requires both to
    # share a dtype, otherwise the kernel reads the fp16/bf16 residual buffer
    # with an fp32 stride and corrupts the add.
    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        weight = self.weight.data.float() + 1.0
        if residual is None:
            out, _ = fused_add_rms_norm(x.float(), None, weight, self.variance_epsilon)
            return out.to(orig_dtype)
        out, residual_out = fused_add_rms_norm(
            x.float(), residual.float(), weight, self.variance_epsilon
        )
        return out.to(orig_dtype), residual_out.to(orig_dtype)

    # [ILU override] GemmaRMSNorm.forward_cuda — delta: 2D fast path via ixformer
    # gemma_rms_norm (+ optional dynamic-int8 fusion); extra trailing kwargs are
    # Iluvatar-only and default-valued (prefix-compatible with community).
    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        fused_int8_quant: bool = False,
        return_origin_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if len(x.shape) == 2:
            if fused_int8_quant:
                return ixf_ops.gemma_rms_norm_dynamic_int8(
                    input=x,
                    weight=self.weight.data,
                    eps=self.variance_epsilon,
                    residual=residual,
                    return_origin_output=return_origin_output,
                )
            return ixf_ops.gemma_rms_norm(
                input=x,
                weight=self.weight.data,
                eps=self.variance_epsilon,
                residual=residual,
            )
        assert not fused_int8_quant and not return_origin_output, (
            "Dynamic int8 quantization and returning origin output are only "
            "supported for 2D inputs."
        )
        return self.forward_native(x, residual)
