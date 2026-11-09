# Source: layernorm.py at 47fb804; ixaite norms and storage stay independent.
# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/model_executor/layers/layernorm.py

Iluvatar OOT CustomOp overrides for RMS-style normalization layers, plus the
plugin-owned ``RMSNormQK`` op used by the fused Qwen3 attention.

Target   : vllm.model_executor.layers.layernorm.{RMSNorm, GemmaRMSNorm,
           RMSNormGated} (+ new RMSNormQK, plugin-owned)
Baseline : main_6e448d0
Why      : route RMSNorm residual through the provider-selected
           vllm._custom_ops.fused_add_rms_norm and GemmaRMSNorm's 2D path
           through the provider-selected vllm._custom_ops.gemma_rms_norm
           (the fork's delta to community layernorm.py), provide a fused
           q/k RMSNorm op for Qwen3 attention, and route the GDN gated RMSNorm
           through the provider-selected vllm._custom_ops.rms_norm_gated.
How      : @RMSNorm.register_oot / @GemmaRMSNorm.register_oot /
           @RMSNormGated.register_oot subclasses that override the dispatched
           forward; RMSNormQK is a fresh CustomOp.
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

import torch
from vllm import envs, ir
from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm,
    RMSNorm,
    RMSNormGated,
)

from vllm_iluvatar.model_executor.layers.layernorm import RMSNormQK as RMSNormQK


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


# Registered once at the process-wide backend boundary.
class IluRMSNorm(RMSNorm):
    """RMSNorm override: residual path through the Iluvatar fused kernel.

    Community ``forward_cuda`` (inherited) delegates to ``forward_native``, so
    overriding ``forward_native`` is sufficient and matches the fork's delta.
    """

    # [ILU override] RMSNorm.forward_native — delta: residual branch uses the
    # Iluvatar fused_add_rms_norm (vllm._custom_ops) instead of
    # ir.ops.fused_add_rms_norm.maybe_inplace. No-residual branch uses the
    # provider-selected vllm._custom_ops.rms_norm (ixaite / ixformer).
    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            if self.variance_size_override is not None:
                return ir.ops.rms_norm(
                    x,
                    self.weight.data if self.pass_weight else None,
                    self.variance_epsilon,
                    self.variance_size_override,
                )
            from vllm import _custom_ops as ops

            out = torch.empty_like(x)
            ops.rms_norm(
                out,
                x,
                self.weight.data if self.pass_weight else None,
                self.variance_epsilon,
            )
            return out
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


# Registered once at the process-wide backend boundary.
class IluGemmaRMSNorm(GemmaRMSNorm):
    """GemmaRMSNorm override: 2D path through provider-routed gemma_rms_norm."""

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

    # [ILU override] GemmaRMSNorm.forward_cuda — delta: 2D residual/norm path
    # through the provider-routed gemma_rms_norm with the raw Gemma weight
    # (the selected kernel applies (1+w) in fp32; no per-call weight folding);
    # dynamic-int8 fusion is unavailable. Extra trailing kwargs are
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
                raise RuntimeError(
                    "ixaite does not support fused Gemma RMSNorm int8 quantization"
                )
            # residual=None is still valid: the routed op runs plain Gemma RMSNorm.
            if residual is not None and residual.dtype != x.dtype:
                residual = residual.to(x.dtype)
            from vllm import _custom_ops as ops

            return ops.gemma_rms_norm(
                x, self.weight.data, self.variance_epsilon, residual
            )
        assert not fused_int8_quant and not return_origin_output, (
            "Dynamic int8 quantization and returning origin output are only "
            "supported for 2D inputs."
        )
        return self.forward_native(x, residual)


# Config envelope of the ixaite FlyDSL rms_norm_gated kernel
# (ixaite.ops.rms_norm_gated): one dtype for x/z/weight/out, no bias,
# group_size=None, contiguous rows — the GDN output-norm configuration.
_IXAITE_RMS_NORM_GATED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_IXAITE_RMS_NORM_GATED_ACTIVATIONS = ("swish", "silu", "sigmoid")


# Registered once at the process-wide backend boundary.
class IluRMSNormGated(RMSNormGated):
    """GDN output normalization with explicit rejection of unsupported layouts."""

    # [ILU override] RMSNormGated.forward_cuda — delta: route the GDN
    # output-norm configuration through the provider-selected
    # vllm._custom_ops.rms_norm_gated instead of calling FLA rmsnorm_fn
    # directly.
    def forward_cuda(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        if (
            self.bias is None
            and self.group_size is None
            and self.activation in _IXAITE_RMS_NORM_GATED_ACTIVATIONS
            and x.dtype in _IXAITE_RMS_NORM_GATED_DTYPES
            and self.weight.dtype == x.dtype
            and (z is None or z.dtype == x.dtype)
            and x.is_contiguous()
            and (z is None or z.is_contiguous())
            and self.weight.is_contiguous()
        ):
            from vllm import _custom_ops as ops

            out = torch.empty_like(x)
            ops.rms_norm_gated(
                out,
                x,
                z,
                self.weight.data,
                float(self.eps),
                bool(self.norm_before_gate),
                self.activation,
            )
            return out
        raise ValueError(
            "ixaite gated RMSNorm does not support this layout or configuration"
        )
