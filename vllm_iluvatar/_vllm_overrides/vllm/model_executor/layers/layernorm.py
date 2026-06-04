# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom normalization layers."""

import ixformer.inference.functions as ixf_ops
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import kernels
import vllm.kernels  # noqa: F401
from vllm import envs, ir
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant

logger = init_logger(__name__)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def poly_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    out = torch.empty_like(x)
    ops.poly_norm(  # type: ignore[attr-defined]
        out,
        x,
        weight,
        bias,
        variance_epsilon,
    )
    return out


def rms_norm_qk(
    input_q: torch.Tensor,
    input_k: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm import _custom_ops as ops

    output_q = torch.empty_like(input_q)
    output_k = torch.empty_like(input_k)
    ops.rms_norm_qk(output_q, output_k, input_q, input_k, weight_q, weight_k, epsilon)
    return output_q, output_k


@CustomOp.register("rms_norm_qk")
class RMSNormQK(CustomOp):
    """Root Mean Square Normalization for Query/Key tensors.

    Computes:
        q -> w_q * q / sqrt(E[q^2] + eps)
        k -> w_k * k / sqrt(E[k^2] + eps)
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


# --8<-- [start:rms_norm]
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    # --8<-- [end:rms_norm]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

        # Do not pass identity weight to native implementation (causes issue on TPU).
        # Other implementations require weight to be passed even if all ones.
        # Cheat and predict if native will be dispatched to:
        #  1) if native is first in priority list
        #  2) if variance_size_override is given (only supported by native impl)
        # TODO(luka): address weight passing inconsistency:
        # https://github.com/vllm-project/vllm/issues/39370
        priority = get_current_vllm_config().kernel_config.ir_op_priority
        var_override = self.variance_size_override is not None
        native_rms_norm = priority.rms_norm[0] == "native" or var_override
        native_add_rms_norm = priority.fused_add_rms_norm[0] == "native" or var_override
        self.pass_weight = self.has_weight or not native_rms_norm
        self.pass_weight_add = self.has_weight or not native_add_rms_norm

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        if residual is None:
            return ir.ops.rms_norm(
                x,
                self.weight.data if self.pass_weight else None,
                self.variance_epsilon,
                self.variance_size_override,
            )
        else:
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
        if (
            envs.VLLM_BATCH_INVARIANT
            and residual is None
            and self.variance_size_override is None
        ):
            return rms_norm_batch_invariant(x, self.weight.data, self.variance_epsilon)

        return self.forward_native(x, residual)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(x, residual)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


# --8<-- [start:gemma_rms_norm]
@CustomOp.register("gemma_rms_norm")
class GemmaRMSNorm(CustomOp):
    """RMS normalization for Gemma.

    Two differences from the above RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    # --8<-- [end:gemma_rms_norm]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        weight = self.weight.data.float() + 1.0
        out, residual = fused_add_rms_norm(
            x.float(), residual, weight, self.variance_epsilon
        )
        return (
            out.to(orig_dtype) if residual is None else (out.to(orig_dtype), residual)
        )

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
            else:
                return ixf_ops.gemma_rms_norm(
                    input=x,
                    weight=self.weight.data,
                    eps=self.variance_epsilon,
                    residual=residual,
                )
        assert (
            not fused_int8_quant and not return_origin_output
        ), "Dynamic int8 quantization and returning origin output are only supported for 2D inputs."
        return self.forward_native(x, residual)


# --8<-- [start:rms_norm_gated]
@CustomOp.register("rms_norm_gated")
class RMSNormGated(CustomOp):
    """RMS Normalization with optional gating.

    This is a native PyTorch implementation that supports:
    - Standard RMS normalization
    - Group RMS normalization
    - Optional gating with SiLU activation
    """

    # --8<-- [end:rms_norm_gated]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        activation: str = "swish",
    ):
        """Initialize RMSNormGated.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon for numerical stability
            group_size: If not None, do GroupNorm with each group
                        having group_size elements.
                        group_size=None is equivalent to group_size=hidden_size
                        (i.e. there's only 1 group).
            norm_before_gate: If True and z is provided: out = norm(x) * silu(z)
                              If False and z is provided: out = norm(x * silu(z))
            device: Device to create parameters on
            dtype: Data type for parameters
            activation: Activation function name for gating
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    @staticmethod
    def forward_static(
        x: torch.Tensor,
        z: torch.Tensor | None,
        weight: torch.Tensor,
        epsilon: float,
        orig_dtype: torch.dtype,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        activation: str = "swish",
    ) -> torch.Tensor:
        """Pure-PyTorch RMS normalization with optional gating.

        This static method contains the full native logic so that both
        ``forward_native`` and ``MatcherRMSNormGated`` (used by the
        compilation pattern matcher) can share the same implementation.

        If *z* is not None and *norm_before_gate* is True:
            ``out = rms_norm(x) * act(z)``
        If *z* is not None and *norm_before_gate* is False:
            ``out = rms_norm(x * act(z))``
        """
        x = x.float()
        weight = weight.float()
        if z is not None:
            z = z.float()

        assert activation in ["silu", "sigmoid", "swish"]
        act_fn = F.sigmoid if activation == "sigmoid" else F.silu

        if z is not None and not norm_before_gate:
            x = x * act_fn(z)

        if group_size is None:
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + epsilon)
            out = x_normed * weight
        else:
            from einops import rearrange

            x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
            variance = x_group.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_group * torch.rsqrt(variance + epsilon)
            out = rearrange(x_normed, "... g d -> ... (g d)") * weight

        if z is not None and norm_before_gate:
            out = out * act_fn(z)

        return out.to(orig_dtype)

    def forward_native(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return self.forward_static(
            x,
            z,
            self.weight,
            self.eps,
            x.dtype,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
        )

    def forward_cuda(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

        return rmsnorm_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
        )

    def forward_xpu(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward_cuda(x, z)


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)
