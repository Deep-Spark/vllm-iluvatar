# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MXFP8 dense linear kernel (per-channel requant stage).

Corresponds to: vllm/model_executor/kernels/linear/mxfp8/

Target   : vllm.model_executor.kernels.linear.Mxfp8LinearKernel implementations
           selected by ``init_mxfp8_linear_kernel``.
Baseline : community CUDA list = [FlashInferCutedslMxfp8LinearKernel,
           FlashInferCutlassMxfp8LinearKernel, MarlinMxfp8LinearKernel,
           EmulationMxfp8LinearKernel, HummingMxfp8LinearKernel]
Why      : none of them run on Iluvatar. FlashInfer needs CUTLASS/CuTeDSL on
           SM100, Marlin needs ``torch.ops._C.gptq_marlin_repack``, Humming
           needs its own CUDA op, and emulation dequantizes to bf16 (correct but
           it throws away the FP8 tensor cores). ``IluPlatform._enum`` is CUDA so
           ``FlashInferCutlassMxfp8LinearKernel`` is selected and then fails.
How      : requantize the per-32 UE8M0 block scale to one fp32 scale per output
           channel at load time (``ixfops.fp8_weight_block_to_channel_requant``
           with ``group_n=1, group_k=32`` -- MXFP8's grid exactly), leaving the
           weight e4m3 ``[N, K]`` in place, then run
           ``fp8_per_channel_mm`` / ``fp8_per_channel_gemv`` in forward. Wired in
           by ``ilu_patch/general_patches/patch_mxfp8_linear_kernel.py``.
Scope    : all MXFP8 dense linears on Iluvatar corex30 (MiniMax-M3 MXFP8
           attention / MLP / shared-expert projections).
Remove   : when ixmm grows a native MXFP8 GEMM that consumes the per-32 UE8M0
           scale directly, removing the per-channel requant loss.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.utils import replace_parameter

from vllm_iluvatar.ilu_util.arch import is_ivcore30

# Module-level zero-arg arch predicate, resolved as a module global at call time
# so tests can monkeypatch it (mirrors ``ilu_fp8_block_scaled_mm.py``).
_is_ivcore30_fn = is_ivcore30

# ``fp8_per_channel_gemv`` (format="TN") requires N % 32 == 0 and A/B row strides
# that are multiples of 64; above this M the GEMM is faster anyway.
_IXMM_FP8_GEMV_MAX_M = 16


class IluMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 dense GEMM for Iluvatar, via a load-time per-channel requant."""

    # Set once ``process_weights_after_loading`` has confirmed the GEMV shape
    # constraints for a given layer; read in ``apply_weights``.
    _GEMV_FLAG: ClassVar[str] = "_ilu_mxfp8_can_use_gemv"

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        # Arch gate only: this kernel is inserted into the CUDA list solely on
        # corex30 (the patch is is_ivcore30()-gated), so reaching here already
        # means an Iluvatar device.
        if not _is_ivcore30_fn():
            return False, "requires Iluvatar ivcore30 (corex30) architecture."
        return True, None

    # Named in the errors below, which abort the load rather than falling
    # through to the next candidate. ``--linear-backend`` is the escape hatch
    # that actually lands somewhere runnable: ``_filter_kernels_by_backend``
    # narrows the candidate list to ``_LINEAR_BACKEND_KERNEL_MAP["emulation"]``,
    # which holds ``EmulationMxfp8LinearKernel``. Note ``VLLM_DISABLED_KERNELS``
    # alone is *not* enough: it only skips this kernel, and the next community
    # candidate, ``FlashInferCutlassMxfp8LinearKernel``, claims support because
    # ``IluPlatform.has_device_capability`` returns True unconditionally -- the
    # very misselection this kernel exists to prevent.
    _DISABLE_HINT: ClassVar[str] = (
        "Pass --linear-backend emulation to fall back to the community "
        "emulation kernel (correct, but dequantises to bf16 and drops the FP8 "
        "tensor cores)."
    )

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        # ``Mxfp8LinearLayerConfig`` carries no fields: every MXFP8 layer is
        # e4m3 weights with uint8 E8M0 scales at block 32, so there is nothing
        # here to reject on. The per-layer shape constraints are only knowable
        # in ``process_weights_after_loading``, and raising there aborts the
        # load rather than falling through to the next candidate -- hence the
        # ``_DISABLE_HINT`` on those messages. Move the gate up here if
        # ``Mxfp8LinearLayerConfig`` ever grows shape fields.
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight
        scale = layer.weight_scale
        assert weight.dtype is MXFP8_VALUE_DTYPE, (
            f"weight expected {MXFP8_VALUE_DTYPE}, got {weight.dtype}"
        )
        assert scale.dtype in (MXFP8_SCALE_DTYPE, torch.float8_e8m0fnu), (
            f"weight_scale expected {MXFP8_SCALE_DTYPE} or float8_e8m0fnu, "
            f"got {scale.dtype}"
        )

        n, k = int(weight.shape[0]), int(weight.shape[1])
        if k % 64 != 0:
            raise ValueError(
                "IluMxfp8LinearKernel requires K % 64 == 0 for the per-channel "
                f"requant / fp8_per_channel_mm path, got K={k}. "
                f"{self._DISABLE_HINT}"
            )
        expected_scale_shape = (n, k // MXFP8_BLOCK_SIZE)
        if tuple(scale.shape) != expected_scale_shape:
            raise ValueError(
                f"weight_scale shape {tuple(scale.shape)} does not match a "
                f"[{n}, {k}] MXFP8 weight at block [1, {MXFP8_BLOCK_SIZE}]; "
                f"expected {expected_scale_shape}. {self._DISABLE_HINT}"
            )

        # Deferred import keeps ixformer off the non-ivcore30 eager import path
        # (matches the patch / is_supported arch gate).
        import ixformer.inference.functions as ixfops

        # The op accepts a float32 or float8_e8m0fnu block scale. The checkpoint
        # dtype is uint8, so take a zero-copy 1-byte reinterpretation rather than
        # ``.to(float32)``, which would read the exponent byte as an integer
        # magnitude (0x7F -> 127.0 instead of 1.0).
        block_scale = (
            scale.data
            if scale.dtype is torch.float8_e8m0fnu
            else scale.data.view(torch.float8_e8m0fnu)
        )

        # Requant in place on ``weight.data`` (a plain tensor) rather than the
        # Parameter, so the in-place write never trips the leaf-requires-grad
        # guard regardless of the Parameter's ``requires_grad``. The weight stays
        # e4m3 [N, K]; only the scale parameter changes shape.
        _w_fp8, channel_scale = ixfops.fp8_weight_block_to_channel_requant(
            weight.data,
            block_scale,
            1,
            MXFP8_BLOCK_SIZE,
            fp8_output=weight.data,
        )
        replace_parameter(layer, "weight_scale", channel_scale)

        # GEMV needs N % 32 == 0 (format "TN"); K % 64 == 0 is already asserted
        # above, which also satisfies the row-stride requirement.
        setattr(layer, self._GEMV_FLAG, n % 32 == 0)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale_w = layer.weight_scale
        # A layer that skipped the load-time requant would still carry the 2-D
        # block scale; feeding that to fp8_per_channel_mm (which wants [N]) is
        # silently wrong rather than an error.
        if scale_w.dim() != 1:
            raise RuntimeError(
                "IluMxfp8LinearKernel.apply_weights expects a per-channel [N] "
                f"weight scale, got shape {list(scale_w.shape)}. The load-time "
                "requant did not run for this layer."
            )

        # ``fp8_per_channel_mm`` / ``fp8_per_channel_gemv`` only write bf16, so a
        # non-bf16 activation would be silently rounded to 8 mantissa bits and
        # widened back -- a precision regression no log would show. MXFP8
        # checkpoints are bf16 in practice; say so rather than degrade.
        if x.dtype is not torch.bfloat16:
            raise NotImplementedError(
                "IluMxfp8LinearKernel only runs bf16 activations (the ixmm "
                f"per-channel FP8 GEMM accumulates into bf16), got {x.dtype}. "
                f"{self._DISABLE_HINT}"
            )

        import ixformer.inference.functions as ixfops

        x_2d = x.reshape(-1, x.shape[-1])
        # ``fp8_per_token_dynamic_scaled`` only requires the last dim contiguous,
        # which ``reshape`` guarantees here.
        qx, scale_x = ixfops.fp8_per_token_dynamic_scaled(x_2d)

        n = int(layer.weight.shape[0])
        output = torch.empty(
            (qx.shape[0], n), dtype=torch.bfloat16, device=qx.device
        )
        bias_bf16 = None if bias is None else bias.to(torch.bfloat16)

        use_gemv = (
            getattr(layer, self._GEMV_FLAG, False)
            and qx.shape[0] <= _IXMM_FP8_GEMV_MAX_M
        )
        if use_gemv:
            ixfops.fp8_per_channel_gemv(
                qx,
                layer.weight,
                scale_x,
                scale_w,
                output,
                bias_bf16,
                format="TN",
                backend="ixmm",
            )
        else:
            ixfops.fp8_per_channel_mm(
                qx, layer.weight, scale_x, scale_w, output, bias_bf16, backend="ixmm"
            )

        return output.view(*x.shape[:-1], n)
