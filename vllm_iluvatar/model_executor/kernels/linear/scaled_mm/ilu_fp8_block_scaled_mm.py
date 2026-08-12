# SPDX-License-Identifier: Apache-2.0
"""Iluvatar FP8 block-scaled dense linear kernel.

New backend implementation living in the upstream-mirrored "by-backend"
directory ``vllm/model_executor/kernels/linear/scaled_mm/`` (siblings:
``triton.py`` / ``cutlass.py`` / ``deep_gemm.py``); per the plugin naming rules a
new backend file in such a mirror gets the ``ilu_`` prefix and the class carries
the ``Ilu`` prefix aligned with its community sibling
``TritonFp8BlockScaledMMKernel``.

Design: the DeepSeek-V4 raw-FP8 dense linears arrive as 128x128 block-scaled fp8
weights. At load time ``process_weights_after_loading`` **requantizes** them to a
per-output-channel fp8 layout (``B_fp8[N,K]`` e4m3 + ``Bs[N]`` fp32) via the
ixformer op ``ixfops.fp8_weight_block_to_channel_requant`` (multi-backend:
ivcore30-native CUDA kernel + torch reference, so corex11 / non-ivcore30 fall
back automatically). The op decodes the block scale (fp32 or e8m0) and handles
strided inputs internally, and is called in place (``fp8_output=weight``) so the
requant reuses the weight buffer. At runtime ``apply_block_scaled_mm``
dynamically per-token-quantizes the bf16 activation
(``ixfops.fp8_per_token_dynamic_scaled``) and runs the per-channel GEMM
(``ixfops.fp8_per_channel_mm``); the ixformer registry handles the ixmm-native
vs torch-reference dispatch, so no arch branching lives here. This is the
standard W8A8 recipe (per-token activation x per-channel weight): a small
precision drop from the 128x128 block scale, sufficient for "跑通 + 单句正确".

Grouped ``wo_a``-style weights (marked ``is_bmm=True`` and carrying a
``bmm_geom``) are also requantized to per-channel fp8 at load time, but are then
reshaped into the grouped views ``bmm_b_fp8`` [G, D, R] + ``bmm_bs`` [G, D] that
the DeepSeek-V4 fp8 grouped-einsum ``_o_proj`` consumes
(``ds4_inv_rope_fp8_quant`` + ``fp8_per_channel_einsum``), instead of going
through ``apply_block_scaled_mm``. Unlike NV's ``deep_gemm.fp8_einsum`` (which
eats the 128-block scale natively, so wo_a needs no weight transform), the
ixformer per-channel einsum requires this one extra block->per-channel requant.
An ``is_bmm`` layer *without* ``bmm_geom`` has no einsum consumer and keeps the
block layout untouched.

Constraint: per-channel requant + ``fp8_per_channel_mm`` require ``K % 64 == 0``
(satisfied by every DeepSeek-V4 dense linear, whose K is a multiple of 128); a
non-conforming K raises at load time rather than silently falling back. The pure
torch block dequant+matmul math is retained as ``_apply_block_scaled_mm_torch``
for a future block-GEMM reference; it is NOT on the runtime path (it needs the
un-requantized block layout).

Registration is done by the general patch
``ilu_patch.general_patches.patch_fp8_block_scaled_kernel`` (this module holds no
``register()``); on Iluvatar Triton and the other CUDA candidates cannot run, so
after that patch inserts this kernel at the head of the CUDA list it is the sole
viable block-FP8 dense path.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.utils import replace_parameter

from vllm_iluvatar.ilu_util.arch import is_ivcore30

# Module-level zero-arg arch predicate, resolved as a module global at call time
# so tests can monkeypatch it (mirrors
# ``distributed/parallel_state.py::_is_ivcore30_fn`` and the mxfp4 oracle).
_is_ivcore30_fn = is_ivcore30


def _e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Torch-equivalent decode of an E8M0 (exponent-only) scale to fp32.

    E8M0 stores only the 8-bit biased exponent (bias=127); reconstruct fp32 by
    placing those bits into the exponent field (bits 23-30), sign=0, mantissa=0.
    Kept as a local torch implementation rather than importing the community
    ``fp8_utils._upcast_e8m0_to_fp32`` (a private helper not treated as a stable
    interface).
    """
    exp_bits = scale.view(torch.uint8).to(torch.int32)
    return (exp_bits << 23).view(torch.float32)


class IluFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """FP8 block-scaled dense GEMM for Iluvatar (per-channel requant stage)."""

    # Iluvatar has no ``torch.ops._C.per_token_group_fp8_quant`` op, so the base
    # ``apply_weights`` activation-quant step (``self.quant_fp8``) is unavailable.
    # Take the sanctioned base-class escape hatch (same as FlashInfer): skip the
    # activation quant and feed the bf16 activation straight into
    # ``apply_block_scaled_mm``, which does its own per-token dynamic quant via
    # ``ixfops.fp8_per_token_dynamic_scaled``.
    apply_input_quant: ClassVar[bool] = False

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        # Arch gate only: this kernel is inserted into the CUDA list solely on
        # corex30 (patch is is_ivcore30()-gated), so reaching here already means
        # an Iluvatar device — no separate cuda_alike check needed.
        if not _is_ivcore30_fn():
            return False, "requires Iluvatar ivcore30 (corex30) architecture."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Base first: produces the block-scaled fp8 ``weight`` [N, K] and the
        # 128x128 block scale (``weight_scale_inv`` for Fp8LinearMethod, else
        # ``weight_scale`` for compressed-tensors — mirror the base selection).
        super().process_weights_after_loading(layer)

        # ``wo_a``-style grouped weights (``is_bmm=True``) are consumed via the
        # fp8 grouped-einsum path (``fp8_attention._o_proj`` ->
        # ``ds4_inv_rope_fp8_quant`` + ``fp8_per_channel_einsum``), which -- like
        # every dense linear here -- needs per-channel fp8 weights, not the
        # 128x128 block layout. Unlike NV's ``deep_gemm.fp8_einsum`` (native
        # 128-block scale, no weight transform), ixformer's per-channel einsum
        # requires one extra block->per-channel requant; we do it here at load
        # time and attach the grouped views the einsum consumes so ``_o_proj``
        # does zero weight work in forward (plan §3.1). A ``is_bmm`` layer
        # *without* ``bmm_geom`` has no einsum consumer and is left in the block
        # layout untouched.
        is_bmm = bool(getattr(layer, "is_bmm", False))
        bmm_geom = getattr(layer, "bmm_geom", None)
        if is_bmm and bmm_geom is None:
            return

        params = self._get_layer_params(layer)
        weight = params.weight
        if params.weight_scale_inv is not None:
            block_scale = params.weight_scale_inv
            scale_attr_name = params.WEIGHT_SCALE_INV
        else:
            block_scale = params.weight_scale
            scale_attr_name = params.WEIGHT_SCALE

        K = int(weight.shape[1])
        if K % 64 != 0:
            raise ValueError(
                "IluFp8BlockScaledMMKernel requires K % 64 == 0 for the "
                f"per-channel requant / fp8_per_channel_mm path, got K={K}."
            )

        # Deferred import keeps ixformer off the non-ivcore30 eager import path
        # (matches the patch / is_supported arch gate).
        import ixformer.inference.functions as ixfops

        # The ixformer op decodes the block scale (fp32 or e8m0) and handles
        # strided inputs internally; ``fp8_output=weight`` requantizes in place
        # (weight stays e4m3 [N, K]), so only the scale parameter is replaced.
        # Requant in place on ``weight.data`` (a plain tensor) rather than the
        # Parameter, so an in-place ``copy_`` never trips the leaf-requires-grad
        # guard regardless of the Parameter's ``requires_grad``.
        gn, gk = int(self.weight_group_shape[0]), int(self.weight_group_shape[1])
        _b_fp8, bs = ixfops.fp8_weight_block_to_channel_requant(
            weight.data, block_scale.data, gn, gk, fp8_output=weight.data
        )

        # Overwrite the block scale parameter: base ``apply_weights`` re-reads the
        # weight (requantized in place above) and (weight_scale_inv or
        # weight_scale) and hands the latter in as ``Bs`` — now the [N]
        # per-channel scale — so no new weight buffer / flag is needed.
        replace_parameter(layer, scale_attr_name, bs)

        if not is_bmm:
            return

        # ``wo_a`` (is_bmm) with geometry: build the load-time grouped zero-copy
        # views the fp8 einsum consumes. The flat weight is
        # ``[N, R] = [G*D, heads_per_group*head_dim]`` and ``bs`` is the flat
        # per-channel ``[N]`` scale.
        #   bmm_b_fp8 [G, D, R] e4m3, contiguous (fp8_per_channel_einsum ``b``).
        #   bmm_bs    [G, D]   fp32, read-safe row stride (``Bs``).
        G = int(bmm_geom.n_local_groups)
        D = int(bmm_geom.o_lora_rank)
        R = int(bmm_geom.heads_per_group) * int(bmm_geom.head_dim)
        N = G * D
        if R % 64 != 0 or R < 128:
            raise ValueError(
                "IluFp8BlockScaledMMKernel wo_a (is_bmm) requires R = "
                f"heads_per_group*head_dim to be a multiple of 64 and >= 128, "
                f"got R={R}."
            )
        if tuple(weight.shape) != (N, R):
            raise ValueError(
                "IluFp8BlockScaledMMKernel wo_a (is_bmm) weight shape "
                f"{tuple(weight.shape)} does not match bmm_geom "
                f"(n_local_groups={G}, o_lora_rank={D}, R={R}) -> expected "
                f"[{N}, {R}]."
            )

        layer.bmm_b_fp8 = weight.data.view(G, D, R)

        # Read-safe ``Bs`` for fp8_per_channel_einsum: last dim contiguous and
        # row stride >= ceil(D/128)*128. ``D = o_lora_rank`` is 128-aligned in
        # DeepSeek-V4, so the contiguous ``[G, D]`` view already qualifies; only
        # a non-128-aligned ``D`` needs a copy into a padded read-safe buffer.
        bs_gd = bs.view(G, D)
        safe_stride = ((D + 127) // 128) * 128
        if bs_gd.stride(1) == 1 and bs_gd.stride(0) >= safe_stride:
            layer.bmm_bs = bs_gd
        else:
            bs_safe = ixfops.alloc_readsafe_scale(G, D, device=bs_gd.device)
            bs_safe.copy_(bs_gd)
            layer.bmm_bs = bs_safe

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # Layouts handed in by the base ``apply_weights`` after the load-time
        # requant:
        #   A  [M, K]      bf16/fp16   raw activation (apply_input_quant=False,
        #                              so ``As`` is an unused placeholder).
        #   B  [N, K]      fp8_e4m3    per-output-channel requantized weight.
        #   Bs [N]         fp32        per-output-channel weight scale.
        # A layer skipped by ``process_weights_after_loading`` still carries a
        # 2-D block scale here; feeding that to ``fp8_per_channel_mm`` (which
        # wants ``[N]``) would be silently wrong rather than an error.
        if Bs.dim() != 1:
            raise RuntimeError(
                "IluFp8BlockScaledMMKernel.apply_block_scaled_mm expects a "
                f"per-channel [N] weight scale, got shape {list(Bs.shape)}. The "
                "load-time requant did not run for this layer."
            )
        # Deferred import keeps ixformer off the non-ivcore30 eager import path
        # (matches the patch/is_supported arch gate).
        import ixformer.inference.functions as ixfops

        # ``fp8_per_token_dynamic_scaled`` only requires the last dim contiguous
        # (``stride(-1) == 1``), which the base ``apply_weights`` guarantees via
        # ``x.view(-1, x.shape[-1])`` — no explicit ``.contiguous()`` needed.
        a_fp8, a_scale = ixfops.fp8_per_token_dynamic_scaled(A)
        out = ixfops.fp8_per_channel_mm(a_fp8, B, a_scale, Bs)
        return out.to(self.config.out_dtype)

    def _apply_block_scaled_mm_torch(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-torch block-scaled dequant+matmul reference (NOT on the runtime
        path).

        Retained for a future block-GEMM reference. It expects the original
        block-scaled layout (``B`` e4m3 [N, K], ``Bs`` block scale
        [N/gn, K/gk]) and is therefore incompatible with the per-channel weights
        produced by ``process_weights_after_loading``; construct a kernel without
        requant (or feed block-layout tensors directly) to use it.

        Semantics match the community base class / Triton kernel
        (``output = dequant(A) @ dequant(B).T``): within each K-block the scalar
        scale factors out of the dot product, so dequantizing up front then a
        single matmul is numerically identical to the per-block tiled
        dequant+accumulate (see
        ``tests/kernels/quant_utils.py::native_w8a8_block_matmul``).
        """
        gn, gk = int(self.weight_group_shape[0]), int(self.weight_group_shape[1])

        # E8M0 scales are exponent-only; Iluvatar does not auto-decode them, so
        # upcast to fp32 in the adapter via a local torch equivalent.
        if Bs.dtype == torch.float8_e8m0fnu:
            Bs = _e8m0_to_fp32(Bs)
        Bs = Bs.to(torch.float32)

        a = A.to(torch.float32)
        b = B.to(torch.float32)
        M, K = a.shape
        N = b.shape[0]

        bs_full = Bs.repeat_interleave(gn, dim=0).repeat_interleave(gk, dim=1)[:N, :K]
        b = b * bs_full

        # Only dequantize the activation when it was actually fp8-quantized. On
        # the runtime (apply_input_quant=False) path A is already real bf16 and
        # As is a dummy placeholder, so it must be ignored.
        if self.apply_input_quant:
            if As.dtype == torch.float8_e8m0fnu:
                As = _e8m0_to_fp32(As)
            As = As.to(torch.float32)
            as_full = As.repeat_interleave(gk, dim=1)[:, :K]
            a = a * as_full

        out = torch.matmul(a, b.t())
        return out.to(self.config.out_dtype)
