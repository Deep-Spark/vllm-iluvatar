# SPDX-License-Identifier: Apache-2.0
"""Iluvatar FP8 block-scaled dense linear kernel.

New backend implementation living in the upstream-mirrored "by-backend"
directory ``vllm/model_executor/kernels/linear/scaled_mm/`` (siblings:
``triton.py`` / ``cutlass.py`` / ``deep_gemm.py``); per the plugin naming rules a
new backend file in such a mirror gets the ``ilu_`` prefix and the class carries
the ``Ilu`` prefix aligned with its community sibling
``TritonFp8BlockScaledMMKernel``.

Stage note: ``apply_block_scaled_mm`` is implemented as a pure **torch
equivalent** (dequantize + matmul) to get the DeepSeek-V4 raw-FP8 dense linears
running first; it does NOT call ixformer yet. The semantics match the community
base class / Triton kernel (``output = dequant(A) @ dequant(B).T``) and the
tensor / scale layouts are exactly what
``Fp8BlockScaledMMLinearKernel.apply_weights`` hands in, so no transpose to any
ixformer layout is needed. The ixformer ``fp8_block_scaled_mm`` kernelization is
a later convergence step that only swaps the matmul body.

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
    """FP8 block-scaled dense GEMM for Iluvatar (torch-equivalent stage)."""

    # Iluvatar has no ``torch.ops._C.per_token_group_fp8_quant`` op, so the base
    # ``apply_weights`` activation-quant step (``self.quant_fp8``) is unavailable.
    # Take the sanctioned base-class escape hatch (same as FlashInfer): skip the
    # activation quant and feed the bf16 activation straight into
    # ``apply_block_scaled_mm``. Since the matmul already dequantizes the weight,
    # this becomes a weight-only-fp8 path (bf16 activation x dequant(weight)) —
    # numerically higher precision than fp8xfp8 and enough for "跑通+单句正确".
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

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # Layouts handed in by the base ``apply_weights``:
        #   A  [M, K]                 activation. With ``apply_input_quant=False``
        #                             (our default) this is the raw bf16 input and
        #                             ``As`` is an unused placeholder; when True it
        #                             is fp8_e4m3 per-token-group quantized.
        #   As [M, K/gk]   fp32       per-token / per-K-group activation scale
        #                             (only meaningful when apply_input_quant).
        #   B  [N, K]      fp8_e4m3   (block-quantized weight)
        #   Bs [N/gn, K/gk]           (per-block weight scale)
        # weight_group_shape = (gn, gk); DeepSeek-V4 uses (128, 128).
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

        # Broadcast the group scales back to full [., K] resolution. This is the
        # vectorized equivalent of the per-block tiled dequant+accumulate: within
        # each K-block the scalar scale factors out of the dot product, so
        # dequantizing up front then a single matmul is numerically identical
        # (see tests/kernels/quant_utils.py::native_w8a8_block_matmul).
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
