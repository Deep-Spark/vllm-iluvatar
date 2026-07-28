# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar FP8 block-scaled dense linear kernel.

Corresponds to: vllm/model_executor/kernels/linear/__init__.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The kernel implementation lives in
``vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm``;
this module only performs register-time wiring.

Target   : vllm.model_executor.kernels.linear._POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.CUDA]
Baseline : community list = [FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
           DeepGemmFp8BlockScaledMMKernel, CutlassFp8BlockScaledMMKernel,
           MarlinFP8ScaledMMLinearKernel, TritonFp8BlockScaledMMKernel,
           HummingFP8ScaledMMLinearKernel]
Why      : none of the community CUDA block-FP8 kernels run on Iluvatar (DeepGEMM
           / FlashInfer / Cutlass / Marlin unavailable, Triton block GEMM does
           not run on corex), so the DeepSeek-V4 raw-FP8 dense linears have no
           working kernel. ``IluFp8BlockScaledMMKernel`` provides a torch
           equivalent (ixformer kernelization later).
How      : data-structure insert — ``list.insert(0, IluFp8BlockScaledMMKernel)``
           on the module-global CUDA list. ``IluPlatform._enum == CUDA`` so
           ``choose_scaled_mm_linear_kernel`` reads this list. The list object
           identity is preserved and ``init_fp8_linear_kernel`` references the
           module global, so an in-place insert is visible to every reader; no
           ``rebind_module_func`` and no method-body copy, hence no fingerprint /
           signature guard. Gated on ``is_ivcore30()`` (corex30 only).
Scope    : all block-FP8 dense linears on Iluvatar corex30 (DeepSeek-V4 raw-FP8:
           gate, shared-expert, wo_b, lm_head, attention linears).
Remove   : when an official plugin registration point for FP8 block kernels
           lands, or ixformer becomes the community default.

Timing   : ``choose_scaled_mm_linear_kernel`` runs while ``Fp8LinearMethod``
           builds each layer's kernel (model construction / weight load), well
           after ``register_general`` -> ``apply_general_patches()``. A patch
           applied later would miss the selection; unit tests must call
           ``register()`` before selecting a kernel.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p

_CLS_NAME = "IluFp8BlockScaledMMKernel"


def register() -> None:
    """Register the Iluvatar FP8 block-scaled kernel at the head of the CUDA list.

    Delegates to ``register_once`` (idempotent + announcement) and is called
    automatically by ``apply_general_patches()``; the patch-specific work lives
    in ``_apply()``.
    """
    p.register_once(
        __name__,
        _apply,
        detail=f"{_CLS_NAME} -> _POSSIBLE_FP8_BLOCK_KERNELS[CUDA][0] on corex30",
    )


def _apply() -> None:
    # Arch gate: only patch on Iluvatar corex30 (ivcore30). ``is_ivcore30`` is
    # imported here (not at module top) so auto-discovery import never pulls in
    # ixformer eagerly.
    from vllm_iluvatar.ilu_util.arch import is_ivcore30

    if not is_ivcore30():
        return

    import vllm.model_executor.kernels.linear as k
    from vllm.platforms import PlatformEnum

    from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (  # noqa: E501
        IluFp8BlockScaledMMKernel,
    )

    cuda_kernels = k._POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.CUDA]

    # [ILU override] _POSSIBLE_FP8_BLOCK_KERNELS[CUDA] — delta: prepend the
    # Iluvatar torch-equivalent block-FP8 kernel so it is selected ahead of the
    # (non-functional on Iluvatar) community candidates. Idempotent: skip if
    # already present.
    if IluFp8BlockScaledMMKernel not in cuda_kernels:
        cuda_kernels.insert(0, IluFp8BlockScaledMMKernel)
