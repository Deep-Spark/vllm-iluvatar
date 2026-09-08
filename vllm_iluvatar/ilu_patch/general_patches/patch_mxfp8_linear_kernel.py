# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar MXFP8 dense linear kernel.

Corresponds to: vllm/model_executor/kernels/linear/__init__.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The kernel implementation lives in
``vllm_iluvatar.model_executor.kernels.linear.mxfp8.ilu_mxfp8_linear``; this
module only performs register-time wiring.

Target   : vllm.model_executor.kernels.linear._POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA]
Baseline : community list = [FlashInferCutedslMxfp8LinearKernel,
           FlashInferCutlassMxfp8LinearKernel, MarlinMxfp8LinearKernel,
           EmulationMxfp8LinearKernel, HummingMxfp8LinearKernel]
Why      : none of the community CUDA MXFP8 kernels run on Iluvatar, yet
           ``IluPlatform._enum == CUDA`` makes them all candidates -- in practice
           ``FlashInferCutlassMxfp8LinearKernel`` is selected and the layer then
           fails for want of CUTLASS. ``register_linear_kernel`` only *appends*,
           which would leave those candidates ahead of ours, so prepend instead.
How      : data-structure insert -- ``list.insert(0, IluMxfp8LinearKernel)`` on
           the module-global CUDA list, exactly as
           ``patch_fp8_block_scaled_kernel.py`` does for block-FP8. The list
           object identity is preserved and ``init_mxfp8_linear_kernel``
           references the module global, so an in-place insert is visible to
           every reader; no ``rebind_module_func`` and no method-body copy, hence
           no fingerprint / signature guard. Gated on ``is_ivcore30()``.
Scope    : all MXFP8 dense linears on Iluvatar corex30 (MiniMax-M3 MXFP8).
Remove   : when community grows a vendor MXFP8 kernel priority hook, or ixformer
           becomes the community default.

Timing   : ``init_mxfp8_linear_kernel`` runs while ``ModelOptMxFp8LinearMethod``
           is constructed (model construction / weight load), well after
           ``register_general`` -> ``apply_general_patches()``. A patch applied
           later would miss the selection; unit tests must call ``register()``
           before selecting a kernel.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p

_CLS_NAME = "IluMxfp8LinearKernel"


def register() -> None:
    """Register the Iluvatar MXFP8 kernel at the head of the CUDA list.

    Delegates to ``register_once`` (idempotent + announcement) and is called
    automatically by ``apply_general_patches()``; the patch-specific work lives
    in ``_apply()``.
    """
    p.register_once(
        __name__,
        _apply,
        detail=f"{_CLS_NAME} -> _POSSIBLE_MXFP8_KERNELS[CUDA][0] on corex30",
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

    from vllm_iluvatar.model_executor.kernels.linear.mxfp8.ilu_mxfp8_linear import (
        IluMxfp8LinearKernel,
    )

    cuda_kernels = k._POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA]

    # [ILU override] _POSSIBLE_MXFP8_KERNELS[CUDA] — delta: prepend the Iluvatar
    # ixmm-backed MXFP8 kernel so it is selected ahead of the (non-functional on
    # Iluvatar) community candidates. Idempotent: skip if already present.
    if IluMxfp8LinearKernel not in cuda_kernels:
        cuda_kernels.insert(0, IluMxfp8LinearKernel)
