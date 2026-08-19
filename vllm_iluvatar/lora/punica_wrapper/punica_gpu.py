# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/lora/punica_wrapper/punica_gpu.py

ILU override implementation for ``PunicaWrapperGPU.add_lora_linear``. This
module holds ONLY the override implementation; the wiring (register/_apply,
signature guard, fallback binding, and class-level replacement) lives in the
centralized ``ilu_patch.general_patches.patch_punica_gpu`` module and is
auto-discovered during ``register_general``.

Target   : vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU.add_lora_linear
Baseline : main_6e448d0
Why      : use ixformer fused LoRA GEMV on Iluvatar when
           VLLM_ILU_USE_LORA_FUSION / legacy VLLM_USE_LORA_FUSION is enabled
           and the request shape satisfies ixformer's optimized condition.
How      : wrapper fast path (delta in the [ILU override] block below) first
           tries ixformer.inference.functions.add_lora_linear; otherwise it
           delegates to the live community add_lora_linear captured before the
           class-level replacement.
Scope    : Punica GPU LoRA linear path on Iluvatar.
Remove   : when upstream exposes a Punica LoRA backend hook or the ixformer
           fused LoRA path is no longer needed.

The wrapper references the live community ``add_lora_linear`` through
``_ORIGINAL_ADD_LORA_LINEAR``. It is a placeholder here and is bound by
``patch_punica_gpu._apply()`` at register-time (kept lazy so importing this module
never pulls in vllm).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# Community fallback used by patch_punica_gpu._apply()
# before PunicaWrapperGPU.add_lora_linear is replaced. Kept lazy so this impl
# module can be imported without importing vllm.
_ORIGINAL_ADD_LORA_LINEAR: Callable | None = None


def set_original_add_lora_linear(fn: Callable) -> None:
    """Store the community method captured before class-level replacement."""
    global _ORIGINAL_ADD_LORA_LINEAR
    _ORIGINAL_ADD_LORA_LINEAR = fn


# [ILU override] PunicaWrapperGPU.add_lora_linear — delta: optionally run the
# ixformer fused LoRA GEMV path before delegating to the community Punica
# shrink+expand implementation. This is a wrapper, not a full-method copy.
def add_lora_linear(
    self,
    y: torch.Tensor,
    x: torch.Tensor,
    lora_a_stacked: tuple[torch.Tensor, ...],
    lora_b_stacked: tuple[torch.Tensor, ...],
    scale: float,
    output_slices: tuple[int, ...],
    *,
    buffer: torch.Tensor | None = None,
    **kwargs,
) -> None:
    """Try ixformer fused LoRA GEMV, then fall back to community Punica."""
    import vllm.envs as envs

    # >>> ILU MODIFICATION >>>
    # Fast path from the fork: when the plugin env reader is enabled and the
    # LoRA geometry is suitable, bypass Punica's triton shrink+expand sequence
    # and call ixformer fused add_lora_linear directly.
    if getattr(envs, "VLLM_ILU_USE_LORA_FUSION", False):
        import ixformer.inference.functions as ixf_ops

        # Geometry matches the original fork condition:
        #   num_token x m input, m x k LoRA-A, k x n LoRA-B/output.
        num_token, m = x.size(0), x.size(-1)
        k, n = lora_b_stacked[0].size(-1), y.size(-1)
        if len(lora_a_stacked) == 1 and ixf_ops.lora_gemv_optim_condition(
            num_token, m, k, n
        ):
            ixf_ops.add_lora_linear(
                y,
                x,
                lora_a_stacked,
                lora_b_stacked,
                lora_bias_stacked=None,
                scale=1.0,
                output_slices=(1,),
            )
            return
    # <<< ILU MODIFICATION <<<

    # Community fallback. Because the original method is captured at
    # register-time, upstream body changes are inherited and no fingerprint
    # baseline is needed for this wrapper patch.
    if _ORIGINAL_ADD_LORA_LINEAR is None:
        raise RuntimeError("ILU PunicaWrapperGPU.add_lora_linear patch not installed")
    return _ORIGINAL_ADD_LORA_LINEAR(
        self,
        y,
        x,
        lora_a_stacked,
        lora_b_stacked,
        scale,
        output_slices,
        buffer=buffer,
        **kwargs,
    )
