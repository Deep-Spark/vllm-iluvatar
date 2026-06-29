# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Punica GPU ILU LoRA-fusion override.

Corresponds to: vllm/lora/punica_wrapper/punica_gpu.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementation lives in
``vllm_iluvatar.lora.punica_wrapper.punica_gpu``; this module only does the
register-time wiring: signature guard, binding the live community method as the
fallback, and installing the wrapper onto ``PunicaWrapperGPU``.

Target   : vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU.add_lora_linear
Baseline : main_e0871ad2259
How      : add_lora_linear -> wrapper delegating to the community method when
                              ixformer fusion is disabled or unsupported.
Fingerprint: none. The community method body is not copied; the wrapper stores
           and delegates to the live method. Only a strict signature guard is
           needed, gated by VLLM_ILU_PATCH_CHECK_SIGNATURE.
"""

from __future__ import annotations

import importlib
import inspect

import vllm_iluvatar.ilu_patch.general_patch as p

# Verbatim community-baseline source of
# PunicaWrapperGPU.add_lora_linear (4-space method indent, trailing newline),
# used as the drift anchor by the fingerprint guard.
#
# Baseline: main_e0871ad2259
ADD_LORA_LINEAR = '''    def add_lora_linear(
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
        """
        Applicable to linear-related lora.

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)
        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight.
            scale (float): Scaling factor.
            output_slices (tuple[int, ...]): Every slice's size.
            buffer (Optional[torch.Tensor]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)

        assert buffer is None, (
            "To minimize overhead, the buffer should be created by "
            ".add_lora_linear() instead of being passed in."
        )
        r = lora_b_stacked[0].size(-1)
        # We set the buffer to be float32 by default, refer to:
        # https://github.com/triton-lang/triton/issues/1387
        # Note: buffer is zeroed inside the shrink op
        buffer = torch.empty(
            (len(output_slices), x.size(0), r), dtype=torch.float32, device=x.device
        )
        add_inputs = kwargs.pop("add_inputs", True)
        self.add_shrink(
            buffer,  # type: ignore
            x,
            lora_a_stacked,
            scale,
            **kwargs,
        )
        self.add_expand(
            y,
            buffer,  # type: ignore
            lora_b_stacked,
            output_slices,
            add_inputs=add_inputs,
            **kwargs,
        )
'''


def register() -> None:
    """Register the ILU Punica GPU LoRA fusion wrapper."""
    p.register_once(
        __name__,
        _apply,
        detail="PunicaWrapperGPU.add_lora_linear ixformer fusion",
    )


def _apply() -> None:
    community = importlib.import_module("vllm.lora.punica_wrapper.punica_gpu")
    impl = importlib.import_module(
        "vllm_iluvatar.lora.punica_wrapper.punica_gpu"
    )
    PunicaWrapperGPU = community.PunicaWrapperGPU

    # Drift guard for the wrapper baseline method source
    # (VLLM_ILU_PATCH_CHECK_FINGERPRINT).
    p.check_fingerprint(
        "PunicaWrapperGPU.add_lora_linear",
        inspect.getsource(PunicaWrapperGPU.add_lora_linear),
        ADD_LORA_LINEAR,
        resync_hint=(
            "Re-sync the ADD_LORA_LINEAR baseline in "
            "ilu_patch/general_patches/patch_punica_gpu.py to the new "
            "community source of PunicaWrapperGPU.add_lora_linear."
        ),
    )

    # Strict signature validation for the wrapper
    # (VLLM_ILU_PATCH_CHECK_SIGNATURE).
    p.check_signatures(PunicaWrapperGPU, impl, ["add_lora_linear"], strict=True)

    # Preserve the live community method as the fallback before installing the
    # wrapper onto the class. Instances resolve the method through the class, so
    # a class-level rebind is sufficient.
    impl.set_original_add_lora_linear(PunicaWrapperGPU.add_lora_linear)
    PunicaWrapperGPU.add_lora_linear = impl.add_lora_linear
