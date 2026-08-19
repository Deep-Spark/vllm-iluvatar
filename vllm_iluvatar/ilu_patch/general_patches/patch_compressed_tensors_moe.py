# SPDX-License-Identifier: Apache-2.0
"""Wiring for compressed-tensors MoE method selection on Iluvatar.

Corresponds to:
    vllm/model_executor/layers/quantization/compressed_tensors/
    compressed_tensors_moe/compressed_tensors_moe.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The implementation lives in
``vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.
compressed_tensors_moe.ilu_compressed_tensors_moe``; this module only installs
the wrapper and preserves the live community method as fallback.

Target   : vllm.model_executor.layers.quantization.compressed_tensors.
           compressed_tensors_moe.compressed_tensors_moe.CompressedTensorsMoEMethod
Baseline : main_6e448d0
Why      : community vLLM has no vendor hook for selecting Iluvatar W4A8/W8A8
           MoE methods from compressed-tensors configs.
How      : install an Iluvatar wrapper for get_moe_method(); unsupported cases
           fall back to the original community method.
Scope    : late general-plugin patch during register_general().
Remove   : when upstream exposes vendor MoE method selection through an official
           extension point.
"""

from __future__ import annotations

import importlib

from vllm_iluvatar.ilu_patch import general_patch as p

_TARGET_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe"
)
_IMPL_MODULE = (
    "vllm_iluvatar.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.ilu_compressed_tensors_moe"
)


def register() -> None:
    """Register Iluvatar compressed-tensors MoE method selection."""
    p.register_once(
        __name__,
        _apply,
        detail="CompressedTensorsMoEMethod.get_moe_method Iluvatar wrapper",
    )


def _apply() -> None:
    community = importlib.import_module(_TARGET_MODULE)
    ilu_impl = importlib.import_module(_IMPL_MODULE)

    p.check_signatures(
        community.CompressedTensorsMoEMethod,
        ilu_impl.IluCompressedTensorsMoEMethod,
        ["get_moe_method"],
        strict=True,
    )

    original_get_moe_method = community.CompressedTensorsMoEMethod.get_moe_method
    ilu_impl.check_scheme_discovery_fingerprint(original_get_moe_method)
    ilu_impl.set_original_get_moe_method(original_get_moe_method)
    community.CompressedTensorsMoEMethod.get_moe_method = staticmethod(
        ilu_impl.IluCompressedTensorsMoEMethod.get_moe_method
    )
