# SPDX-License-Identifier: Apache-2.0
"""Wiring for compressed-tensors linear scheme selection on Iluvatar.

Corresponds to:
    vllm/model_executor/layers/quantization/compressed_tensors/
    compressed_tensors.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The implementation lives in
``vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.
compressed_tensors``; this module only installs the wrapper and preserves the
live community method as fallback.

Target   : vllm.model_executor.layers.quantization.compressed_tensors.
           compressed_tensors.CompressedTensorsConfig._get_scheme_from_parts
Baseline : main_4875b4456b5
Why      : community vLLM has no vendor hook for selecting Iluvatar W4A8/W8A8
           linear schemes from compressed-tensors configs.
How      : install an Iluvatar wrapper for _get_scheme_from_parts(); unsupported
           cases fall back to the original community method.
Scope    : late general-plugin patch during register_general().
Remove   : when upstream exposes vendor linear scheme selection through an
           official extension point.
"""

from __future__ import annotations

import importlib

from vllm_iluvatar.ilu_patch import general_patch as p

_TARGET_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors"
)
_IMPL_MODULE = (
    "vllm_iluvatar.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors"
)


def register() -> None:
    """Register Iluvatar compressed-tensors linear scheme selection."""
    p.register_once(
        __name__,
        _apply,
        detail="CompressedTensorsConfig._get_scheme_from_parts Iluvatar wrapper",
    )


def _apply() -> None:
    community = importlib.import_module(_TARGET_MODULE)
    ilu_impl = importlib.import_module(_IMPL_MODULE)

    p.check_signatures(
        community.CompressedTensorsConfig,
        ilu_impl.IluCompressedTensorsConfig,
        ["_get_scheme_from_parts"],
        strict=True,
    )

    original_get_scheme_from_parts = (
        community.CompressedTensorsConfig._get_scheme_from_parts
    )
    ilu_impl.set_original_get_scheme_from_parts(original_get_scheme_from_parts)
    community.CompressedTensorsConfig._get_scheme_from_parts = (
        ilu_impl.IluCompressedTensorsConfig._get_scheme_from_parts
    )
