# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT quantization configs.

The top-level ``_registration`` module is auto-imported by
``vllm_iluvatar.ilu_register`` and deterministically imports nested config and
online-method registrars. No community-source edits to
``vllm/model_executor/layers/quantization/__init__.py`` or
``vllm/config/model.py`` are required.
"""

from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.registry import (
    IluvatarQuantRegistry,
)

__all__ = ["IluvatarQuantRegistry"]
