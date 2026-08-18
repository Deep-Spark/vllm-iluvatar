# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT quantization configs.

Modules here are auto-imported by ``vllm_iluvatar.ilu_register`` (the package is
listed in ``_OOT_OP_PACKAGES``); each module registers its config via the
official ``@register_quantization_config`` decorator at import time, so no
community-source edits to ``vllm/model_executor/layers/quantization/__init__.py``
or ``vllm/config/model.py`` are required.
"""

from vllm_iluvatar.model_executor.layers.quantization.ilu_quant_registry import (
    IluvatarQuantRegistry,
)

__all__ = ["IluvatarQuantRegistry"]
