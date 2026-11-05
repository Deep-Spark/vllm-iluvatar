# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""Ixaite MiniMax-M3 architecture registration.

Flash (mxfp8 + NVFP4) text / VL entry points resolve under
``backends.ixaite.models.minimax_m3``. The W4A8 ``IluMiniMax*`` names stay
registered so ModelRegistry never falls through to unsupported/vendor paths;
constructing them fails closed with a Flash-scope message until a dedicated
W4A8 milestone exists.
"""

from __future__ import annotations

import torch.nn as nn

_FLASH = "vllm_iluvatar.backends.ixaite.models.minimax_m3.flash"
_MODEL = "vllm_iluvatar.backends.ixaite.models.minimax_m3.model"

_FLASH_REGISTRATIONS = (
    ("MiniMaxM3SparseForCausalLM", f"{_FLASH}:MiniMaxM3SparseForCausalLM"),
    (
        "MiniMaxM3SparseForConditionalGeneration",
        f"{_FLASH}:MiniMaxM3SparseForConditionalGeneration",
    ),
)

_W4A8_REGISTRATIONS = (
    ("IluMiniMaxM3SparseForCausalLM", f"{_MODEL}:IluMiniMaxM3SparseForCausalLM"),
    (
        "IluMiniMaxM3SparseForConditionalGeneration",
        f"{_MODEL}:IluMiniMaxM3SparseForConditionalGeneration",
    ),
)


class IluMiniMaxM3SparseForCausalLM(nn.Module):
    """W4A8 compressed-tensors MiniMax is outside the Flash implementation."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        del vllm_config, prefix
        raise RuntimeError(
            "ixaite MiniMax W4A8 compressed-tensors arch is not enabled; "
            "Ixaite supports MiniMaxM3Sparse* Flash (mxfp8 + NVFP4)"
        )


class IluMiniMaxM3SparseForConditionalGeneration(nn.Module):
    """W4A8 VL MiniMax is outside the Flash implementation."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        del vllm_config, prefix
        raise RuntimeError(
            "ixaite MiniMax W4A8 compressed-tensors arch is not enabled; "
            "Ixaite supports MiniMaxM3Sparse* Flash (mxfp8 + NVFP4)"
        )


def register_model() -> None:
    from vllm import ModelRegistry

    for architecture, class_path in _FLASH_REGISTRATIONS + _W4A8_REGISTRATIONS:
        ModelRegistry.register_model(architecture, class_path)
