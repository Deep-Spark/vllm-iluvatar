# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT overrides for convolution layers.

Corresponds to: vllm/model_executor/layers/conv.py

Target   : vllm.model_executor.layers.conv.Conv3dLayer
Baseline : main_6e448d0
Why      : the fork forced the Conv3D CUDA path to use the linear/matmul
           implementation on Iluvatar.
How      : @Conv3dLayer.register_oot subclass overriding only ``forward_cuda``.
Scope    : Conv3D CustomOp instances on Iluvatar.
Remove   : when CUDNN Conv3D is validated or upstream exposes a backend choice.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.conv import Conv3dLayer


@Conv3dLayer.register_oot
class IluConv3dLayer(Conv3dLayer):
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # [ILU override] Conv3dLayer.forward_cuda - delta: always use the
        # matmul implementation selected by the fork.
        return self._forward_mulmat(x)
