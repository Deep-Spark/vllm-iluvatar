# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT override for rotary embedding helper op.

Corresponds to: vllm/model_executor/layers/rotary_embedding/common.py

Target   : vllm.model_executor.layers.rotary_embedding.common.ApplyRotaryEmb
Baseline : main_4875b4456b5
Why      : the flash-attn rotary CUDA path is not used on Iluvatar; the fork
           forced the native PyTorch implementation.
How      : @ApplyRotaryEmb.register_oot subclass overriding ``forward_cuda`` to
           delegate to ``forward_native``. No method body copy.
Scope    : rotary helper CustomOp instances on Iluvatar.
Remove   : when Iluvatar supports the community flash-attn rotary CUDA path.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb


@ApplyRotaryEmb.register_oot
class IluApplyRotaryEmb(ApplyRotaryEmb):
    def forward_cuda(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # [ILU override] ApplyRotaryEmb.forward_cuda - delta: skip flash-attn
        # rotary path and use the native implementation.
        return self.forward_native(x, cos, sin)
