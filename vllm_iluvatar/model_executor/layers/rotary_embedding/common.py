# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT override for rotary embedding helper op.

Corresponds to: vllm/model_executor/layers/rotary_embedding/common.py

Target   : vllm.model_executor.layers.rotary_embedding.common.ApplyRotaryEmb
Baseline : main_6e448d0
Why      : the community ``forward_cuda`` imports
           ``from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb``
           (a flash-attn kernel absent on Iluvatar/corex). Rather than fake that
           whole upstream sub-package via an import shim, keep this single-class
           OOT override (the sanctioned CustomOp extension point) and route it to
           the plugin's own native ``torch.ops._C.apply_rotary_emb`` kernel.
How      : @ApplyRotaryEmb.register_oot subclass overriding ``forward_cuda`` to
           reuse the community ``_pre_process`` / ``_post_process`` and call the
           plugin ``_C`` kernel in between (no method body copy of the math).
Scope    : rotary helper CustomOp instances on Iluvatar.
Remove   : when the community flash-attn rotary CUDA path is available on
           Iluvatar (then this override and the ``_C`` op can both be dropped).
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
        # [ILU override] ApplyRotaryEmb.forward_cuda - delta: replace the
        # flash-attn rotary import with the plugin's native _C kernel. Reuse the
        # community pre/post processing (3D->4D reshape and enable_fp32_compute
        # handling) so only the kernel call differs from upstream.
        x, cos, sin, origin_shape, origin_dtype = self._pre_process(x, cos, sin)
        interleaved = not self.is_neox_style
        output = torch.ops._C.apply_rotary_emb(x, cos, sin, interleaved)
        return self._post_process(output, origin_shape, origin_dtype)
