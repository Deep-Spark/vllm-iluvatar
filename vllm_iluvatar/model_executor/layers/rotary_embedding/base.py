# SPDX-License-Identifier: Apache-2.0
"""Iluvatar OOT overrides for base rotary embedding.

Corresponds to: vllm/model_executor/layers/rotary_embedding/base.py

Target   : RotaryEmbedding and rope subclasses inheriting its CUDA path
Baseline : main_4875b4456b5
Why      : Iluvatar's rotary custom op needs the actual ``rotary_dim``; community
           ``RotaryEmbedding.forward_cuda`` does not pass it.
How      : @<RopeClass>.register_oot subclasses overriding only ``forward_cuda``
           to pass ``self.rotary_dim`` to ``vllm._custom_ops``.
Scope    : base RotaryEmbedding-style CUDA path on Iluvatar.
Remove   : when community forwards ``rotary_dim`` or Iluvatar no longer needs it.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.rotary_embedding.dynamic_ntk_alpha_rope import (
    DynamicNTKAlphaRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.dynamic_ntk_scaling_rope import (
    DynamicNTKScalingRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.fope import FourierRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.gemma4_rope import (
    Gemma4RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.linear_scaling_rope import (
    LinearScalingRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.llama3_rope import (
    Llama3RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.ntk_scaling_rope import (
    NTKScalingRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.telechat3_scaling_rope import (
    TeleChat3RoPEScaledRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
    YaRNScalingRotaryEmbedding,
)


def forward_cuda(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from vllm import _custom_ops as ops

    cos_sin_cache = self._match_cos_sin_cache_dtype(query)

    # [ILU override] RotaryEmbedding.forward_cuda - delta: pass rotary_dim
    # required by the Iluvatar rotary kernel.
    ops.rotary_embedding(
        positions,
        query,
        key,
        self.head_size,
        cos_sin_cache,
        self.is_neox_style,
        self.rotary_dim,
    )
    return query, key


@RotaryEmbedding.register_oot
class IluRotaryEmbedding(RotaryEmbedding):
    forward_cuda = forward_cuda


@Llama3RotaryEmbedding.register_oot
class IluLlama3RotaryEmbedding(Llama3RotaryEmbedding):
    forward_cuda = forward_cuda


@LinearScalingRotaryEmbedding.register_oot
class IluLinearScalingRotaryEmbedding(LinearScalingRotaryEmbedding):
    forward_cuda = forward_cuda


@NTKScalingRotaryEmbedding.register_oot
class IluNTKScalingRotaryEmbedding(NTKScalingRotaryEmbedding):
    forward_cuda = forward_cuda


@DynamicNTKAlphaRotaryEmbedding.register_oot
class IluDynamicNTKAlphaRotaryEmbedding(DynamicNTKAlphaRotaryEmbedding):
    forward_cuda = forward_cuda


@DynamicNTKScalingRotaryEmbedding.register_oot
class IluDynamicNTKScalingRotaryEmbedding(DynamicNTKScalingRotaryEmbedding):
    forward_cuda = forward_cuda


@FourierRotaryEmbedding.register_oot
class IluFourierRotaryEmbedding(FourierRotaryEmbedding):
    forward_cuda = forward_cuda


@YaRNScalingRotaryEmbedding.register_oot
class IluYaRNScalingRotaryEmbedding(YaRNScalingRotaryEmbedding):
    forward_cuda = forward_cuda


@TeleChat3RoPEScaledRotaryEmbedding.register_oot
class IluTeleChat3RoPEScaledRotaryEmbedding(TeleChat3RoPEScaledRotaryEmbedding):
    forward_cuda = forward_cuda


@Gemma4RotaryEmbedding.register_oot
class IluGemma4RotaryEmbedding(Gemma4RotaryEmbedding):
    forward_cuda = forward_cuda
