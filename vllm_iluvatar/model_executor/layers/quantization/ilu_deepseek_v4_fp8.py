# SPDX-License-Identifier: Apache-2.0
"""Iluvatar quantization config for the raw DeepSeek-V4 FP8 checkpoint.

Corresponds to: vllm/models/deepseek_v4/quant_config.py

Target   : vllm.models.deepseek_v4.quant_config.DeepseekV4FP8Config
           (registered under the name ``deepseek_v4_fp8``)
Baseline : main_9c12a81ac
Why      : the community config routes MXFP4 routed experts to
           ``Mxfp4MoEMethod``, whose every backend is NVIDIA/ROCm-only. Iluvatar
           needs the same checkpoint parsing with an Iluvatar MoE method.
How      : ``@register_quantization_config("deepseek_v4_fp8")`` -- registering
           the community name on purpose, see below -- on a subclass that
           overrides exactly one branch of ``get_quant_method``: routed experts
           with ``expert_dtype == "fp4"`` get ``IluMxfp4MoEMethod``. Everything
           else (linear layers, attention, the ``expert_dtype == "fp8"``
           fall-through) delegates to ``super()``.
Scope    : DeepSeek-V4 FP8 checkpoints on Iluvatar.
Remove   : when community exposes a vendor MXFP4 MoE backend hook, at which
           point the community config can be used directly.

Why register the community name instead of a new ``ilu_*`` one
-------------------------------------------------------------
The DeepSeek-V4 FP8 weights are the official release: their
``quantization_config.quant_method`` is ``"fp8"``, and community
``ModelConfig._verify_quantization`` resolves that to ``"deepseek_v4_fp8"``
through the inherited ``DeepseekV4FP8Config.override_quantization_method``
(``quant_config.py:120-133``). This is therefore the "community format, Iluvatar
implementation" case, exactly like ``IluAutoGPTQConfig`` covering ``gptq`` and
``IluAutoAWQConfig`` covering ``awq`` -- both of which also keep the community
``get_name()``. New ``ilu_*`` names are reserved for Iluvatar-produced
checkpoints (``ilu_deepseek_v4_w4a8``).

``register_quantization_config`` permits shadowing a built-in name; the collision
only emits one debug line (``quantization/__init__.py:85-94``) and
``get_quantization_config`` resolves through
``_CUSTOMIZED_METHOD_TO_QUANT_CONFIG`` last, so our class wins. Keeping the name
also means every code path that branches on ``quant_config.get_name()`` behaves
identically to a community run.

Named after the **checkpoint**, not the format, on purpose: ``expert_dtype`` /
``moe_quant_algo`` / ``is_scale_e8m0`` are DeepSeek-V4-specific fields, which is
why community itself keeps this class under ``vllm/models/deepseek_v4/`` rather
than in ``layers/quantization/``. The format-level pieces live in
``ilu_mxfp4.py`` and ``fused_moe/ilu_mxfp4_fused_moe.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

if TYPE_CHECKING:
    import torch


@register_quantization_config("deepseek_v4_fp8")
class IluDeepseekV4FP8Config(DeepseekV4FP8Config):
    """DeepSeek-V4 FP8 config whose MXFP4 experts land on Iluvatar kernels.

    Inherits ``__init__`` / ``from_config`` / ``get_name`` /
    ``override_quantization_method`` / the ``expert_dtype`` &
    ``is_scale_e8m0`` properties unchanged, so checkpoint detection and the FP8
    linear scale dtype stay community behaviour.
    """

    # [ILU override] DeepseekV4FP8Config.get_quant_method — delta: the routed
    # MXFP4 branch returns IluMxfp4MoEMethod instead of community
    # Mxfp4MoEMethod. The skip / NVFP4 / expert_dtype=="fp8" branches and all
    # non-MoE layers are left to super().
    #
    # The ``is_layer_skipped`` and NVFP4 checks are re-stated rather than
    # delegated because community reaches our branch only *after* them
    # (``quant_config.py:135-153``); calling super() first would already have
    # built a community ``Mxfp4MoEMethod``.
    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, RoutedExperts) and self.expert_dtype == "fp4":
            skipped = is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            )
            if not skipped and self.moe_quant_algo != "NVFP4":
                from vllm_iluvatar.model_executor.layers.quantization.ilu_mxfp4 import (  # noqa: E501
                    IluMxfp4MoEMethod,
                )

                return IluMxfp4MoEMethod(layer.moe_config)
        return super().get_quant_method(layer, prefix)
