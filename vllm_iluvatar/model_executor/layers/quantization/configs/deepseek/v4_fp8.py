# SPDX-License-Identifier: Apache-2.0
"""Iluvatar quantization config for the raw DeepSeek-V4 FP8 checkpoint.

Corresponds to: vllm/models/deepseek_v4/quant_config.py

Target   : vllm.models.deepseek_v4.quant_config.DeepseekV4FP8Config
           (registered under the name ``deepseek_v4_fp8``)
Baseline : v0.31.0 (db9527a468)
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
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
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
        if isinstance(layer, LinearBase):
            method = self._int8_linear_method(layer, prefix)
            if method is not None:
                return method
        if isinstance(layer, RoutedExperts) and self.expert_dtype == "fp4":
            skipped = is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            )
            if not skipped and self.moe_quant_algo != "NVFP4":
                from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.mxfp4_moe import (  # noqa: E501
                    IluMxfp4MoEMethod,
                )
                from vllm_iluvatar.model_executor.layers.quantization.ilu_runtime.config import (
                    resolve_ilu_runtime_method,
                )

                original = IluMxfp4MoEMethod(layer.moe_config)
                runtime = resolve_ilu_runtime_method(
                    base_quant_config=self,
                    original_method=original,
                    original_disposition="quantized",
                    layer=layer,
                    prefix=prefix,
                )
                from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.mxfp4_moe import (
                    select_mxfp4_method,
                )

                return select_mxfp4_method(layer, original, runtime)
        return super().get_quant_method(layer, prefix)

    def _int8_linear_method(self, layer: torch.nn.Module, prefix: str):
        """ivcore11 W8A8 INT8 dense execution from the block-FP8 source, or ``None``.

        Only ivcore11 is claimed. Every other architecture gets ``None`` and
        stays on ``super()`` -- community ``Fp8LinearMethod`` plus the
        ``IluFp8BlockScaledMMKernel`` prepended by
        ``patch_fp8_block_scaled_kernel``, which is the ivcore30 FP8 execution
        path. ``fp8_linear_backend()`` is still consulted so an explicit
        ``ilu_compute.linear.fp8_backend`` is validated rather than silently
        ignored; on ivcore11 it can only resolve to ``w8a8_int8``.

        There is no ``ilu_quant`` overlay here: that entry point is ivcore30
        only (``require_ivcore30``), so on ivcore11 the checkpoint dispatch is
        the whole story.

        V4 and V4.1 share the source format (block-FP8 weight + UE8M0
        ``weight_scale_inv``) and therefore the loaders, only the block shape
        differs: ``[128, 128]`` here against ``[32, 32]`` on V4.1.
        """
        from vllm_iluvatar.config.ilu_compute import fp8_linear_backend
        from vllm_iluvatar.ilu_util.arch import cuda_arch

        if cuda_arch() != "ivcore11" or fp8_linear_backend() != "w8a8_int8":
            return None
        if is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return UnquantizedLinearMethod()
        if prefix.endswith(".wo_a"):
            # wo_a is block-diagonal over o_groups and attention contracts it
            # per group, never as a flat linear, so it needs the grouped
            # loader regardless of TP. (The class name says V4.1; the method
            # is the shared block-FP8 source adapter for both.)
            from vllm_iluvatar.model_executor.layers.quantization.backends.grouped_bmm.deepseek_v4_wo_a import (  # noqa: E501
                IluDeepseekV41Fp8GroupedInt8Method,
            )

            return IluDeepseekV41Fp8GroupedInt8Method(self)
        from vllm_iluvatar.model_executor.layers.quantization.backends.fp8.linear import (  # noqa: E501
            IluFp8LinearMethod,
        )

        return IluFp8LinearMethod(self)
