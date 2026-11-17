# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MXFP8 config for NVIDIA ModelOpt-tagged checkpoints.

Corresponds to: vllm/model_executor/layers/quantization/modelopt.py

Target   : ``ModelOptMxFp8Config`` under ``quant_method="modelopt"`` +
           ``quantization.quant_algo="MXFP8"``, which
           ``ModelOptMxFp8Config.override_quantization_method`` resolves to the
           registry name **``modelopt_mxfp8``**.
Baseline : ``ilu_mxfp8.py`` registers only ``"mxfp8"`` (MiniMax-style tagging)
           and its docstring states that ``modelopt_mxfp8`` is deliberately not
           shadowed. Hy4-preview uses the ModelOpt tagging, so it lands on the
           community class.
Why      : two independent failures on the community class.
           1. Routed experts go to ``ModelOptMxFp8FusedMoE`` ->
              ``select_mxfp8_moe_backend``, whose auto priority reaches MARLIN
              because ``MarlinExperts`` only gates on "CUDA SM75+" and
              ``IluPlatform._enum`` is CUDA. Load then dies in
              ``prepare_mxfp8_moe_layer_for_marlin`` ->
              ``torch.ops._C.gptq_marlin_repack``, which this build lacks.
           2. ``lm_head`` is in Hy4's ``exclude_modules``, and the inherited
              ``get_quant_method`` hands a ``ParallelLMHead`` an
              ``UnquantizedLinearMethod``. HY V4 runs an fp32 head
              (``enable_lm_head_fp32``), and community ``LogitsProcessor``
              accepts only ``UnquantizedEmbeddingMethod`` for a head whose dtype
              differs from the model dtype, so it raises.
How      : subclass, swap ``FusedMoEMethodCls`` to ``IluMxFp8ToFp4OnlineMoE``
           (load-time MXFP8 -> ix30 FP4; the host-staging ``IluMxFp8FusedMoE``
           peak does not fit Hy4, see that module's docstring), and narrow the
           excluded-``ParallelLMHead`` case to ``UnquantizedEmbeddingMethod``.
           Both are one-line deltas; no method body is copied. Dense linears
           are untouched -- they go through
           the inherited ``ModelOptMxFp8LinearMethod``, which picks up
           ``IluMxfp8LinearKernel`` from the patched kernel list.
Scope    : ModelOpt MXFP8 checkpoints on Iluvatar corex30 (Hy4-preview).
Remove   : when community exposes a vendor MXFP8 MoE backend hook, and when
           ``LogitsProcessor`` accepts ``UnquantizedLinearMethod`` for an
           unquantized head (upstream #54160 does the latter).
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

from vllm_iluvatar.model_executor.layers.quantization.backends.fp4.mxfp8_moe import (  # noqa: E501
    IluMxFp8ToFp4OnlineMoE,
)


@register_quantization_config("modelopt_mxfp8")
class IluModelOptMxFp8Config(ModelOptMxFp8Config):
    """ModelOpt MXFP8 whose routed experts use the Iluvatar ix30 compute path.

    Sibling of ``IluMxFp8Config`` (``ilu_mxfp8.py``), which covers the
    MiniMax-style ``quant_method: "mxfp8"`` tagging. The two differ only in the
    registry name they claim; the expert method is shared.
    """

    def get_name(self) -> Any:
        return "modelopt_mxfp8"

    # [ILU override] ModelOptQuantConfigBase.get_quant_method — delta: an
    # excluded ``ParallelLMHead`` gets ``UnquantizedEmbeddingMethod`` instead of
    # ``UnquantizedLinearMethod``. Both are unquantized and both expose the same
    # ``apply``; the difference is only that community ``LogitsProcessor``
    # isinstance-checks for the embedding one before taking the fp32-accumulate
    # path that HY V4's ``head_dtype="float32"`` selects. Everything else
    # delegates.
    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Any:
        if isinstance(layer, ParallelLMHead) and self.is_layer_excluded(prefix):
            return UnquantizedEmbeddingMethod()
        return super().get_quant_method(layer, prefix)


IluModelOptMxFp8Config.FusedMoEMethodCls = IluMxFp8ToFp4OnlineMoE
