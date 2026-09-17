# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 OOT architecture registration.

Quant implementations stay in ``w4a8/`` and ``mxfp8/``. This
module is the package-level ``register_model()`` hook so
``vllm_iluvatar.models.minimax_m3`` can be listed in ``_OOT_MODEL_PACKAGES``
like the other OOT models.
"""


def register_model() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "IluMiniMaxM3SparseForCausalLM",
        "vllm_iluvatar.models.minimax_m3.w4a8.model:IluMiniMaxM3SparseForCausalLM",
    )
    ModelRegistry.register_model(
        "IluMiniMaxM3SparseForConditionalGeneration",
        "vllm_iluvatar.models.minimax_m3.w4a8.model:"
        "IluMiniMaxM3SparseForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForCausalLM",
        "vllm_iluvatar.models.minimax_m3.mxfp8.model:MiniMaxM3SparseForCausalLM",
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForConditionalGeneration",
        "vllm_iluvatar.models.minimax_m3.mxfp8.model:"
        "MiniMaxM3SparseForConditionalGeneration",
    )
