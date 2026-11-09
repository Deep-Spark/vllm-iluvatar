# SPDX-License-Identifier: Apache-2.0
"""Register explicit construction failures for model kernels absent from ixaite."""

import torch.nn as nn

_REASONS = {
    "DeepseekV32ForCausalLM": "sparse MLA cache insertion and grouped MoE routing",
    "DeepseekV32MTPModel": "sparse MLA cache insertion and grouped MoE routing",
    "GlmMoeDsaForCausalLM": "FP8 DSA attention and quantized MoE",
    "IluGlmMoeDsaFp8MTPModel": "FP8 DSA attention and quantized MoE",
    "IluGlmMoeDsaForCausalLM": "DSA fused insertion and Marlin weight repack",
    "IluGlmMoeDsaMTPModel": "DSA fused insertion and Marlin weight repack",
    "HYV4ForCausalLM": "FP8 attention, iHC and grouped MoE routing",
    "HYV4MTPModel": "FP8 attention, iHC and grouped MoE routing",
    "Qwen4ExpForCausalLM": "QSA attention, PLE and hyperconnection",
    "Qwen4ExpForConditionalGeneration": "QSA attention, PLE and hyperconnection",
    "Qwen4ExpMTP": "QSA attention, PLE and hyperconnection",
}


# Registry inspection needs the generation interface before construction can reject.
class UnsupportedIxaiteModel(nn.Module):
    # Fail before creating any parameters or importing a vendor model implementation.
    def __init__(self, *, vllm_config, prefix=""):
        architectures = vllm_config.model_config.hf_config.architectures
        architecture = next(
            (name for name in architectures if name in _REASONS), architectures[0]
        )
        raise RuntimeError(
            f"ixaite model {architecture} requires unavailable {_REASONS.get(architecture, 'model kernels')}"
        )

    # These methods only describe the text-generation interface to ModelRegistry.
    def embed_input_ids(self, input_ids):
        raise RuntimeError("unsupported ixaite model cannot be constructed")

    # No execution path exists after the constructor failure.
    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        raise RuntimeError("unsupported ixaite model cannot be constructed")

    # No logits path exists after the constructor failure.
    def compute_logits(self, hidden_states):
        raise RuntimeError("unsupported ixaite model cannot be constructed")


# Use the official registry so unsupported architectures cannot resolve baseline classes.
def register_model() -> None:
    from vllm import ModelRegistry

    for architecture in _REASONS:
        ModelRegistry.register_model(architecture, f"{__name__}:UnsupportedIxaiteModel")
