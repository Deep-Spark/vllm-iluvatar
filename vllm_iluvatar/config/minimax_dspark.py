# SPDX-License-Identifier: Apache-2.0
"""Normalize the MiniMax DSpark checkpoint envelope before model construction."""

import copy

from transformers import PretrainedConfig


def normalize_dspark_config(config):
    """Flatten the checkpoint's VL envelope for the text-only draft runner."""
    if getattr(config, "_ilu_minimax_dspark", False):
        config.architectures = ["DSparkMiniMaxDraftModel"]
        config.model_type = "minimax_m3_dspark"
        return config
    raw = config.to_dict()
    text = raw.pop("text_config", None) or {}
    raw.update(text)
    raw.pop("vision_config", None)
    raw.update(
        architectures=["DSparkMiniMaxDraftModel"],
        num_local_experts=0,
        n_shared_experts=0,
        sparse_attention_config={},
        use_aux_hidden_state=True,
        markov_rank=raw.get("dspark_markov_rank", 256),
        mask_token_id=raw.get("dspark_noise_token_id", 200058),
        swiglu_beta=raw.get("swiglu_beta", 1.0),
        is_neox_style=True,
        _ilu_minimax_dspark=True,
    )
    raw["rope_parameters"] = dict(
        rope_type="default",
        rope_theta=raw["rope_theta"],
        partial_rotary_factor=raw["partial_rotary_factor"],
    )
    # Tolerate drafts that carry no quantization_config at all, and extend
    # ``ignored_layers`` only for the ModelOpt/MXFP8 schema that field belongs
    # to: a compressed-tensors doc (e.g. the W8A8 DSpark recipe) uses
    # ``ignore`` instead and must not get a foreign key injected.
    quant = raw.get("quantization_config") or None
    if quant is None:
        raw.pop("quantization_config", None)
    else:
        quant = copy.deepcopy(quant)
        if "ignored_layers" in quant or quant.get("quant_method") == "mxfp8":
            quant["ignored_layers"] = list(quant.get("ignored_layers", [])) + [
                "model.fc",
                "markov_head",
                "confidence_head",
                "lm_head",
                "embed_tokens",
            ]
        raw["quantization_config"] = quant
    result = PretrainedConfig(**raw)
    result.model_type = "minimax_m3_dspark"
    return result
