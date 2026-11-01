# SPDX-License-Identifier: Apache-2.0
"""Shared model predicates for Iluvatar DeepSeek-V4.1 patches."""

from __future__ import annotations

from typing import Any

ILU_DSV41_ARCHS = frozenset(
    {
        "DeepseekV41ForCausalLM",
        "IluDeepseekV41ForCausalLM",
        "IluDeepseekV41VLForCausalLM",
    }
)


def is_ilu_dsv41_model_config(model_config: Any | None) -> bool:
    if model_config is None:
        return False
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if getattr(hf_config, "model_type", None) in ("deepseek_v41", "deepseek_v41_text") or getattr(
        hf_text_config, "model_type", None
    ) in ("deepseek_v41", "deepseek_v41_text"):
        return True
    try:
        architectures = tuple(
            getattr(model_config, "architectures", None)
            or getattr(hf_config, "architectures", None)
            or ()
        )
    except Exception:
        architectures = ()
    return any(arch in ILU_DSV41_ARCHS for arch in architectures)


def is_ilu_dsv41_config(vllm_config: Any | None) -> bool:
    return is_ilu_dsv41_model_config(getattr(vllm_config, "model_config", None))
