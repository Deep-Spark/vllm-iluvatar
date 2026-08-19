# SPDX-License-Identifier: Apache-2.0
"""Iluvatar ``SpeculativeConfig.hf_config_override`` override.

Corresponds to: vllm/config/speculative.py ::
``SpeculativeConfig.hf_config_override``

The community/public method maps ``glm_moe_dsa`` -> ``DeepSeekMTPModel`` and
``deepseek_v4`` -> ``DeepSeekV4MTPModel``. On Iluvatar the W4A8 forks of those
models ship dedicated, self-contained MTP draft archs
(``IluGlmMoeDsaMTPModel`` / ``IluDeepSeekV4MTPModel`` /
``DeepseekV32MTPModel``); this wrapper re-points the draft architecture to
them *after* the public routing runs, keyed on the target model's original
architecture, so the community MTP path stays untouched for non-ilu models.

Baseline : main_6e448d0
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig

# Target causal-LM arch -> ilu MTP draft arch.
_ILU_MTP_ARCH_MAP = {
    "DeepseekV32ForCausalLM": "DeepseekV32MTPModel",
    "IluGlmMoeDsaForCausalLM": "IluGlmMoeDsaMTPModel",
    "IluDeepseekV4ForCausalLM": "IluDeepSeekV4MTPModel",
}


def make_hf_config_override(
    orig: Callable[[PretrainedConfig], PretrainedConfig],
) -> Callable[[PretrainedConfig], PretrainedConfig]:
    """Wrap the community ``hf_config_override`` staticmethod.

    Captures the target model's original architecture, delegates to the public
    routing, then swaps the resulting draft arch to the ilu MTP model when the
    target is an ilu W4A8 fork. Non-ilu targets are returned unchanged.
    """

    @functools.wraps(orig)
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        architectures = getattr(hf_config, "architectures", None) or []
        initial_architecture = architectures[0] if architectures else None

        hf_config = orig(hf_config)

        ilu_mtp_arch = _ILU_MTP_ARCH_MAP.get(initial_architecture)
        if ilu_mtp_arch is not None:
            hf_config.update({"architectures": [ilu_mtp_arch]})
        return hf_config

    return hf_config_override
