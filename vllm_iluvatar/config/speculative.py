# SPDX-License-Identifier: Apache-2.0
"""Iluvatar ``SpeculativeConfig`` draft-arch routing overrides.

Corresponds to: vllm/config/speculative.py ::
``SpeculativeConfig.hf_config_override`` / ``SpeculativeConfig.__post_init__``

Some ilu models ship dedicated, self-contained MTP draft archs
(``IluGlmMoeDsaMTPModel`` / ``DeepseekV32MTPModel`` / ``HYV4MTPModel``).
``make_hf_config_override`` re-points the draft architecture at them: the
community/public method maps ``glm_moe_dsa`` -> ``DeepSeekMTPModel`` and
``deepseek_v4`` -> ``DeepSeekV4MTPModel``, and the wrapper remaps that result
keyed on the target model's original architecture, so the community path stays
untouched for non-ilu models.

``make_post_init`` hooks the later of the two community entry points -- it runs
after ``ModelConfig`` construction, i.e. after ``hf_config_override`` -- for the
things that can only be decided there: the step5 MTP quantization reset, the
MiniMax DSpark config normalization, and the DSpark draft-arch log line. DSpark
needs no arch remap: every ilu DSpark draft is registered under the community
``DSparkDraftModel`` name by its own model module.

``create_draft_parallel_config`` keeps the draft on a single PP stage: ilu MTP
drafts are one layer and do not implement ``SupportsPP``. Community
(v0.31.0) already hardcodes ``pipeline_parallel_size=1`` in that
function, so no ilu override is needed for it.

Baseline : v0.31.0 (db9527a468)
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from transformers import PretrainedConfig


# Target causal-LM arch -> ilu MTP draft arch.
#
# ``GlmMoeDsaForCausalLM`` is the *community* arch name, declared by the official
# ZhipuAI FP8 checkpoint (the W4A8 fork renames itself to ``Ilu...``). Community
# routes it to the generic ``DeepSeekMTPModel``, which real DeepSeek checkpoints
# share, so the FP8 GLM draft has to be re-pointed here rather than by
# overriding that name in the registry.
_ILU_MTP_ARCH_MAP = {
    "DeepseekV32ForCausalLM": "DeepseekV32MTPModel",
    "IluGlmMoeDsaForCausalLM": "IluGlmMoeDsaMTPModel",
    "GlmMoeDsaForCausalLM": "IluGlmMoeDsaFp8MTPModel",
    "HYV4ForCausalLM": "HYV4MTPModel",
    "HYV4VLForConditionalGeneration": "HYV4MTPModel",
}


def make_post_init(orig: Callable[..., None]) -> Callable[..., None]:
    """Apply the ilu fixups that need the draft ``ModelConfig``.

    Community builds it inside ``__post_init__``, after ``hf_config_override``
    has already run, so none of this can happen there: the step5 MTP
    quantization reset, the MiniMax DSpark config normalization, and reporting
    the architecture the DSpark draft actually resolved to.
    """

    @functools.wraps(orig)
    def __post_init__(self) -> None:
        orig(self)
        if (
            self.method == "mtp"
            and self.draft_model_config is not None
            and getattr(self.draft_model_config.hf_config, "model_type", None)
            == "step5_mtp"
        ):
            self.quantization = None
            self.draft_model_config.quantization = None
        if self.method != "dspark" or self.draft_model_config is None:
            return
        if getattr(self.draft_model_config.hf_config, "_ilu_minimax_dspark", False):
            from vllm_iluvatar.config.minimax_dspark import normalize_dspark_config

            self.draft_model_config.hf_config = normalize_dspark_config(
                self.draft_model_config.hf_config
            )
            self.update_arch_()
            self.draft_parallel_config.enable_expert_parallel = False
            logger.info(
                "MiniMax DSpark draft: five dense layers, target auxiliary states, shared vocabulary"
            )
            return

        # The draft ModelConfig logs "Resolved architecture" while the MTP
        # remap is still in effect, so that line names the wrong arch for
        # DSpark. Report what the draft actually ends up as.
        logger.info(
            "DSpark draft architecture: %s",
            list(self.draft_model_config.architectures),
        )

    return __post_init__


def make_hf_config_override(
    orig: Callable[[PretrainedConfig], PretrainedConfig],
) -> Callable[[PretrainedConfig], PretrainedConfig]:
    """Wrap the community ``hf_config_override`` staticmethod.

    Captures the target model's original architecture, delegates to the public
    routing, then swaps the resulting draft arch to the ilu MTP model when the
    target is an ilu W4A8 fork. Non-ilu targets are returned unchanged. DSpark
    is not handled here: community assigns that arch later, in
    ``__post_init__`` (see ``make_post_init``).
    """

    @functools.wraps(orig)
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        architectures = getattr(hf_config, "architectures", None) or []
        initial_architecture = architectures[0] if architectures else None

        if initial_architecture == "DSparkMiniMaxDraftModel":
            from vllm_iluvatar.config.minimax_dspark import normalize_dspark_config

            return normalize_dspark_config(hf_config)

        if getattr(hf_config, "model_type", None) in (
            "step5",
            "step5_vl",
            "step5_mtp",
        ) or initial_architecture in (
            "Step5ForCausalLM",
            "Step5ForConditionalGeneration",
            "Step5MTP",
        ):
            from vllm_iluvatar.models.step5.config import Step5MTPConfig

            return Step5MTPConfig.from_target(hf_config)
        hf_config = orig(hf_config)

        ilu_mtp_arch = _ILU_MTP_ARCH_MAP.get(initial_architecture)
        if ilu_mtp_arch is not None:
            hf_config.update({"architectures": [ilu_mtp_arch]})
            if initial_architecture in (
                "HYV4ForCausalLM",
                "HYV4VLForConditionalGeneration",
            ):
                # The baseline's MTPModelTypes predates HY V4. Use the generic
                # DeepSeek tuple-MTP model_type only on the draft config so
                # SpeculativeConfig accepts method="mtp"; the architecture
                # remains HYV4MTPModel and retains all HY4 config fields.
                hf_config.update({"model_type": "deepseek_mtp"})
        return hf_config

    return hf_config_override
