# SPDX-License-Identifier: Apache-2.0
"""Iluvatar ``SpeculativeConfig`` draft-arch routing and draft-PP overrides.

Corresponds to: vllm/config/speculative.py ::
``SpeculativeConfig.hf_config_override`` / ``SpeculativeConfig.__post_init__`` /
``SpeculativeConfig.create_draft_parallel_config``

On Iluvatar the W4A8 forks of the speculative draft models ship dedicated,
self-contained draft archs (``IluGlmMoeDsaMTPModel`` / ``IluDeepSeekV4MTPModel`` /
``DeepseekV32MTPModel`` / ``IluDSparkDraftModel``). Two wrappers re-point the
draft architecture at them, each hooking the community function that actually
assigns the arch, so the community path stays untouched for non-ilu models:

- MTP (``make_hf_config_override``): the community/public method maps
  ``glm_moe_dsa`` -> ``DeepSeekMTPModel`` and ``deepseek_v4`` ->
  ``DeepSeekV4MTPModel``; the wrapper remaps the result keyed on the target
  model's original architecture.
- DSpark (``make_post_init``): community assigns ``DSparkDraftModel`` in
  ``__post_init__``, after ``ModelConfig`` construction and therefore after
  ``hf_config_override`` has run, so that remap has to happen there instead.

``create_draft_parallel_config`` keeps the draft on a single PP stage: ilu MTP
drafts are one layer and do not implement ``SupportsPP``.

Baseline : main_6e448d0
"""

from __future__ import annotations

import copy
import functools
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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
    "IluDeepseekV4ForCausalLM": "IluDeepSeekV4MTPModel",
}

# Community draft arch -> ilu draft arch, keyed by the architecture community
# assigns in SpeculativeConfig.__post_init__. DSpark reuses the DeepSeek-V4
# stack but is a separate draft arch from MTP, so it must never be routed
# through _ILU_MTP_ARCH_MAP.
_ILU_DSPARK_ARCH_MAP = {
    "DSparkDraftModel": "IluDSparkDraftModel",
}

# Target architectures whose DSpark draft should use the ilu implementation.
_ILU_DSPARK_TARGET_ARCHS = (
    "IluDeepseekV4ForCausalLM",
    "DeepseekV4ForCausalLM",
)


def make_post_init(orig: Callable[..., None]) -> Callable[..., None]:
    """Re-point the DSpark draft architecture at the ilu implementation.

    Community assigns ``["DSparkDraftModel"]`` inside ``__post_init__``, after
    ``ModelConfig`` construction and therefore after ``hf_config_override`` has
    already run, so the remap has to happen here to survive.
    """

    @functools.wraps(orig)
    def __post_init__(self) -> None:
        orig(self)
        if self.method != "dspark" or self.draft_model_config is None:
            return
        target_archs = (
            []
            if self.target_model_config is None
            else list(self.target_model_config.architectures)
        )
        if any(arch in _ILU_DSPARK_TARGET_ARCHS for arch in target_archs):
            hf_config = self.draft_model_config.hf_config
            current = list(getattr(hf_config, "architectures", None) or [])
            remapped = [_ILU_DSPARK_ARCH_MAP.get(arch, arch) for arch in current]
            if remapped != current:
                hf_config.architectures = remapped
                self.update_arch_()

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

        hf_config = orig(hf_config)

        ilu_mtp_arch = _ILU_MTP_ARCH_MAP.get(initial_architecture)
        if ilu_mtp_arch is not None:
            hf_config.update({"architectures": [ilu_mtp_arch]})
        return hf_config

    return hf_config_override


def make_create_draft_parallel_config(orig: Callable[..., Any]) -> Callable[..., Any]:
    """Force draft PP=1 when the target uses PP>1.

    Newer vLLM takes an explicit draft PP (already resolved to the target PP
    when unset). Older builds copy target PP. Either way the ilu MTP draft
    cannot be pipeline-split, so override to a single stage.
    """
    takes_draft_pp = (
        "speculative_draft_pipeline_parallel_size" in inspect.signature(orig).parameters
    )

    @functools.wraps(orig)
    def create_draft_parallel_config(
        target_parallel_config: Any,
        speculative_draft_tensor_parallel_size: int,
        *_args: Any,
    ) -> Any:
        target_pp = target_parallel_config.pipeline_parallel_size
        draft_pp = 1 if target_pp > 1 else target_pp
        if takes_draft_pp:
            return orig(
                target_parallel_config,
                speculative_draft_tensor_parallel_size,
                draft_pp,
            )
        # Old signature copies PP from the target. Do not dataclasses.replace
        # the returned config: pydantic Field(init=False) members (world_size)
        # leak back as init kwargs.
        if target_pp != draft_pp:
            target_parallel_config = copy.copy(target_parallel_config)
            target_parallel_config.pipeline_parallel_size = draft_pp
        return orig(target_parallel_config, speculative_draft_tensor_parallel_size)

    return create_draft_parallel_config


def make_create_speculative_config(orig: Callable[..., Any]) -> Callable[..., Any]:
    """Drop ``draft_pipeline_parallel_size`` on older SpeculativeConfig.

    Extra keys are forbidden; the draft PP override above does not read this
    field, so stripping it is always safe.
    """

    @functools.wraps(orig)
    def create_speculative_config(self: Any, *args: Any, **kwargs: Any) -> Any:
        cfg = getattr(self, "speculative_config", None)
        if isinstance(cfg, dict):
            cfg.pop("draft_pipeline_parallel_size", None)
        return orig(self, *args, **kwargs)

    return create_speculative_config
