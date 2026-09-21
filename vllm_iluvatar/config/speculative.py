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

if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = init_logger("vllm." + __name__)

# Target causal-LM arch -> ilu MTP draft arch.
_ILU_MTP_ARCH_MAP = {
    "DeepseekV32ForCausalLM": "DeepseekV32MTPModel",
    "IluGlmMoeDsaForCausalLM": "IluGlmMoeDsaMTPModel",
    "IluDeepseekV4ForCausalLM": "IluDeepSeekV4MTPModel",
}

# The draft arch community assigns in SpeculativeConfig.__post_init__ for every
# DeepSeek-family DSpark. DSpark reuses the DeepSeek-V4 stack but is a separate
# draft arch from MTP, so it must never be routed through _ILU_MTP_ARCH_MAP.
_ILU_DSPARK_COMMUNITY_ARCHS = ("DSparkDraftModel",)

# Target arch -> ilu DSpark draft arch. V4 and V4.1 share the DSpark scheme but
# not the decoder layer (V4.1 defers its FFN hc_post and has no global MHC head),
# so the draft class depends on which target is loaded.
_ILU_DSPARK_DRAFT_BY_TARGET = {
    "IluDeepseekV4ForCausalLM": "IluDSparkDraftModel",
    "DeepseekV4ForCausalLM": "IluDSparkDraftModel",
    "DeepseekV41ForCausalLM": "IluDSparkV41DraftModel",
}

# Targets that read ``dspark_target_layer_ids`` as the layers whose *input* feeds
# the draft. Community's aux-layer resolution shifts those ids by one, which is
# the V4 convention; naming them explicitly under the key it checks first
# (``eagle_aux_hidden_state_layer_ids``) bypasses the shift.
_ILU_DSPARK_UNSHIFTED_AUX_TARGETS = ("DeepseekV41ForCausalLM",)


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
        draft_arch = next(
            (
                _ILU_DSPARK_DRAFT_BY_TARGET[arch]
                for arch in target_archs
                if arch in _ILU_DSPARK_DRAFT_BY_TARGET
            ),
            None,
        )
        if draft_arch is not None:
            hf_config = self.draft_model_config.hf_config
            current = list(getattr(hf_config, "architectures", None) or [])
            remapped = [
                draft_arch if arch in _ILU_DSPARK_COMMUNITY_ARCHS else arch
                for arch in current
            ]
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


def _fixup_v41_dspark_config(hf_config: PretrainedConfig) -> None:
    """Patch the two DSpark fields V4.1 reads differently from V4.

    Both have to land here rather than in ``__post_init__``: community reads
    ``n_predict`` while resolving ``num_speculative_tokens`` inside
    ``__post_init__`` itself, so a later write is too late.

    ``n_predict``
        V4.1 drafts a block of ``dspark_block_size`` (5) tokens per step, not
        ``num_nextn_predict_layers`` (3). Nothing sets ``n_predict`` for
        ``model_type == "deepseek_v41"`` -- none of community's branches match it
        -- so ``num_speculative_tokens`` would be left for the user to supply and
        then rejected by the ``>= dspark_block_size`` check. V4.1 has no MTP
        draft registered, so claiming this field for DSpark is unambiguous.

    ``eagle_aux_hidden_state_layer_ids``
        ``dspark_target_layer_ids`` names the layers whose *input* feeds the
        draft, but community's aux resolution shifts DSpark ids by one (the V4
        convention, where they name the layers to capture after). It checks
        ``eagle_aux_hidden_state_layer_ids`` first and returns it unshifted, so
        writing the ids there is what pins the capture points.
    """
    block_size = getattr(hf_config, "dspark_block_size", None)
    if block_size is not None and getattr(hf_config, "n_predict", None) is None:
        hf_config.update({"n_predict": block_size})

    layer_ids = getattr(hf_config, "dspark_target_layer_ids", None)
    if layer_ids and not getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None):
        hf_config.update({"eagle_aux_hidden_state_layer_ids": list(layer_ids)})


def make_hf_config_override(
    orig: Callable[[PretrainedConfig], PretrainedConfig],
) -> Callable[[PretrainedConfig], PretrainedConfig]:
    """Wrap the community ``hf_config_override`` staticmethod.

    Captures the target model's original architecture, delegates to the public
    routing, then swaps the resulting draft arch to the ilu MTP model when the
    target is an ilu W4A8 fork. Non-ilu targets are returned unchanged. DSpark's
    *arch* is not handled here -- community assigns that later, in
    ``__post_init__`` (see ``make_post_init``) -- but the V4.1 DSpark config
    fields are, because ``__post_init__`` already depends on them.
    """

    @functools.wraps(orig)
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        architectures = getattr(hf_config, "architectures", None) or []
        initial_architecture = architectures[0] if architectures else None

        if initial_architecture in _ILU_DSPARK_UNSHIFTED_AUX_TARGETS:
            _fixup_v41_dspark_config(hf_config)

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
