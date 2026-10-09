# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar ``SpeculativeConfig`` draft-arch overrides.

Corresponds to: vllm/config/speculative.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementations live in ``vllm_iluvatar.config.speculative``; this
module only performs register-time wiring.

Target   : vllm.config.speculative.SpeculativeConfig.hf_config_override
           vllm.config.speculative.SpeculativeConfig.__post_init__
Baseline : v0.31.0 (db9527a468)
Why      : re-point the draft architecture to the ilu W4A8 self-contained draft
           models (DeepseekV32MTPModel / IluGlmMoeDsaMTPModel /
           DeepSeekV4MTPModel for MTP, IluDSparkDraftModel for DSpark) for the
           ilu target archs, without modifying community source. The two hooks
           differ because community assigns the MTP arch in
           ``hf_config_override`` but the DSpark arch later, in
           ``__post_init__``. ``create_draft_parallel_config`` and
           ``EngineArgs.create_speculative_config`` are not overridden: community
           already hardcodes draft ``pipeline_parallel_size=1`` and no longer
           accepts a ``draft_pipeline_parallel_size`` field.
How      : wrap the staticmethod with ``make_hf_config_override`` and setattr it
           back onto the class as a staticmethod; wrap ``__post_init__`` with
           ``make_post_init`` so the DSpark remap runs after the community body.
Scope    : SpeculativeConfig MTP/DSpark draft-arch routing on Iluvatar.
Remove   : when upstream exposes a platform hook for draft arch selection.

Timing   : ``load_general_plugins()`` runs inside ``create_engine_config``,
           before any ``SpeculativeConfig`` is instantiated, so both wrappers
           are in place before the wrapped functions are called.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the Iluvatar SpeculativeConfig wrappers."""
    p.register_once(
        __name__,
        _apply,
        detail="SpeculativeConfig ilu MTP/DSpark draft-arch routing",
    )


def _apply() -> None:
    from vllm.config.speculative import SpeculativeConfig

    from vllm_iluvatar.config.speculative import (
        make_hf_config_override,
        make_post_init,
    )

    orig_hf = SpeculativeConfig.hf_config_override
    orig_post_init = SpeculativeConfig.__post_init__

    # [ILU override] hf_config_override — delta: route iluvatar target archs
    # to their dedicated MTP draft models after the community routing runs.
    wrapped_hf = make_hf_config_override(orig_hf)
    # [ILU override] __post_init__ — delta: re-point the DSpark draft arch at
    # IluDSparkDraftModel after community assigns DSparkDraftModel.
    wrapped_post_init = make_post_init(orig_post_init)

    p.check_signatures(
        SpeculativeConfig,
        SimpleNamespace(
            hf_config_override=wrapped_hf,
            __post_init__=wrapped_post_init,
        ),
        ["hf_config_override", "__post_init__"],
        strict=True,
    )

    SpeculativeConfig.hf_config_override = staticmethod(wrapped_hf)
    SpeculativeConfig.__post_init__ = wrapped_post_init
