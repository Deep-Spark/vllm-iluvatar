# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar ``SpeculativeConfig`` draft-arch and draft-PP overrides.

Corresponds to: vllm/config/speculative.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementations live in ``vllm_iluvatar.config.speculative``; this
module only performs register-time wiring.

Target   : vllm.config.speculative.SpeculativeConfig.hf_config_override
           vllm.config.speculative.SpeculativeConfig.__post_init__
           vllm.config.speculative.SpeculativeConfig.create_draft_parallel_config
           vllm.engine.arg_utils.EngineArgs.create_speculative_config
Baseline : main_6e448d0
Why      : re-point the draft architecture to the ilu W4A8 self-contained draft
           models (DeepseekV32MTPModel / IluGlmMoeDsaMTPModel /
           DeepSeekV4MTPModel for MTP, IluDSparkDraftModel for DSpark) for the
           ilu target archs, without modifying community source. The two hooks
           differ because community assigns the MTP arch in
           ``hf_config_override`` but the DSpark arch later, in
           ``__post_init__``. Also keep draft PP at 1 because those MTP models
           do not implement SupportsPP.
How      : wrap the staticmethod with ``make_hf_config_override`` and setattr it
           back onto the class as a staticmethod; wrap ``__post_init__`` with
           ``make_post_init`` so the DSpark remap runs after the community body;
           wrap ``create_draft_parallel_config`` / ``create_speculative_config``
           for the draft PP=1 default.
Scope    : SpeculativeConfig MTP/DSpark draft-arch routing and draft PP on
           Iluvatar.
Remove   : when upstream exposes a platform hook for draft arch selection
           and draft PP independent of the target.

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
        detail="SpeculativeConfig ilu MTP/DSpark draft-arch routing and draft PP",
    )


def _apply() -> None:
    from vllm.config.speculative import SpeculativeConfig
    from vllm.engine.arg_utils import EngineArgs

    from vllm_iluvatar.config.speculative import (
        make_create_draft_parallel_config,
        make_create_speculative_config,
        make_hf_config_override,
        make_post_init,
    )

    orig_hf = SpeculativeConfig.hf_config_override
    orig_post_init = SpeculativeConfig.__post_init__
    orig_draft_pp = SpeculativeConfig.create_draft_parallel_config
    orig_create = EngineArgs.create_speculative_config

    # [ILU override] hf_config_override — delta: route iluvatar target archs
    # to their dedicated MTP draft models after the community routing runs.
    wrapped_hf = make_hf_config_override(orig_hf)
    # [ILU override] __post_init__ — delta: re-point the DSpark draft arch at
    # IluDSparkDraftModel after community assigns DSparkDraftModel.
    wrapped_post_init = make_post_init(orig_post_init)
    # [ILU override] create_draft_parallel_config — delta: draft PP=1 when
    # target PP>1.
    wrapped_draft_pp = make_create_draft_parallel_config(orig_draft_pp)
    # [ILU override] EngineArgs.create_speculative_config — delta: drop
    # draft_pipeline_parallel_size on older SpeculativeConfig.
    wrapped_create = make_create_speculative_config(orig_create)

    p.check_signatures(
        SpeculativeConfig,
        SimpleNamespace(
            hf_config_override=wrapped_hf,
            __post_init__=wrapped_post_init,
            create_draft_parallel_config=wrapped_draft_pp,
        ),
        ["hf_config_override", "__post_init__", "create_draft_parallel_config"],
        strict=True,
    )
    p.check_signatures(
        EngineArgs,
        SimpleNamespace(create_speculative_config=wrapped_create),
        ["create_speculative_config"],
        strict=True,
    )

    SpeculativeConfig.hf_config_override = staticmethod(wrapped_hf)
    SpeculativeConfig.__post_init__ = wrapped_post_init
    SpeculativeConfig.create_draft_parallel_config = staticmethod(wrapped_draft_pp)
    EngineArgs.create_speculative_config = wrapped_create
