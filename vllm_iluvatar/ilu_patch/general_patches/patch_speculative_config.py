# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar ``SpeculativeConfig.hf_config_override`` override.

Corresponds to: vllm/config/speculative.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementation lives in ``vllm_iluvatar.config.speculative``; this
module only performs register-time wiring.

Target   : vllm.config.speculative.SpeculativeConfig.hf_config_override
Baseline : main_dcfebf93f4e
Why      : re-point the MTP draft architecture to the ilu W4A8 self-contained
           MTP models (IluGlmMoeDsaMTPModel / IluDeepSeekV4MTPModel) for the ilu
           target archs, without modifying community source.
How      : wrap the staticmethod with ``make_hf_config_override`` and setattr it
           back onto the class as a staticmethod.
Scope    : SpeculativeConfig MTP draft-arch routing on Iluvatar.
Remove   : when upstream exposes a platform hook for MTP draft arch selection.

Timing   : ``load_general_plugins()`` runs inside ``create_engine_config``,
           before any ``SpeculativeConfig`` is instantiated, so the wrapped
           staticmethod is in place before ``hf_config_override`` is called.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the Iluvatar hf_config_override wrapper."""
    p.register_once(
        __name__,
        _apply,
        detail="SpeculativeConfig.hf_config_override ilu MTP draft-arch routing",
    )


def _apply() -> None:
    from vllm.config.speculative import SpeculativeConfig

    from vllm_iluvatar.config.speculative import make_hf_config_override

    # Accessing the staticmethod on the class yields the underlying function.
    orig = SpeculativeConfig.hf_config_override
    # [ILU override] hf_config_override - delta: route ilu target archs
    # (IluGlmMoeDsaForCausalLM / IluDeepseekV4ForCausalLM) to their dedicated
    # ilu MTP draft models after the community routing runs.
    SpeculativeConfig.hf_config_override = staticmethod(make_hf_config_override(orig))
