# SPDX-License-Identifier: Apache-2.0
"""Wiring for Iluvatar model config verification overrides.

Corresponds to: vllm/model_executor/models/config.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementation lives in ``vllm_iluvatar.model_executor.models.config``;
this module only performs register-time wiring.

Target   : vllm.model_executor.models.config.Gemma4Config.verify_and_update_config
Baseline : main_6e448d0
Why      : keep community vLLM on its upstream Gemma4 fallback while selecting the
           Iluvatar-preferred FLASH_ATTN fallback from the plugin package.
How      : wrap the staticmethod after community config module import; the wrapper
           delegates to community first, then rewrites only the auto-selected
           TRITON_ATTN fallback for heterogeneous Gemma4 head dimensions.
Scope    : Gemma4 model config verification on Iluvatar.
Remove   : when community backend selection is suitable for Iluvatar Gemma4.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register the Iluvatar Gemma4 config verifier wrapper."""
    p.register_once(
        __name__,
        _apply,
        detail="Gemma4Config.verify_and_update_config FLASH_ATTN fallback",
    )


def _apply() -> None:
    import importlib

    from vllm_iluvatar.model_executor.models.config import (
        make_gemma4_verify_and_update_config,
    )

    community_config = importlib.import_module("vllm.model_executor.models.config")
    Gemma4Config = community_config.Gemma4Config

    orig = Gemma4Config.verify_and_update_config
    Gemma4Config.verify_and_update_config = staticmethod(
        make_gemma4_verify_and_update_config(orig)
    )
