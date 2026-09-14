# SPDX-License-Identifier: Apache-2.0
"""Wiring for Iluvatar version strings in community serve / collect-env logs.

Corresponds to: vllm/collect_env.py, vllm/entrypoints/serve/utils/api_utils.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

This patch is a pure wrapper with no override implementation of its own, so no
upstream-mirrored implementation sibling is needed.

Target   : vllm.collect_env.get_vllm_version,
           vllm.entrypoints.serve.utils.api_utils.log_version_and_model
Baseline : main_6e448d0
Why      : community printers only know vLLM's own version; do not patch vLLM.
How      : wrap those module-level functions with ``rebind_module_func`` and log
           ``vllm_iluvatar {version} {git_id}``. Delegation wrappers; no
           fingerprint check.
Scope    : serve banner and collect-env after general plugins load.
Remove   : if upstream adds an OOT hook for plugin version strings.
"""

from __future__ import annotations

import logging

from vllm_iluvatar.ilu_patch import general_patch as p
from vllm_iluvatar.version import format_identity

logger = logging.getLogger("vllm_iluvatar")


def register() -> None:
    """Register community version-printer wrappers."""
    p.register_once(
        __name__,
        _apply,
        detail="vllm_iluvatar version in serve/collect-env",
    )


def _wrap_collect_env(mod: object) -> None:
    orig = getattr(mod, "get_vllm_version", None)
    if orig is None:
        return

    def get_vllm_version() -> str:
        base = orig()
        ident = format_identity()
        marker = f"vllm_iluvatar {ident}"
        if marker in base:
            return base
        return f"{base} ; {marker}"

    # [ILU override] collect_env.get_vllm_version — delta: append plugin identity
    p.rebind_module_func(mod, "get_vllm_version", get_vllm_version)


def _wrap_log_version_and_model(mod: object) -> None:
    orig = getattr(mod, "log_version_and_model", None)
    if orig is None:
        return

    def log_version_and_model(lgr, version: str, model_name: str) -> None:
        orig(lgr, version, model_name)
        lgr.info("vllm_iluvatar %s", format_identity())

    # [ILU override] api_utils.log_version_and_model — delta: extra plugin line
    p.rebind_module_func(mod, "log_version_and_model", log_version_and_model)


def _apply() -> None:
    ident = format_identity()
    try:
        from vllm.logger import init_logger

        init_logger("vllm_iluvatar").info("vllm_iluvatar %s", ident)
    except Exception:
        logger.info("vllm_iluvatar %s", ident)

    import vllm.collect_env as collect_env
    from vllm.entrypoints.serve.utils import api_utils

    _wrap_collect_env(collect_env)
    _wrap_log_version_and_model(api_utils)
