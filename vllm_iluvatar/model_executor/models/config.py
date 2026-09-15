# SPDX-License-Identifier: Apache-2.0
"""Iluvatar model config override helpers.

Corresponds to: vllm/model_executor/models/config.py
Baseline : main_6e448d0
Target   : vllm.model_executor.models.config.Gemma4Config.verify_and_update_config
Why      : Gemma4 heterogeneous head-dim models should avoid the community
           TRITON_ATTN fallback on Iluvatar/CoreX when FA4 is unavailable.
How      : Wrap the community verifier, preserve FA4 selection and explicit user
           backend choices, then rewrite only the community auto fallback from
           TRITON_ATTN to FLASH_ATTN.
Scope    : Gemma4 config verification on Iluvatar.
Remove   : when community backend selection is suitable for Iluvatar Gemma4.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from vllm.logger import init_logger

logger = init_logger(__name__)


def _has_heterogeneous_gemma4_head_dims(
    vllm_config,
) -> tuple[bool, int | None, int | None]:
    hf_text_config = vllm_config.model_config.hf_text_config
    head_dim = getattr(hf_text_config, "head_dim", None)
    global_head_dim = getattr(hf_text_config, "global_head_dim", None)
    return (
        head_dim is not None
        and global_head_dim is not None
        and head_dim != global_head_dim,
        head_dim,
        global_head_dim,
    )


def make_gemma4_verify_and_update_config(
    community_verify_and_update_config: Callable[[object], None],
) -> Callable[[object], None]:
    """Build the Iluvatar wrapper for Gemma4 config verification."""

    def _ilu_gemma4_verify_and_update_config(vllm_config) -> None:
        registry = importlib.import_module("vllm.v1.attention.backends.registry")
        AttentionBackendEnum = registry.AttentionBackendEnum

        attention_config = vllm_config.attention_config
        before_backend = attention_config.backend

        community_verify_and_update_config(vllm_config)

        is_heterogeneous, head_dim, global_head_dim = (
            _has_heterogeneous_gemma4_head_dims(vllm_config)
        )
        if (
            is_heterogeneous
            and before_backend is None
            and attention_config.backend == AttentionBackendEnum.TRITON_ATTN
        ):
            # [ILU override] Gemma4Config.verify_and_update_config - delta:
            # preserve community FA4 handling, but use FLASH_ATTN for the
            # auto-selected fallback on Iluvatar.
            attention_config.backend = AttentionBackendEnum.FLASH_ATTN
            logger.info(
                "Gemma4 model has heterogeneous head dimensions "
                "(head_dim=%d, global_head_dim=%d). FA4 not available, "
                "forcing FLASH_ATTN backend on Iluvatar.",
                head_dim,
                global_head_dim,
            )

    return _ilu_gemma4_verify_and_update_config
