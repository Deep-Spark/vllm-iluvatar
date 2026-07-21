# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for Iluvatar model config patch helpers."""

from __future__ import annotations

from types import SimpleNamespace

from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_iluvatar.model_executor.models.config import (
    make_gemma4_verify_and_update_config,
)


def _gemma4_vllm_config(
    *,
    backend=None,
    head_dim: int | None = 256,
    global_head_dim: int | None = 512,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                head_dim=head_dim,
                global_head_dim=global_head_dim,
            )
        ),
        attention_config=SimpleNamespace(backend=backend),
    )


def _community_gemma4_fallback(vllm_config) -> None:
    if vllm_config.attention_config.backend is None:
        vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN


def test_gemma4_config_wrapper_rewrites_auto_triton_fallback_to_flash_attn():
    verify = make_gemma4_verify_and_update_config(_community_gemma4_fallback)
    vllm_config = _gemma4_vllm_config()

    verify(vllm_config)

    assert vllm_config.attention_config.backend == AttentionBackendEnum.FLASH_ATTN


def test_gemma4_config_wrapper_preserves_explicit_backend():
    verify = make_gemma4_verify_and_update_config(_community_gemma4_fallback)
    vllm_config = _gemma4_vllm_config(backend=AttentionBackendEnum.TRITON_ATTN)

    verify(vllm_config)

    assert vllm_config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN


def test_gemma4_config_wrapper_preserves_homogeneous_head_dims():
    verify = make_gemma4_verify_and_update_config(_community_gemma4_fallback)
    vllm_config = _gemma4_vllm_config(head_dim=256, global_head_dim=256)

    verify(vllm_config)

    assert vllm_config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
