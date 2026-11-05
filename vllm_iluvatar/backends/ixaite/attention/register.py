# SPDX-License-Identifier: Apache-2.0
"""Attention registry paths selected before allocating layers or metadata."""


# Register dotted paths so platform discovery does not import executable kernels.
def register_attention_backends() -> None:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        MambaAttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        "vllm_iluvatar.backends.ixaite.attention.flash_attn.IxaiteFlashAttentionBackend",
    )
    register_backend(
        MambaAttentionBackendEnum.GDN_ATTN,
        "vllm_iluvatar.backends.ixaite.attention.gdn.IluGDNAttentionBackend",
        is_mamba=True,
    )
