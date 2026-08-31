# SPDX-License-Identifier: Apache-2.0
"""Cap MLA chunked-prefill workspace size via env.

Corresponds to:
    vllm/model_executor/layers/attention/mla_attention.py
        MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size
    vllm/model_executor/layers/attention/sparse_mla_attention.py
        SparseMLACommonMetadataBuilder.determine_chunked_prefill_workspace_size

Target   : community MLA / sparse-MLA metadata builders.
Baseline : main_6e448d0
Why      : Iluvatar deployments sometimes need a smaller workspace cap than
           the community 64k formula (e.g. W8A8 kv_b_proj peak memory).
How      : replace both builders'
           ``determine_chunked_prefill_workspace_size`` so every caller
           (builder init, ``MLAAttention`` property, profiling) sees
           ``VLLM_ILU_MLA_CHUNKED_PREFILL_MAX_WS_SIZE`` (0 = no change).
Scope    : dense and sparse MLA chunked-prefill workspace sizing.
Remove   : when upstream exposes a configurable workspace size cap.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="MLA chunked-prefill workspace size cap",
    )


def _apply() -> None:
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonMetadataBuilder,
    )
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLACommonMetadataBuilder,
    )

    from vllm_iluvatar.model_executor.layers.attention.mla_attention import (
        wrap_determine_chunked_prefill_workspace_size,
    )

    MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size = staticmethod(
        wrap_determine_chunked_prefill_workspace_size(
            MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size
        )
    )
    SparseMLACommonMetadataBuilder.determine_chunked_prefill_workspace_size = (
        staticmethod(
            wrap_determine_chunked_prefill_workspace_size(
                SparseMLACommonMetadataBuilder.determine_chunked_prefill_workspace_size
            )
        )
    )
