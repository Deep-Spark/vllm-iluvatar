# SPDX-License-Identifier: Apache-2.0
"""Use storage_block_size in compressed Ilu indexer metadata planning.

Corresponds to: vllm/v1/attention/backends/mla/indexer.py

Target   : vllm_iluvatar.v1.attention.backends.mla.ilu_indexer.
           IluDeepseekV32IndexerMetadataBuilder.build
Baseline : v0.31.0 (db9527a468)
Why      : compressed (V4) indexer pages hold storage_block_size tokens, while
           the builder's kv_cache_spec.block_size is the logical block size.
How      : wrap the Ilu fork of the community
           DeepseekV32IndexerMetadataBuilder.build so it sees the physical
           page size when compress_ratio > 1.
Scope    : DeepSeek-V4 compressed indexer metadata on Iluvatar.
Remove   : when upstream plans compressed indexer metadata with the storage
           block size.
Fingerprint: the community DeepseekV32IndexerMetadataBuilder.build source is
           SHA-checked at register-time; gated by
           VLLM_ILU_PATCH_CHECK_FINGERPRINT.
"""

from __future__ import annotations

import inspect
import types

from vllm_iluvatar.ilu_patch import general_patch as p

_COMMUNITY_BUILD = (
    "vllm.v1.attention.backends.mla.indexer.DeepseekV32IndexerMetadataBuilder.build"
)
_COMMUNITY_BUILD_SOURCE_SHA = (
    "6f6a33809a13dab8fddca0a5abe7d1c3ab6619617db9ee6a4bc0551d3ff7f1d3"
)


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="Ilu indexer compressed storage block-size wrapper",
    )


def _check_community_build(build: object) -> None:
    if not p._enabled(p.ENV_CHECK_FINGERPRINT):
        return
    current_sha = p.source_sha(inspect.getsource(build))
    if current_sha != _COMMUNITY_BUILD_SOURCE_SHA:
        raise RuntimeError(
            f"ILU patch: community {_COMMUNITY_BUILD} source changed vs "
            f"baseline v0.31.0. Expected {_COMMUNITY_BUILD_SOURCE_SHA}, got "
            f"{current_sha}. Re-sync IluDeepseekV32IndexerMetadataBuilder.build "
            "and make_storage_block_size_build in "
            "vllm_iluvatar/v1/attention/backends/mla/ilu_indexer.py, then "
            "refresh the baseline SHA in patch_ilu_indexer.py."
        )


def _apply() -> None:
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )

    from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
        IluDeepseekV32IndexerMetadataBuilder,
        make_storage_block_size_build,
    )

    _check_community_build(DeepseekV32IndexerMetadataBuilder.build)

    replacement = make_storage_block_size_build(
        IluDeepseekV32IndexerMetadataBuilder.build
    )
    p.check_signatures(
        IluDeepseekV32IndexerMetadataBuilder,
        types.SimpleNamespace(build=replacement),
        ["build"],
        strict=True,
    )
    IluDeepseekV32IndexerMetadataBuilder.build = replacement
