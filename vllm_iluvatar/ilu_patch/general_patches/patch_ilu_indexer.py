# SPDX-License-Identifier: Apache-2.0
"""Use storage_block_size in compressed Ilu indexer metadata planning."""

from __future__ import annotations

import inspect
import types

from vllm_iluvatar.ilu_patch import general_patch as p

# v0.31.0 IluDeepseekV32IndexerMetadataBuilder.build.
_BUILD_SOURCE_SHA = "b0cb28a43984d927657cad9ddde40e27a265f22fa72640adb72eca6be6ca38d1"


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="Ilu indexer compressed storage block-size wrapper",
    )


def _apply() -> None:
    from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
        IluDeepseekV32IndexerMetadataBuilder,
        make_storage_block_size_build,
    )

    original = IluDeepseekV32IndexerMetadataBuilder.build
    current_sha = p.source_sha(inspect.getsource(original))
    if current_sha != _BUILD_SOURCE_SHA:
        raise RuntimeError(
            "ILU patch: plugin "
            "IluDeepseekV32IndexerMetadataBuilder.build changed vs the "
            "v0.31.0 baseline. Re-sync the focused storage_block_size "
            "wrapper before running."
        )

    replacement = make_storage_block_size_build(original)
    p.check_signatures(
        IluDeepseekV32IndexerMetadataBuilder,
        types.SimpleNamespace(build=replacement),
        ["build"],
        strict=True,
    )
    IluDeepseekV32IndexerMetadataBuilder.build = replacement
