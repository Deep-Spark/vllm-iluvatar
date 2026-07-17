# SPDX-License-Identifier: Apache-2.0
"""Unit tests verifying that Iluvatar MLA attention backends are importable.

These files were migrated from the vllm source tree into the plugin repo
(vllm_iluvatar/v1/attention/backends/mla/). This test suite ensures the
modules and their key exported symbols remain importable after the move.

CPU-only — no GPU or ixformer required for the import-level checks.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_path",
    [
        "vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse",
        "vllm_iluvatar.v1.attention.backends.mla.ilu_mla",
        "vllm_iluvatar.v1.attention.backends.mla.ilu_indexer",
        "vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa",
    ],
)
def test_module_importable(module_path: str):
    """Each migrated module must be importable without errors."""
    mod = importlib.import_module(module_path)
    assert mod is not None


_EXPECTED_SYMBOLS = {
    "vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse": [
        "IluFlashMLASparseBackend",
        "IluDeepseekV4FlashMLASparseBackend",
        "IluFlashMLASparseMetadataBuilder",
        "IluFlashMLASparseImpl",
    ],
    "vllm_iluvatar.v1.attention.backends.mla.ilu_mla": [
        "IluMLABackend",
        "IluMLAImpl",
    ],
    "vllm_iluvatar.v1.attention.backends.mla.ilu_indexer": [
        "IluDeepseekV32IndexerBackend",
        "IluDeepseekV4IndexerBackend",
        "IluDeepseekV32IndexerMetadataBuilder",
    ],
    "vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa": [
        "IluDeepseekV4SWACache",
        "IluDeepseekSparseSWABackend",
    ],
}


@pytest.mark.parametrize(
    "module_path, symbols",
    [
        (mod, syms)
        for mod, syms in _EXPECTED_SYMBOLS.items()
    ],
    ids=list(_EXPECTED_SYMBOLS.keys()),
)
def test_key_symbols_exported(module_path: str, symbols: list[str]):
    """Key backend classes must be accessible as module attributes."""
    mod = importlib.import_module(module_path)
    for sym in symbols:
        assert hasattr(mod, sym), (
            f"{module_path} is missing expected symbol '{sym}'"
        )


def test_attention_py_imports_resolve():
    """The plugin's attention.py must be able to import the migrated symbols."""
    from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (  # noqa: F401
        IluDeepseekV4FlashMLASparseBackend,
        IluFlashMLASparseMetadata,
    )
    from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (  # noqa: F401
        IluDeepseekV4IndexerBackend,
        IluDeepseekV32IndexerMetadata,
        get_max_prefill_buffer_size,
    )
    from vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa import (  # noqa: F401
        IluDeepseekV4SWACache,
        IluDeepseekSparseSWABackend,
    )

    assert IluDeepseekV4FlashMLASparseBackend is not None
    assert IluDeepseekV4IndexerBackend is not None
    assert callable(get_max_prefill_buffer_size)
    assert IluDeepseekV4SWACache is not None
    assert IluDeepseekSparseSWABackend is not None
