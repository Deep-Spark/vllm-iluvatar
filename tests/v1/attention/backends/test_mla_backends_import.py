# SPDX-License-Identifier: Apache-2.0
"""Unit tests verifying that Iluvatar MLA attention backends are importable.

These files were migrated from the vllm source tree into the plugin repo
(vllm_iluvatar/v1/attention/backends/mla/). This test suite ensures the
modules and their key exported symbols remain importable after the move.

CPU-only — no GPU or ixformer required for the import-level checks.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "module_path",
    [
        "vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse",
        "vllm_iluvatar.v1.attention.backends.mla.ilu_mla",
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
    "vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa": [
        "IluDeepseekV4SWACache",
        "IluDeepseekSparseSWABackend",
    ],
}


@pytest.mark.parametrize(
    "module_path, symbols",
    [(mod, syms) for mod, syms in _EXPECTED_SYMBOLS.items()],
    ids=list(_EXPECTED_SYMBOLS.keys()),
)
def test_key_symbols_exported(module_path: str, symbols: list[str]):
    """Key backend classes must be accessible as module attributes."""
    mod = importlib.import_module(module_path)
    for sym in symbols:
        assert hasattr(mod, sym), f"{module_path} is missing expected symbol '{sym}'"


def test_attention_py_imports_resolve():
    """The plugin's attention.py must be able to import the migrated symbols."""
    from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (  # noqa: F401
        IluDeepseekV4FlashMLASparseBackend,
        IluFlashMLASparseMetadata,
    )
    from vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa import (  # noqa: F401
        IluDeepseekSparseSWABackend,
        IluDeepseekV4SWACache,
    )

    assert IluDeepseekV4FlashMLASparseBackend is not None
    assert IluDeepseekV4SWACache is not None
    assert IluDeepseekSparseSWABackend is not None


def test_ilu_mla_spec_decode_expand_pads_graph_rows():
    """Spec decode metadata must stay token-level when CUDA graph pads rows."""
    from vllm_iluvatar.v1.attention.backends.mla.ilu_mla import (
        IluMLACommonMetadataBuilder,
    )

    builder = IluMLACommonMetadataBuilder.__new__(IluMLACommonMetadataBuilder)
    builder.decode_use_graph = True
    builder.use_full_cuda_graph = False
    builder._spec_decode_expand_block_table_buf = None
    builder._spec_decode_expand_seq_lens_buf = None
    builder.device = torch.device("cpu")
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16)
    )

    block_table = torch.tensor([[10, 11], [20, 21], [0, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([10, 12, 0], dtype=torch.int32)
    query_start_loc_cpu = torch.tensor([0, 2, 4, 4], dtype=torch.int32)

    expanded_block_table, expanded_seq_lens = builder._expand_spec_decode_metadata(
        block_table,
        seq_lens,
        query_start_loc_cpu,
        num_decode_tokens=6,
    )

    assert expanded_block_table.tolist() == [
        [10, 11],
        [10, 11],
        [20, 21],
        [20, 21],
        [0, 0],
        [0, 0],
    ]
    assert expanded_seq_lens.tolist() == [9, 10, 11, 12, 0, 0]
