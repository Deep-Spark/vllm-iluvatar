# SPDX-License-Identifier: Apache-2.0
"""QSA plugin wiring tests; ixformer owns numerical kernel coverage."""

from __future__ import annotations

import inspect

import ixformer._C as ixf_c_ops

from vllm_iluvatar.models.qwen4_exp import attention, indexer, qsa_cache


def test_qsa_runtime_has_no_triton_or_torch_backend() -> None:
    source = "\n".join(
        inspect.getsource(module) for module in (attention, indexer, qsa_cache)
    )
    assert "vllm.triton_utils" not in source
    assert 'backend="torch"' not in source
    assert "build_qsa_metadata_triton" not in source


def test_qsa_runtime_forces_ixformer_native_dispatch() -> None:
    source = inspect.getsource(attention) + inspect.getsource(indexer)
    assert 'backend="ixformer_native"' in source


def test_qsa_required_ixformer_cuda_ops_are_available() -> None:
    required = (
        "qwen_qsa_build_metadata",
        "qwen_qsa_indexer_mqa_logits",
        "qwen_qsa_indexer_q_norm_rope",
        "qwen_qsa_indexer_topk_expand",
        "qwen_qsa_kpool_compress_store",
        "qwen_qsa_kpool_tail_update",
        "qwen_qsa_kv_store",
        "qwen_qsa_qk_norm_rope_gate",
        "qwen_qsa_sparse_paged_fmha",
    )
    missing = [name for name in required if not hasattr(ixf_c_ops.infer, name)]
    assert missing == []
