# SPDX-License-Identifier: Apache-2.0
"""Tests for the sparse MLA warmup compatibility shim."""

from __future__ import annotations

import sys
from types import ModuleType

from vllm_iluvatar.ilu_patch.general_patches import patch_sparse_mla_warmup


def _install_fake_sparse_swa(monkeypatch, sparse_swa):
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    v1 = ModuleType("vllm.v1")
    v1.__path__ = []
    attention = ModuleType("vllm.v1.attention")
    attention.__path__ = []
    backends = ModuleType("vllm.v1.attention.backends")
    backends.__path__ = []
    mla = ModuleType("vllm.v1.attention.backends.mla")
    mla.__path__ = []

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.attention", attention)
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends", backends)
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.mla", mla)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.attention.backends.mla.sparse_swa",
        sparse_swa,
    )
    setattr(mla, "sparse_swa", sparse_swa)


def _install_fake_ixformer(monkeypatch, calls):
    ixformer = ModuleType("ixformer")
    ixformer.__path__ = []
    inference = ModuleType("ixformer.inference")
    inference.__path__ = []
    functions = ModuleType("ixformer.inference.functions")

    def fake_ds4_compute_swa_prefill_gather_lens(**kwargs):
        calls.append(kwargs)

    functions.ds4_compute_swa_prefill_gather_lens = (
        fake_ds4_compute_swa_prefill_gather_lens
    )

    monkeypatch.setitem(sys.modules, "ixformer", ixformer)
    monkeypatch.setitem(sys.modules, "ixformer.inference", inference)
    monkeypatch.setitem(sys.modules, "ixformer.inference.functions", functions)


def test_apply_injects_kernel_like_prefill_metadata_shim(monkeypatch):
    sparse_swa = ModuleType("vllm.v1.attention.backends.mla.sparse_swa")
    calls = []
    _install_fake_sparse_swa(monkeypatch, sparse_swa)
    _install_fake_ixformer(monkeypatch, calls)

    patch_sparse_mla_warmup._apply()

    kernel = sparse_swa._compute_prefill_metadata_kernel
    kernel[(1,)](
        "prefill_gather_lens",
        "seq_lens",
        "query_start_loc",
        2,
        1,
        4096,
        BLOCK_SIZE=2,
    )
    assert calls == [
        {
            "prefill_gather_lens": "prefill_gather_lens",
            "seq_lens": "seq_lens",
            "query_start_loc": "query_start_loc",
            "num_prefills": 2,
            "num_decodes": 1,
            "window_size": 4096,
        }
    ]


def test_apply_keeps_existing_prefill_metadata_kernel(monkeypatch):
    sparse_swa = ModuleType("vllm.v1.attention.backends.mla.sparse_swa")
    existing_kernel = object()
    sparse_swa._compute_prefill_metadata_kernel = existing_kernel
    _install_fake_sparse_swa(monkeypatch, sparse_swa)

    patch_sparse_mla_warmup._apply()

    assert sparse_swa._compute_prefill_metadata_kernel is existing_kernel
