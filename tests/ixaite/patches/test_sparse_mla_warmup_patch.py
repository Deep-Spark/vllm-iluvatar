# SPDX-License-Identifier: Apache-2.0
"""Tests for the sparse MLA warmup compatibility shim."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from vllm_iluvatar.backends.ixaite.register import _apply_warmup
from vllm_iluvatar.ilu_patch.general_patches import patch_sparse_mla_warmup


# Isolate the upstream symbol without loading its full attention dependencies.
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
    mla.sparse_swa = sparse_swa


# Each shim must call its own backend even though the launch ABI is shared.
def _install_fake_ilu_ops(monkeypatch, calls, mode):
    # Record the launch contract without invoking a device kernel.
    def fake_ds4_compute_swa_prefill_gather_lens(**kwargs):
        calls.append(kwargs)

    target = (
        "vllm_iluvatar.backends.ixaite.ops.ds4_compute_swa_prefill_gather_lens"
        if mode == "ixaite"
        else "ixformer.inference.functions.ds4_compute_swa_prefill_gather_lens"
    )
    monkeypatch.setattr(target, fake_ds4_compute_swa_prefill_gather_lens)


# Backend selection happens once when installing the upstream warmup symbol.
@pytest.mark.parametrize("mode", ["baseline", "ixaite"])
def test_apply_injects_kernel_like_prefill_metadata_shim(monkeypatch, mode):
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", mode)
    sparse_swa = ModuleType("vllm.v1.attention.backends.mla.sparse_swa")
    calls = []
    _install_fake_sparse_swa(monkeypatch, sparse_swa)
    _install_fake_ilu_ops(monkeypatch, calls, mode)

    apply = _apply_warmup if mode == "ixaite" else patch_sparse_mla_warmup._apply
    apply()

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


# Both installation paths must preserve an upstream kernel that already exists.
@pytest.mark.parametrize("mode", ["baseline", "ixaite"])
def test_apply_keeps_existing_prefill_metadata_kernel(monkeypatch, mode):
    sparse_swa = ModuleType("vllm.v1.attention.backends.mla.sparse_swa")
    existing_kernel = object()
    sparse_swa._compute_prefill_metadata_kernel = existing_kernel
    _install_fake_sparse_swa(monkeypatch, sparse_swa)

    apply = _apply_warmup if mode == "ixaite" else patch_sparse_mla_warmup._apply
    apply()

    assert sparse_swa._compute_prefill_metadata_kernel is existing_kernel
