# SPDX-License-Identifier: Apache-2.0
"""Tests for the Triton jit compatibility patch."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from vllm_iluvatar.ilu_patch.general_patches import patch_triton_jit_compat


def _install_fake_vllm_triton(monkeypatch, triton_module):
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    triton_utils = ModuleType("vllm.triton_utils")
    triton_utils.triton = triton_module

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", triton_utils)
    monkeypatch.setitem(sys.modules, "triton", triton_module)


def test_apply_wraps_old_triton_jit_and_drops_alignment_kwarg(monkeypatch):
    seen = []

    def old_jit(fn=None, **kwargs):
        if "do_not_specialize_on_alignment" in kwargs:
            raise TypeError("jit() got an unexpected keyword argument")
        seen.append(kwargs)
        if fn is None:
            return lambda f: f
        return fn

    triton_module = SimpleNamespace(jit=old_jit)
    _install_fake_vllm_triton(monkeypatch, triton_module)

    patch_triton_jit_compat._apply()

    assert triton_module.jit is not old_jit
    assert getattr(
        triton_module.jit,
        patch_triton_jit_compat._ILU_WRAPPED_ATTR,
    )
    decorator = triton_module.jit(
        do_not_specialize=["x"],
        do_not_specialize_on_alignment=["seq_lens"],
    )
    assert decorator(lambda: None).__name__ == "<lambda>"
    assert seen == [{"do_not_specialize": ["x"]}]


def test_apply_leaves_new_triton_jit_unchanged(monkeypatch):
    seen = []

    def new_jit(
        fn=None,
        *,
        do_not_specialize=None,
        do_not_specialize_on_alignment=None,
    ):
        seen.append((do_not_specialize, do_not_specialize_on_alignment))
        if fn is None:
            return lambda f: f
        return fn

    triton_module = SimpleNamespace(jit=new_jit)
    _install_fake_vllm_triton(monkeypatch, triton_module)

    patch_triton_jit_compat._apply()

    assert triton_module.jit is new_jit
    decorator = triton_module.jit(do_not_specialize_on_alignment=["seq_lens"])
    assert decorator(lambda: None).__name__ == "<lambda>"
    assert seen == [(None, ["seq_lens"])]
