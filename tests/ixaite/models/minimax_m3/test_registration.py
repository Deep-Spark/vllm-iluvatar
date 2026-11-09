# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""ixaite MiniMax OOT registration contracts."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
import vllm

from vllm_iluvatar.backends.ixaite.models import unsupported as unsupported_mod
from vllm_iluvatar.backends.ixaite.models.minimax_m3 import model as minimax_model
from vllm_iluvatar.backends.ixaite.oot_registration import _SELECTED_MODEL_MODULES

_REPO = Path(__file__).resolve().parents[4]
_IXAITE_MINIMAX = (
    _REPO / "vllm_iluvatar" / "backends" / "ixaite" / "models" / "minimax_m3"
)

_FLASH_ARCHES = (
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
)
_W4A8_ARCHES = (
    "IluMiniMaxM3SparseForCausalLM",
    "IluMiniMaxM3SparseForConditionalGeneration",
)


def test_minimax_arches_are_not_in_unsupported_reasons():
    for architecture in _FLASH_ARCHES + _W4A8_ARCHES:
        assert architecture not in unsupported_mod._REASONS


def test_oot_maps_baseline_minimax_model_to_ixaite():
    assert (
        _SELECTED_MODEL_MODULES["vllm_iluvatar.models.minimax_m3.model"]
        == "vllm_iluvatar.backends.ixaite.models.minimax_m3.model"
    )


def test_register_model_points_at_ixaite_classes(monkeypatch):
    calls: list[tuple[str, str]] = []

    class _Registry:
        @staticmethod
        def register_model(arch, path):
            calls.append((arch, path))

    monkeypatch.setattr(vllm, "ModelRegistry", _Registry)
    minimax_model.register_model()
    by_arch = dict(calls)
    for architecture in _FLASH_ARCHES:
        path = by_arch[architecture]
        module_name, class_name = path.split(":")
        assert module_name == "vllm_iluvatar.backends.ixaite.models.minimax_m3.flash"
        assert class_name == architecture
        # Class bodies live in flash.py; avoid importing the full graph here.
        flash_src = (_IXAITE_MINIMAX / "flash.py").read_text()
        assert f"class {class_name}" in flash_src
    for architecture in _W4A8_ARCHES:
        path = by_arch[architecture]
        module_name, class_name = path.split(":")
        assert module_name == "vllm_iluvatar.backends.ixaite.models.minimax_m3.model"
        assert class_name == architecture
        assert hasattr(minimax_model, class_name)


def test_w4a8_ilu_arches_fail_closed_with_flash_scope_message():
    with pytest.raises(RuntimeError, match="Ixaite supports MiniMaxM3Sparse"):
        minimax_model.IluMiniMaxM3SparseForCausalLM(vllm_config=object())
    with pytest.raises(RuntimeError, match="Ixaite supports MiniMaxM3Sparse"):
        minimax_model.IluMiniMaxM3SparseForConditionalGeneration(vllm_config=object())


def test_ixaite_minimax_package_has_no_vendor_or_ixformer_imports():
    forbidden_prefixes = (
        "vllm.models.minimax_m3.nvidia",
        "vllm.models.minimax_m3.amd",
        "ixformer",
    )
    for source_file in _IXAITE_MINIMAX.rglob("*.py"):
        tree = ast.parse(source_file.read_text(), filename=str(source_file))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        bad = [
            name
            for name in imported
            for prefix in forbidden_prefixes
            if name == prefix or name.startswith(f"{prefix}.")
        ]
        assert not bad, f"{source_file}: {bad}"


def test_flash_attention_scaffold_exports():
    attention = importlib.import_module(
        "vllm_iluvatar.backends.ixaite.models.minimax_m3.attention"
    )
    assert hasattr(attention, "MiniMaxM3Attention")
    assert hasattr(attention, "MiniMaxM3SparseAttention")
    assert hasattr(attention, "MiniMAXGemmaRMSNorm")
    assert callable(attention.fused_minimax_m3_qknorm_rope_kv_insert)
    from vllm_iluvatar.backends.ixaite.models.minimax_m3.sparse_attention import (
        MiniMaxM3SparseBackend,
    )

    assert attention.MiniMaxM3SparseBackend is MiniMaxM3SparseBackend
