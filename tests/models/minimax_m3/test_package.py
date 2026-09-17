# SPDX-License-Identifier: Apache-2.0
"""Quant-agnostic MiniMax-M3 package contracts."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import vllm_iluvatar.ilu_register as ilu_register

PACKAGE = "vllm_iluvatar.models.minimax_m3"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_minimax_m3_packages_are_explicitly_discovered():
    assert PACKAGE in ilu_register._OOT_MODEL_PACKAGES
    assert f"{PACKAGE}.w4a8" not in ilu_register._OOT_MODEL_PACKAGES
    assert f"{PACKAGE}.mxfp8" not in ilu_register._OOT_MODEL_PACKAGES


def test_minimax_m3_package_init_does_not_eager_import_model():
    init_path = _REPO_ROOT / "vllm_iluvatar" / "models" / "minimax_m3" / "__init__.py"
    tree = ast.parse(init_path.read_text(), filename=str(init_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in {"model", "mtp"}
            if node.module:
                assert not node.module.endswith(".model")
                assert not node.module.endswith(".mtp")


def test_minimax_m3_runtime_has_no_hardware_upstream_imports():
    package_dir = _REPO_ROOT / "vllm_iluvatar" / "models" / "minimax_m3"
    forbidden = {
        "vllm.models.minimax_m3.nvidia",
        "vllm.models.minimax_m3.amd",
    }
    for source_file in package_dir.rglob("*.py"):
        tree = ast.parse(source_file.read_text(), filename=str(source_file))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        forbidden_modules = forbidden | {"nvidia", "amd"}
        assert not any(
            name == prefix or name.startswith(f"{prefix}.")
            for name in imported
            for prefix in forbidden_modules
        ), source_file


def test_no_general_patch_touches_community_minimax_m3():
    """The local package wires the ixformer ops itself. A surviving patch would
    additionally rebind the community module functions process-wide."""
    patch_dir = _REPO_ROOT / "vllm_iluvatar" / "ilu_patch" / "general_patches"
    assert not list(patch_dir.glob("*minimax_m3*.py"))


def test_community_config_still_supplies_m3_swiglu_defaults():
    """``swiglu_beta`` is absent from the checkpoint, so the config default is
    what M3 actually runs with. If community ever changes it, M3's routed and
    shared experts both silently change activation."""
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

    defaults = inspect.signature(MiniMaxM3TextConfig.__init__).parameters
    assert defaults["swiglu_alpha"].default == 1.702
    assert defaults["swiglu_beta"].default == 1.0
    assert defaults["swiglu_limit"].default == 7.0
