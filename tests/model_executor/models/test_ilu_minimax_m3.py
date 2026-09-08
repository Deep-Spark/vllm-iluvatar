# SPDX-License-Identifier: Apache-2.0
"""CPU-only registration and import contracts for MiniMax-M3."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest
import vllm

import vllm_iluvatar.ilu_register as ilu_register

PACKAGE = "vllm_iluvatar.models.minimax_m3"
EXPECTED_REGISTRATIONS = [
    (
        "MiniMaxM3SparseForCausalLM",
        f"{PACKAGE}.model:MiniMaxM3SparseForCausalLM",
    ),
    (
        "MiniMaxM3SparseForConditionalGeneration",
        f"{PACKAGE}.model:MiniMaxM3SparseForConditionalGeneration",
    ),
]


def test_minimax_m3_package_is_explicitly_discovered():
    assert PACKAGE in ilu_register._OOT_MODEL_PACKAGES


def test_minimax_m3_package_init_does_not_eager_import_model():
    init_path = (
        Path(__file__).parents[3]
        / "vllm_iluvatar"
        / "models"
        / "minimax_m3"
        / "__init__.py"
    )
    tree = ast.parse(init_path.read_text(), filename=str(init_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in {"model", "mtp"}
            if node.module:
                assert not node.module.endswith(".model")
                assert not node.module.endswith(".mtp")


def test_minimax_m3_registers_checkpoint_arches_to_local_classes(monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        vllm.ModelRegistry,
        "register_model",
        lambda arch, path: calls.append((arch, path)),
    )

    model = importlib.import_module(f"{PACKAGE}.model")
    model.register_model()

    assert calls == EXPECTED_REGISTRATIONS
    for _, class_path in calls:
        module_name, class_name = class_path.split(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        assert cls.__module__.startswith(PACKAGE)


def test_minimax_m3_registry_entries_resolve_to_local_module():
    model = importlib.import_module(f"{PACKAGE}.model")
    assert model.fused_minimax_m3_qknorm_rope_kv_insert.__module__ == (
        f"{PACKAGE}.model"
    )
    model.register_model()

    for architecture, class_path in EXPECTED_REGISTRATIONS:
        expected_module, expected_class = class_path.split(":")
        entry = vllm.ModelRegistry.models[architecture]
        assert entry.module_name == expected_module
        assert entry.class_name == expected_class


def test_minimax_m3_reuses_community_common_graph():
    """Indexer / sparse-attention metadata live in community ``common``."""
    indexer = importlib.import_module(f"{PACKAGE}.indexer")
    sparse = importlib.import_module(f"{PACKAGE}.sparse_attention")
    common_indexer = importlib.import_module("vllm.models.minimax_m3.common.indexer")
    common_sparse = importlib.import_module(
        "vllm.models.minimax_m3.common.sparse_attention"
    )

    assert indexer.MiniMaxM3IndexerCache is common_indexer.MiniMaxM3IndexerCache
    assert indexer.MiniMaxM3IndexerMetadata is common_indexer.MiniMaxM3IndexerMetadata
    assert (
        indexer.MiniMaxM3IndexerTritonMetadataBuilder
        is common_indexer.MiniMaxM3IndexerTritonMetadataBuilder
    )
    assert sparse.MiniMaxM3SparseMetadata is common_sparse.MiniMaxM3SparseMetadata
    assert (
        sparse.MiniMaxM3SparseMetadataBuilder
        is common_sparse.MiniMaxM3SparseMetadataBuilder
    )
    assert issubclass(
        indexer.MiniMaxM3IndexerBackend, common_indexer.MiniMaxM3IndexerBackend
    )
    assert issubclass(
        sparse.MiniMaxM3SparseBackend, common_sparse.MiniMaxM3SparseBackend
    )


def test_minimax_m3_runtime_has_no_hardware_upstream_imports():
    package_dir = Path(__file__).parents[3] / "vllm_iluvatar" / "models" / "minimax_m3"
    forbidden = {
        "vllm.models.minimax_m3.nvidia",
        "vllm.models.minimax_m3.amd",
    }
    for source_file in package_dir.glob("*.py"):
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
    patch_dir = (
        Path(__file__).parents[3] / "vllm_iluvatar" / "ilu_patch" / "general_patches"
    )
    assert not list(patch_dir.glob("*minimax_m3*.py"))


def test_routed_experts_receive_the_config_swiglu_parameters():
    """Pin the config -> RoutedExperts hop for limit / alpha / beta.

    ``IluModelOptMxFp8FusedMoE.apply`` now refuses to run without all three, so
    a dropped keyword here is a loud failure rather than a silently different
    activation -- but only if the keyword is actually passed. Checked
    statically: constructing ``MiniMaxM3MoE`` needs a distributed group.
    """
    package_dir = Path(__file__).parents[3] / "vllm_iluvatar" / "models" / "minimax_m3"
    tree = ast.parse((package_dir / "model.py").read_text())

    moe_cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxM3MoE"
    )
    factory = next(
        node
        for node in ast.walk(moe_cls)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FusedMoEFactory"
    )
    passed = {
        kw.arg: ast.unparse(kw.value) for kw in factory.keywords if kw.arg is not None
    }

    for name in ("swiglu_limit", "swiglu_alpha", "swiglu_beta"):
        assert passed.get(name) == f"config.{name}", (
            f"MiniMaxM3MoE must forward {name} to the routed experts"
        )
    # The activation string selects which of the two SwiGLU-OAI layouts runs;
    # w13 is loaded packed, so it must be the uninterleaved one.
    assert passed.get("activation") == "'swigluoai_uninterleave'"


def test_community_config_still_supplies_m3_swiglu_defaults():
    """``swiglu_beta`` is absent from the checkpoint, so the config default is
    what M3 actually runs with. If community ever changes it, M3's routed and
    shared experts both silently change activation."""
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

    defaults = inspect.signature(MiniMaxM3TextConfig.__init__).parameters
    assert defaults["swiglu_alpha"].default == 1.702
    assert defaults["swiglu_beta"].default == 1.0
    assert defaults["swiglu_limit"].default == 7.0


def test_require_unquantized_m3_caches_rejects_fp8():
    import torch

    from vllm_iluvatar.models.minimax_m3.model import _require_unquantized_m3_caches

    _require_unquantized_m3_caches("auto", torch.bfloat16, "bf16")
    _require_unquantized_m3_caches("bfloat16", torch.bfloat16, "bf16")
    with pytest.raises(NotImplementedError, match="kv_cache_dtype"):
        _require_unquantized_m3_caches("fp8", torch.float8_e4m3fn, "bf16")
    with pytest.raises(NotImplementedError, match="indexer_kv_dtype"):
        _require_unquantized_m3_caches("auto", torch.bfloat16, "fp8")
    # ``auto`` follows --dtype, so a float16 model must be caught on the
    # resolved dtype rather than sailing past a string-only check.
    with pytest.raises(NotImplementedError, match="resolved to"):
        _require_unquantized_m3_caches("auto", torch.float16, "bf16")
